"""
dashboard.py — HAI Digital Twin real-time monitoring dashboard.

Five intelligence layers:
  1. Live Anomaly Score  — RF attack probability gauge (0–1)
  2. Twin vs Actual      — predicted vs real PV overlay
  3. Reconstruction Error— per-step MSE timeline
  4. Root Cause Analysis — top PVs ranked by residual contribution
  5. Alert Feed          — debounced event log

Usage:
    streamlit run 05_detect/dashboard.py -- \
        --ckpt outputs/pipeline/gru_normal_only/gru_plant.pt
"""

import sys
import argparse
import numpy as np
import torch
import joblib
from pathlib import Path
from collections import deque
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCSequenceModel
from config import LOOPS, PV_COLS, HAIEND_COLS, PROCESSED_DATA_DIR

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH      = 128
CTRL_LOOPS = ['PC', 'LC', 'FC', 'TC', 'CC']

SCENARIO_NAMES  = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}
SCENARIO_COLORS = {0: "#2196F3", 1: "#FF5722", 2: "#E91E63", 3: "#9C27B0"}

OUT_DIR = ROOT / "outputs" / "monitor"

EXTRA_CHANNELS = {
    'PC': ['P1_PCV02D', 'P1_FT01',   'P1_TIT01'],
    'LC': ['P1_FT03',   'P1_FCV03D', 'P1_PCV01D'],
    'FC': ['P1_PIT01',  'P1_LIT01',  'P1_TIT03'],
    'TC': ['P1_FT02',   'P1_PIT02',  'P1_TIT02'],
    'CC': ['P1_PP04D',  'P1_FCV03D', 'P1_PCV02D'],
}

ATTACK_THRESHOLD = 0.50   # RF probability threshold for alert
DEBOUNCE_WINDOWS = 5      # suppress repeated alerts within this many windows


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_everything(ckpt_path_str: str):
    ckpt_path = Path(ckpt_path_str)

    data        = load_and_prepare_data()
    plant_data  = data['plant']
    ctrl_data   = data['ctrl']
    sensor_cols = data['metadata']['sensor_cols']

    plant_scaler = joblib.load(f"{PROCESSED_DATA_DIR}/scaler.pkl")
    npz = {s: np.load(f"{PROCESSED_DATA_DIR}/{s}_data.npz") for s in ("train", "val", "test")}
    col_idx = {c: i for i, c in enumerate(sensor_cols)}
    for ln, extra_cols in EXTRA_CHANNELS.items():
        for ec in extra_cols:
            if ec not in col_idx:
                continue
            ei = col_idx[ec]
            mean_e, scale_e = plant_scaler.mean_[ei], plant_scaler.scale_[ei]
            for split, arr in npz.items():
                raw = arr['X'][:, :, [ei]].astype(np.float32)
                ctrl_data[ln][f'X_{split}'] = np.concatenate(
                    [ctrl_data[ln][f'X_{split}'], (raw - mean_e) / scale_e], axis=-1)

    N_PLANT_IN  = plant_data['n_plant_in']
    N_PV        = plant_data['n_pv']
    N_HAIEND    = plant_data['n_haiend']
    N_SCENARIOS = data['metadata']['n_scenarios']
    TARGET_LEN  = data['metadata']['target_len']

    ckpt        = torch.load(ckpt_path, map_location=DEVICE)
    hidden      = ckpt.get('hidden', 512)
    layers      = ckpt.get('layers', 2)
    n_haiend    = ckpt.get('n_haiend', N_HAIEND)

    plant_model = GRUPlant(
        n_plant_in=N_PLANT_IN, n_pv=N_PV,
        hidden=hidden, layers=layers,
        n_scenarios=N_SCENARIOS, dropout=0.0,
        n_haiend=n_haiend,
    ).to(DEVICE)
    plant_model.load_state_dict(ckpt['model_state'], strict=False)
    plant_model.eval()

    CTRL_HIDDEN = {'PC': 64, 'LC': 64, 'FC': 128, 'TC': 64, 'CC': 64}
    ctrl_models = {}
    ckpt_dir = ckpt_path.parent
    for ln in CTRL_LOOPS:
        n_in = ctrl_data[ln]['X_train'].shape[-1]
        h    = CTRL_HIDDEN[ln]
        if ln == 'CC':
            m = CCSequenceModel(n_inputs=n_in, hidden=h, layers=2, dropout=0.0, output_len=TARGET_LEN).to(DEVICE)
        else:
            m = GRUController(n_inputs=n_in, hidden=h, layers=2, dropout=0.0, output_len=TARGET_LEN).to(DEVICE)
        p = ckpt_dir / f"gru_ctrl_{ln.lower()}.pt"
        if p.exists():
            c = torch.load(p, map_location=DEVICE)
            m.load_state_dict(c['model_state'], strict=False)
        m.eval()
        ctrl_models[ln] = m

    pv_set      = set(PV_COLS)
    non_pv_cols = [c for c in sensor_cols if c not in pv_set]
    col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}
    ctrl_cv_col_idx = {ln: col_to_idx[LOOPS[ln].cv]
                       for ln in CTRL_LOOPS if LOOPS[ln].cv in col_to_idx}

    return data, plant_model, ctrl_models, ctrl_cv_col_idx, TARGET_LEN, N_PV, N_HAIEND


@st.cache_resource
def compute_all_residuals(_data, _plant_model, _ctrl_models, _ctrl_cv_col_idx, TARGET_LEN, N_PV):
    plant_data = _data['plant']
    ctrl_data  = _data['ctrl']

    def run_inference(X, X_cv_tgt, pv_init, split):
        N = len(X)
        out = np.zeros((N, TARGET_LEN, N_PV), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, N, BATCH):
                sl        = slice(i, i + BATCH)
                x_cv_b    = torch.tensor(X[sl]).float().to(DEVICE)
                xct_b     = torch.tensor(X_cv_tgt[sl]).float().to(DEVICE).clone()
                pv_init_b = torch.tensor(pv_init[sl]).float().to(DEVICE)
                sc_b      = torch.zeros(x_cv_b.size(0), dtype=torch.long).to(DEVICE)
                for ln in CTRL_LOOPS:
                    if ln not in _ctrl_cv_col_idx:
                        continue
                    Xc      = torch.tensor(ctrl_data[ln][f'X_{split}'][sl]).float().to(DEVICE)
                    cv_pred = _ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
                    ci      = _ctrl_cv_col_idx[ln]
                    xct_b[:, :, ci:ci+1] = cv_pred
                pv_seq, _ = _plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
                out[i:i + x_cv_b.size(0)] = pv_seq.cpu().numpy()
        return out

    pred_val  = run_inference(plant_data['X_val'],  plant_data['X_cv_target_val'],  plant_data['pv_init_val'],  'val')
    pred_test = run_inference(plant_data['X_test'], plant_data['X_cv_target_test'], plant_data['pv_init_test'], 'test')

    res_val  = plant_data['pv_target_val']  - pred_val
    res_test = plant_data['pv_target_test'] - pred_test

    # threshold from normal val windows
    normal_mean_abs = np.abs(res_val).mean(axis=-1).mean(axis=-1)
    threshold       = float(np.percentile(normal_mean_abs, 99.5))

    return res_val, res_test, pred_test, threshold


def extract_features(res: np.ndarray) -> np.ndarray:
    """(N, T, K) → (N, K*5) feature matrix for RF classifier."""
    feats = []
    for k in range(res.shape[-1]):
        r = res[:, :, k]
        feats.append(np.abs(r).mean(axis=1))
        feats.append(np.abs(r).max(axis=1))
        feats.append(r.std(axis=1))
        feats.append(r.mean(axis=1))
        feats.append(np.diff(r, axis=1).mean(axis=1))
    return np.stack(feats, axis=1)


# ── Gauge plot ─────────────────────────────────────────────────────────────────

def gauge_fig(prob: float) -> plt.Figure:
    """Draw a half-circle speedometer gauge for attack probability."""
    fig, ax = plt.subplots(figsize=(3.2, 1.8), subplot_kw=dict(aspect='equal'))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')

    # background arc
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='#e0e0e0', linewidth=12, solid_capstyle='round')

    # coloured arc up to prob
    theta_fill = np.linspace(np.pi, np.pi - prob * np.pi, 200)
    color = '#4CAF50' if prob < 0.35 else ('#FF9800' if prob < ATTACK_THRESHOLD else '#F44336')
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color, linewidth=12, solid_capstyle='round')

    # needle
    angle = np.pi - prob * np.pi
    ax.plot([0, 0.7 * np.cos(angle)], [0, 0.7 * np.sin(angle)], color='#333', linewidth=2.5)
    ax.add_patch(plt.Circle((0, 0), 0.06, color='#333'))

    ax.text(0, 0.38, f"{prob:.0%}", ha='center', va='center', fontsize=18, fontweight='bold', color=color)
    ax.text(0, 0.15, "Attack Probability", ha='center', va='center', fontsize=7, color='#666')
    ax.text(-1.05, -0.05, "0%",   fontsize=7, color='#888')
    ax.text( 0.85, -0.05, "100%", fontsize=7, color='#888')
    ax.axvline(0, color='#FF9800', linewidth=0.8, linestyle='--', ymin=0.05, ymax=0.55)
    ax.text(0.05, -0.15, "50%", fontsize=6, color='#FF9800')

    fig.patch.set_facecolor('none')
    fig.tight_layout(pad=0)
    return fig


# ── Main app ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="outputs/pipeline/gru_normal_only/gru_plant.pt")
    try:
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace(ckpt="outputs/pipeline/gru_normal_only/gru_plant.pt")

    ckpt_path = ROOT / args.ckpt

    st.set_page_config(
        page_title="HAI Digital Twin — Security Monitor",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ─────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .alert-card { background:#fff3cd; border-left:4px solid #FF9800;
                  border-radius:6px; padding:10px 14px; margin:4px 0; font-size:0.85em; }
    .critical-card { background:#fde8e8; border-left:4px solid #F44336;
                     border-radius:6px; padding:10px 14px; margin:4px 0; font-size:0.85em; }
    .ok-card { background:#e8f5e9; border-left:4px solid #4CAF50;
               border-radius:6px; padding:10px 14px; margin:4px 0; font-size:0.85em; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏭 HAI Digital Twin")
        st.caption("Cyber-Physical Security Monitor")
        st.divider()

        st.markdown(f"**Checkpoint:** `{ckpt_path.name}`")
        st.markdown(f"**Device:** `{DEVICE}`")
        st.divider()

        st.markdown("### Playback Controls")
        auto_play = st.toggle("Auto-play (live simulation)", value=False)
        play_speed = st.slider("Speed (windows/sec)", 0.5, 5.0, 1.0, step=0.5)
        st.divider()

        st.markdown("### Detection Settings")
        pct_slider = st.slider("Threshold percentile", 90, 99, 99,
                               help="Computed from normal validation residuals")

    # ── Load data & models ─────────────────────────────────────────────────────
    with st.spinner("Loading GRU digital twin and running inference…"):
        data, plant_model, ctrl_models, ctrl_cv_col_idx, TARGET_LEN, N_PV, N_HAIEND = \
            load_everything(str(ckpt_path))
        res_val, res_test, pred_test, threshold_default = \
            compute_all_residuals(data, plant_model, ctrl_models, ctrl_cv_col_idx, TARGET_LEN, N_PV)

    # Recompute threshold from slider
    normal_mean_abs = np.abs(res_val).mean(axis=-1).mean(axis=-1)
    threshold = float(np.percentile(normal_mean_abs, pct_slider))

    plant_data    = data['plant']
    scenario_test = plant_data['scenario_test']
    pv_target     = plant_data['pv_target_test']   # (N, T, N_PV) actual
    N_WINDOWS     = len(res_test)

    # Per-window summary MSE (scalar per window)
    per_window_mse = (res_test ** 2).mean(axis=(1, 2))  # (N,)

    # RF attack probability per window
    clf_path = OUT_DIR / "type_classifier.pkl"
    if clf_path.exists():
        clf_bundle  = joblib.load(clf_path)
        clf         = clf_bundle['clf']
        scaler_clf  = clf_bundle['scaler']
        X_feats     = extract_features(res_test)
        X_s         = scaler_clf.transform(X_feats)
        # binary probability: P(attack) = P(not Normal)
        proba_all   = clf.predict_proba(X_s)
        # class 0 = Normal; attack prob = 1 - P(Normal)
        normal_class_idx = list(clf.classes_).index(0) if 0 in clf.classes_ else 0
        attack_prob_all  = 1.0 - proba_all[:, normal_class_idx]
        attack_type_all  = clf.predict(X_s)
    else:
        attack_prob_all = (per_window_mse - per_window_mse.min()) / \
                          (per_window_mse.max() - per_window_mse.min() + 1e-8)
        attack_type_all = np.zeros(N_WINDOWS, dtype=int)

    # ── Session state ──────────────────────────────────────────────────────────
    if 'window_idx' not in st.session_state:
        st.session_state.window_idx   = 0
        st.session_state.alert_log    = []   # list of dicts
        st.session_state.last_alert_w = -DEBOUNCE_WINDOWS - 1

    # Playback slider (always shown)
    st.markdown("## Live Simulation Playback")
    col_slider, col_scenario = st.columns([4, 1])
    with col_slider:
        window_idx = st.slider("Current window", 0, N_WINDOWS - 1,
                               st.session_state.window_idx, key="window_slider")
    with col_scenario:
        sc_id   = int(scenario_test[window_idx])
        sc_name = SCENARIO_NAMES[sc_id]
        sc_color = SCENARIO_COLORS[sc_id]
        st.markdown(
            f"<div style='padding:8px;border-radius:6px;background:{sc_color}22;"
            f"border:1px solid {sc_color};text-align:center'>"
            f"<b style='color:{sc_color}'>Ground truth</b><br>{sc_name}</div>",
            unsafe_allow_html=True,
        )

    st.session_state.window_idx = window_idx

    # Current window data
    prob_now  = float(attack_prob_all[window_idx])
    mse_now   = float(per_window_mse[window_idx])
    res_now   = res_test[window_idx]          # (T, N_PV)
    actual_now = pv_target[window_idx]        # (T, N_PV)
    pred_now   = actual_now - res_now         # (T, N_PV)

    # Alert logic (debounce)
    under_attack = prob_now >= ATTACK_THRESHOLD
    if under_attack and (window_idx - st.session_state.last_alert_w) > DEBOUNCE_WINDOWS:
        top_sensor = PV_COLS[int(np.abs(res_now).mean(axis=0).argmax())]
        st.session_state.alert_log.append({
            'window':     window_idx,
            'prob':       prob_now,
            'type':       SCENARIO_NAMES.get(int(attack_type_all[window_idx]), "?"),
            'top_sensor': top_sensor,
        })
        st.session_state.last_alert_w = window_idx

    # ── Layer 1: Gauge ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Layer 1 — Live Anomaly Score")
    g_col, status_col = st.columns([1, 2])
    with g_col:
        fig_g = gauge_fig(prob_now)
        st.pyplot(fig_g, use_container_width=False)
        plt.close(fig_g)
    with status_col:
        if prob_now >= ATTACK_THRESHOLD:
            st.markdown(
                f'<div class="critical-card">🚨 <b>ATTACK DETECTED</b> — '
                f'Probability: {prob_now:.1%}<br>'
                f'Predicted type: <b>{SCENARIO_NAMES.get(int(attack_type_all[window_idx]), "?")}</b>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="ok-card">✅ <b>NORMAL OPERATION</b> — '
                f'Anomaly probability: {prob_now:.1%}</div>',
                unsafe_allow_html=True)
        st.caption(f"Threshold: {ATTACK_THRESHOLD:.0%} | "
                   f"Step MSE: {mse_now:.5f} | "
                   f"Detection threshold: {threshold:.4f}")

    # ── Layer 2: Twin vs Actual ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### Layer 2 — Digital Twin vs Actual (Key PV Readings)")

    fig2, axes2 = plt.subplots(1, N_PV, figsize=(3.2 * N_PV, 3), squeeze=False)
    t_ax = np.arange(TARGET_LEN)
    for k, pv_name in enumerate(PV_COLS):
        ax = axes2[0, k]
        ax.plot(t_ax, actual_now[:, k], color=sc_color, linewidth=2, label='Actual')
        ax.plot(t_ax, pred_now[:, k],   color='black',   linewidth=1.4,
                linestyle='--', label='Twin prediction')
        ax.set_title(pv_name, fontsize=8)
        ax.set_xlabel("t (s)", fontsize=7)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.set_ylabel("Scaled value", fontsize=7)
            ax.legend(fontsize=6)
    fig2.suptitle("Solid = real plant  |  Dashed = digital twin prediction", fontsize=8)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Layer 3: Reconstruction Error (MSE timeline) ───────────────────────────
    st.divider()
    st.markdown("### Layer 3 — Reconstruction Error (Step MSE)")

    history_len = min(window_idx + 1, 200)
    history_idx = np.arange(max(0, window_idx - history_len + 1), window_idx + 1)
    mse_history = per_window_mse[history_idx]
    sc_history  = scenario_test[history_idx]

    fig3, ax3 = plt.subplots(figsize=(12, 2.5))
    ax3.fill_between(history_idx, mse_history,
                     alpha=0.25, color='steelblue')
    # colour by scenario
    for sc_id_h, sc_col_h in SCENARIO_COLORS.items():
        mask_h = (sc_history == sc_id_h)
        if mask_h.any():
            ax3.scatter(history_idx[mask_h], mse_history[mask_h],
                        color=sc_col_h, s=8, zorder=3, label=SCENARIO_NAMES[sc_id_h])
    ax3.axvline(window_idx, color='red', linewidth=1.5, linestyle='--', label='Now')
    ax3.set_xlabel("Window index", fontsize=8)
    ax3.set_ylabel("MSE", fontsize=8)
    ax3.set_title("Mean Squared Error between Twin and Plant (last 200 windows)", fontsize=9)
    ax3.legend(fontsize=6, ncol=5)
    ax3.tick_params(labelsize=7)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Layer 4: Root Cause Analysis ───────────────────────────────────────────
    st.divider()
    st.markdown("### Layer 4 — Root Cause Analysis (Top PVs by Residual Contribution)")

    sensor_contrib = np.abs(res_now).mean(axis=0)  # (N_PV,) mean |residual| per PV
    sorted_idx     = np.argsort(sensor_contrib)[::-1]
    sorted_names   = [PV_COLS[i] for i in sorted_idx]
    sorted_vals    = sensor_contrib[sorted_idx]
    bar_colors     = ['#F44336' if v > threshold else '#4CAF50' for v in sorted_vals]

    fig4, ax4 = plt.subplots(figsize=(8, 2.8))
    bars = ax4.barh(sorted_names[::-1], sorted_vals[::-1], color=bar_colors[::-1], edgecolor='white')
    ax4.axvline(threshold, color='orange', linestyle='--', linewidth=1.2, label=f'Threshold ({threshold:.4f})')
    ax4.set_xlabel("Mean |residual|", fontsize=8)
    ax4.set_title("Process variables ranked by deviation from digital twin", fontsize=9)
    ax4.legend(fontsize=7)
    ax4.tick_params(labelsize=7)
    for bar, val in zip(bars[::-1], sorted_vals[::-1]):
        ax4.text(val + 0.0003, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va='center', fontsize=7)
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.caption(
        "🔴 Red bars exceed the detection threshold — these sensors deviate from "
        "what the digital twin predicts. Inspect the corresponding physical equipment."
    )

    # ── Layer 5: Alert Feed ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Layer 5 — Alert Feed")

    if not st.session_state.alert_log:
        st.markdown('<div class="ok-card">No alerts triggered so far.</div>',
                    unsafe_allow_html=True)
    else:
        for entry in reversed(st.session_state.alert_log[-20:]):
            st.markdown(
                f'<div class="critical-card">'
                f'🚨 <b>Window {entry["window"]}</b> — '
                f'Attack probability: <b>{entry["prob"]:.1%}</b> | '
                f'Predicted type: <b>{entry["type"]}</b> | '
                f'Most-affected sensor: <b>{entry["top_sensor"]}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if st.button("Clear alert log"):
            st.session_state.alert_log = []
            st.rerun()

    # ── Auto-play ──────────────────────────────────────────────────────────────
    if auto_play:
        time.sleep(1.0 / play_speed)
        next_idx = (window_idx + 1) % N_WINDOWS
        st.session_state.window_idx = next_idx
        st.rerun()


if __name__ == "__main__":
    main()
