"""
dashboard.py — HAI Virtual Power Plant: Monitoring + Simulation + Attack Lab

Three tabs:
  Monitor              -- explore any test window, live threshold tuning, score timeline
  Virtual Plant        -- interactive process simulator: adjust SPs, observe PV response
                          P1 boiler physical topology diagram with live state coloring
  Attack Lab           -- inject synthetic cyber attacks, observe twin detection response

Run:
    streamlit run 03_model/dashboard.py
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from digital_twin import DigitalTwin
from config import PV_COLS, LOOPS
from sklearn.metrics import roc_auc_score, f1_score

GRU_DIR    = ROOT / "outputs" / "gru_scenario_weighted" / "gru_scenario_weighted"
DATA_DIR   = ROOT / "outputs" / "scaled_split"
PV_LABELS  = [c.replace("P1_", "") for c in PV_COLS]
CTRL_LOOPS = ['PC', 'LC', 'FC', 'TC', 'CC']

LOOP_PV = {
    'PC': ('PIT01', 0),
    'LC': ('LIT01', 1),
    'FC': ('FT03Z', 2),
    'TC': ('TIT01', 3),
    'CC': ('TIT03', 4),
}
LOOP_NAMES = {
    'PC': 'Pressure Control',
    'LC': 'Level Control',
    'FC': 'Flow Control',
    'TC': 'Temperature Control',
    'CC': 'Cooling Control',
}

st.set_page_config(
    page_title="HAI Virtual Power Plant",
    page_icon="🏭",
    layout="wide",
)


# ── Cached: load models + precompute all test predictions ─────────────────────

@st.cache_resource(show_spinner="Loading GRU Digital Twin … (first run only)")
def load_everything():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data   = load_and_prepare_data()
    twin   = DigitalTwin(GRU_DIR, device=device, data=data)

    raw_val  = np.load(str(DATA_DIR / "val_data.npz"))["X"]
    raw_test = np.load(str(DATA_DIR / "test_data.npz"))["X"]

    twin.calibrate(data, fpr_target=0.05, raw_val=raw_val, use_controllers=True)

    plant       = data['plant']
    sc_normal   = np.zeros_like(plant['scenario_test'])
    ctrl_test_6 = twin.build_ctrl_inputs(raw_test, data['ctrl'])

    results = twin.run_batch(
        plant['X_test'],
        plant['X_cv_target_test'],
        plant['pv_init_test'],
        sc_normal,
        plant['pv_target_test'],
        ctrl_data=ctrl_test_6,
    )
    return twin, data, plant['pv_target_test'], plant['attack_test'], results, ctrl_test_6


twin, data, pv_actual, attack_labels, results, ctrl_test_6 = load_everything()

pv_pred   = results["pv_pred"]
residuals = results["residuals"]
scores    = results["scores"]
N         = len(scores)

plant_data  = data['plant']
sensor_cols = data['metadata']['sensor_cols']
pv_set      = set(PV_COLS)
non_pv_cols = [c for c in sensor_cols if c not in pv_set]
input_cols  = non_pv_cols   # alias

# Precompute NRMSE
nrmse_vals = []
for i in range(len(PV_COLS)):
    pv_t = pv_actual[:, :, i]; pv_h = pv_pred[:, :, i]
    rng  = float(pv_t.max() - pv_t.min())
    nrmse_vals.append(float(np.sqrt(np.mean((pv_t - pv_h) ** 2)) / (rng + 1e-8)))
mean_nrmse = float(np.mean(nrmse_vals))

# SP value ranges (for simulator sliders)
sp_ranges = {}
for ln in CTRL_LOOPS:
    if ln in ctrl_test_6:
        all_sp = ctrl_test_6[ln]['X_test'][:, :, 0]
        sp_ranges[ln] = (float(all_sp.min()), float(all_sp.max()), float(all_sp.mean()))


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("HAI Digital Twin")
    st.markdown("---")
    st.markdown("**Model:** GRU-Scenario-Weighted")
    st.markdown(f"**Val loss:** `{twin._plant_val_loss:.6f}`")
    st.markdown(f"**Test windows:** {N}")
    st.markdown(f"**True attacks:** {int(attack_labels.sum())}")
    st.markdown("---")

    threshold = float(twin.threshold)

    is_attack_live = scores > threshold
    auroc = roc_auc_score(attack_labels, scores)
    f1    = f1_score(attack_labels, is_attack_live, zero_division=0)

    st.markdown("---")
    st.metric("AUROC",     f"{auroc:.4f}")
    st.metric("F1 Score",  f"{f1:.4f}")
    st.metric("Threshold", f"{threshold:.6f}")
    st.metric("Alerts",    f"{int(is_attack_live.sum())} / {int(attack_labels.sum())}")
    st.markdown("---")
    st.caption("HAI dataset — P1 Boiler (Emerson Ovation DCS)")
    st.caption("5 PVs: PIT01 · LIT01 · FT03Z · TIT01 · TIT03")
    st.caption("5 Loops: PC · LC · FC · TC · CC")


# ── Physical topology diagram ─────────────────────────────────────────────────

def _health_color(residual: float, max_res: float = 0.06) -> str:
    """Green → yellow → red based on absolute residual magnitude."""
    if residual is None:
        return '#4a9eff'
    r = min(1.0, max(0.0, abs(residual) / max_res))
    if r < 0.5:
        t = r * 2
        R, G, B = int(255 * t), 200, int(50 * (1 - t))
    else:
        t = (r - 0.5) * 2
        R, G, B = 255, int(200 * (1 - t)), 0
    return f'#{max(0,min(255,R)):02x}{max(0,min(255,G)):02x}{max(0,min(255,B)):02x}'


def draw_process_flow(
    cv_norm: dict = None,
    pv_res:  dict = None,
    score:   float = None,
    thresh:  float = None,
) -> plt.Figure:
    """
    Draw the P1 boiler physical process flow diagram.

    cv_norm : {col_name: 0..1}  valve/pump openness (coloured green→red)
    pv_res  : {pv_label: abs_residual}  sensor health coloring
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 15); ax.set_ylim(0, 7)
    ax.set_aspect('equal'); ax.axis('off')
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    cv  = cv_norm or {}
    res = pv_res  or {}

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _box(x, y, w, h, label, sub='', color='#1e3a5f', fs=8):
        r = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle='round,pad=0.06',
            facecolor=color, edgecolor='#4a9eff', linewidth=1.2, zorder=3,
        )
        ax.add_patch(r)
        ax.text(x, y + (0.09 if sub else 0), label,
                ha='center', va='center', color='white',
                fontsize=fs, fontweight='bold', zorder=4)
        if sub:
            ax.text(x, y - 0.18, sub,
                    ha='center', va='center', color='#999999', fontsize=6.5, zorder=4)

    def _tank(x, y, w, h, label, level=0.55):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle='square,pad=0',
            facecolor='#151525', edgecolor='#4a9eff', linewidth=1.5, zorder=2))
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - w/2 + 0.04, y - h/2 + 0.04), w - 0.08, (h - 0.08) * level,
            boxstyle='square,pad=0',
            facecolor='#1e5fa0', edgecolor='none', alpha=0.7, zorder=2))
        ax.text(x, y + h/2 + 0.18, label, ha='center', va='bottom',
                color='white', fontsize=8.5, fontweight='bold', zorder=4)

    def _valve(x, y, label, openness=0.5):
        color = plt.cm.RdYlGn(float(np.clip(openness, 0, 1)))
        diam  = plt.Polygon(
            [(x, y+0.20), (x+0.20, y), (x, y-0.20), (x-0.20, y)],
            facecolor=color, edgecolor='white', linewidth=0.8, zorder=3,
        )
        ax.add_patch(diam)
        ax.text(x, y - 0.36, label, ha='center', va='top',
                color='#cccccc', fontsize=6.5, zorder=4)

    def _sensor(x, y, label, color='#4a9eff'):
        ax.add_patch(plt.Circle((x, y), 0.22, color=color, zorder=3, alpha=0.9))
        ax.text(x, y, label, ha='center', va='center',
                color='white', fontsize=6.5, fontweight='bold', zorder=4)

    def _pipe(x1, y1, x2, y2, w=2.5):
        ax.plot([x1, x2], [y1, y2], color='#2a5f8f', lw=w, zorder=1, solid_capstyle='round')

    def _arr(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#4a9eff', lw=1.4), zorder=2)

    # ── Sensor colors from residuals ──────────────────────────────────────────
    c_pit01 = _health_color(res.get('PIT01'))
    c_lit01 = _health_color(res.get('LIT01'))
    c_ft03  = _health_color(res.get('FT03Z'))
    c_tit01 = _health_color(res.get('TIT01'))
    c_tit03 = _health_color(res.get('TIT03'))

    # Valve openness (0..1 from normalised CV)
    o_pcv01 = float(np.clip(cv.get('P1_PCV01D', 0.5), 0, 1))
    o_pcv02 = float(np.clip(cv.get('P1_PCV02D', 0.5), 0, 1))
    o_lcv01 = float(np.clip(cv.get('P1_LCV01D', 0.5), 0, 1))
    o_fcv03 = float(np.clip(cv.get('P1_FCV03D', 0.5), 0, 1))

    # ── Main water loop pipes ─────────────────────────────────────────────────
    _pipe(1.3, 3.5, 2.2, 3.5)          # TK01 → PP01
    _pipe(2.9, 3.5, 3.9, 3.5)          # PP01 → HEX01T
    _pipe(4.7, 3.5, 5.3, 3.5)          # HEX01T → PCV01
    _pipe(5.9, 3.5, 6.7, 3.5)          # PCV01 → sensors
    _pipe(6.7, 3.5, 6.7, 2.8)          # down to LCV01
    _pipe(6.7, 2.2, 6.7, 1.55)         # LCV01 → TK03
    _pipe(6.7, 1.0, 5.5, 1.0)          # TK03 → FCV03
    _pipe(5.0, 1.0, 3.5, 1.0)          # FCV03 → return
    _pipe(3.5, 1.0, 1.0, 1.0)          # return → TK01
    _pipe(1.0, 1.0, 1.0, 3.0)          # TK01 recirculation

    # PCV02 bypass
    _pipe(2.55, 3.25, 2.55, 2.4)
    _pipe(2.55, 2.4, 1.0, 2.4)
    _pipe(1.0, 2.4, 1.0, 3.0)

    # ── Heating circuit pipes ─────────────────────────────────────────────────
    _pipe(10.0, 5.8, 10.0, 4.7)        # HT01 → TK02
    _pipe(10.0, 3.85, 10.0, 3.1)       # TK02 → PP02
    _pipe(10.0, 2.6, 10.0, 2.0)        # PP02 → FCV01
    _pipe(10.0, 1.6, 8.3, 1.6)         # FCV01 → HEX01S
    _pipe(7.8, 1.6, 7.8, 3.5)          # HEX01S → HEX01T area
    _pipe(7.8, 3.5, 4.7, 3.5)          # connects to HEX01T

    # ── Components ────────────────────────────────────────────────────────────

    # Main water circuit
    _tank(1.0, 3.5, 0.85, 1.1, 'TK01\nMain Tank', level=0.60)
    _box(2.55, 3.5, 0.65, 0.42, 'PP01A/B', 'Pumps', color='#1a3560')
    _box(4.3, 3.5, 0.68, 0.9,  'HEX01T', 'Tube Side', color='#1f4a20')

    _valve(5.6, 3.5, 'PCV01', openness=o_pcv01)
    _valve(2.55, 2.6, 'PCV02\nbypass', openness=o_pcv02)

    _sensor(7.1, 3.85, 'PIT01', color=c_pit01)
    _sensor(7.1, 3.15, 'TIT01', color=c_tit01)

    _valve(6.7, 2.5, 'LCV01', openness=o_lcv01)
    _tank(6.7, 1.25, 0.75, 0.7, 'TK03\nReturn', level=0.40)
    _sensor(7.7, 1.25, 'LIT01', color=c_lit01)

    _valve(5.25, 1.0, 'FCV03', openness=o_fcv03)
    _sensor(4.3, 0.4, 'FT03', color=c_ft03)
    _pipe(5.25, 0.80, 4.3, 0.62)

    # Heating circuit
    _box(10.0, 6.1, 0.8, 0.42,  'HT01', 'Heater', color='#5a1515')
    _tank(10.0, 4.3, 0.85, 0.95, 'TK02\nHot Tank', level=0.50)
    _sensor(11.1, 4.3, 'TIT02', color='#4a9eff')
    _box(10.0, 2.85, 0.65, 0.42, 'PP02', 'Pump', color='#1a3560')
    _box(10.0, 1.8, 0.65, 0.35,  'FCV01', 'Heat Flow', color='#1a4f5f')
    _box(8.05, 1.6, 0.58, 0.55,  'HEX01S', 'Shell\nSide', color='#1f4a20')
    _sensor(11.1, 1.8, 'FT02',  color='#4a9eff')
    _sensor(11.1, 1.2, 'PIT02', color='#4a9eff')

    # Cooling
    _box(1.0, 5.5, 0.72, 0.42, 'PP04', 'Cool Pump', color='#1a3560')
    _sensor(2.5, 5.5, 'TIT03', color=c_tit03)
    _pipe(1.36, 5.5, 2.28, 5.5)
    _pipe(1.0, 5.28, 1.0, 4.08)

    # ── Flow arrows ───────────────────────────────────────────────────────────
    _arr(1.7, 3.5,  2.25, 3.5)
    _arr(2.88, 3.5, 3.95, 3.5)
    _arr(4.65, 3.5, 5.4, 3.5)
    _arr(5.78, 3.5, 6.68, 3.5)
    _arr(6.7, 2.75, 6.7, 2.2)
    _arr(6.7, 1.62, 6.7, 1.62)
    _arr(6.3, 1.0,  5.45, 1.0)
    _arr(4.8, 1.0,  3.6,  1.0)
    _arr(10.0, 5.9, 10.0, 4.78)
    _arr(10.0, 3.85, 10.0, 3.12)
    _arr(10.0, 2.62, 10.0, 1.98)
    _arr(9.68, 1.6, 8.34, 1.6)

    # ── Status badge ──────────────────────────────────────────────────────────
    if score is not None and thresh is not None:
        is_atk = score > thresh
        badge_color = '#cc2222' if is_atk else '#22cc55'
        badge_text  = 'ATTACK DETECTED' if is_atk else 'NORMAL'
        ax.text(13.0, 6.3, badge_text, ha='center', va='center',
                color=badge_color, fontsize=15, fontweight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.45', facecolor='#111111',
                          edgecolor=badge_color, linewidth=2.2))
        ax.text(13.0, 5.6, f'Score:  {score:.5f}',
                ha='center', color='#aaaaaa', fontsize=9, zorder=5)
        ax.text(13.0, 5.3, f'Thresh: {thresh:.5f}',
                ha='center', color='#666666', fontsize=9, zorder=5)

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.text(0.3, 6.85, 'P1 Boiler — Physical Process Topology',
            color='white', fontsize=11, fontweight='bold', zorder=5)
    ax.text(0.3, 6.5,
            '◆ Valve: green=open, red=closed    '
            '● Sensor: green=normal, red=high residual    '
            '□ Tank: blue level = fill fraction',
            color='#777777', fontsize=7, zorder=5)

    fig.tight_layout(pad=0.3)
    return fig


# ── Single-window simulate with custom SPs ────────────────────────────────────

def simulate_custom_sp(base_idx: int, sp_deltas: dict, scenario_id: int = 0) -> dict:
    """Re-run closed-loop for one window with shifted SP values and optional scenario."""
    X      = plant_data['X_test'][[base_idx]].copy()
    Xct    = plant_data['X_cv_target_test'][[base_idx]].copy()
    pv_ini = plant_data['pv_init_test'][[base_idx]]
    sc     = np.full(1, scenario_id, dtype=np.int64)
    pv_act = plant_data['pv_target_test'][[base_idx]]

    ctrl_mod = {}
    for ln in CTRL_LOOPS:
        if ln in ctrl_test_6:
            arr = ctrl_test_6[ln]['X_test'][[base_idx]].copy()
            if ln in sp_deltas and abs(sp_deltas[ln]) > 1e-9:
                arr[:, :, 0] = arr[:, :, 0] + sp_deltas[ln]
            ctrl_mod[ln] = {'X_test': arr}

    return twin.run_batch(X, Xct, pv_ini, sc, pv_act, ctrl_data=ctrl_mod)


def _baseline_1win(idx: int) -> dict:
    """Run (or reuse cached) baseline for a single window."""
    ctrl_1 = {ln: {'X_test': ctrl_test_6[ln]['X_test'][[idx]]}
               for ln in ctrl_test_6}
    return twin.run_batch(
        plant_data['X_test'][[idx]],
        plant_data['X_cv_target_test'][[idx]],
        plant_data['pv_init_test'][[idx]],
        np.zeros(1, dtype=np.int64),
        plant_data['pv_target_test'][[idx]],
        ctrl_data=ctrl_1,
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_monitor, tab_virt, tab_assist, tab_attack = st.tabs([
    "Monitor", "Virtual Plant Simulator", "Assistive Analysis", "Attack Lab"
])


# =============================================================================
# TAB 1 — MONITOR
# =============================================================================

with tab_monitor:
    st.title("HAI Industrial Process Digital Twin")
    st.caption(
        f"GRU-Scenario-Weighted + 6-input causal controllers  |  "
        f"Residual anomaly detection  |  AUROC = {auroc:.4f}"
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUROC",      f"{auroc:.4f}")
    c2.metric("F1 Score",   f"{f1:.4f}")
    c3.metric("Threshold",  f"{threshold:.6f}  (val-calibrated)")
    c4.metric("Mean NRMSE", f"{mean_nrmse*100:.3f}%")
    st.markdown("---")

    win_idx       = st.slider("Window index", 0, N - 1, 0, step=1, key="mon_win")
    win_score     = scores[win_idx]
    win_det       = win_score > threshold
    win_true      = bool(attack_labels[win_idx])

    cb, cs = st.columns([2, 1])
    with cb:
        if win_det:
            st.error(f"ATTACK DETECTED  —  Window {win_idx}  (score = {win_score:.6f})")
        else:
            st.success(f"NORMAL  —  Window {win_idx}  (score = {win_score:.6f})")
    with cs:
        st.metric("Anomaly Score", f"{win_score:.6f}",
                  delta=f"True: {'ATTACK' if win_true else 'NORMAL'}", delta_color="off")

    st.markdown("---")

    # PV plots
    st.subheader("Process Variables — Twin Prediction vs Actual")
    t = np.arange(pv_actual.shape[1])
    fig, axes = plt.subplots(1, len(PV_LABELS), figsize=(16, 3.2))
    for i, (ax, lbl) in enumerate(zip(axes, PV_LABELS)):
        ax.plot(t, pv_actual[win_idx, :, i], color='darkorange', lw=1.5, label='Actual')
        ax.plot(t, pv_pred[win_idx, :, i],   color='steelblue',  lw=1.5, ls='--', label='Twin')
        ax.fill_between(t, pv_actual[win_idx, :, i], pv_pred[win_idx, :, i],
                        alpha=0.2, color='red')
        ax.set_title(lbl, fontsize=11, fontweight='bold')
        ax.set_xlabel('Timestep', fontsize=8)
        ax.grid(True, ls='--', alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Residual plots
    st.subheader("Residuals — Actual minus Predicted")
    fig, axes = plt.subplots(1, len(PV_LABELS), figsize=(16, 2.5))
    for i, (ax, lbl) in enumerate(zip(axes, PV_LABELS)):
        res = residuals[win_idx, :, i]
        ax.plot(t, res, color='dimgray', lw=1.2)
        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.fill_between(t, res, 0, where=(res > 0), alpha=0.35, color='crimson')
        ax.fill_between(t, res, 0, where=(res < 0), alpha=0.35, color='steelblue')
        ax.set_title(lbl, fontsize=11, fontweight='bold')
        ax.grid(True, ls='--', alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # Score timeline
    st.subheader("Anomaly Score Timeline — All Test Windows")
    fig, ax = plt.subplots(figsize=(14, 2.8))
    x = np.arange(N)
    ax.plot(x, scores, color='steelblue', lw=0.9, alpha=0.9)
    ax.fill_between(x, scores, threshold, where=(scores > threshold),
                    color='crimson', alpha=0.45, label='Detected')
    in_att = False; s = 0
    for j, att in enumerate(attack_labels):
        if att and not in_att:
            s, in_att = j, True
        elif not att and in_att:
            ax.axvspan(s - 0.5, j - 0.5, color='red', alpha=0.12)
            in_att = False
    if in_att:
        ax.axvspan(s - 0.5, N - 0.5, color='red', alpha=0.12)
    ax.axhline(threshold, color='black', lw=1.3, ls='--',
               label=f'threshold = {threshold:.6f}')
    ax.axvline(win_idx, color='gold', lw=2.0, label=f'Window {win_idx}')
    ax.set_xlabel('Window index'); ax.set_ylabel('MSE score')
    ax.set_title('Red shading = true attack windows  |  Gold line = selected window')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, ls='--', alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Simulation Quality — NRMSE per Signal")
    fig, ax = plt.subplots(figsize=(8, 2.5))
    bars = ax.bar(PV_LABELS, nrmse_vals, color='steelblue', alpha=0.85)
    for bar, v in zip(bars, nrmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(nrmse_vals) * 0.01,
                f'{v*100:.2f}%', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('NRMSE')
    ax.set_title(f'Closed-Loop NRMSE  (mean = {mean_nrmse*100:.3f}%)')
    ax.grid(axis='y', ls='--', alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── PREDICTIVE: Drift Forecast ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Predictive Drift Forecast")
    st.caption(
        "Looks at the last 60 windows around the selected point and fits a linear trend. "
        "If the score is rising toward the threshold, it estimates how many windows remain."
    )

    look_back = 60
    w_start   = max(0, win_idx - look_back)
    w_end     = min(N, win_idx + look_back)
    local_x   = np.arange(w_start, w_end)
    local_s   = scores[w_start:w_end]

    # Fit linear trend on the local window
    coeffs = np.polyfit(local_x, local_s, 1)   # slope, intercept
    slope, intercept = coeffs
    trend_y = np.polyval(coeffs, local_x)

    # Forecast 40 windows ahead
    fcast_x = np.arange(w_end, min(N, w_end + 40))
    fcast_y = np.polyval(coeffs, fcast_x)

    # Time-to-threshold estimate
    if slope > 1e-10:
        steps_to_thresh = max(0.0, (threshold - float(np.polyval(coeffs, win_idx))) / slope)
        eta_str = f"~{int(steps_to_thresh)} windows"
    else:
        steps_to_thresh = None
        eta_str = "Not trending toward threshold"

    d1, d2, d3 = st.columns(3)
    d1.metric("Current Score",    f"{win_score:.6f}")
    d2.metric("Trend (per win)",  f"{slope:+.2e}",
              delta="rising" if slope > 1e-10 else "stable/falling",
              delta_color="inverse")
    d3.metric("ETA to threshold", eta_str,
              delta=f"threshold = {threshold:.5f}", delta_color="off")

    if slope > 1e-10 and steps_to_thresh is not None and steps_to_thresh < 100:
        st.warning(
            f"Score is rising at {slope:.2e}/window. "
            f"Threshold may be reached in ~{int(steps_to_thresh)} windows. "
            "Check the Assistive Analysis tab for root cause."
        )

    fig, ax = plt.subplots(figsize=(14, 2.8))
    ax.plot(local_x, local_s, color='steelblue', lw=1.0, alpha=0.85, label='Score')
    ax.plot(local_x, trend_y, color='gold',      lw=1.5, ls='--', label='Trend')
    if len(fcast_x):
        ax.plot(fcast_x, fcast_y, color='gold', lw=1.2, ls=':', alpha=0.6, label='Forecast')
        ax.fill_between(fcast_x, 0, fcast_y, alpha=0.08, color='gold')
    ax.fill_between(local_x, local_s, threshold,
                    where=(local_s > threshold), color='crimson', alpha=0.4)
    ax.axhline(threshold, color='black', lw=1.2, ls='--',
               label=f'threshold = {threshold:.5f}')
    ax.axvline(win_idx,   color='white', lw=1.5, alpha=0.6, label='Selected window')
    ax.set_xlabel('Window index'); ax.set_ylabel('MSE score')
    ax.set_title('Local score trend + 40-window forecast  (gold dashed = linear extrapolation)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, ls='--', alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =============================================================================
# TAB 2 — VIRTUAL PLANT SIMULATOR
# =============================================================================

with tab_virt:
    st.title("Virtual Power Plant Simulator")
    st.caption(
        "Modify setpoints for any P1 control loop and watch the GRU plant model "
        "simulate the boiler's physical response in real time."
    )
    st.markdown("---")

    # Window selector + topology
    col_win, col_stat = st.columns([3, 1])
    with col_win:
        virt_win = st.slider("Base test window", 0, N - 1, 0, step=1, key="virt_base")
    with col_stat:
        st.metric("Window score", f"{scores[virt_win]:.6f}",
                  delta="ATTACK" if scores[virt_win] > threshold else "NORMAL",
                  delta_color="off")

    # Build CV values for topology coloring from the precomputed test window
    cv_for_topo = {}
    for cv_col in ['P1_PCV01D', 'P1_PCV02D', 'P1_LCV01D', 'P1_FCV03D']:
        if cv_col in non_pv_cols:
            idx = non_pv_cols.index(cv_col)
            raw_cv = plant_data['X_cv_target_test'][virt_win, :, idx]
            cv_for_topo[cv_col] = float(np.clip(np.mean(raw_cv), 0, 1))

    res_mean = np.mean(np.abs(residuals[virt_win]), axis=0)   # (n_pv,)
    pv_res_dict = {PV_LABELS[i]: float(res_mean[i]) for i in range(len(PV_COLS))}

    st.subheader("Physical Process Topology — Current State")
    topo_fig = draw_process_flow(
        cv_norm=cv_for_topo,
        pv_res=pv_res_dict,
        score=scores[virt_win],
        thresh=threshold,
    )
    st.pyplot(topo_fig, use_container_width=True)
    plt.close(topo_fig)

    st.markdown("---")

    # ── SP controls ───────────────────────────────────────────────────────────

    st.subheader("Control Loop Setpoints — Adjust and Simulate")
    st.info(
        "Shift the setpoint (SP) for each loop below. The GRU controllers recompute "
        "the control outputs (CVs), and the GRU plant predicts the new PV trajectory. "
        "Compare **Simulated** (green) vs **Baseline** (blue) vs **Actual** (orange).",
        icon="⚙",
    )

    # ── GENERATIVE: Scenario Embedding Selector ───────────────────────────────
    SCENARIO_OPTIONS = {
        "Normal operation (Scenario 0)":           0,
        "Attack present — no plant response (1)":  1,
        "Attack present — plant responds (2)":     2,
        "AE attack — no plant response (3)":       3,
    }
    with st.expander("Generative Mode — Scenario Embedding", expanded=False):
        st.markdown(
            "The GRU plant was trained with **4 scenario embeddings**. "
            "Switch the embedding to generate synthetic sensor trajectories "
            "for different fault/attack conditions — without touching the real plant."
        )
        scenario_label = st.selectbox("Scenario embedding", list(SCENARIO_OPTIONS.keys()))
        scenario_id    = SCENARIO_OPTIONS[scenario_label]
        if scenario_id != 0:
            st.warning(
                f"Scenario {scenario_id} active — the plant model will generate PVs "
                "as if the plant were in that fault/attack condition."
            )
        else:
            st.success("Normal operation (default)")
    # ─────────────────────────────────────────────────────────────────────────

    cols_sp = st.columns(len(CTRL_LOOPS))
    sp_deltas = {}
    for j, ln in enumerate(CTRL_LOOPS):
        pv_name, _   = LOOP_PV[ln]
        lo, hi, _    = sp_ranges.get(ln, (-3.0, 3.0, 0.0))
        rng          = max(hi - lo, 0.1)
        with cols_sp[j]:
            st.markdown(f"**{ln}** — {LOOP_NAMES[ln]}")
            st.caption(f"→ {pv_name}")
            delta = st.slider(
                f"SP delta ({ln})",
                min_value=float(-rng * 0.6),
                max_value=float( rng * 0.6),
                value=0.0,
                step=float(rng / 80),
                format="%.3f",
                key=f"sp_{ln}",
                label_visibility="collapsed",
            )
            sp_deltas[ln] = delta
            if abs(delta) > 0.005:
                direction = "▲" if delta > 0 else "▼"
                st.caption(f"{direction} {delta:+.3f}")
            else:
                st.caption("No change")

    run_sim = st.button("Run Simulation", type="primary", use_container_width=True)

    # Session state for persisting simulation results
    if 'sim_result' not in st.session_state:
        st.session_state.sim_result   = None
        st.session_state.sim_win      = -1
        st.session_state.sim_deltas   = {}

    if run_sim:
        with st.spinner("Running closed-loop simulation…"):
            res_sim = simulate_custom_sp(virt_win, sp_deltas, scenario_id=scenario_id)
        st.session_state.sim_result = res_sim
        st.session_state.sim_win    = virt_win
        st.session_state.sim_deltas = sp_deltas.copy()

    if st.session_state.sim_result is not None:
        res_sim   = st.session_state.sim_result
        sim_win   = st.session_state.sim_win
        sim_delt  = st.session_state.sim_deltas

        score_base = float(scores[sim_win])
        score_sim  = float(res_sim["scores"][0])
        det_base   = score_base > threshold
        det_sim    = score_sim  > threshold

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline Score",  f"{score_base:.6f}",
                  delta="ATTACK" if det_base else "NORMAL", delta_color="off")
        m2.metric("Simulated Score", f"{score_sim:.6f}",
                  delta=f"{score_sim - score_base:+.6f}", delta_color="inverse")
        m3.metric("Twin Decision",
                  "ATTACK" if det_sim else "NORMAL",
                  delta=f"threshold = {threshold:.5f}", delta_color="off")
        m4.metric("True Label",
                  "ATTACK" if attack_labels[sim_win] else "NORMAL",
                  delta=f"Window {sim_win}", delta_color="off")

        st.markdown("---")

        # PV trajectory comparison
        st.markdown("**PV Trajectories — Baseline vs Simulated vs Actual**")
        t = np.arange(plant_data['pv_target_test'].shape[1])
        pv_act_w = plant_data['pv_target_test'][sim_win]

        fig, axes = plt.subplots(1, len(PV_LABELS), figsize=(16, 3.5))
        for i, (ax, lbl) in enumerate(zip(axes, PV_LABELS)):
            ax.plot(t, pv_act_w[:, i],
                    color='darkorange', lw=1.4, alpha=0.8, label='Actual')
            ax.plot(t, pv_pred[sim_win, :, i],
                    color='steelblue', lw=1.5, ls='--', label='Baseline')
            ax.plot(t, res_sim["pv_pred"][0, :, i],
                    color='limegreen', lw=2.0, ls=':', label='Simulated')
            ax.set_title(lbl, fontsize=11, fontweight='bold')
            ax.set_xlabel('Timestep', fontsize=8)
            ax.grid(True, ls='--', alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
                ax.set_ylabel('Norm. PV', fontsize=8)
        fig.suptitle(
            f"Window {sim_win}  |  Baseline: {score_base:.5f}  →  "
            f"Simulated: {score_sim:.5f}",
            fontsize=10, fontweight='bold',
        )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Residuals
        st.markdown("**Residuals — Baseline vs Simulated**")
        fig, axes = plt.subplots(1, len(PV_LABELS), figsize=(16, 2.8))
        for i, (ax, lbl) in enumerate(zip(axes, PV_LABELS)):
            r_b = residuals[sim_win, :, i]
            r_s = res_sim["residuals"][0, :, i]
            ax.plot(t, r_b, color='steelblue', lw=1.2, alpha=0.8, label='Baseline')
            ax.plot(t, r_s, color='limegreen',  lw=1.5, ls='--', label='Simulated')
            ax.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
            ax.set_title(lbl, fontsize=10, fontweight='bold')
            ax.grid(True, ls='--', alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8); ax.set_ylabel('Residual', fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # SP delta bar chart (only if something changed)
        active = {ln: v for ln, v in sim_delt.items() if abs(v) > 1e-4}
        if active:
            st.markdown("**Applied Setpoint Changes**")
            fig, ax = plt.subplots(figsize=(8, 2.5))
            lns  = list(active.keys())
            vals = [active[ln] for ln in lns]
            clrs = ['crimson' if v > 0 else 'steelblue' for v in vals]
            bars = ax.bar([LOOP_NAMES[ln] for ln in lns], vals, color=clrs, alpha=0.8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.002,
                        f'{v:+.3f}', ha='center', va='bottom',
                        fontsize=9, color='white')
            ax.axhline(0, color='white', lw=0.8)
            ax.set_ylabel('SP delta (normalised)')
            ax.set_title('Setpoint Perturbations Applied to P1 Control Loops')
            ax.grid(axis='y', ls='--', alpha=0.4)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Topology updated with simulated residuals
        st.markdown("**Updated Process Topology — Post-Simulation State**")
        sim_res_mean = np.mean(np.abs(res_sim["residuals"][0]), axis=0)
        sim_res_dict = {PV_LABELS[i]: float(sim_res_mean[i]) for i in range(len(PV_COLS))}
        topo2 = draw_process_flow(
            cv_norm=cv_for_topo,
            pv_res=sim_res_dict,
            score=score_sim,
            thresh=threshold,
        )
        st.pyplot(topo2, use_container_width=True)
        plt.close(topo2)

    else:
        st.info("Adjust setpoints above and click **Run Simulation** to see the plant respond.")
        with st.expander("What does each control loop do?"):
            st.markdown("""
| Loop | Signal | Physical Component | Function |
|------|--------|--------------------|----------|
| **PC** Pressure Control | PIT01 | PCV01 + PCV02 valves | Maintains boiler pressure by opening/closing pressure control valves |
| **LC** Level Control    | LIT01 | LCV01 valve          | Keeps TK03 return tank level stable via flow valve |
| **FC** Flow Control     | FT03Z | FCV03 valve          | Regulates recirculation flow rate back to main tank |
| **TC** Temperature Control | TIT01 | FCV01 + FCV02 | Controls heat exchanger shell-side flow to reach target tube temp |
| **CC** Cooling Control  | TIT03 | PP04 pump            | Activates cooling pump to maintain safe buffer tank temperature |

**Physical chain:** HT01 heats TK02 → PP02 pumps hot water through HEX01S (shell) →
HEX01T heats main water → PP01A/B circulate through PCV01 → LCV01 fills TK03 →
FCV03 recirculates back to TK01.
            """)


# =============================================================================
# TAB 3 — ASSISTIVE ANALYSIS
# =============================================================================

# PV index → controlling loop
PV_TO_LOOP = {0: 'PC', 1: 'LC', 2: 'FC', 3: 'TC', 4: 'CC'}
LOOP_CV    = {'PC': 'PCV01/PCV02', 'LC': 'LCV01', 'FC': 'FCV03',
              'TC': 'FCV01/FCV02', 'CC': 'PP04'}


def sp_recommendation(win_idx: int, target_loop: str, n_steps: int = 15) -> list:
    """Grid-search SP delta for target_loop to minimise anomaly score."""
    lo, hi, _ = sp_ranges.get(target_loop, (-3.0, 3.0, 0.0))
    rng    = max(hi - lo, 0.1)
    deltas = np.linspace(-rng * 0.5, rng * 0.5, n_steps)
    out    = []
    for d in deltas:
        r = simulate_custom_sp(win_idx, {target_loop: float(d)})
        out.append((float(d), float(r["scores"][0]), r))
    return out


with tab_assist:
    st.title("Assistive Analysis")
    st.caption(
        "When an anomaly is detected, this tab traces which control loop is the "
        "root cause and recommends a setpoint correction to bring the plant back to normal."
    )
    st.markdown("---")

    # Window selector
    ass_win = st.slider("Select window to analyse", 0, N - 1,
                        int(np.where(attack_labels)[0][0]) if attack_labels.sum() > 0 else 0,
                        step=1, key="ass_win")

    ass_score = scores[ass_win]
    ass_det   = ass_score > threshold
    ass_true  = bool(attack_labels[ass_win])

    ca, cb_col = st.columns([2, 1])
    with ca:
        if ass_det:
            st.error(f"ATTACK DETECTED — Window {ass_win}  (score = {ass_score:.6f})")
        else:
            st.success(f"NORMAL — Window {ass_win}  (score = {ass_score:.6f})")
    with cb_col:
        st.metric("True label", "ATTACK" if ass_true else "NORMAL",
                  delta=f"score = {ass_score:.6f}", delta_color="off")

    st.markdown("---")

    # ── Root Cause Attribution ────────────────────────────────────────────────
    st.subheader("Root Cause Attribution — Which Loop Is Responsible?")
    st.caption(
        "Per-PV mean-squared residual. The loop controlling the highest-residual PV "
        "is most likely the source of the deviation."
    )

    per_pv_mse = np.mean(residuals[ass_win] ** 2, axis=0)   # (n_pv,)
    per_pv_abs = np.mean(np.abs(residuals[ass_win]), axis=0)

    # Build attribution table
    attr_rows = []
    for pv_i, ln in PV_TO_LOOP.items():
        attr_rows.append({
            'Loop': ln,
            'PV Signal': PV_LABELS[pv_i],
            'Controller': LOOP_NAMES[ln],
            'Actuator': LOOP_CV[ln],
            'MSE': float(per_pv_mse[pv_i]),
            'Mean |residual|': float(per_pv_abs[pv_i]),
        })
    attr_rows.sort(key=lambda r: r['MSE'], reverse=True)
    root_loop = attr_rows[0]['Loop']

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 3))
    loop_lbls = [r['Loop'] for r in attr_rows]
    mse_vals  = [r['MSE'] for r in attr_rows]
    abs_vals  = [r['Mean |residual|'] for r in attr_rows]
    clrs = ['crimson'] + ['steelblue'] * (len(attr_rows) - 1)

    bars = axes[0].bar(loop_lbls, mse_vals, color=clrs, alpha=0.85)
    for bar, v in zip(bars, mse_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(mse_vals) * 0.01,
                     f'{v:.5f}', ha='center', va='bottom', fontsize=8)
    axes[0].set_title('Per-Loop MSE Contribution  (red = highest = root cause)')
    axes[0].set_ylabel('MSE'); axes[0].grid(axis='y', ls='--', alpha=0.4)

    bars2 = axes[1].bar(loop_lbls, abs_vals, color=clrs, alpha=0.85)
    for bar, v in zip(bars2, abs_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(abs_vals) * 0.01,
                     f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    axes[1].set_title('Per-Loop Mean |Residual|')
    axes[1].set_ylabel('Mean |residual|'); axes[1].grid(axis='y', ls='--', alpha=0.4)

    fig.suptitle(
        f"Window {ass_win}  |  Root cause: {root_loop} ({LOOP_NAMES[root_loop]}) "
        f"— actuator: {LOOP_CV[root_loop]}",
        fontsize=11, fontweight='bold',
    )
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Residual timeseries per PV
    st.markdown("**Residual Timeseries — Which PV Diverges First?**")
    t = np.arange(residuals.shape[1])
    fig, axes = plt.subplots(1, len(PV_LABELS), figsize=(16, 2.8))
    for i, (ax, lbl) in enumerate(zip(axes, PV_LABELS)):
        res_w = residuals[ass_win, :, i]
        color = 'crimson' if PV_TO_LOOP[i] == root_loop else 'steelblue'
        ax.plot(t, res_w, color=color, lw=1.4)
        ax.fill_between(t, res_w, 0, where=(res_w > 0), alpha=0.3, color='crimson')
        ax.fill_between(t, res_w, 0, where=(res_w < 0), alpha=0.3, color='steelblue')
        ax.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
        ax.set_title(f"{lbl}\n({PV_TO_LOOP[i]})", fontsize=9, fontweight='bold',
                     color='crimson' if PV_TO_LOOP[i] == root_loop else 'white')
        ax.grid(True, ls='--', alpha=0.3)
    fig.suptitle('Red = root cause loop', fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # ── SP Recommendation ─────────────────────────────────────────────────────
    st.subheader("SP Recommendation — Suggested Correction")
    st.caption(
        f"Grid-searching SP adjustments for the root cause loop "
        f"**{root_loop} ({LOOP_NAMES[root_loop]})** to find the delta that minimises the anomaly score."
    )

    rec_loop = st.selectbox(
        "Loop to optimise (default = root cause)",
        options=CTRL_LOOPS,
        index=CTRL_LOOPS.index(root_loop),
        key="rec_loop",
    )

    run_rec = st.button("Find Optimal Setpoint", type="primary", use_container_width=True)

    if 'rec_result' not in st.session_state:
        st.session_state.rec_result = None
        st.session_state.rec_win    = -1
        st.session_state.rec_loop   = ''

    if run_rec:
        with st.spinner(f"Searching SP space for {rec_loop}…"):
            rec_grid = sp_recommendation(ass_win, rec_loop, n_steps=15)
        st.session_state.rec_result = rec_grid
        st.session_state.rec_win    = ass_win
        st.session_state.rec_loop   = rec_loop

    if st.session_state.rec_result is not None and st.session_state.rec_win == ass_win:
        rec_grid  = st.session_state.rec_result
        rec_deltas  = [r[0] for r in rec_grid]
        rec_scores  = [r[1] for r in rec_grid]
        best_i      = int(np.argmin(rec_scores))
        best_delta  = rec_deltas[best_i]
        best_score  = rec_scores[best_i]
        orig_score  = float(scores[ass_win])

        r1, r2, r3 = st.columns(3)
        r1.metric("Original score",  f"{orig_score:.6f}")
        r2.metric("Best score found", f"{best_score:.6f}",
                  delta=f"{best_score - orig_score:+.6f}", delta_color="inverse")
        r3.metric("Recommended SP delta",
                  f"{best_delta:+.4f}",
                  delta=f"for loop {st.session_state.rec_loop}", delta_color="off")

        if best_score < threshold:
            st.success(
                f"Correction found: adjust **{st.session_state.rec_loop}** SP by "
                f"**{best_delta:+.4f}** → score drops from "
                f"{orig_score:.5f} to {best_score:.5f} (below threshold)."
            )
        else:
            st.warning(
                f"Best SP delta = {best_delta:+.4f} reduces score to {best_score:.5f} "
                "but does not fully recover — the anomaly may require multi-loop correction."
            )

        # Score vs SP delta curve
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(rec_deltas, rec_scores, color='steelblue', lw=2, marker='o', ms=5)
        ax.axhline(threshold, color='black', lw=1.2, ls='--',
                   label=f'threshold = {threshold:.5f}')
        ax.axhline(orig_score, color='darkorange', lw=1.0, ls=':',
                   label=f'original score = {orig_score:.5f}')
        ax.axvline(best_delta, color='limegreen', lw=1.5,
                   label=f'best delta = {best_delta:+.4f}')
        ax.scatter([best_delta], [best_score], color='limegreen', s=80, zorder=5)
        ax.fill_between(rec_deltas, rec_scores, threshold,
                        where=np.array(rec_scores) < threshold,
                        color='limegreen', alpha=0.15, label='Safe zone')
        ax.set_xlabel(f'SP delta for {st.session_state.rec_loop} (normalised units)')
        ax.set_ylabel('Anomaly score (MSE)')
        ax.set_title(f'SP Sweep — {st.session_state.rec_loop} ({LOOP_NAMES[st.session_state.rec_loop]})')
        ax.legend(fontsize=9)
        ax.grid(True, ls='--', alpha=0.4)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Show PV response at best delta vs original
        best_res = rec_grid[best_i][2]
        st.markdown("**PV Response at Recommended SP vs Original**")
        t = np.arange(plant_data['pv_target_test'].shape[1])
        fig, axes = plt.subplots(1, len(PV_LABELS), figsize=(16, 3.2))
        for i, (ax, lbl) in enumerate(zip(axes, PV_LABELS)):
            ax.plot(t, plant_data['pv_target_test'][ass_win, :, i],
                    color='darkorange', lw=1.2, alpha=0.7, label='Actual')
            ax.plot(t, pv_pred[ass_win, :, i],
                    color='steelblue', lw=1.4, ls='--', label='Original twin')
            ax.plot(t, best_res["pv_pred"][0, :, i],
                    color='limegreen', lw=1.8, ls=':', label='With correction')
            ax.set_title(lbl, fontsize=10, fontweight='bold')
            ax.grid(True, ls='--', alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7); ax.set_ylabel('Norm. PV', fontsize=8)
        fig.suptitle(
            f'SP correction: {st.session_state.rec_loop} delta = {best_delta:+.4f}  |  '
            f'Score: {orig_score:.5f} → {best_score:.5f}',
            fontsize=10, fontweight='bold',
        )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    else:
        st.info("Click **Find Optimal Setpoint** to run the SP grid search.")
        with st.expander("How the recommendation works"):
            st.markdown("""
The assistant sweeps **15 SP values** between −50% and +50% of the loop's operating range.

For each candidate SP it:
1. Rebuilds the controller input with the shifted setpoint
2. Runs the GRU controllers → new CV sequence
3. Runs the GRU plant → new PV trajectory
4. Computes the anomaly score (MSE between predicted and actual PVs)

The SP value that produces the **lowest score** (closest to normal plant behaviour)
is returned as the recommended correction.

If the score drops below the threshold → the correction is sufficient.
If not → the anomaly involves multiple loops and multi-loop correction may be needed.
            """)


# =============================================================================
# TAB 4 — ATTACK LAB
# =============================================================================

with tab_attack:
    st.title("Attack Lab — Cyber Attack Simulation")
    st.caption(
        "Inject a synthetic attack on any sensor channel and observe how the "
        "digital twin's residual-based detector responds."
    )
    st.markdown("---")

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.subheader("Attack Configuration")
        st.info(
            "The digital twin runs on clean data and predicts normal plant behavior. "
            "Here you secretly corrupt one sensor reading. "
            "If the residual between prediction and tampered data exceeds the threshold, "
            "an alert fires.",
            icon="⚠",
        )

        atk_win = st.slider("Window index", 0, N - 1, 0, step=1, key="atk_win")

        SENSITIVE = [
            'P1_FT03', 'P1_FCV01Z', 'P1_FCV03Z', 'P1_FT01', 'P1_FT02',
            'P1_PIT02', 'P1_TIT02', 'P1_PCV01D', 'P1_LCV01D', 'P1_FCV03D',
        ]
        top_cols   = [c for c in SENSITIVE   if c in input_cols]
        other_cols = [c for c in input_cols  if c not in top_cols]
        sel_col    = st.selectbox("Sensor to attack", top_cols + other_cols)
        sel_idx    = input_cols.index(sel_col)

        atk_type = st.radio(
            "Attack type",
            ["Offset (+delta)", "Scale (×f)", "Zero (flatline)", "Constant"],
        )

        if atk_type == "Offset (+delta)":
            magnitude = st.slider("Offset (normalized units)", -5.0, 5.0, 3.0, 0.1)
        elif atk_type == "Scale (×f)":
            magnitude = st.slider("Scale factor (0=zero, 1=no change)", 0.0, 3.0, 0.0, 0.1)
        elif atk_type == "Constant":
            magnitude = st.slider("Constant value", -3.0, 3.0, 0.0, 0.1)
        else:
            magnitude = 0.0

        adv = st.checkbox("Also perturb SP of associated loop", value=False)
        sp_perturb = 0.0
        if adv:
            sp_perturb = st.slider("SP shift (normalized)", -3.0, 3.0, 1.0, 0.1)

        run_attack = st.button("Launch Attack", type="primary", use_container_width=True)

    with col_r:
        st.subheader("Detection Results")

        if run_attack:
            X_orig   = plant_data['X_test'][[atk_win]].copy()
            Xct_orig = plant_data['X_cv_target_test'][[atk_win]].copy()
            X_mod    = X_orig.copy()
            Xct_mod  = Xct_orig.copy()

            def _apply(arr, idx, atype, mag):
                if atype == "Offset (+delta)":    arr[0, :, idx] += mag
                elif atype == "Scale (×f)":       arr[0, :, idx] *= mag
                elif atype == "Zero (flatline)":  arr[0, :, idx]  = 0.0
                elif atype == "Constant":         arr[0, :, idx]  = mag

            _apply(X_mod,   sel_idx, atk_type, magnitude)
            _apply(Xct_mod, sel_idx, atk_type, magnitude)

            pv_ini_w  = plant_data['pv_init_test'][[atk_win]]
            sc_w      = np.zeros(1, dtype=np.int64)
            pv_act_w  = plant_data['pv_target_test'][[atk_win]]

            ctrl_clean = {ln: {'X_test': ctrl_test_6[ln]['X_test'][[atk_win]]}
                          for ln in ctrl_test_6}
            ctrl_atk   = {ln: {'X_test': ctrl_test_6[ln]['X_test'][[atk_win]].copy()}
                          for ln in ctrl_test_6}
            if adv and abs(sp_perturb) > 1e-9:
                for ln in ctrl_atk:
                    ctrl_atk[ln]['X_test'][:, :, 0] += sp_perturb

            res_clean = twin.run_batch(X_orig,  Xct_orig, pv_ini_w, sc_w, pv_act_w,
                                       ctrl_data=ctrl_clean)
            res_atk   = twin.run_batch(X_mod,   Xct_mod,  pv_ini_w, sc_w, pv_act_w,
                                       ctrl_data=ctrl_atk)

            s_clean = float(res_clean["scores"][0])
            s_atk   = float(res_atk["scores"][0])
            d_clean = s_clean > threshold
            d_atk   = s_atk   > threshold
            delta   = s_atk - s_clean

            a1, a2, a3 = st.columns(3)
            a1.metric("Score — clean",  f"{s_clean:.6f}",
                      delta="FLAGGED" if d_clean else "normal", delta_color="inverse")
            a2.metric("Score — attack", f"{s_atk:.6f}",
                      delta=f"{delta:+.6f}", delta_color="inverse")
            a3.metric("Twin decision",
                      "ATTACK DETECTED" if d_atk else "NOT DETECTED",
                      delta=f"threshold = {threshold:.5f}", delta_color="off")

            if d_atk and not d_clean:
                st.success(
                    f"Twin caught the attack on **{sel_col}**!  "
                    f"Score: {s_clean:.5f} → {s_atk:.5f}"
                )
            elif d_atk and d_clean:
                st.warning("Window was already flagged before attack injection.")
            else:
                st.error(
                    f"Attack not detected — score {s_atk:.5f} < threshold {threshold:.5f}.  "
                    "Try **P1_FT03** with Offset +3 or higher."
                )

            st.markdown("---")

            t = np.arange(pv_act_w.shape[1])
            fig, axes = plt.subplots(2, len(PV_LABELS), figsize=(16, 6))
            for i, lbl in enumerate(PV_LABELS):
                ax  = axes[0, i]
                ax.plot(t, res_clean["pv_pred"][0, :, i],
                        color='steelblue', lw=1.5, label='Clean twin')
                ax.plot(t, res_atk["pv_pred"][0, :, i],
                        color='crimson',   lw=1.5, ls='--', label='Attack twin')
                ax.plot(t, pv_act_w[0, :, i],
                        color='darkorange', lw=1.0, alpha=0.6, label='Actual')
                ax.set_title(lbl, fontsize=10, fontweight='bold')
                ax.grid(True, ls='--', alpha=0.3)
                if i == 0:
                    ax.set_ylabel('PV', fontsize=9); ax.legend(fontsize=7)

                ax2 = axes[1, i]
                ax2.plot(t, res_clean["residuals"][0, :, i],
                         color='steelblue', lw=1.2, label='Clean residual')
                ax2.plot(t, res_atk["residuals"][0, :, i],
                         color='crimson', lw=1.2, ls='--', label='Attack residual')
                ax2.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
                ax2.grid(True, ls='--', alpha=0.3)
                if i == 0:
                    ax2.set_ylabel('Residual', fontsize=9); ax2.legend(fontsize=7)

            fig.suptitle(
                f"'{sel_col}'  {atk_type}  mag={magnitude:.2f}  |  "
                f"Score: {s_clean:.5f} → {s_atk:.5f}",
                fontsize=11, fontweight='bold',
            )
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("**MSE increase per PV channel**")
            ch_delta = (np.mean(res_atk["residuals"][0]**2, axis=0)
                        - np.mean(res_clean["residuals"][0]**2, axis=0))
            fig, ax = plt.subplots(figsize=(8, 2.5))
            clrs = ['crimson' if d > 0 else 'steelblue' for d in ch_delta]
            bars = ax.bar(PV_LABELS, ch_delta, color=clrs, alpha=0.85)
            for bar, v in zip(bars, ch_delta):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(abs(ch_delta).max(), 1e-9) * 0.02,
                        f'{v:.4f}', ha='center', va='bottom', fontsize=9)
            ax.axhline(0, color='white', lw=0.8)
            ax.set_ylabel('MSE increase')
            ax.set_title('Which PV signals are most affected?')
            ax.grid(axis='y', ls='--', alpha=0.4)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        else:
            st.info("Configure the attack on the left and click **Launch Attack**.")
            st.markdown("""
### HAI Cyber Attack Taxonomy

The HAI testbed has **47 AP attack scenarios** (direct signal manipulation)
and **8 AE attacks** (internal DCS block injection), all targeting P1.

---

### P1 Physical Attack Surfaces

| Sensor | Loop | Physical Node | Sensitivity |
|--------|------|---------------|-------------|
| **P1_FT03** | FC | FCV03 → FT03 flow path | **Very High** |
| **P1_PCV01D** | PC | PCV01 demand signal | **High** |
| **P1_LCV01D** | LC | LCV01 demand signal | **High** |
| **P1_FCV01Z** | TC | FCV01 feedback | Medium |
| **P1_TIT02** | TC | TK02 temperature | Medium |
| **P1_PIT02** | TC | FCV01 outlet pressure | Medium |

---

### Recommended Experiments

1. **P1_FT03** + Offset **+3.0** — strongest flow manipulation signal
2. **P1_PCV01D** + Zero flatline — pressure valve stuck open
3. **P1_LCV01D** + Constant 0.0 — level valve locked shut
4. Any sensor + Advanced SP perturb — combined stealthy attack

---

### How the DCS is Structured (from dcs_1001h)

```
PIT01 (sensor) → DCS block 1001.1 → PID chain
HMI injects SP → block 1001.5 → comparator
EWS can override → blocks 1001.14, 1001.16, 1001.21, 1001.23
Output → PCV01 demand + PCV02 demand
```
Attacks can enter at PIT01 (PV spoofing), SP (setpoint hijack),
or EWS override (direct CV manipulation).
            """)
