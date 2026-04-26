"""
monitor.py — Predictive attack monitor for the HAI digital twin.

Three outputs per sliding window:
  1. WHEN  — first timestep where residual exceeds normal bounds (early warning time)
  2. WHAT  — attack type classifier: Normal / AP_no / AP_with / AE_no
  3. HOW   — per-sensor residual trajectory showing which sensors are affected

Usage:
    python 05_detect/monitor.py --ckpt outputs/pipeline/gru_normal_only/gru_plant.pt
    python 05_detect/monitor.py --ckpt outputs/pipeline/gru_normal_scratch/gru_plant.pt
"""

import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCSequenceModel
from config import LOOPS, PV_COLS, HAIEND_COLS, PROCESSED_DATA_DIR

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH     = 128
CTRL_LOOPS = ['PC', 'LC', 'FC', 'TC', 'CC']
SCENARIO_NAMES = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}

OUT_DIR = ROOT / "outputs" / "monitor"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXTRA_CHANNELS = {
    'PC': ['P1_PCV02D', 'P1_FT01',   'P1_TIT01'],
    'LC': ['P1_FT03',   'P1_FCV03D', 'P1_PCV01D'],
    'FC': ['P1_PIT01',  'P1_LIT01',  'P1_TIT03'],
    'TC': ['P1_FT02',   'P1_PIT02',  'P1_TIT02'],
    'CC': ['P1_PP04D',  'P1_FCV03D', 'P1_PCV02D'],
}


# ── helpers ────────────────────────────────────────────────────────────────────

def augment_ctrl_data(ctrl_data, sensor_cols):
    plant_scaler = joblib.load(f"{PROCESSED_DATA_DIR}/scaler.pkl")
    npz = {s: np.load(f"{PROCESSED_DATA_DIR}/{s}_data.npz")
           for s in ("train", "val", "test")}
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


def load_models(ckpt_path: Path, data):
    plant_data  = data['plant']
    ctrl_data   = data['ctrl']
    N_PLANT_IN  = plant_data['n_plant_in']
    N_PV        = plant_data['n_pv']
    N_HAIEND    = plant_data['n_haiend']
    N_SCENARIOS = data['metadata']['n_scenarios']
    TARGET_LEN  = data['metadata']['target_len']

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    hidden     = ckpt.get('hidden', 512)
    layers     = ckpt.get('layers', 2)
    n_haiend   = ckpt.get('n_haiend', N_HAIEND)

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
            m = CCSequenceModel(n_inputs=n_in, hidden=h, layers=2,
                                dropout=0.0, output_len=TARGET_LEN).to(DEVICE)
        else:
            m = GRUController(n_inputs=n_in, hidden=h, layers=2,
                              dropout=0.0, output_len=TARGET_LEN).to(DEVICE)
        p = ckpt_dir / f"gru_ctrl_{ln.lower()}.pt"
        if p.exists():
            c = torch.load(p, map_location=DEVICE)
            m.load_state_dict(c['model_state'], strict=False)
        m.eval()
        ctrl_models[ln] = m

    return plant_model, ctrl_models, TARGET_LEN, N_PV, n_haiend


# ── 1. Streaming inference — residual per window ───────────────────────────────

def run_inference(plant_model, ctrl_models, ctrl_cv_col_idx,
                  X, X_cv_tgt, pv_init, scenario, ctrl_data, split, TARGET_LEN, N_PV):
    """Return (N, TARGET_LEN, N_PV) PV predictions for a split."""
    N = len(X)
    pv_preds = np.zeros((N, TARGET_LEN, N_PV), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl        = slice(i, i + BATCH)
            x_cv_b    = torch.tensor(X[sl]).float().to(DEVICE)
            xct_b     = torch.tensor(X_cv_tgt[sl]).float().to(DEVICE).clone()
            pv_init_b = torch.tensor(pv_init[sl]).float().to(DEVICE)
            sc_b      = torch.zeros(x_cv_b.size(0), dtype=torch.long).to(DEVICE)

            for ln in CTRL_LOOPS:
                if ln not in ctrl_cv_col_idx:
                    continue
                Xc      = torch.tensor(ctrl_data[ln][f'X_{split}'][sl]).float().to(DEVICE)
                cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
                ci      = ctrl_cv_col_idx[ln]
                xct_b[:, :, ci:ci+1] = cv_pred

            pv_seq, _ = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
            pv_preds[i:i + x_cv_b.size(0)] = pv_seq.cpu().numpy()
    return pv_preds


# ── 2. Alert time estimator ────────────────────────────────────────────────────

def estimate_alert_time(residuals: np.ndarray, threshold: float) -> np.ndarray:
    """
    For each window, find the first timestep where mean absolute residual
    across all PVs exceeds threshold.

    residuals : (N, TARGET_LEN, N_PV)
    threshold : float — computed from normal val windows (e.g. mean + 3*std)
    returns   : (N,) int — timestep of first alert (-1 = no alert)
    """
    mean_abs = np.abs(residuals).mean(axis=-1)   # (N, TARGET_LEN)
    alert_times = np.full(len(residuals), -1, dtype=np.int32)
    for i, row in enumerate(mean_abs):
        crossing = np.where(row > threshold)[0]
        if len(crossing) > 0:
            alert_times[i] = crossing[0]
    return alert_times


# ── 3. Residual feature extractor for classifier ───────────────────────────────

def extract_residual_features(residuals: np.ndarray) -> np.ndarray:
    """
    Summarize (N, TARGET_LEN, N_PV) residuals into a flat feature vector per window.

    Features per PV: mean_abs, max_abs, std, mean (signed), rate_of_change
    → N_PV * 5 features total
    """
    feats = []
    for k in range(residuals.shape[-1]):
        r = residuals[:, :, k]                        # (N, TARGET_LEN)
        feats.append(np.abs(r).mean(axis=1))          # mean absolute
        feats.append(np.abs(r).max(axis=1))           # max absolute
        feats.append(r.std(axis=1))                   # std
        feats.append(r.mean(axis=1))                  # signed mean
        feats.append(np.diff(r, axis=1).mean(axis=1)) # rate of change
    return np.stack(feats, axis=1)                    # (N, N_PV*5)


# ── main ───────────────────────────────────────────────────────────────────────

def main(ckpt_path: Path):
    print("=" * 60)
    print("HAI Digital Twin — Predictive Attack Monitor")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    data        = load_and_prepare_data()
    plant_data  = data['plant']
    ctrl_data   = data['ctrl']
    sensor_cols = data['metadata']['sensor_cols']
    TARGET_LEN  = data['metadata']['target_len']

    augment_ctrl_data(ctrl_data, sensor_cols)

    pv_set      = set(PV_COLS)
    non_pv_cols = [c for c in sensor_cols if c not in pv_set]
    col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}
    ctrl_cv_col_idx = {ln: col_to_idx[LOOPS[ln].cv]
                       for ln in CTRL_LOOPS if LOOPS[ln].cv in col_to_idx}

    # ── Load models ────────────────────────────────────────────────────────────
    print(f"\n[2] Loading model from {ckpt_path.name}...")
    plant_model, ctrl_models, TARGET_LEN, N_PV, N_HAIEND = load_models(ckpt_path, data)

    # ── Run inference on val (normal) and test (all scenarios) ─────────────────
    print("\n[3] Running inference...")

    pv_preds_val = run_inference(
        plant_model, ctrl_models, ctrl_cv_col_idx,
        plant_data['X_val'], plant_data['X_cv_target_val'],
        plant_data['pv_init_val'], plant_data['scenario_val'],
        ctrl_data, 'val', TARGET_LEN, N_PV)

    pv_preds_test = run_inference(
        plant_model, ctrl_models, ctrl_cv_col_idx,
        plant_data['X_test'], plant_data['X_cv_target_test'],
        plant_data['pv_init_test'], plant_data['scenario_test'],
        ctrl_data, 'test', TARGET_LEN, N_PV)

    residuals_val  = plant_data['pv_target_val']  - pv_preds_val
    residuals_test = plant_data['pv_target_test'] - pv_preds_test

    # ── Compute normal threshold ───────────────────────────────────────────────
    # Val windows are all Normal; test Normal windows may differ due to operating
    # conditions.  Combine both to get a stable threshold, then use the 99th
    # percentile so the false-positive rate on Normal is ≤ 1%.
    print("\n[4] Computing normal residual threshold...")
    normal_mean_abs_val  = np.abs(residuals_val).mean(axis=-1).mean(axis=-1)
    normal_mask_test     = (plant_data['scenario_test'] == 0)
    normal_mean_abs_test = np.abs(residuals_test[normal_mask_test]).mean(axis=-1).mean(axis=-1)
    normal_mean_abs      = np.concatenate([normal_mean_abs_val, normal_mean_abs_test])
    threshold = float(np.percentile(normal_mean_abs, 99))
    print(f"    Threshold (99th pct of val+test-normal residuals): {threshold:.4f}")

    # ── WHEN: alert time per test window ──────────────────────────────────────
    print("\n[5] Estimating alert times...")
    alert_times = estimate_alert_time(residuals_test, threshold)

    scenario_test = plant_data['scenario_test']
    print(f"\n    Alert time (seconds into forecast window) per scenario:")
    print(f"    {'Scenario':<12} {'Windows':<10} {'Alerted':<10} {'Recall':<10} {'Avg alert time (s)'}")
    alert_summary = {}
    for sc_id, sc_name in SCENARIO_NAMES.items():
        mask    = (scenario_test == sc_id)
        if mask.sum() == 0:
            continue
        at      = alert_times[mask]
        alerted = (at >= 0).sum()
        recall  = alerted / mask.sum()
        avg_t   = at[at >= 0].mean() if alerted > 0 else float('nan')
        alert_summary[sc_name] = {
            'windows': int(mask.sum()), 'alerted': int(alerted),
            'recall': float(recall), 'avg_alert_time_s': float(avg_t)
        }
        print(f"    {sc_name:<12} {mask.sum():<10} {alerted:<10} {recall:<10.2%} {avg_t:.1f}")

    # ── WHAT: type classifier ─────────────────────────────────────────────────
    # Attack windows only exist in the test split (model trained on normal-only
    # data, so the train split has no attack labels).  Use a stratified 50/50
    # split of the test set: train the RF on one half, evaluate on the other.
    print("\n[6] Training attack type classifier...")

    X_all = extract_residual_features(residuals_test)
    y_all = scenario_test

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    clf_tr_idx, clf_te_idx = next(sss.split(X_all, y_all))

    X_tr, y_tr = X_all[clf_tr_idx], y_all[clf_tr_idx]
    X_te, y_te = X_all[clf_te_idx], y_all[clf_te_idx]

    scaler_clf = StandardScaler()
    X_tr_s = scaler_clf.fit_transform(X_tr)
    X_te_s = scaler_clf.transform(X_te)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    clf.fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_te_s)

    print("\n    Classification report:")
    target_names = [SCENARIO_NAMES[i] for i in sorted(SCENARIO_NAMES)]
    print(classification_report(y_te, y_pred, target_names=target_names, zero_division=0))

    joblib.dump({'clf': clf, 'scaler': scaler_clf}, OUT_DIR / "type_classifier.pkl")

    # ── HOW: per-sensor residual plot per scenario ─────────────────────────────
    print("\n[7] Plotting sensor impact per scenario...")
    fig, axes = plt.subplots(len(SCENARIO_NAMES), N_PV,
                             figsize=(4 * N_PV, 3 * len(SCENARIO_NAMES)),
                             sharey='row')
    for row, (sc_id, sc_name) in enumerate(SCENARIO_NAMES.items()):
        mask = (scenario_test == sc_id)
        if mask.sum() == 0:
            continue
        mean_res = np.abs(residuals_test[mask]).mean(axis=0)  # (TARGET_LEN, N_PV)
        for col, pv_name in enumerate(PV_COLS):
            ax = axes[row, col]
            ax.plot(mean_res[:, col], color='crimson' if sc_id > 0 else 'steelblue')
            ax.axhline(threshold, color='orange', linestyle='--', linewidth=0.8, label='threshold')
            ax.set_title(f"{sc_name}\n{pv_name}", fontsize=8)
            ax.set_xlabel("t (s)")
            if col == 0:
                ax.set_ylabel("mean |residual|")
    fig.suptitle("Sensor Impact per Scenario (mean absolute residual over forecast window)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sensor_impact.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {OUT_DIR}/sensor_impact.png")

    # ── Evaluation plots ───────────────────────────────────────────────────────
    print("\n[8] Generating evaluation plots...")

    SC_COLORS = {0: '#2196F3', 1: '#FF5722', 2: '#E91E63', 3: '#9C27B0'}

    # Plot 1: Alert time histogram per scenario
    fig_at, axes_at = plt.subplots(1, len(SCENARIO_NAMES), figsize=(14, 3.5), sharey=False)
    for i, (sc_id, sc_name) in enumerate(SCENARIO_NAMES.items()):
        mask  = (scenario_test == sc_id)
        at    = alert_times[mask]
        valid = at[at >= 0]
        ax    = axes_at[i]
        if len(valid) > 0:
            ax.hist(valid, bins=20, color=SC_COLORS[sc_id], edgecolor='white', linewidth=0.5)
        recall = len(valid) / mask.sum() if mask.sum() > 0 else 0
        ax.set_title(f"{sc_name}\nRecall={recall:.0%}", fontsize=9)
        ax.set_xlabel("Alert timestep (s)", fontsize=8)
        ax.set_ylabel("Windows", fontsize=8)
        ax.tick_params(labelsize=7)
    fig_at.suptitle("WHEN — Distribution of alert times per scenario", fontsize=10)
    fig_at.tight_layout()
    fig_at.savefig(OUT_DIR / "alert_time_histogram.png", dpi=150, bbox_inches='tight')
    plt.close(fig_at)
    print(f"    Saved: {OUT_DIR}/alert_time_histogram.png")

    # Plot 2: Residual distribution boxplot per scenario
    fig_bp, ax_bp = plt.subplots(figsize=(8, 4))
    per_window_mean_abs = np.abs(residuals_test).mean(axis=-1).mean(axis=-1)
    bp_data  = [per_window_mean_abs[scenario_test == sc_id] for sc_id in SCENARIO_NAMES]
    bp_labels = list(SCENARIO_NAMES.values())
    bp = ax_bp.boxplot(bp_data, patch_artist=True, notch=False)
    for patch, sc_id in zip(bp['boxes'], SCENARIO_NAMES):
        patch.set_facecolor(SC_COLORS[sc_id])
        patch.set_alpha(0.7)
    ax_bp.axhline(threshold, color='orange', linestyle='--', linewidth=1.2, label=f'Threshold ({threshold:.4f})')
    ax_bp.set_xticklabels(bp_labels, fontsize=9)
    ax_bp.set_ylabel("Mean |residual|", fontsize=9)
    ax_bp.set_title("Residual distribution by scenario", fontsize=10)
    ax_bp.legend(fontsize=8)
    fig_bp.tight_layout()
    fig_bp.savefig(OUT_DIR / "residual_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close(fig_bp)
    print(f"    Saved: {OUT_DIR}/residual_boxplot.png")

    # Plot 3: Confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    target_names_cm = [SCENARIO_NAMES[i] for i in sorted(SCENARIO_NAMES)]
    cm  = confusion_matrix(y_te, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names_cm)
    disp.plot(ax=ax_cm, colorbar=False, cmap='Blues')
    ax_cm.set_title("WHAT — Attack type classifier (confusion matrix)", fontsize=10)
    fig_cm.tight_layout()
    fig_cm.savefig(OUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close(fig_cm)
    print(f"    Saved: {OUT_DIR}/confusion_matrix.png")

    # Plot 4: Feature importance (top 15)
    feat_names = []
    for pv in PV_COLS:
        for fn in ['mean_abs', 'max_abs', 'std', 'signed_mean', 'rate_of_change']:
            feat_names.append(f"{pv}_{fn}")
    importances = clf.feature_importances_
    top_idx  = np.argsort(importances)[::-1][:15]
    top_names = [feat_names[i] for i in top_idx]
    top_vals  = importances[top_idx]
    fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
    ax_fi.barh(top_names[::-1], top_vals[::-1], color='steelblue')
    ax_fi.set_xlabel("Importance", fontsize=8)
    ax_fi.set_title("Top 15 residual features driving attack classification", fontsize=10)
    ax_fi.tick_params(labelsize=7)
    fig_fi.tight_layout()
    fig_fi.savefig(OUT_DIR / "feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close(fig_fi)
    print(f"    Saved: {OUT_DIR}/feature_importance.png")

    # Plot 5: Recall summary bar chart
    sc_names_list  = list(alert_summary.keys())
    recall_vals    = [alert_summary[s]['recall'] for s in sc_names_list]
    avg_alert_vals = [alert_summary[s]['avg_alert_time_s'] for s in sc_names_list]
    colors = [SC_COLORS[i] for i in range(len(sc_names_list))]

    fig_rec, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(10, 4))
    ax_r1.bar(sc_names_list, recall_vals, color=colors, edgecolor='white')
    ax_r1.set_ylabel("Recall", fontsize=9)
    ax_r1.set_title("Detection recall per scenario", fontsize=10)
    ax_r1.set_ylim(0, 1.05)
    for i, v in enumerate(recall_vals):
        ax_r1.text(i, v + 0.02, f"{v:.0%}", ha='center', fontsize=8)

    valid_at = [(s, t) for s, t in zip(sc_names_list, avg_alert_vals) if not np.isnan(t)]
    if valid_at:
        names_at, vals_at = zip(*valid_at)
        bar_colors_at = [SC_COLORS[list(SCENARIO_NAMES.values()).index(n)] for n in names_at]
        ax_r2.bar(names_at, vals_at, color=bar_colors_at, edgecolor='white')
        ax_r2.set_ylabel("Avg alert time (s)", fontsize=9)
        ax_r2.set_title("Average early-warning time per scenario", fontsize=10)
        for i, v in enumerate(vals_at):
            ax_r2.text(i, v + 1, f"{v:.1f}s", ha='center', fontsize=8)

    fig_rec.suptitle("WHEN — Detection performance summary", fontsize=11)
    fig_rec.tight_layout()
    fig_rec.savefig(OUT_DIR / "detection_summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig_rec)
    print(f"    Saved: {OUT_DIR}/detection_summary.png")

    # ── Plot 6: Predicted vs Actual during attack windows ─────────────────────
    print("\n[9] Plotting predicted vs actual signals during attacks...")

    for sc_id, sc_name in SCENARIO_NAMES.items():
        if sc_id == 0:
            continue  # skip normal
        mask = np.where(scenario_test == sc_id)[0]
        if len(mask) == 0:
            continue

        # pick the window with the highest residual (most visible attack)
        best_win = mask[np.abs(residuals_test[mask]).mean(axis=(1, 2)).argmax()]
        actual   = plant_data['pv_target_test'][best_win]   # (TARGET_LEN, N_PV)
        pred     = actual - residuals_test[best_win]        # (TARGET_LEN, N_PV)
        at_w     = int(alert_times[best_win])
        t_ax     = np.arange(actual.shape[0])

        fig_pa, axes_pa = plt.subplots(1, N_PV, figsize=(3.5 * N_PV, 3.5), squeeze=False)
        color = SC_COLORS[sc_id]
        for k, pv_name in enumerate(PV_COLS):
            ax = axes_pa[0, k]
            ax.plot(t_ax, actual[:, k], color=color,   linewidth=2,   label='Actual (real plant)')
            ax.plot(t_ax, pred[:, k],   color='black', linewidth=1.4, linestyle='--', label='Predicted (twin)')
            ax.fill_between(t_ax, actual[:, k], pred[:, k],
                            alpha=0.15, color='red', label='Residual')
            if at_w >= 0:
                ax.axvline(at_w, color='orange', linewidth=1.5,
                           linestyle=':', label=f'Alert @ t={at_w}s')
            ax.set_title(pv_name, fontsize=8)
            ax.set_xlabel("t (s)", fontsize=7)
            ax.tick_params(labelsize=6)
            if k == 0:
                ax.set_ylabel("Scaled value", fontsize=7)
                ax.legend(fontsize=5.5)

        fig_pa.suptitle(
            f"Predicted vs Actual — {sc_name}  "
            f"(window {best_win}, alert @ t={at_w}s)",
            fontsize=10, color=color
        )
        fig_pa.tight_layout()
        fname = OUT_DIR / f"pred_vs_actual_{sc_name}.png"
        fig_pa.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig_pa)
        print(f"    Saved: {fname}")

    # ── Save summary ───────────────────────────────────────────────────────────
    summary = {
        "checkpoint": str(ckpt_path),
        "threshold":  float(threshold),
        "alert_summary": alert_summary,
    }
    with open(OUT_DIR / "monitor_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n    Saved: {OUT_DIR}/monitor_results.json")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="outputs/pipeline/gru_normal_only/gru_plant.pt",
                        help="Path to gru_plant.pt checkpoint")
    args = parser.parse_args()
    main(Path(args.ckpt))
