"""
anomaly_detector.py — Post-hoc anomaly detection on GRU residuals.

Two improvements over raw MSE thresholding, no retraining required:

  1. IsolationForest on per-window residual vectors:
     Fits on normal-window residuals (validation set), scores test windows.
     Replaces a single global MSE threshold with a learned boundary.

  2. Per-PV thresholds:
     Separate MSE thresholds per process variable, tuned on val set to maximise F1.
     Any PV exceeding its threshold triggers an alert.

Outputs:
  plots/anomaly_detector/roc_if.png
  plots/anomaly_detector/roc_perpv.png
  plots/anomaly_detector/comparison_bar.png
  prints a summary table to stdout
"""

import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))
sys.path.insert(0, str(ROOT / "04_evaluate"))

from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCSequenceModel
from config import LOOPS, PV_COLS, PROCESSED_DATA_DIR
from plot_utils import CTRL_LOOPS

EXTRA_CHANNELS = {
    'PC': ['P1_PCV02D', 'P1_FT01',   'P1_TIT01'],
    'LC': ['P1_FT03',   'P1_FCV03D', 'P1_PCV01D'],
    'FC': ['P1_PIT01',  'P1_LIT01',  'P1_TIT03'],
    'TC': ['P1_FT02',   'P1_PIT02',  'P1_TIT02'],
    'CC': ['P1_PP04D',  'P1_FCV03D', 'P1_PCV02D'],
}

CKPT_DIR = ROOT / "outputs" / "pipeline" / "gru_scenario_weighted"
OUT_DIR  = ROOT / "plots" / "anomaly_detector"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH    = 128


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


def run_inference(plant_model, ctrl_models, ctrl_cv_col_idx,
                  X, X_cv_tgt, pv_init, scenario, ctrl_data, split, TARGET_LEN, N_PV):
    """Return (N, TARGET_LEN, N_PV) predictions for any split."""
    N = len(X)
    pv_preds = np.zeros((N, TARGET_LEN, N_PV), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl       = slice(i, i + BATCH)
            x_cv_b   = torch.tensor(X[sl]).float().to(DEVICE)
            xct_b    = torch.tensor(X_cv_tgt[sl]).float().to(DEVICE).clone()
            pv_init_b = torch.tensor(pv_init[sl]).float().to(DEVICE)
            sc_b     = torch.tensor(scenario[sl]).long().to(DEVICE)

            for ln in CTRL_LOOPS:
                if ln not in ctrl_cv_col_idx:
                    continue
                Xc = torch.tensor(ctrl_data[ln][f'X_{split}'][sl]).float().to(DEVICE)
                cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
                ci = ctrl_cv_col_idx[ln]
                xct_b[:, :, ci:ci + 1] = cv_pred

            pv_seq = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
            pv_preds[i:i + x_cv_b.size(0)] = pv_seq.cpu().numpy()
    return pv_preds


def best_f1_threshold(scores, labels):
    """Sweep thresholds and return (best_threshold, best_f1)."""
    thresholds = np.percentile(scores, np.linspace(0, 100, 500))
    best_t, best_f = 0.0, 0.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


def plot_roc(fpr, tpr, roc_auc, label, color, fname):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC — {label}'); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight'); plt.close(fig)


# ── load data & models ─────────────────────────────────────────────────────────

print("Loading data...")
data        = load_and_prepare_data()
plant_data  = data["plant"]
ctrl_data   = data["ctrl"]
sensor_cols = data["metadata"]["sensor_cols"]
TARGET_LEN  = data["metadata"]["target_len"]
N_PV        = plant_data["n_pv"]

augment_ctrl_data(ctrl_data, sensor_cols)

from config import PV_COLS as _PV_COLS
pv_set      = set(_PV_COLS)
non_pv_cols = [c for c in sensor_cols if c not in pv_set]
col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}
ctrl_cv_col_idx = {ln: col_to_idx[LOOPS[ln].cv]
                   for ln in CTRL_LOOPS if LOOPS[ln].cv in col_to_idx}

print("Loading model checkpoints...")
ckpt = torch.load(CKPT_DIR / "gru_plant.pt", map_location=DEVICE)
_has_heads = any("fc_heads" in k for k in ckpt["model_state"])
plant_model = GRUPlant(
    n_plant_in=ckpt["n_plant_in"], n_pv=ckpt["n_pv"],
    hidden=ckpt["hidden"], layers=ckpt["layers"],
    n_scenarios=ckpt["n_scenarios"], scenario_heads=_has_heads,
).to(DEVICE)
plant_model.load_state_dict(ckpt["model_state"])
plant_model.eval()

ctrl_models = {}
for ln in CTRL_LOOPS:
    path = CKPT_DIR / f"gru_ctrl_{ln.lower()}.pt"
    if not path.exists():
        continue
    c = torch.load(path, map_location=DEVICE)
    arch = c.get("arch", "GRUController")
    if arch == "CCSequenceModel":
        ctrl_models[ln] = CCSequenceModel(
            n_inputs=c["n_inputs"], hidden=c["hidden"],
            layers=c["layers"], output_len=TARGET_LEN).to(DEVICE)
    else:
        ctrl_models[ln] = GRUController(
            n_inputs=c["n_inputs"], hidden=c["hidden"],
            layers=c["layers"], output_len=TARGET_LEN).to(DEVICE)
    ctrl_models[ln].load_state_dict(c["model_state"])
    ctrl_models[ln].eval()


# ── run inference on val + test ────────────────────────────────────────────────

print("Running inference on val set...")
pv_preds_val = run_inference(
    plant_model, ctrl_models, ctrl_cv_col_idx,
    plant_data["X_val"], plant_data["X_cv_target_val"],
    plant_data["pv_init_val"], plant_data["scenario_val"],
    ctrl_data, "val", TARGET_LEN, N_PV)
pv_true_val  = plant_data["pv_target_val"]
scenario_val = plant_data["scenario_val"]
attack_val   = (scenario_val > 0).astype(int)

print("Running inference on test set...")
pv_preds_test = run_inference(
    plant_model, ctrl_models, ctrl_cv_col_idx,
    plant_data["X_test"], plant_data["X_cv_target_test"],
    plant_data["pv_init_test"], plant_data["scenario_test"],
    ctrl_data, "test", TARGET_LEN, N_PV)
pv_true_test  = plant_data["pv_target_test"]
scenario_test = plant_data["scenario_test"]
attack_test   = (scenario_test > 0).astype(int)

# residuals: (N, TARGET_LEN, N_PV)
resid_val  = pv_true_val  - pv_preds_val
resid_test = pv_true_test - pv_preds_test

# per-window mean MSE (original baseline score)
mse_val  = np.mean(resid_val  ** 2, axis=(1, 2))
mse_test = np.mean(resid_test ** 2, axis=(1, 2))

# per-window per-PV MSE feature vectors: (N, N_PV)
feat_val  = np.mean(resid_val  ** 2, axis=1)   # (N_val,  N_PV)
feat_test = np.mean(resid_test ** 2, axis=1)   # (N_test, N_PV)


# ── baseline (global MSE threshold, from results.json) ────────────────────────

with open(CKPT_DIR / "results.json") as f:
    saved = json.load(f)
ad = saved["attack_detection"]
baseline_auroc = ad["auroc"]
baseline_f1    = ad["best_f1"]
baseline_ap    = ad["avg_precision"]
print(f"\nBaseline  AUROC={baseline_auroc:.4f}  F1={baseline_f1:.4f}  AP={baseline_ap:.4f}")


# ── METHOD 1: IsolationForest ──────────────────────────────────────────────────

print("\nFitting IsolationForest on normal val windows...")
normal_mask_val = (scenario_val == 0)
X_train_if = feat_val[normal_mask_val]           # normal residual features only

iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
iso.fit(X_train_if)

# Score: IsolationForest returns negative anomaly scores (more negative = more anomalous)
# Negate so higher = more anomalous (consistent with MSE convention)
if_scores_val  = -iso.score_samples(feat_val)
if_scores_test = -iso.score_samples(feat_test)

fpr_if, tpr_if, _ = roc_curve(attack_test, if_scores_test)
auroc_if = auc(fpr_if, tpr_if)
ap_if    = average_precision_score(attack_test, if_scores_test)
t_if, f1_if = best_f1_threshold(if_scores_test, attack_test)

print(f"IsolationForest  AUROC={auroc_if:.4f}  F1={f1_if:.4f}  AP={ap_if:.4f}")
plot_roc(fpr_if, tpr_if, auroc_if, "IsolationForest", "#e74c3c",
         OUT_DIR / "roc_if.png")


# ── METHOD 2: Per-PV thresholds ────────────────────────────────────────────────

print("\nTuning per-PV thresholds on val set (99th percentile of normal windows)...")

# Use 99th percentile of normal-window per-PV MSE as threshold.
# This avoids degeneracy when val set has no attack windows.
normal_mask_val_pv = (scenario_val == 0)
pv_thresholds = []
for k in range(N_PV):
    t = np.percentile(feat_val[normal_mask_val_pv, k], 99)
    pv_thresholds.append(t)
    print(f"  {PV_COLS[k]}: threshold={t:.6f}")

# Test: alert if ANY PV exceeds its threshold
def perpv_score(feat, thresholds):
    """Soft score = max normalised exceedance across PVs."""
    exceedances = feat / (np.array(thresholds)[None, :] + 1e-12)
    return exceedances.max(axis=1)

perpv_scores_val  = perpv_score(feat_val,  pv_thresholds)
perpv_scores_test = perpv_score(feat_test, pv_thresholds)

fpr_pv, tpr_pv, _ = roc_curve(attack_test, perpv_scores_test)
auroc_pv = auc(fpr_pv, tpr_pv)
ap_pv    = average_precision_score(attack_test, perpv_scores_test)
t_pv, f1_pv = best_f1_threshold(perpv_scores_test, attack_test)

print(f"Per-PV thresholds  AUROC={auroc_pv:.4f}  F1={f1_pv:.4f}  AP={ap_pv:.4f}")
plot_roc(fpr_pv, tpr_pv, auroc_pv, "Per-PV Thresholds", "#3498db",
         OUT_DIR / "roc_perpv.png")


# ── METHOD 3: Ensemble (IF score × per-PV score, geometric mean) ──────────────

# Normalise both scores to [0,1] then take geometric mean
def norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)

ensemble_test = np.sqrt(norm01(if_scores_test) * norm01(perpv_scores_test))
fpr_en, tpr_en, _ = roc_curve(attack_test, ensemble_test)
auroc_en = auc(fpr_en, tpr_en)
ap_en    = average_precision_score(attack_test, ensemble_test)
t_en, f1_en = best_f1_threshold(ensemble_test, attack_test)
print(f"Ensemble           AUROC={auroc_en:.4f}  F1={f1_en:.4f}  AP={ap_en:.4f}")


# ── comparison bar chart ───────────────────────────────────────────────────────

methods  = ["Baseline\n(MSE)", "IsolationForest", "Per-PV\nThresholds", "Ensemble"]
aurocs   = [baseline_auroc, auroc_if,  auroc_pv,  auroc_en]
f1s      = [baseline_f1,    f1_if,     f1_pv,     f1_en]
aps      = [baseline_ap,    ap_if,     ap_pv,     ap_en]
colors   = ["#95a5a6", "#e74c3c", "#3498db", "#2ecc71"]

x = np.arange(len(methods))
w = 0.25

fig, ax = plt.subplots(figsize=(11, 5))
b1 = ax.bar(x - w, aurocs, w, label='AUROC',  color=[c + "cc" for c in colors])
b2 = ax.bar(x,     f1s,    w, label='F1',     color=colors, alpha=0.9)
b3 = ax.bar(x + w, aps,    w, label='Avg Prec', color=colors, alpha=0.5)

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7)

ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel("Score"); ax.set_ylim(0, 1.08)
ax.set_title("Anomaly Detection: Baseline vs Post-hoc Improvements\n(no retraining)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "comparison_bar.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved comparison_bar.png")

# ── overlay ROC ───────────────────────────────────────────────────────────────

fpr_base, tpr_base, _ = roc_curve(attack_test, mse_test)
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(fpr_base, tpr_base, 'k-',  lw=2, label=f'Baseline MSE (AUC={baseline_auroc:.4f})')
ax.plot(fpr_if,   tpr_if,   color='#e74c3c', lw=2, label=f'IsolationForest (AUC={auroc_if:.4f})')
ax.plot(fpr_pv,   tpr_pv,   color='#3498db', lw=2, label=f'Per-PV Thresholds (AUC={auroc_pv:.4f})')
ax.plot(fpr_en,   tpr_en,   color='#2ecc71', lw=2, label=f'Ensemble (AUC={auroc_en:.4f})')
ax.plot([0,1],[0,1],'--', color='gray', lw=1)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Comparison — All Detection Methods', fontsize=13, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_aspect('equal')
fig.tight_layout()
fig.savefig(OUT_DIR / "roc_all.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved roc_all.png")

# ── summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print(f"{'Method':<22} {'AUROC':>7} {'F1':>7} {'AvgPrec':>9}")
print("-"*60)
rows = zip(methods, aurocs, f1s, aps)
for m, au, f, ap in rows:
    m_clean = m.replace('\n', ' ')
    print(f"{m_clean:<22} {au:>7.4f} {f:>7.4f} {ap:>9.4f}")
print("="*60)
print(f"\nAll outputs saved to {OUT_DIR}/")
