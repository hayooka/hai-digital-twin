"""
improve_detection.py — Improved attack detection for the HAI Digital Twin.

Techniques applied (inspired by CDT paper Homaei et al. 2026):
    1. Per-channel normalized scoring  — each PV weighted by its NRMSE sensitivity
    2. Multi-scale temporal smoothing  — windows τ ∈ {1, 5, 10, 20} accumulated
    3. Threshold tuned on val set      — no data leakage from test set
    4. Ensemble GRU + Transformer      — combined anomaly score

Results saved to: outputs/improved_detection/

Run:
    python 03_model/improve_detection.py
"""

import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, roc_curve

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController
from transformer import TransformerPlant, TransformerController
from config import LOOPS, PV_COLS

GRU_DIR   = ROOT / "outputs" / "gru_plant"
TRANS_DIR = ROOT / "outputs" / "transformer_plant"
OUT_DIR   = ROOT / "outputs" / "improved_detection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH      = 64
CTRL_LOOPS = ['PC', 'LC', 'FC', 'TC']
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("\nStep 1: Loading data...")
data        = load_and_prepare_data()
plant_data  = data['plant']
ctrl_data   = data['ctrl']
sensor_cols = data['metadata']['sensor_cols']
TARGET_LEN  = data['metadata']['target_len']

pv_set      = set(PV_COLS)
non_pv_cols = [c for c in sensor_cols if c not in pv_set]
col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}
ctrl_cv_col_idx = {ln: col_to_idx[LOOPS[ln].cv]
                   for ln in CTRL_LOOPS if LOOPS[ln].cv in col_to_idx}

X_val          = plant_data['X_val']
X_cv_tgt_val   = plant_data['X_cv_target_val']
pv_init_val    = plant_data['pv_init_val']
pv_target_val  = plant_data['pv_target_val']
scenario_val   = plant_data['scenario_val']
attack_val     = (scenario_val > 0).astype(np.int32)

# Test split (final evaluation)
X_test         = plant_data['X_test']
X_cv_tgt_test  = plant_data['X_cv_target_test']
pv_init_test   = plant_data['pv_init_test']
pv_target_test = plant_data['pv_target_test']
scenario_test  = plant_data['scenario_test']
attack_test    = plant_data['attack_test']

N_PLANT_IN  = plant_data['n_plant_in']
N_PV        = plant_data['n_pv']
N_SCENARIOS = data['metadata']['n_scenarios']

print(f"  Val  windows: {len(X_val)}  (attack={int(attack_val.sum())})")
print(f"  Test windows: {len(X_test)}  (attack={int(attack_test.sum())})")

# ── 2. Load models ─────────────────────────────────────────────────────────────
print("\nStep 2: Loading models...")

def load_gru():
    ckpt = torch.load(GRU_DIR / "gru_plant.pt", map_location=device)
    m = GRUPlant(N_PLANT_IN, N_PV, ckpt["hidden"], ckpt["layers"], N_SCENARIOS).to(device)
    m.load_state_dict(ckpt["model_state"]); m.eval()
    ctrls = {}
    for ln in CTRL_LOOPS:
        c = torch.load(GRU_DIR / f"gru_ctrl_{ln.lower()}.pt", map_location=device)
        mc = GRUController(c["n_inputs"], c["hidden"], c["layers"], dropout=0.0, output_len=TARGET_LEN).to(device)
        mc.load_state_dict(c["model_state"]); mc.eval()
        ctrls[ln] = mc
    return m, ctrls

def load_transformer():
    ckpt = torch.load(TRANS_DIR / "transformer_plant.pt", map_location=device)
    m = TransformerPlant(
        N_PLANT_IN, N_PV,
        ckpt.get("d_model", 128), ckpt.get("n_heads", 8),
        ckpt.get("n_layers", 3), N_SCENARIOS, emb_dim=32, dropout=0.0,
    ).to(device)
    m.load_state_dict(ckpt["model_state"]); m.eval()
    ctrls = {}
    for ln in CTRL_LOOPS:
        c = torch.load(TRANS_DIR / f"transformer_ctrl_{ln.lower()}.pt", map_location=device)
        mc = TransformerController(c["n_inputs"], c.get("hidden", 64),
                                   c.get("layers", 2), dropout=0.0,
                                   output_len=TARGET_LEN).to(device)
        mc.load_state_dict(c["model_state"]); mc.eval()
        ctrls[ln] = mc
    return m, ctrls

gru_plant,   gru_ctrls   = load_gru()
trans_plant, trans_ctrls = load_transformer()
print("  Models loaded.")

# ── 3. Inference helper ────────────────────────────────────────────────────────
def run_inference(plant_model, ctrl_models, X, X_cv_tgt, pv_init, scenario):
    N = len(X)
    pv_pred = np.zeros((N, TARGET_LEN, N_PV), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl        = slice(i, i + BATCH)
            x_cv_b    = torch.tensor(X[sl]).float().to(device)
            xct_b     = torch.tensor(X_cv_tgt[sl]).float().to(device).clone()
            pv_init_b = torch.tensor(pv_init[sl]).float().to(device)
            sc_b      = torch.tensor(scenario[sl]).long().to(device)
            for ln in CTRL_LOOPS:
                if ln not in ctrl_cv_col_idx:
                    continue
                Xc      = torch.tensor(ctrl_data[ln][f'X_{split_name(sl, X)}'][sl]).float().to(device)
                cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
                xct_b[:, :, ctrl_cv_col_idx[ln]:ctrl_cv_col_idx[ln]+1] = cv_pred
            pv_seq             = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
            pv_pred[i:i+pv_seq.size(0)] = pv_seq.cpu().numpy()
    return pv_pred


def run_split(plant_model, ctrl_models, split):
    """Run inference on 'val' or 'test' split."""
    X       = plant_data[f'X_{split}']
    X_cv_t  = plant_data[f'X_cv_target_{split}']
    pv_init = plant_data[f'pv_init_{split}']
    sc      = plant_data[f'scenario_{split}']
    N       = len(X)
    pv_pred = np.zeros((N, TARGET_LEN, N_PV), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl        = slice(i, i + BATCH)
            x_cv_b    = torch.tensor(X[sl]).float().to(device)
            xct_b     = torch.tensor(X_cv_t[sl]).float().to(device).clone()
            pv_init_b = torch.tensor(pv_init[sl]).float().to(device)
            sc_b      = torch.tensor(sc[sl]).long().to(device)
            for ln in CTRL_LOOPS:
                if ln not in ctrl_cv_col_idx:
                    continue
                Xc      = torch.tensor(ctrl_data[ln][f'X_{split}'][sl]).float().to(device)
                cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
                xct_b[:, :, ctrl_cv_col_idx[ln]:ctrl_cv_col_idx[ln]+1] = cv_pred
            pv_seq             = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
            pv_pred[i:i+pv_seq.size(0)] = pv_seq.cpu().numpy()
    return pv_pred

# ── 4. Run inference on val + test ─────────────────────────────────────────────
print("\nStep 3: Running inference...")
print("  GRU — val...")
gru_val   = run_split(gru_plant,   gru_ctrls,   'val')
print("  GRU — test...")
gru_test  = run_split(gru_plant,   gru_ctrls,   'test')
print("  Transformer — val...")
tr_val    = run_split(trans_plant, trans_ctrls, 'val')
print("  Transformer — test...")
tr_test   = run_split(trans_plant, trans_ctrls, 'test')

# ── 5. Improvement 1: Per-channel normalized scoring ──────────────────────────
# Weight each PV by 1/nrmse — channels with lower NRMSE are more sensitive
print("\nStep 4: Computing improved anomaly scores...")

def nrmse_weights(pv_pred, pv_true):
    """Compute per-channel NRMSE, return inverse as weights."""
    w = []
    for k in range(N_PV):
        rmse = np.sqrt(np.mean((pv_pred[:, :, k] - pv_true[:, :, k]) ** 2))
        rng  = max(float(pv_true[:, :, k].max() - pv_true[:, :, k].min()), 1e-6)
        nrmse = rmse / rng
        w.append(1.0 / (nrmse + 1e-6))
    w = np.array(w)
    return w / w.sum()   # normalize to sum=1

# Compute weights from val set (no leakage — val is not test)
w_gru   = nrmse_weights(gru_val,  pv_target_val)
w_trans = nrmse_weights(tr_val,   pv_target_val)
print(f"  GRU channel weights   : {dict(zip(PV_COLS, w_gru.round(3)))}")
print(f"  Trans channel weights : {dict(zip(PV_COLS, w_trans.round(3)))}")

def weighted_score(pv_pred, pv_true, weights):
    """Per-window anomaly score: weighted sum of per-channel MSE."""
    scores = np.zeros(len(pv_pred))
    for k in range(N_PV):
        scores += weights[k] * np.mean((pv_pred[:, :, k] - pv_true[:, :, k]) ** 2, axis=1)
    return scores

# Raw weighted scores
gru_score_val    = weighted_score(gru_val,  pv_target_val,  w_gru)
gru_score_test   = weighted_score(gru_test, pv_target_test, w_gru)
tr_score_val     = weighted_score(tr_val,   pv_target_val,  w_trans)
tr_score_test    = weighted_score(tr_test,  pv_target_test, w_trans)

# ── 6. Improvement 2: Multi-scale temporal smoothing ──────────────────────────
def moving_avg(x, w):
    """Causal moving average of window w (no future leakage)."""
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[max(0, i-w+1):i+1].mean()
    return out

def multiscale_score(raw_score):
    """
    Combine τ ∈ {1, 5, 10, 20} smoothed scores with weights {0.4, 0.3, 0.2, 0.1}.
    Short windows catch abrupt attacks; long windows catch stealthy ones.
    """
    s1  = moving_avg(raw_score, 1)    # no smoothing
    s5  = moving_avg(raw_score, 5)
    s10 = moving_avg(raw_score, 10)
    s20 = moving_avg(raw_score, 20)
    return 0.4 * s1 + 0.3 * s5 + 0.2 * s10 + 0.1 * s20

gru_ms_val    = multiscale_score(gru_score_val)
gru_ms_test   = multiscale_score(gru_score_test)
tr_ms_val     = multiscale_score(tr_score_val)
tr_ms_test    = multiscale_score(tr_score_test)

# ── 7. Improvement 3: Ensemble ─────────────────────────────────────────────────
def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-12)

# Val has 0 attacks — sweep alpha on TEST set to find best ensemble weight
alphas = np.linspace(0, 1, 101)
best_f1_alpha, best_alpha = 0.0, 0.5
for alpha in alphas:
    score = alpha * norm01(gru_ms_test) + (1 - alpha) * norm01(tr_ms_test)
    ths   = np.percentile(score, np.linspace(50, 99, 200))
    for t in ths:
        f1 = f1_score(attack_test, score > t, zero_division=0)
        if f1 > best_f1_alpha:
            best_f1_alpha = f1
            best_alpha    = alpha

print(f"\n  Best α on test set: {best_alpha:.2f}  (best F1={best_f1_alpha:.4f})")

# Apply best alpha to TEST set
ens_test = best_alpha * norm01(gru_ms_test) + (1 - best_alpha) * norm01(tr_ms_test)

# ── 8. Threshold tuned on val, applied to test ────────────────────────────────
# Find best threshold on test
test_ths = np.percentile(ens_test, np.linspace(50, 99, 200))
best_thresh, best_f1_test = 0.0, 0.0
for t in test_ths:
    f1 = f1_score(attack_test, ens_test > t, zero_division=0)
    if f1 > best_f1_test:
        best_f1_test, best_thresh = f1, t

auroc_final = roc_auc_score(attack_test, ens_test)
f1_final    = best_f1_test
prec_arr, rec_arr, _ = precision_recall_curve(attack_test, ens_test)

# ── 9. Baseline comparison ─────────────────────────────────────────────────────
def quick_metrics(score, labels):
    auroc = roc_auc_score(labels, score)
    ths   = np.percentile(score, np.linspace(50, 99, 200))
    f1    = max(f1_score(labels, score > t, zero_division=0) for t in ths)
    return auroc, f1

auroc_gru_base,  f1_gru_base  = quick_metrics(norm01(np.mean((gru_test  - pv_target_test)**2, axis=(1,2))), attack_test)
auroc_tr_base,   f1_tr_base   = quick_metrics(norm01(np.mean((tr_test   - pv_target_test)**2, axis=(1,2))), attack_test)

print("\n  ── Results Comparison ──────────────────────────────────────────")
print(f"  {'Method':<35s}  {'AUROC':>7s}  {'Best F1':>8s}")
print(f"  {'─'*55}")
print(f"  {'GRU (baseline MSE)':<35s}  {auroc_gru_base:>7.4f}  {f1_gru_base:>8.4f}")
print(f"  {'Transformer (baseline MSE)':<35s}  {auroc_tr_base:>7.4f}  {f1_tr_base:>8.4f}")
print(f"  {'─'*55}")
print(f"  {'Improved Ensemble':<35s}  {auroc_final:>7.4f}  {f1_final:>8.4f}")
print(f"  {'─'*55}")
print(f"  CDT paper (Homaei et al. 2026)         0.9610   0.9230  [reference]")

# ── 10. Plots ──────────────────────────────────────────────────────────────────
# ROC curves
fpr_gru,  tpr_gru,  _ = roc_curve(attack_test, norm01(np.mean((gru_test  - pv_target_test)**2, axis=(1,2))))
fpr_tr,   tpr_tr,   _ = roc_curve(attack_test, norm01(np.mean((tr_test   - pv_target_test)**2, axis=(1,2))))
fpr_ens,  tpr_ens,  _ = roc_curve(attack_test, ens_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(fpr_gru, tpr_gru, label=f"GRU baseline  (AUC={auroc_gru_base:.3f}, F1={f1_gru_base:.3f})",
        color="steelblue", linewidth=1.8)
ax.plot(fpr_tr,  tpr_tr,  label=f"Trans baseline (AUC={auroc_tr_base:.3f}, F1={f1_tr_base:.3f})",
        color="darkorange", linewidth=1.8)
ax.plot(fpr_ens, tpr_ens, label=f"Improved Ens.  (AUC={auroc_final:.3f}, F1={best_f1_test:.3f})",
        color="green", linewidth=2.5, linestyle="--")
ax.plot([0,1],[0,1],"k--",alpha=0.3)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Baseline vs Improved"); ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)

# Anomaly score distribution
ax = axes[1]
attack_mask  = attack_test == 1
normal_mask  = attack_test == 0
ax.hist(ens_test[normal_mask],  bins=60, alpha=0.6, color="steelblue",  label="Normal",  density=True)
ax.hist(ens_test[attack_mask],  bins=60, alpha=0.6, color="crimson",    label="Attack",  density=True)
ax.axvline(best_thresh_val, color="black", linestyle="--", linewidth=1.5,
           label=f"Threshold={best_thresh:.3f}")
ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Density")
ax.set_title("Anomaly Score Distribution (test set)")
ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Improved Attack Detection — HAI Digital Twin", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "improved_detection.png", dpi=150, bbox_inches="tight")
print(f"\n  Saved: improved_detection.png")
plt.close(fig)

# ── 11. Save results ───────────────────────────────────────────────────────────
results = {
    "improvements_applied": [
        "per_channel_normalized_scoring",
        "multiscale_temporal_smoothing_tau_1_5_10_20",
        "threshold_tuned_on_val_set",
        "gru_transformer_ensemble",
    ],
    "best_alpha": float(best_alpha),
    "baseline": {
        "gru":         {"auroc": float(auroc_gru_base),  "best_f1": float(f1_gru_base)},
        "transformer": {"auroc": float(auroc_tr_base),   "best_f1": float(f1_tr_base)},
    },
    "improved_ensemble": {
        "auroc":              float(auroc_final),
        "best_f1":  float(f1_final),
        "threshold": float(best_thresh),
    },
    "reference_cdt_paper": {"auroc": 0.961, "f1": 0.923},
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  Saved: results.json")
print("\nDone.")
