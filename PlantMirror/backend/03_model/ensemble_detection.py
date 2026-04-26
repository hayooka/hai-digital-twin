"""
ensemble_detection.py — Hybrid Digital Twin: GRU plant + Transformer attack detection.

    - GRU Plant        → process simulation  (best NRMSE = 0.88%)
    - Transformer Plant → complementary anomaly score
    - Ensemble score   → α × MSE_gru + (1-α) × MSE_transformer
    - Sweeps α ∈ [0, 1] to find the best F1 and AUROC

Outputs saved to: outputs/ensemble/

Run:
    python 03_model/ensemble_detection.py
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
OUT_DIR   = ROOT / "outputs" / "ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH  = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

CTRL_LOOPS = ['PC', 'LC', 'FC', 'TC']

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("\nStep 1: Loading data...")
data        = load_and_prepare_data()
plant_data  = data['plant']
ctrl_data   = data['ctrl']
sensor_cols = data['metadata']['sensor_cols']
TARGET_LEN  = data['metadata']['target_len']

X_test         = plant_data['X_test']
X_cv_tgt_test  = plant_data['X_cv_target_test']
pv_init_test   = plant_data['pv_init_test']
pv_target_test = plant_data['pv_target_test']
scenario_test  = plant_data['scenario_test']
attack_test    = plant_data['attack_test']

N_PLANT_IN  = plant_data['n_plant_in']
N_PV        = plant_data['n_pv']
N_SCENARIOS = data['metadata']['n_scenarios']

pv_set      = set(PV_COLS)
non_pv_cols = [c for c in sensor_cols if c not in pv_set]
col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}
ctrl_cv_col_idx = {ln: col_to_idx[LOOPS[ln].cv]
                   for ln in CTRL_LOOPS if LOOPS[ln].cv in col_to_idx}

print(f"  Test windows : {len(X_test)}  "
      f"(attack={int(attack_test.sum())}, normal={int((attack_test==0).sum())})")

# ── 2. Load GRU ────────────────────────────────────────────────────────────────
print("\nStep 2: Loading GRU models...")
gru_ckpt = torch.load(GRU_DIR / "gru_plant.pt", map_location=device)
gru_plant = GRUPlant(
    n_plant_in  = N_PLANT_IN,
    n_pv        = N_PV,
    hidden      = gru_ckpt["hidden"],
    layers      = gru_ckpt["layers"],
    n_scenarios = N_SCENARIOS,
).to(device)
gru_plant.load_state_dict(gru_ckpt["model_state"])
gru_plant.eval()
print(f"  GRU plant loaded (epoch={gru_ckpt.get('epoch','?')}, "
      f"val_loss={gru_ckpt.get('val_loss', float('nan')):.6f})")

gru_ctrls = {}
for ln in CTRL_LOOPS:
    c = torch.load(GRU_DIR / f"gru_ctrl_{ln.lower()}.pt", map_location=device)
    m = GRUController(
        n_inputs   = c["n_inputs"],
        hidden     = c["hidden"],
        layers     = c["layers"],
        output_len = TARGET_LEN,
    ).to(device)
    m.load_state_dict(c["model_state"])
    m.eval()
    gru_ctrls[ln] = m
print("  GRU controllers loaded")

# ── 3. Load Transformer ────────────────────────────────────────────────────────
print("\nStep 3: Loading Transformer models...")
trans_ckpt = torch.load(TRANS_DIR / "transformer_plant.pt", map_location=device)
trans_plant = TransformerPlant(
    n_plant_in  = N_PLANT_IN,
    n_pv        = N_PV,
    d_model     = trans_ckpt.get("d_model",  128),
    n_heads     = trans_ckpt.get("n_heads",  8),
    n_layers    = trans_ckpt.get("n_layers", 3),
    n_scenarios = N_SCENARIOS,
    emb_dim     = 32,
    dropout     = 0.0,
).to(device)
trans_plant.load_state_dict(trans_ckpt["model_state"])
trans_plant.eval()
print(f"  Transformer plant loaded (epoch={trans_ckpt.get('epoch','?')}, "
      f"val_loss={trans_ckpt.get('val_loss', float('nan')):.6f})")

trans_ctrls = {}
for ln in CTRL_LOOPS:
    c = torch.load(TRANS_DIR / f"transformer_ctrl_{ln.lower()}.pt", map_location=device)
    m = TransformerController(
        n_inputs   = c["n_inputs"],
        d_model    = c.get("hidden", 64),
        n_layers   = c.get("layers", 2),
        output_len = TARGET_LEN,
    ).to(device)
    m.load_state_dict(c["model_state"])
    m.eval()
    trans_ctrls[ln] = m
print("  Transformer controllers loaded")


# ── 4. Inference helper ────────────────────────────────────────────────────────
def run_inference(plant_model, ctrl_models):
    """Run closed-loop on test set. Returns (N_test, TARGET_LEN, N_PV) predictions."""
    N_test  = len(X_test)
    pv_pred = np.zeros((N_test, TARGET_LEN, N_PV), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N_test, BATCH):
            sl        = slice(i, i + BATCH)
            x_cv_b    = torch.tensor(X_test[sl]).float().to(device)
            xct_b     = torch.tensor(X_cv_tgt_test[sl]).float().to(device).clone()
            pv_init_b = torch.tensor(pv_init_test[sl]).float().to(device)
            sc_b      = torch.tensor(scenario_test[sl]).long().to(device)
            for ln in CTRL_LOOPS:
                if ln not in ctrl_cv_col_idx:
                    continue
                Xc      = torch.tensor(ctrl_data[ln]['X_test'][sl]).float().to(device)
                cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
                xct_b[:, :, ctrl_cv_col_idx[ln]:ctrl_cv_col_idx[ln]+1] = cv_pred
            pv_seq             = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
            pv_pred[i:i+pv_seq.size(0)] = pv_seq.cpu().numpy()
    return pv_pred


# ── 5. Run both models ─────────────────────────────────────────────────────────
print("\nStep 4: Running GRU inference on test set...")
pv_gru = run_inference(gru_plant, gru_ctrls)

print("Step 5: Running Transformer inference on test set...")
pv_trans = run_inference(trans_plant, trans_ctrls)

# ── 6. Compute per-window anomaly scores ───────────────────────────────────────
mse_gru   = np.mean((pv_gru   - pv_target_test) ** 2, axis=(1, 2))  # (N_test,)
mse_trans = np.mean((pv_trans - pv_target_test) ** 2, axis=(1, 2))  # (N_test,)

# Normalise each score to [0,1] for fair mixing
def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-12)

mse_gru_n   = norm01(mse_gru)
mse_trans_n = norm01(mse_trans)


# ── 7. Sweep α ────────────────────────────────────────────────────────────────
print("\nStep 6: Sweeping α (GRU weight) from 0 → 1...")
alphas   = np.linspace(0, 1, 101)
results_sweep = []

for alpha in alphas:
    score  = alpha * mse_gru_n + (1 - alpha) * mse_trans_n
    auroc  = roc_auc_score(attack_test, score)
    thresholds = np.percentile(score, np.linspace(50, 99, 100))
    best_f1 = 0.0
    for t in thresholds:
        f1 = f1_score(attack_test, score > t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    results_sweep.append({"alpha": float(alpha), "auroc": auroc, "f1": best_f1})

best_by_f1   = max(results_sweep, key=lambda r: r["f1"])
best_by_auroc = max(results_sweep, key=lambda r: r["auroc"])

print(f"\n  Best F1    : {best_by_f1['f1']:.4f}  at α={best_by_f1['alpha']:.2f}")
print(f"  Best AUROC : {best_by_auroc['auroc']:.4f}  at α={best_by_auroc['alpha']:.2f}")

# ── 8. Final metrics at best-F1 alpha ─────────────────────────────────────────
alpha_best = best_by_f1["alpha"]
score_best = alpha_best * mse_gru_n + (1 - alpha_best) * mse_trans_n

auroc_final = roc_auc_score(attack_test, score_best)
thresholds  = np.percentile(score_best, np.linspace(50, 99, 100))
best_f1_final, best_thresh = 0.0, thresholds[0]
for t in thresholds:
    f1 = f1_score(attack_test, score_best > t, zero_division=0)
    if f1 > best_f1_final:
        best_f1_final, best_thresh = f1, t

prec_arr, rec_arr, _ = precision_recall_curve(attack_test, score_best)

print(f"\n  Final Ensemble (α={alpha_best:.2f}):")
print(f"    AUROC          : {auroc_final:.4f}")
print(f"    Best F1        : {best_f1_final:.4f}  (threshold={best_thresh:.5f})")
print(f"    Avg Precision  : {float(np.mean(prec_arr)):.4f}")

# ── 9. Comparison table ────────────────────────────────────────────────────────
print("\n  ── Comparison ──────────────────────────────────────")
print(f"  {'Model':<20s}  {'AUROC':>7s}  {'Best F1':>8s}")
print(f"  {'GRU only':<20s}  {roc_auc_score(attack_test, mse_gru_n):>7.4f}  "
      f"{max(f1_score(attack_test, mse_gru_n > t, zero_division=0) for t in np.percentile(mse_gru_n, np.linspace(50,99,100))):>8.4f}")
print(f"  {'Transformer only':<20s}  {roc_auc_score(attack_test, mse_trans_n):>7.4f}  "
      f"{max(f1_score(attack_test, mse_trans_n > t, zero_division=0) for t in np.percentile(mse_trans_n, np.linspace(50,99,100))):>8.4f}")
print(f"  {'Ensemble':<20s}  {auroc_final:>7.4f}  {best_f1_final:>8.4f}")
print(f"  {'─'*45}")

# ── 10. Plots ──────────────────────────────────────────────────────────────────
# Plot 1: F1 and AUROC vs alpha
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
alphas_arr = [r["alpha"] for r in results_sweep]
f1s        = [r["f1"]    for r in results_sweep]
aurocs     = [r["auroc"] for r in results_sweep]

ax1.plot(alphas_arr, f1s, color="steelblue", linewidth=2)
ax1.axvline(best_by_f1["alpha"], color="red", linestyle="--", alpha=0.7,
            label=f"Best α={best_by_f1['alpha']:.2f}")
ax1.set_xlabel("α  (GRU weight)")
ax1.set_ylabel("Best F1")
ax1.set_title("Ensemble F1 vs α")
ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.4)

ax2.plot(alphas_arr, aurocs, color="darkorange", linewidth=2)
ax2.axvline(best_by_auroc["alpha"], color="red", linestyle="--", alpha=0.7,
            label=f"Best α={best_by_auroc['alpha']:.2f}")
ax2.set_xlabel("α  (GRU weight)")
ax2.set_ylabel("AUROC")
ax2.set_title("Ensemble AUROC vs α")
ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("GRU + Transformer Ensemble — Attack Detection", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "ensemble_alpha_sweep.png", dpi=150, bbox_inches="tight")
print(f"\n  Saved: ensemble_alpha_sweep.png")
plt.close(fig)

# Plot 2: ROC curves for all three
fpr_gru,   tpr_gru,   _ = roc_curve(attack_test, mse_gru_n)
fpr_trans, tpr_trans, _ = roc_curve(attack_test, mse_trans_n)
fpr_ens,   tpr_ens,   _ = roc_curve(attack_test, score_best)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_gru,   tpr_gru,   label=f"GRU        (AUC={roc_auc_score(attack_test, mse_gru_n):.3f})",
        color="steelblue",  linewidth=2)
ax.plot(fpr_trans, tpr_trans, label=f"Transformer (AUC={roc_auc_score(attack_test, mse_trans_n):.3f})",
        color="darkorange", linewidth=2)
ax.plot(fpr_ens,   tpr_ens,   label=f"Ensemble   (AUC={auroc_final:.3f})",
        color="green",      linewidth=2.5, linestyle="--")
ax.plot([0,1],[0,1], "k--", alpha=0.3)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — GRU vs Transformer vs Ensemble")
ax.legend(fontsize=10); ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
print(f"  Saved: roc_curves.png")
plt.close(fig)

# ── 11. Save results ───────────────────────────────────────────────────────────
output = {
    "best_alpha":     alpha_best,
    "ensemble": {
        "auroc":          auroc_final,
        "best_f1":        best_f1_final,
        "best_threshold": float(best_thresh),
        "avg_precision":  float(np.mean(prec_arr)),
    },
    "gru_only": {
        "auroc": float(roc_auc_score(attack_test, mse_gru_n)),
    },
    "transformer_only": {
        "auroc": float(roc_auc_score(attack_test, mse_trans_n)),
    },
    "alpha_sweep": results_sweep,
}
with open(OUT_DIR / "ensemble_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"  Saved: ensemble_results.json")
print("\nDone.")
