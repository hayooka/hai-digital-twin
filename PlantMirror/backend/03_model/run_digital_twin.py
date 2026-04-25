"""
run_digital_twin.py — Offline evaluation of the GRU Digital Twin on the HAI test set.

Uses the teammate's GRU-Scenario-Weighted model (AUROC=0.8989, hidden=512).
Controllers use 6-input causal features for more accurate CV prediction.
Threshold calibrated honestly on the validation set (no test leakage).

Outputs saved to: outputs/digital_twin/
    residual_timeline.png   -- mean |residual| per PV over all test windows (attacks shaded)
    score_histogram.png     -- anomaly score distribution: normal vs attack
    pv_comparison.png       -- PV_pred vs PV_actual: 1 normal + 1 attack window
    summary.json            -- NRMSE, AUROC, F1 metrics

Run:
    python 03_model/run_digital_twin.py
"""

import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from digital_twin import DigitalTwin
from config import PV_COLS

# Teammate's model (causal GRU-Scenario-Weighted, AUROC=0.8989)
GRU_DIR  = ROOT / "outputs" / "gru_scenario_weighted" / "gru_scenario_weighted"
DATA_DIR = ROOT / "outputs" / "scaled_split"
OUT_DIR  = ROOT / "outputs" / "digital_twin"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("\nStep 1: Loading data...")
data           = load_and_prepare_data()
plant_data     = data['plant']
ctrl_data      = data['ctrl']

X_test         = plant_data['X_test']
X_cv_tgt_test  = plant_data['X_cv_target_test']
pv_init_test   = plant_data['pv_init_test']
pv_target_test = plant_data['pv_target_test']
scenario_test  = plant_data['scenario_test']
attack_test    = plant_data['attack_test']

# Raw full-sensor arrays (all cols including PVs) for causal controller inputs
print("  Loading raw sensor arrays for causal controller inputs...")
raw_test = np.load(DATA_DIR / 'test_data.npz')['X']
raw_val  = np.load(DATA_DIR / 'val_data.npz')['X']

print(f"  Test windows : {len(X_test)}  "
      f"(attack={int(attack_test.sum())}, normal={int((attack_test == 0).sum())})")

# ── 2. Instantiate Digital Twin ───────────────────────────────────────────────
print("\nStep 2: Loading Digital Twin (GRU-Scenario-Weighted)...")
twin = DigitalTwin(GRU_DIR, device=device, data=data)

# ── 3. Calibrate threshold on validation set ──────────────────────────────────
print("\nStep 3: Calibrating threshold on validation set (normal windows only)...")
print("  Threshold = 95th percentile of val scores => at most 5% FPR")
twin.calibrate(data, fpr_target=0.05, raw_val=raw_val, use_controllers=True)

# ── 4. Run closed-loop simulation on TEST set ─────────────────────────────────
print("\nStep 4: Running closed-loop simulation on test set...")
scenario_normal = np.zeros_like(scenario_test)
ctrl_test       = twin.build_ctrl_inputs(raw_test, ctrl_data)

results   = twin.run_batch(
    X_test, X_cv_tgt_test, pv_init_test, scenario_normal,
    pv_target_test, ctrl_data=ctrl_test,
)
pv_pred   = results["pv_pred"]
residuals = results["residuals"]
scores    = results["scores"]
is_attack = results["is_attack"]

# ── 5. Compute metrics ────────────────────────────────────────────────────────
pv_labels    = [c.replace("P1_", "") for c in PV_COLS]
nrmse_per_pv = {}

for i, col in enumerate(PV_COLS):
    pv_true = pv_target_test[:, :, i]
    pv_hat  = pv_pred[:, :, i]
    rng     = float(pv_true.max() - pv_true.min())
    nrmse   = float(np.sqrt(np.mean((pv_true - pv_hat) ** 2)) / (rng + 1e-8))
    nrmse_per_pv[col] = nrmse

mean_nrmse = float(np.mean(list(nrmse_per_pv.values())))
auroc      = roc_auc_score(attack_test, scores)

best_thresh  = twin.threshold
pred_attack  = scores > best_thresh
best_f1      = f1_score(attack_test, pred_attack, zero_division=0)
TP = int(((pred_attack == 1) & (attack_test == 1)).sum())
FP = int(((pred_attack == 1) & (attack_test == 0)).sum())
FN = int(((pred_attack == 0) & (attack_test == 1)).sum())

print("\n== GRU-Scenario-Weighted Digital Twin -- Test Results ==")
print(f"  {'Signal':<12}  NRMSE")
for col, v in nrmse_per_pv.items():
    print(f"  {col:<12}  {v*100:.3f}%")
print(f"  {'Mean':<12}  {mean_nrmse*100:.3f}%")
print(f"\n  AUROC          : {auroc:.4f}  (threshold-independent)")
print(f"  F1 @ val thresh: {best_f1:.4f}  (threshold={best_thresh:.6f})")
print(f"  True attacks detected  : {TP} / {int(attack_test.sum())}  "
      f"(recall={TP/max(int(attack_test.sum()),1):.2%})")
print(f"  False alerts           : {FP} / {int((attack_test==0).sum())} normal windows  "
      f"(FPR={FP/max(int((attack_test==0).sum()),1):.2%})")
print(f"  Missed attacks         : {FN}")
print(f"\n  NOTE: threshold calibrated on val set (no test leakage). AUROC is the")
print(f"  most trustworthy metric -- F1 depends on threshold.")
print("=" * 56)

# ── 6. Plot 1 -- Residual timeline ────────────────────────────────────────────
print("\nPlot 1: Residual timeline...")
fig, axes = plt.subplots(len(PV_COLS), 1, figsize=(14, 10), sharex=True)

for i, (ax, label) in enumerate(zip(axes, pv_labels)):
    ch_res = np.mean(np.abs(residuals[:, :, i]), axis=1)
    ax.plot(ch_res, color="steelblue", linewidth=0.9, alpha=0.85)

    in_attack = False
    start = 0
    for j, att in enumerate(attack_test):
        if att and not in_attack:
            start, in_attack = j, True
        elif not att and in_attack:
            ax.axvspan(start - 0.5, j - 0.5, color="red", alpha=0.25)
            in_attack = False
    if in_attack:
        ax.axvspan(start - 0.5, len(attack_test) - 0.5, color="red", alpha=0.25)

    ax.set_ylabel(label, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

axes[-1].set_xlabel("Test window index")
axes[0].set_title(
    "GRU-Scenario-Weighted Digital Twin -- Mean |Residual| per PV  (red = true attacks)",
    fontsize=11, fontweight="bold",
)
fig.tight_layout()
fig.savefig(OUT_DIR / "residual_timeline.png", dpi=150, bbox_inches="tight")
print(f"  Saved: residual_timeline.png")
plt.close(fig)

# ── 7. Plot 2 -- Score histogram ──────────────────────────────────────────────
print("Plot 2: Score histogram...")
normal_scores = scores[attack_test == 0]
attack_scores = scores[attack_test == 1]

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(normal_scores, bins=60, alpha=0.65, color="steelblue",
        label=f"Normal (n={len(normal_scores)})", density=True)
ax.hist(attack_scores, bins=25, alpha=0.75, color="crimson",
        label=f"Attack (n={len(attack_scores)})", density=True)
ax.axvline(twin.threshold, color="black", linestyle="--", linewidth=1.5,
           label=f"Threshold = {twin.threshold:.4f}")
ax.set_xlabel("Anomaly Score (MSE)")
ax.set_ylabel("Density")
ax.set_title(f"Anomaly Score Distribution  |  AUROC={auroc:.3f}  F1={best_f1:.3f}")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "score_histogram.png", dpi=150, bbox_inches="tight")
print(f"  Saved: score_histogram.png")
plt.close(fig)

# ── 8. Plot 3 -- PV comparison ────────────────────────────────────────────────
print("Plot 3: PV comparison (normal vs attack)...")
normal_idx = int(np.where(attack_test == 0)[0][50])
attack_idx = int(np.where(attack_test == 1)[0][0])
t          = np.arange(pv_target_test.shape[1])

fig, axes = plt.subplots(len(PV_COLS), 2, figsize=(14, 10))
for i, label in enumerate(pv_labels):
    for col, (win_idx, title) in enumerate([(normal_idx, "Normal Window"),
                                             (attack_idx,  "Attack Window")]):
        ax  = axes[i, col]
        act = pv_target_test[win_idx, :, i]
        hat = pv_pred[win_idx, :, i]
        ax.plot(t, act, color="darkorange", linewidth=1.5, label="Actual")
        ax.plot(t, hat, color="steelblue",  linewidth=1.5, linestyle="--", label="Twin")
        ax.fill_between(t, act, hat, alpha=0.2, color="red")
        if i == 0:
            ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        if i == 0 and col == 1:
            ax.legend(fontsize=8)

axes[-1, 0].set_xlabel("Timestep (target horizon)")
axes[-1, 1].set_xlabel("Timestep (target horizon)")
fig.suptitle("GRU-Scenario-Weighted Digital Twin: Predicted vs Actual PVs",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "pv_comparison.png", dpi=150, bbox_inches="tight")
print(f"  Saved: pv_comparison.png")
plt.close(fig)

# ── 9. Save summary ───────────────────────────────────────────────────────────
summary = {
    "model":       "GRU-Scenario-Weighted Digital Twin",
    "nrmse_per_pv": nrmse_per_pv,
    "mean_nrmse":   mean_nrmse,
    "attack_detection": {
        "auroc":            auroc,
        "best_f1":          best_f1,
        "best_threshold":   float(best_thresh),
        "attacks_detected": int(is_attack.sum()),
        "total_attacks":    int(attack_test.sum()),
        "total_windows":    int(len(attack_test)),
        "TP": TP, "FP": FP, "FN": FN,
    },
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: summary.json")
print(f"\nDone. All outputs in: {OUT_DIR}")
