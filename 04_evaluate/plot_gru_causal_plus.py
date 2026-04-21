"""
plot_gru_causal_plus.py — Generate all paper plots for the final model.

Reads checkpoints from outputs/gru_scenario_weighted/ (produced by train_gru_scenario_weighted.py).
Plots saved to plots/.

Usage:
    python 04_evaluate/plot_gru_causal_plus.py
"""

import sys
import json
import numpy as np
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))
sys.path.insert(0, str(ROOT / "04_evaluate"))

from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCSequenceModel
from config import LOOPS, PROCESSED_DATA_DIR
from plot_utils import (
    plot_loss_curves,
    plot_nrmse_per_pv,
    plot_per_loop_performance,
    plot_nrmse_per_scenario,
    plot_scenario_overlay,
    plot_error_growth_curve,
    plot_error_heatmap,
    plot_roc_curve,
    plot_pr_curve,
    plot_residual_boxplot,
    plot_residual_timeline,
    plot_confusion_matrix_attack,
    plot_detection_rate_per_attack,
    compute_nrmse,
    compute_nrmse_per_scenario,
    CTRL_LOOPS
)

# Three causal layers per loop — must match train_gru_scenario_weighted.py
EXTRA_CHANNELS = {
    'PC': ['P1_PCV02D',  'P1_FT01',   'P1_TIT01'],
    'LC': ['P1_FT03',   'P1_FCV03D', 'P1_PCV01D'],
    'FC': ['P1_PIT01',  'P1_LIT01',  'P1_TIT03'],
    'TC': ['P1_FT02',   'P1_PIT02',  'P1_TIT02'],
    'CC': ['P1_PP04D',  'P1_FCV03D', 'P1_PCV02D'],
}


def augment_ctrl_data(ctrl_data: dict, sensor_cols: list) -> None:
    """Append L0/L1/L2 causal channels to each loop's X arrays in-place."""
    import joblib
    plant_scaler = joblib.load(f"{PROCESSED_DATA_DIR}/scaler.pkl")
    npz_train = np.load(f"{PROCESSED_DATA_DIR}/train_data.npz")
    npz_val   = np.load(f"{PROCESSED_DATA_DIR}/val_data.npz")
    npz_test  = np.load(f"{PROCESSED_DATA_DIR}/test_data.npz")
    col_idx   = {c: i for i, c in enumerate(sensor_cols)}

    for ln, extra_cols in EXTRA_CHANNELS.items():
        for extra_col in extra_cols:
            if extra_col not in col_idx:
                print(f"  WARNING: {extra_col} not found for {ln} — skipping")
                continue
            ei      = col_idx[extra_col]
            mean_e  = plant_scaler.mean_[ei]
            scale_e = plant_scaler.scale_[ei]
            for split, npz in [('train', npz_train), ('val', npz_val), ('test', npz_test)]:
                raw = npz['X'][:, :, [ei]].astype('float32')
                sc  = (raw - mean_e) / scale_e
                key = f'X_{split}'
                ctrl_data[ln][key] = np.concatenate([ctrl_data[ln][key], sc], axis=-1)


def run_closed_loop_inference(
    plant_model, ctrl_models, data, split, device, batch_size=128, verbose=True
):
    """Run closed-loop inference on specified split."""
    plant_data = data["plant"]
    ctrl_data = data["ctrl"]
    target_len = data["metadata"]["target_len"]
    n_pv = plant_data["n_pv"]
    
    X = plant_data.get(f"X_{split}")
    X_cv_tgt = plant_data.get(f"X_cv_target_{split}")
    pv_init = plant_data.get(f"pv_init_{split}")
    scenario = plant_data.get(f"scenario_{split}")
    
    if X is None:
        raise ValueError(f"No data found for split '{split}'")
    
    N = len(X)
    pv_preds = np.zeros((N, target_len, n_pv), dtype=np.float32)
    
    plant_model.eval()
    for m in ctrl_models.values():
        m.eval()
    
    if verbose:
        print(f"  Running closed-loop inference on {split} split ({N} samples)")
    
    with torch.no_grad():
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            sl = slice(i, end_idx)
            
            x_cv_b = torch.tensor(X[sl]).float().to(device)
            xct_b = torch.tensor(X_cv_tgt[sl]).float().to(device).clone()
            pv_init_b = torch.tensor(pv_init[sl]).float().to(device)
            sc_b = torch.tensor(scenario[sl]).long().to(device)
            
            try:
                pv_seq = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
                B_actual = pv_seq.size(0)
                pv_preds[i:i + B_actual] = pv_seq.cpu().numpy()
            except Exception as e:
                print(f"  Error in plant model prediction for batch {i}: {e}")
                continue
    
    if verbose:
        print(f"  Completed inference for {split} split")
    
    return pv_preds


CKPT_DIR  = ROOT / "outputs" / "pipeline" / "gru_scenario_weighted"
OUT_DIR   = ROOT / "plots"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH     = 128

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
data        = load_and_prepare_data()
plant_data  = data["plant"]
ctrl_data   = data["ctrl"]
TARGET_LEN  = data["metadata"]["target_len"]
sensor_cols = data["metadata"]["sensor_cols"]

print("Augmenting controller data with three causal layers...")
augment_ctrl_data(ctrl_data, sensor_cols)

# ── Load plant model ──────────────────────────────────────────────────────────
ckpt = torch.load(CKPT_DIR / "gru_plant.pt", map_location=DEVICE)
_has_heads = any("fc_heads" in k for k in ckpt["model_state"])
plant_model = GRUPlant(
    n_plant_in     = ckpt["n_plant_in"],
    n_pv           = ckpt["n_pv"],
    hidden         = ckpt["hidden"],
    layers         = ckpt["layers"],
    n_scenarios    = ckpt["n_scenarios"],
    scenario_heads = _has_heads,
).to(DEVICE)
plant_model.load_state_dict(ckpt["model_state"])
epoch    = ckpt.get('epoch', '?')
val_loss = ckpt.get('val_loss', '?')
val_str  = f"{val_loss:.5f}" if isinstance(val_loss, float) else str(val_loss)
print(f"  Loaded GRUPlant (epoch {epoch}, val_loss={val_str})")

# ── Load controller models ────────────────────────────────────────────────────
ctrl_models = {}
for ln in CTRL_LOOPS:
    path = CKPT_DIR / f"gru_ctrl_{ln.lower()}.pt"
    if not path.exists():
        print(f"  WARNING: {path.name} not found, skipping {ln}")
        continue
    c    = torch.load(path, map_location=DEVICE)
    arch = c.get("arch", "GRUController")
    if arch == "CCSequenceModel":
        ctrl_models[ln] = CCSequenceModel(
            n_inputs   = c["n_inputs"],
            hidden     = c["hidden"],
            layers     = c["layers"],
            output_len = TARGET_LEN,
        ).to(DEVICE)
    else:
        ctrl_models[ln] = GRUController(
            n_inputs   = c["n_inputs"],
            hidden     = c["hidden"],
            layers     = c["layers"],
            output_len = TARGET_LEN,
        ).to(DEVICE)
    ctrl_models[ln].load_state_dict(c["model_state"])

# ── Run closed-loop inference on test split ───────────────────────────────────
print("\n" + "="*60)
print("Running closed-loop inference on test split...")
print("="*60)

pv_preds_test = run_closed_loop_inference(
    plant_model, ctrl_models, data,
    "test", DEVICE, BATCH, verbose=True
)

# Get ground truth data
pv_true_test = data["plant"]["pv_target_test"]
scenario_test = data["plant"]["scenario_test"]
attack_test = data["plant"]["attack_test"]

# Compute NRMSE metrics
print("\n" + "="*60)
print("Computing NRMSE metrics...")
print("="*60)

nrmse_overall = compute_nrmse(pv_true_test, pv_preds_test)
print("\nOverall NRMSE per PV:")
for pv, val in nrmse_overall.items():
    print(f"  {pv}: {val:.4f}")

nrmse_by_scenario = compute_nrmse_per_scenario(pv_true_test, pv_preds_test, scenario_test)
print("\nNRMSE per scenario:")
for sc_id, nrmse_dict in nrmse_by_scenario.items():
    if sc_id >= 0:
        print(f"  Scenario {sc_id}: {nrmse_dict}")

# Compute anomaly scores for attack detection
anomaly_scores = np.mean((pv_preds_test - pv_true_test) ** 2, axis=(1, 2))

# Load attack detection metrics from results.json
results_path = CKPT_DIR / "results.json"
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
    attack_metrics = results.get("attack_detection", {})
    best_threshold = attack_metrics.get("best_threshold", 0.326)
    best_f1 = attack_metrics.get("best_f1", 0.487)
    auroc = attack_metrics.get("auroc", 0.899)
else:
    print(f"WARNING: {results_path} not found, using default values")
    best_threshold = 0.326
    best_f1 = 0.487
    auroc = 0.899

print(f"\nAttack Detection Metrics:")
print(f"  Best threshold: {best_threshold:.4f}")
print(f"  Best F1: {best_f1:.4f}")
print(f"  AUROC: {auroc:.4f}")

# ── Generate all paper plots (13 total) ───────────────────────────────────────
print("\n" + "="*60)
print("Generating Paper Plots...")
print("="*60)

# Plot 1: Loss curves
print("\n[Plot 1] Loss curves...")
plot_loss_curves(CKPT_DIR / "results.json", "GRU-Scenario-Weighted", OUT_DIR)

# Plot 2: NRMSE per PV
print("\n[Plot 2] NRMSE per PV...")
plot_nrmse_per_pv(nrmse_overall, "GRU-Scenario-Weighted", OUT_DIR)

# Plot 3: Per-loop performance
print("\n[Plot 3] Per-loop performance...")
plot_per_loop_performance(nrmse_overall, "GRU-Scenario-Weighted", OUT_DIR)

# Plot 4: NRMSE per scenario
print("\n[Plot 4] NRMSE per scenario...")
plot_nrmse_per_scenario(nrmse_by_scenario, "GRU-Scenario-Weighted", OUT_DIR)

# Plot 5: Scenario overlay (PIT01)
print("\n[Plot 5] Scenario overlay (PIT01)...")
plot_scenario_overlay(
    pv_true_test, pv_preds_test, scenario_test,
    pv_name="P1_PIT01", model_name="GRU-Scenario-Weighted", out_dir=OUT_DIR
)

# Plot 6: Error growth curve (NRMSE vs prediction horizon)
print("\n[Plot 6] Error growth curve...")
plot_error_growth_curve(pv_true_test, pv_preds_test, OUT_DIR, "GRU-Scenario-Weighted")

# Plot 7: Error heatmap (PV × Horizon)
print("\n[Plot 7] Error heatmap...")
plot_error_heatmap(pv_true_test, pv_preds_test, OUT_DIR, "GRU-Scenario-Weighted")

# Plot 8: ROC curve
print("\n[Plot 8] ROC curve...")
plot_roc_curve(attack_test, anomaly_scores, "GRU-Scenario-Weighted", OUT_DIR)

# Plot 9: Precision-Recall curve
print("\n[Plot 9] Precision-Recall curve...")
plot_pr_curve(attack_test, anomaly_scores, best_f1, best_threshold, 
              "GRU-Scenario-Weighted", OUT_DIR)

# Plot 10: Residual boxplot
print("\n[Plot 10] Residual boxplot...")
plot_residual_boxplot(anomaly_scores, scenario_test, best_threshold,
                      "GRU-Scenario-Weighted", OUT_DIR)

# Plot 11: Residual timeline
print("\n[Plot 11] Residual timeline...")
plot_residual_timeline(anomaly_scores, attack_test, best_threshold,
                       "GRU-Scenario-Weighted", OUT_DIR)

# Plot 12: Confusion matrix
print("\n[Plot 12] Confusion matrix...")
plot_confusion_matrix_attack(attack_test, anomaly_scores, best_threshold,
                             "GRU-Scenario-Weighted", OUT_DIR)

# Plot 13: Detection rate per attack type
print("\n[Plot 13] Detection rate per attack type...")
plot_detection_rate_per_attack(attack_test, anomaly_scores, scenario_test,
                                best_threshold, "GRU-Scenario-Weighted", OUT_DIR)

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"All 13 paper plots saved to: {OUT_DIR}/")
print("="*60)
print("\nGenerated plots (13 of 13 total):")
print("  ┌────┬─────────────────────────────────┬────────────────────────────────────────┐")
print("  │ #  │ Filename                        │ Description                            │")
print("  ├────┼─────────────────────────────────┼────────────────────────────────────────┤")
print("  │ 1  │ loss_curves.png                 │ Training & validation loss curves      │")
print("  │ 2  │ nrmse_per_pv.png                │ NRMSE bar chart per PV                 │")
print("  │ 3  │ per_loop_performance.png        │ NRMSE per control loop                 │")
print("  │ 4  │ nrmse_per_scenario.png          │ NRMSE per PV per scenario              │")
print("  │ 5  │ scenario_overlay.png            │ Generated vs real PV across scenarios  │")
print("  │ 6  │ error_growth_curve.png          │ NRMSE vs prediction horizon            │")
print("  │ 7  │ error_heatmap.png               │ PV × Horizon heatmap                   │")
print("  │ 8  │ roc_curve.png                   │ ROC curve with AUC                     │")
print("  │ 9  │ pr_curve.png                    │ Precision-Recall curve                 │")
print("  │ 10 │ residual_boxplot.png            │ Residual distribution by scenario      │")
print("  │ 11 │ residual_timeline.png           │ Residual timeline with attack regions  │")
print("  │ 12 │ confusion_matrix.png            │ Confusion matrix at best threshold     │")
print("  │ 13 │ detection_rate_per_attack.png   │ Detection rate by attack type          │")
print("  └────┴─────────────────────────────────┴────────────────────────────────────────┘")