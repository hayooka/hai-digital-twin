"""
plot_error_growth_fixed.py — Correct per-step NRMSE growth over the 180-step horizon.

For each prediction step t (1..180), compute NRMSE across all normal test windows.
This shows true within-horizon error accumulation without chaining artifacts.

Output: plots/error_growth_curve.png  (overwrites the old incorrect version)
"""

import sys
import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

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
OUT_DIR  = ROOT / "plots"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH    = 128
PV_SHORT = [p.replace("P1_", "") for p in PV_COLS]
PV_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']


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


# ── load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
data        = load_and_prepare_data()
plant_data  = data["plant"]
ctrl_data   = data["ctrl"]
sensor_cols = data["metadata"]["sensor_cols"]
TARGET_LEN  = data["metadata"]["target_len"]
N_PV        = plant_data["n_pv"]

augment_ctrl_data(ctrl_data, sensor_cols)

pv_set      = set(PV_COLS)
non_pv_cols = [c for c in sensor_cols if c not in pv_set]
col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}
ctrl_cv_col_idx = {ln: col_to_idx[LOOPS[ln].cv]
                   for ln in CTRL_LOOPS if LOOPS[ln].cv in col_to_idx}

print("Loading model...")
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

# ── inference on normal test windows only ──────────────────────────────────────
X_test         = plant_data["X_test"]
X_cv_tgt_test  = plant_data["X_cv_target_test"]
pv_init_test   = plant_data["pv_init_test"]
scenario_test  = plant_data["scenario_test"]
pv_true_test   = plant_data["pv_target_test"]

normal_mask = scenario_test == 0
print(f"Normal test windows: {normal_mask.sum()}")

X_n       = X_test[normal_mask]
Xcvt_n    = X_cv_tgt_test[normal_mask]
pvinit_n  = pv_init_test[normal_mask]
sc_n      = scenario_test[normal_mask]
pvtrue_n  = pv_true_test[normal_mask]
N         = len(X_n)

pv_preds = np.zeros((N, TARGET_LEN, N_PV), dtype=np.float32)

print("Running inference...")
with torch.no_grad():
    for i in range(0, N, BATCH):
        sl       = slice(i, i + BATCH)
        x_cv_b   = torch.tensor(X_n[sl]).float().to(DEVICE)
        xct_b    = torch.tensor(Xcvt_n[sl]).float().to(DEVICE).clone()
        pv_init_b = torch.tensor(pvinit_n[sl]).float().to(DEVICE)
        sc_b     = torch.tensor(sc_n[sl]).long().to(DEVICE)

        for ln in CTRL_LOOPS:
            if ln not in ctrl_cv_col_idx:
                continue
            Xc = torch.tensor(ctrl_data[ln]['X_test'][normal_mask][sl]).float().to(DEVICE)
            cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
            ci = ctrl_cv_col_idx[ln]
            xct_b[:, :, ci:ci + 1] = cv_pred

        pv_seq = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
        pv_preds[i:i + x_cv_b.size(0)] = pv_seq.cpu().numpy()

# ── compute per-step NRMSE ─────────────────────────────────────────────────────
# pv_preds: (N, 180, 5), pvtrue_n: (N, 180, 5)
# At each step t, compute NRMSE across all N windows

print("Computing per-step NRMSE...")
steps = np.arange(1, TARGET_LEN + 1)

# per-step, per-PV squared error: (N, 180, 5)
sq_err = (pvtrue_n - pv_preds) ** 2

# cumulative mean squared error up to step t: shape (180, 5)
# cumsum over steps then divide by step index
cumsum_err = np.cumsum(sq_err.mean(axis=0), axis=0)          # (180, 5)
cum_mse    = cumsum_err / steps[:, None]                      # (180, 5) — cumulative MSE

# normalise by signal range per PV (computed over all N windows, all steps)
pv_ranges = pvtrue_n.max(axis=(0, 1)) - pvtrue_n.min(axis=(0, 1))  # (5,)
pv_ranges  = np.maximum(pv_ranges, 1e-8)

cum_nrmse  = np.sqrt(cum_mse) / pv_ranges[None, :]           # (180, 5)
overall    = cum_nrmse.mean(axis=1)                           # (180,)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(steps, overall, 'k-', lw=2.5, label='Overall Average', zorder=5)

for k, (pv, color) in enumerate(zip(PV_SHORT, PV_COLORS)):
    ax.plot(steps, cum_nrmse[:, k], '--', lw=1.8, color=color,
            alpha=0.85, label=pv)

ax.axhline(y=0.10, color='red', linestyle='--', lw=1.5, label='10% Deployment Gate')
ax.axvline(x=TARGET_LEN, color='gray', linestyle=':', lw=1.2,
           label=f'Training Horizon ({TARGET_LEN}s)')

ax.set_xlabel('Prediction Step (seconds)', fontsize=12)
ax.set_ylabel('Cumulative NRMSE', fontsize=12)
ax.set_title('GRU-Scenario-Weighted — Cumulative Error Growth Over Prediction Horizon\n'
             f'(averaged across {N} normal test windows)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, ncol=2)
ax.grid(True, linestyle='--', alpha=0.35)
ax.set_xlim(1, TARGET_LEN)
ax.set_ylim(0, None)

fig.tight_layout()
out_path = OUT_DIR / "error_growth_curve.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_path}")

# ── print summary ──────────────────────────────────────────────────────────────
print("\nCumulative NRMSE at key steps:")
print(f"{'PV':<8} {'step 1':>8} {'step 60':>8} {'step 120':>9} {'step 180':>9}")
print("-" * 42)
for k, pv in enumerate(PV_SHORT):
    print(f"{pv:<8} {cum_nrmse[0,k]:>8.4f} {cum_nrmse[59,k]:>8.4f} "
          f"{cum_nrmse[119,k]:>9.4f} {cum_nrmse[179,k]:>9.4f}")
print(f"{'Overall':<8} {overall[0]:>8.4f} {overall[59]:>8.4f} "
      f"{overall[119]:>9.4f} {overall[179]:>9.4f}")
