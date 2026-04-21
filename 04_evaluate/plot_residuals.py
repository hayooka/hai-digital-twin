"""
plot_residuals.py — Compare normal-only model predictions vs actual readings.

For each attack scenario:
  - Feed the attack window data to the normal-only model (scenario forced = 0)
  - Plot predicted (what should have been) vs actual (what happened)
  - Plot residual (actual - predicted) = the attack footprint

Saved to plots/normal_only/
"""

import sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))
sys.path.insert(0, str(ROOT / "04_evaluate"))

import joblib
from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCSequenceModel
from config import LOOPS, PV_COLS, PROCESSED_DATA_DIR
from plot_utils import CTRL_LOOPS

EXTRA_CHANNELS = {
    'PC': ['P1_PCV02D',  'P1_FT01',   'P1_TIT01'],
    'LC': ['P1_FT03',   'P1_FCV03D', 'P1_PCV01D'],
    'FC': ['P1_PIT01',  'P1_LIT01',  'P1_TIT03'],
    'TC': ['P1_FT02',   'P1_PIT02',  'P1_TIT02'],
    'CC': ['P1_PP04D',  'P1_FCV03D', 'P1_PCV02D'],
}
CTRL_HIDDEN = {'PC': 64, 'LC': 64, 'FC': 128, 'TC': 64, 'CC': 64}
SCENARIO_NAMES = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}
SCENARIO_COLORS = {0: "#2196F3", 1: "#FF9800", 2: "#F44336", 3: "#9C27B0"}

CKPT_DIR = ROOT / "outputs/pipeline/gru_normal_only"
OUT_DIR  = ROOT / "plots/normal_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH    = 128

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
data        = load_and_prepare_data()
plant_data  = data["plant"]
ctrl_data   = data["ctrl"]
sensor_cols = data["metadata"]["sensor_cols"]
TARGET_LEN  = data["metadata"]["target_len"]

def augment_ctrl_data(ctrl_data, sensor_cols):
    plant_scaler = joblib.load(f"{PROCESSED_DATA_DIR}/scaler.pkl")
    npz_train = np.load(f"{PROCESSED_DATA_DIR}/train_data.npz")
    npz_val   = np.load(f"{PROCESSED_DATA_DIR}/val_data.npz")
    npz_test  = np.load(f"{PROCESSED_DATA_DIR}/test_data.npz")
    col_idx   = {c: i for i, c in enumerate(sensor_cols)}
    for ln, extra_cols in EXTRA_CHANNELS.items():
        for extra_col in extra_cols:
            if extra_col not in col_idx: continue
            ei = col_idx[extra_col]
            mean_e, scale_e = plant_scaler.mean_[ei], plant_scaler.scale_[ei]
            for split, npz in [('train', npz_train), ('val', npz_val), ('test', npz_test)]:
                raw = npz['X'][:, :, [ei]].astype(np.float32)
                ctrl_data[ln][f'X_{split}'] = np.concatenate(
                    [ctrl_data[ln][f'X_{split}'], (raw - mean_e) / scale_e], axis=-1)

augment_ctrl_data(ctrl_data, sensor_cols)

pv_set      = set(PV_COLS)
non_pv_cols = [c for c in sensor_cols if c not in pv_set]
col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}
ctrl_cv_col_idx = {ln: col_to_idx[LOOPS[ln].cv]
                   for ln in CTRL_LOOPS if LOOPS[ln].cv in col_to_idx}

X_test         = plant_data['X_test']
X_cv_tgt_test  = plant_data['X_cv_target_test']
pv_init_test   = plant_data['pv_init_test']
pv_target_test = plant_data['pv_target_test']
scenario_test  = plant_data['scenario_test']
N_PV           = plant_data['n_pv']

# ── Load normal-only model ─────────────────────────────────────────────────────
print("Loading normal-only model...")
ckpt = torch.load(CKPT_DIR / "gru_plant.pt", map_location=DEVICE)
plant_model = GRUPlant(
    n_plant_in  = ckpt["n_plant_in"],
    n_pv        = ckpt["n_pv"],
    hidden      = ckpt["hidden"],
    layers      = ckpt["layers"],
    n_scenarios = ckpt["n_scenarios"],
).to(DEVICE)
plant_model.load_state_dict(ckpt["model_state"])
plant_model.eval()

ctrl_models = {}
for ln in CTRL_LOOPS:
    path = CKPT_DIR / f"gru_ctrl_{ln.lower()}.pt"
    if not path.exists(): continue
    c    = torch.load(path, map_location=DEVICE)
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

# ── Run inference — always force scenario=0 ────────────────────────────────────
print("Running inference (scenario forced = Normal)...")
N_test      = len(X_test)
pv_preds    = np.zeros((N_test, TARGET_LEN, N_PV), dtype=np.float32)

with torch.no_grad():
    for i in range(0, N_test, BATCH):
        sl        = slice(i, i + BATCH)
        x_cv_b    = torch.tensor(X_test[sl]).float().to(DEVICE)
        xct_b     = torch.tensor(X_cv_tgt_test[sl]).float().to(DEVICE).clone()
        pv_init_b = torch.tensor(pv_init_test[sl]).float().to(DEVICE)
        B_actual  = x_cv_b.size(0)
        sc_b      = torch.zeros(B_actual, dtype=torch.long).to(DEVICE)

        for ln in CTRL_LOOPS:
            if ln not in ctrl_cv_col_idx: continue
            Xc      = torch.tensor(ctrl_data[ln]['X_test'][sl]).float().to(DEVICE)
            cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
            xct_b[:, :, ctrl_cv_col_idx[ln]:ctrl_cv_col_idx[ln] + 1] = cv_pred

        pv_seq = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
        pv_preds[i:i + B_actual] = pv_seq.cpu().numpy()

residuals = pv_target_test - pv_preds  # actual - normal_prediction

# ── Plot 1: Residual magnitude per scenario per PV ────────────────────────────
print("Plotting residual magnitudes...")
fig, axes = plt.subplots(1, N_PV, figsize=(18, 4), sharey=False)
fig.suptitle("Mean Absolute Residual per Attack Scenario\n(actual − normal prediction)",
             fontsize=13, fontweight='bold')

sc_ids   = [sc for sc in SCENARIO_NAMES if (scenario_test == sc).sum() > 0]
sc_names = [SCENARIO_NAMES[sc] for sc in sc_ids]
colors   = [SCENARIO_COLORS[sc] for sc in sc_ids]

for k, (ax, pv) in enumerate(zip(axes, PV_COLS)):
    values = []
    for sc_id in sc_ids:
        mask = (scenario_test == sc_id)
        values.append(np.abs(residuals[mask, :, k]).mean())
    bars = ax.bar(sc_names, values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title(pv.replace("P1_", ""), fontsize=10)
    ax.set_ylabel("MAE" if k == 0 else "")
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(OUT_DIR / "residual_per_scenario.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: residual_per_scenario.png")

# ── Plot 2: Time-series — predicted normal vs actual for each attack type ──────
print("Plotting time-series comparison per attack type...")

for sc_id in sc_ids:
    if sc_id == 0: continue  # skip normal
    sc_name = SCENARIO_NAMES[sc_id]
    color   = SCENARIO_COLORS[sc_id]
    mask    = np.where(scenario_test == sc_id)[0]
    if len(mask) == 0: continue

    # Pick the window with the largest residual (most visible attack)
    res_mag = np.abs(residuals[mask]).mean(axis=(1, 2))
    idx     = mask[np.argmax(res_mag)]

    actual    = pv_target_test[idx]   # (T, N_PV)
    predicted = pv_preds[idx]          # (T, N_PV)  — normal prediction
    residual  = residuals[idx]         # (T, N_PV)
    t         = np.arange(TARGET_LEN)

    fig, axes = plt.subplots(3, N_PV, figsize=(20, 9), sharex=True)
    fig.suptitle(f"Attack Type: {sc_name}  |  Normal-Only Model Comparison\n"
                 f"(window with largest residual from test set)",
                 fontsize=12, fontweight='bold')

    for k, pv in enumerate(PV_COLS):
        # Row 0: actual vs predicted
        axes[0, k].plot(t, actual[:, k],    color='black',  lw=1.2, label='Actual')
        axes[0, k].plot(t, predicted[:, k], color='steelblue', lw=1.2,
                        linestyle='--', label='Normal pred.')
        axes[0, k].set_title(pv.replace("P1_", ""), fontsize=9)
        if k == 0:
            axes[0, k].set_ylabel("Scaled value", fontsize=8)
            axes[0, k].legend(fontsize=7, loc='upper right')

        # Row 1: residual
        axes[1, k].plot(t, residual[:, k], color=color, lw=1.0)
        axes[1, k].axhline(0, color='gray', lw=0.5, linestyle='--')
        axes[1, k].fill_between(t, 0, residual[:, k], alpha=0.25, color=color)
        if k == 0:
            axes[1, k].set_ylabel("Residual\n(actual−pred)", fontsize=8)

        # Row 2: absolute residual
        axes[2, k].plot(t, np.abs(residual[:, k]), color=color, lw=1.0)
        if k == 0:
            axes[2, k].set_ylabel("|Residual|", fontsize=8)
        axes[2, k].set_xlabel("Time step", fontsize=8)

    plt.tight_layout()
    fname = OUT_DIR / f"comparison_{sc_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: comparison_{sc_name}.png")

# ── Plot 3: Residual heatmap — which PV is most affected per scenario ──────────
print("Plotting residual heatmap...")
fig, ax = plt.subplots(figsize=(10, 4))

sc_labels  = [SCENARIO_NAMES[sc] for sc in sc_ids if sc != 0]
sc_ids_atk = [sc for sc in sc_ids if sc != 0]
pv_labels  = [p.replace("P1_", "") for p in PV_COLS]
matrix     = np.zeros((len(sc_ids_atk), N_PV))

for i, sc_id in enumerate(sc_ids_atk):
    mask = (scenario_test == sc_id)
    matrix[i] = np.abs(residuals[mask]).mean(axis=(0, 1))

# Normalise row-wise so colour shows relative impact within each attack
matrix_norm = matrix / (matrix.max(axis=1, keepdims=True) + 1e-9)

im = ax.imshow(matrix_norm, cmap='Reds', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(N_PV));    ax.set_xticklabels(pv_labels, fontsize=10)
ax.set_yticks(range(len(sc_labels))); ax.set_yticklabels(sc_labels, fontsize=10)
ax.set_title("Relative Attack Footprint per PV\n(row-normalised mean absolute residual)",
             fontsize=11, fontweight='bold')

for i in range(len(sc_ids_atk)):
    for j in range(N_PV):
        ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center',
                fontsize=8, color='black' if matrix_norm[i,j] < 0.6 else 'white')

plt.colorbar(im, ax=ax, label='Normalised residual')
plt.tight_layout()
plt.savefig(OUT_DIR / "residual_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: residual_heatmap.png")

print(f"\nAll plots saved to {OUT_DIR}/")
