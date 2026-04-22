"""
plot_error_growth_longterm.py — Long-horizon NRMSE across chained windows.

Chains consecutive normal test windows autoregressively (feeding predicted PVs
as the next window's initial state). Computes NRMSE at every individual step
— NOT cumulative averaging — so error growth is never masked by easy segments.

Output: plots/error_growth_longterm_with_cumulative.png
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

CKPT_DIR  = ROOT / "outputs" / "pipeline" / "gru_scenario_weighted"
OUT_DIR   = ROOT / "plots"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH     = 128
PV_SHORT  = [p.replace("P1_", "") for p in PV_COLS]
PV_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

# Number of consecutive normal windows to chain (30 × 180s = 5400s = 1.5 hours)
N_CHAIN = 30


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

# Debug: Print available keys in plant_data
print(f"Available keys in plant_data: {plant_data.keys()}")

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

# ── find consecutive normal windows ───────────────────────────────────────────
scenario_test = plant_data["scenario_test"]
normal_indices = np.where(scenario_test == 0)[0]

# Find the longest run of consecutive indices
best_start, best_len, cur_start, cur_len = 0, 0, normal_indices[0], 1
for i in range(1, len(normal_indices)):
    if normal_indices[i] == normal_indices[i-1] + 1:
        cur_len += 1
    else:
        if cur_len > best_len:
            best_len, best_start = cur_len, cur_start
        cur_start, cur_len = normal_indices[i], 1
if cur_len > best_len:
    best_len, best_start = cur_len, cur_start

n_chain = min(N_CHAIN, best_len)
start_idx = best_start
print(f"Chaining {n_chain} consecutive normal windows from index {start_idx} "
      f"({n_chain * TARGET_LEN}s = {n_chain * TARGET_LEN / 60:.1f} min)")

# ── autoregressive chain ───────────────────────────────────────────────────────
X_test        = plant_data["X_test"]
X_cv_tgt_test = plant_data["X_cv_target_test"]
pv_target_test = plant_data["pv_target_test"]

# per-step squared error: shape (n_chain * TARGET_LEN, N_PV)
all_true = []
all_pred = []

with torch.no_grad():
    # encode first window
    x0   = torch.tensor(X_test[start_idx:start_idx+1]).float().to(DEVICE)
    sc0  = torch.tensor(scenario_test[start_idx:start_idx+1]).long().to(DEVICE)
    emb  = plant_model.scenario_emb(sc0).unsqueeze(1).expand(-1, x0.size(1), -1)
    _, h = plant_model.encoder(torch.cat([x0, emb], dim=-1))

    pv_prev = torch.tensor(
        pv_target_test[start_idx:start_idx+1, 0, :]
    ).float().to(DEVICE)

    for w in range(n_chain):
        idx  = start_idx + w
        xct  = torch.tensor(X_cv_tgt_test[idx:idx+1]).float().to(DEVICE).clone()
        sc_w = torch.tensor(scenario_test[idx:idx+1]).long().to(DEVICE)

        # inject controller predictions
        for ln in CTRL_LOOPS:
            if ln not in ctrl_cv_col_idx:
                continue
            Xc = torch.tensor(
                ctrl_data[ln]['X_test'][idx:idx+1]
            ).float().to(DEVICE)
            cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
            ci = ctrl_cv_col_idx[ln]
            xct[:, :, ci:ci+1] = cv_pred

        # step-by-step decode
        win_pred = []
        for t in range(TARGET_LEN):
            dec_in = torch.cat([xct[:, t, :], pv_prev], dim=-1).unsqueeze(1)
            out, h = plant_model.decoder(dec_in, h)
            h_out  = out.squeeze(1)
            if getattr(plant_model, "scenario_heads", False):
                pv_pred = torch.zeros(1, N_PV, device=DEVICE)
                for sc_id in range(plant_model.n_scenarios):
                    mask = (sc_w == sc_id)
                    if mask.any():
                        pv_pred[mask] = plant_model.fc_heads[sc_id](h_out[mask])
            else:
                pv_pred = plant_model.fc(h_out)
            win_pred.append(pv_pred.cpu().numpy())
            pv_prev = pv_pred

        win_pred = np.concatenate(win_pred, axis=0)  # (TARGET_LEN, N_PV)
        all_pred.append(win_pred)
        all_true.append(pv_target_test[idx])         # (TARGET_LEN, N_PV)

all_true = np.concatenate(all_true, axis=0)  # (n_chain*180, N_PV)
all_pred = np.concatenate(all_pred, axis=0)

total_steps = len(all_true)
time_axis   = np.arange(1, total_steps + 1)

# ── FIX: use global PV range (from training data) ────────────────────────────
# Try different possible key names for training PV data
if "pv_train" in plant_data:
    pv_train = plant_data["pv_train"]
elif "pv_target_train" in plant_data:
    pv_train = plant_data["pv_target_train"]
elif "train_pv_target" in plant_data:
    pv_train = plant_data["train_pv_target"]
else:
    # Fallback: use test data ranges (not ideal but better than nothing)
    print("Warning: No training PV data found, using test data for normalization")
    pv_train = pv_target_test

pv_flat  = pv_train.reshape(-1, pv_train.shape[-1])
pv_ranges = pv_flat.max(axis=0) - pv_flat.min(axis=0)
pv_ranges = np.maximum(pv_ranges, 1e-8)

print(f"PV ranges for normalization: {pv_ranges}")

# ── per-step error ────────────────────────────────────────────────────────────
sq_err = (all_true - all_pred) ** 2
step_nrmse = np.sqrt(sq_err) / pv_ranges[None, :]

# ── smoothing ─────────────────────────────────────────────────────────────────
def rolling_mean(x, w=60):
    out = np.convolve(x, np.ones(w)/w, mode='same')
    # fix edge effects
    for i in range(w//2):
        out[i]      = x[:i+w//2+1].mean()
        out[-(i+1)] = x[-(i+w//2+1):].mean()
    return out

step_nrmse_smooth = np.stack(
    [rolling_mean(step_nrmse[:, k]) for k in range(N_PV)], axis=1
)

overall_raw    = step_nrmse.mean(axis=1)
overall_smooth = step_nrmse_smooth.mean(axis=1)

# ── cumulative NRMSE over long horizon ───────────────────────────────────
cumulative_nrmse = np.cumsum(overall_raw) / np.arange(1, total_steps + 1)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))

# per-step (existing)
ax.plot(time_axis, overall_smooth, 'k-', lw=2.5,
        label='Per-step NRMSE (60s smoothed)', zorder=5)

# cumulative
ax.plot(time_axis, cumulative_nrmse, color='red', lw=2.2,
        label='Cumulative NRMSE (long-horizon)', alpha=0.9)

# PV-specific (optional)
for k, (pv, color) in enumerate(zip(PV_SHORT, PV_COLORS)):
    ax.plot(time_axis, step_nrmse_smooth[:, k], '--', lw=1.2,
            color=color, alpha=0.6, label=pv if k == 0 else "")

# window boundaries
for w in range(1, n_chain):
    ax.axvline(x=w * TARGET_LEN, color='gray',
               lw=0.5, alpha=0.3, linestyle=':')

# deployment threshold
ax.axhline(y=0.10, color='red', linestyle='--',
           lw=1.5, label='10% Deployment Gate')

# labels
total_min = total_steps / 60
ax.set_title(
    f'GRU-Scenario-Weighted — Error Behavior over {total_min:.0f}-Minute Horizon\n'
    f'(Per-step vs Cumulative, {n_chain} chained windows)',
    fontsize=12, fontweight='bold'
)

ax.set_xlabel('Time (minutes)', fontsize=12)
ax.set_ylabel('NRMSE', fontsize=12)

# x-axis in minutes
xticks = np.arange(0, total_steps + 1, TARGET_LEN * 5)
ax.set_xticks(xticks)
ax.set_xticklabels([f'{int(x/60)}m' for x in xticks])

ax.legend(loc='upper right', fontsize=9, ncol=2)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_xlim(1, total_steps)
ax.set_ylim(bottom=0)

fig.tight_layout()
out_path = OUT_DIR / "error_growth_longterm_with_cumulative.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Saved: {out_path}")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("KEY METRICS")
print("="*60)
print(f"Final cumulative NRMSE: {cumulative_nrmse[-1]:.4f}")
print(f"Max per-step NRMSE (smoothed): {overall_smooth.max():.4f}")
print(f"Min per-step NRMSE (smoothed): {overall_smooth.min():.4f}")
print(f"Mean per-step NRMSE (smoothed): {overall_smooth.mean():.4f}")

# Check deployment readiness (10% threshold)
below_threshold = overall_smooth < 0.10
if below_threshold.any():
    first_good = np.argmax(below_threshold)
    print(f"\nFirst sustained below 10% threshold: {first_good//60:.1f} minutes")
    # Find longest consecutive run below threshold
    runs = np.diff(np.concatenate(([0], below_threshold, [0])))
    run_starts = np.where(runs == 1)[0]
    run_ends = np.where(runs == -1)[0]
    if len(run_starts) > 0:
        longest_run = max(run_ends - run_starts)
        print(f"Longest consecutive run below 10%: {longest_run//60:.1f} minutes")
else:
    print("\nNever drops below 10% threshold")

print("\n" + "-"*60)
print("PER-STEP NRMSE AT KEY HORIZONS (smoothed):")
print(f"{'Horizon':<12} {'Time(min)':<10} {'NRMSE':<10}")
print("-"*60)

checkpoints = [TARGET_LEN, TARGET_LEN*5, TARGET_LEN*10, TARGET_LEN*20, total_steps-1]
for c in checkpoints:
    if c < len(overall_smooth):
        print(f"{c:3d} steps    {c//60:3d} min       {overall_smooth[c]:.4f}")

print("\n" + "-"*60)
print("PV-SPECIFIC FINAL ERRORS:")
for k, pv in enumerate(PV_SHORT):
    print(f"{pv:<8}: final NRMSE = {step_nrmse_smooth[-1, k]:.4f}")

print("\n" + "="*60)
print(f"Analysis complete! Total horizon: {total_min:.0f} minutes")
print("="*60)