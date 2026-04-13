"""
plot_transformer_ss.py — Multi-horizon closed-loop prediction plots for the Transformer-SS model.

For each scenario and each horizon (300s, 600s, 900s, 1800s):
  - Chains consecutive 180-step predictions to reach the full horizon
  - Shows predicted (red dashed) vs actual (blue solid) for all 5 PVs
  - Fills red shaded area between predicted and actual (error magnitude)
  - Uses 8 consecutive windows per horizon

Saves to: outputs/transformer_plant/plots/multi_horizon/
  scenario_0_normal_300s.png
  scenario_0_normal_600s.png
  ...
  scenario_3_AE_no_1800s.png
"""

from __future__ import annotations
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from transformer import TransformerPlant, TransformerController
from config import LOOPS, PV_COLS

# Load plot_utils via importlib to avoid module issues
_spec = importlib.util.spec_from_file_location(
    "plot_utils", ROOT / "03_model" / "plot_utils.py"
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
CTRL_LOOPS     = _mod.CTRL_LOOPS
SCENARIO_NAMES = _mod.SCENARIO_NAMES
SCENARIO_SHORT = _mod.SCENARIO_SHORT
PV_SHORT       = _mod.PV_SHORT

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = ROOT / "outputs" / "transformer_plant"
OUT_DIR  = CKPT_DIR / "plots" / "multi_horizon"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LEN = 180
N_WINDOWS  = 8
HORIZONS   = [300, 600, 900, 1800]

SCENARIO_ALL = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
data        = load_and_prepare_data()
plant_data  = data["plant"]
ctrl_data   = data["ctrl"]
metadata    = data["metadata"]
sensor_cols = metadata["sensor_cols"]

non_pv_cols     = [c for c in sensor_cols if c not in set(PV_COLS)]
col_to_idx      = {c: i for i, c in enumerate(non_pv_cols)}
ctrl_cv_col_idx = {
    ln: col_to_idx[LOOPS[ln].cv]
    for ln in CTRL_LOOPS if LOOPS[ln].cv in col_to_idx
}

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading checkpoints...")
ckpt = torch.load(CKPT_DIR / "transformer_plant.pt", map_location=DEVICE)
plant_model = TransformerPlant(
    n_plant_in  = ckpt["n_plant_in"],
    n_pv        = ckpt["n_pv"],
    d_model     = ckpt["d_model"],
    n_heads     = ckpt["n_heads"],
    n_layers    = ckpt["n_layers"],
    n_scenarios = ckpt["n_scenarios"],
    emb_dim     = ckpt.get("emb_dim", 32),
).to(DEVICE)
plant_model.load_state_dict(ckpt["model_state"])
plant_model.eval()

ctrl_models = {}
for ln in CTRL_LOOPS:
    path = CKPT_DIR / f"transformer_ctrl_{ln.lower()}.pt"
    if not path.exists():
        print(f"  WARNING: {path.name} not found, skipping {ln}")
        continue
    c = torch.load(path, map_location=DEVICE)
    ctrl_models[ln] = TransformerController(
        n_inputs   = c["n_inputs"],
        d_model    = c["hidden"],
        n_layers   = c["layers"],
        output_len = TARGET_LEN,
    ).to(DEVICE)
    ctrl_models[ln].load_state_dict(c["model_state"])
    ctrl_models[ln].eval()
print("  Models loaded.")


def run_one_window(x_cv_np, x_cv_tgt_np, pv_init_np, sc_np):
    """Run closed-loop for one window. Returns pv_pred (TARGET_LEN, n_pv)."""
    with torch.no_grad():
        x_cv = torch.tensor(x_cv_np[None]).float().to(DEVICE)
        xct  = torch.tensor(x_cv_tgt_np[None]).float().to(DEVICE).clone()
        pv_i = torch.tensor(pv_init_np[None]).float().to(DEVICE)
        sc   = torch.tensor([sc_np]).long().to(DEVICE)

        for ln in CTRL_LOOPS:
            if ln not in ctrl_cv_col_idx or ln not in ctrl_models:
                continue
            cv_pred = ctrl_models[ln].predict(x_cv[:, :, :2], target_len=TARGET_LEN)
            cv_i = ctrl_cv_col_idx[ln]
            xct[:, :, cv_i:cv_i+1] = cv_pred

        pred = plant_model.predict(x_cv, xct, pv_i, sc)
        return pred.cpu().numpy()[0]


def chain_predictions(indices, X, X_cv_tgt, pv_init, scenario, n_steps):
    """Chain n_steps consecutive windows. Returns (true_concat, pred_concat)."""
    true_parts, pred_parts = [], []
    pv_target = plant_data["pv_target_val"] if scenario == 0 \
                else plant_data["pv_target_test"]

    for idx in indices[:n_steps]:
        if idx >= len(X):
            break
        true_parts.append(pv_target[idx])
        pred_parts.append(run_one_window(X[idx], X_cv_tgt[idx], pv_init[idx], scenario))

    return np.concatenate(true_parts, axis=0), np.concatenate(pred_parts, axis=0)


def plot_horizon(true_arr, pred_arr, scenario_id, horizon_s, n_windows, out_path):
    n_pv        = true_arr.shape[1]
    t           = np.arange(true_arr.shape[0])
    horizon_min = horizon_s // 60
    sc_name     = SCENARIO_ALL.get(scenario_id, f"Scenario {scenario_id}")

    fig, _axes = plt.subplots(n_pv, 1, figsize=(16, 3.2 * n_pv), sharex=True)
    axes: list = [_axes] if n_pv == 1 else [_axes[i] for i in range(n_pv)]  # type: ignore[index]

    fig.suptitle(
        f"Transformer-SS — Scenario: {sc_name} | Horizon: {horizon_s}s ({horizon_min} min) | {n_windows} chained windows",
        fontsize=13, fontweight="bold", y=1.01
    )

    for k, (ax, pv_name) in enumerate(zip(axes, PV_SHORT)):
        true_k = true_arr[:, k]
        pred_k = pred_arr[:, k]

        ax.plot(t, true_k, color="steelblue", linewidth=1.5, label="Ground truth", zorder=3)
        ax.plot(t, pred_k, color="crimson",   linewidth=1.2,
                linestyle="--", label="Prediction (closed-loop)", zorder=3)
        ax.fill_between(t, true_k, pred_k,
                        where=np.ones_like(t, dtype=bool), interpolate=True,
                        color="red", alpha=0.18, label="Error magnitude", zorder=2)

        for w in range(1, n_windows):
            ax.axvline(x=w * TARGET_LEN, color="gray",
                       linewidth=0.8, linestyle=":", alpha=0.6)

        rmse  = np.sqrt(np.mean((pred_k - true_k) ** 2))
        rng   = max(float(true_k.max() - true_k.min()), 1e-6)
        nrmse = rmse / rng
        ax.set_title(f"{pv_name}   [NRMSE = {nrmse:.4f}]", fontsize=10)
        ax.set_ylabel("Value (normalised)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Time steps (1 step = 1 second)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Main loop: all scenarios × all horizons ───────────────────────────────────
print(f"\nGenerating multi-horizon plots → {OUT_DIR}/\n")

for sc_id in [0, 1, 2, 3]:
    sc_short = {0: "normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}[sc_id]
    print(f"Scenario {sc_id} ({sc_short})...")

    if sc_id == 0:
        X        = plant_data["X_val"]
        X_cv_tgt = plant_data["X_cv_target_val"]
        pv_init  = plant_data["pv_init_val"]
        scenario_labels = plant_data["scenario_val"]
    else:
        X        = plant_data["X_test"]
        X_cv_tgt = plant_data["X_cv_target_test"]
        pv_init  = plant_data["pv_init_test"]
        scenario_labels = plant_data["scenario_test"]

    indices = np.where(scenario_labels == sc_id)[0]
    if len(indices) == 0:
        print(f"  No windows for scenario {sc_id}, skipping.")
        continue

    variances = plant_data["pv_target_val"][indices].var(axis=(1, 2)) if sc_id == 0 \
                else plant_data["pv_target_test"][indices].var(axis=(1, 2))
    best_start = int(indices[variances.argmax()])

    chain = []
    for idx in range(best_start, len(X)):
        if scenario_labels[idx] == sc_id:
            chain.append(idx)
        if len(chain) >= N_WINDOWS:
            break
    if len(chain) < 2:
        chain = list(indices[:N_WINDOWS])

    print(f"  Using {len(chain)} windows starting at index {chain[0]}")

    for horizon_s in HORIZONS:
        n_steps = min(max(1, int(np.ceil(horizon_s / TARGET_LEN))), len(chain))
        actual_horizon = n_steps * TARGET_LEN

        true_arr, pred_arr = chain_predictions(
            chain, X, X_cv_tgt, pv_init, sc_id, n_steps
        )
        out_path = OUT_DIR / f"scenario_{sc_id}_{sc_short}_{horizon_s}s.png"
        plot_horizon(true_arr, pred_arr, sc_id, actual_horizon, n_steps, out_path)

print(f"\nDone. All plots saved to {OUT_DIR}/")
