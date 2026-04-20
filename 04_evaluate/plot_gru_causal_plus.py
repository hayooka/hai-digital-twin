"""
plot_gru_causal_plus.py — Generate all diagnostic plots for the final model.

Reads checkpoints from outputs/gru_scenario_weighted/ (produced by train_gru_scenario_weighted.py).
Plots saved to plots/.

Usage:
    python 04_evaluate/plot_gru_causal_plus.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))
sys.path.insert(0, str(ROOT / "04_evaluate"))

from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCSequenceModel
from config import LOOPS, PROCESSED_DATA_DIR
from plot_utils import plot_all, CTRL_LOOPS

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
    import joblib, numpy as np
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

# ── Plot ──────────────────────────────────────────────────────────────────────
plot_all(
    plant_model  = plant_model,
    ctrl_models  = ctrl_models,
    data         = data,
    model_name   = "GRU-Scenario-Weighted",
    out_dir      = OUT_DIR,
    results_path = CKPT_DIR / "results.json",
    device       = DEVICE,
    batch_size   = BATCH,
)
