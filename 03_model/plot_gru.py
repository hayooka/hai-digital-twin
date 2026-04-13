"""
plot_gru.py — Generate all diagnostic plots for the trained GRU model.

Plots saved to outputs/gru_plant/plots/:
    loss_curves.png           — train/val loss + scheduled-sampling ratio
    nrmse_per_pv.png          — closed-loop NRMSE per PV (val split)
    scenario_1_AP_no.png      — predicted vs real PV for AP_no attack windows
    scenario_2_AP_with.png    — predicted vs real PV for AP_with attack windows
    scenario_3_AE_no.png      — predicted vs real PV for AE_no attack windows

Usage:
    python 03_model/plot_gru.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCClassifierRegressor
from config import LOOPS
from plot_utils import plot_all, CTRL_LOOPS

CKPT_DIR  = ROOT / "outputs" / "gru_plant"
OUT_DIR   = CKPT_DIR / "plots"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH     = 128

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
data      = load_and_prepare_data()
plant_data = data["plant"]
ctrl_data  = data["ctrl"]
TARGET_LEN = data["metadata"]["target_len"]

# ── Load plant model ──────────────────────────────────────────────────────────
ckpt = torch.load(CKPT_DIR / "gru_plant.pt", map_location=DEVICE)
plant_model = GRUPlant(
    n_plant_in  = ckpt["n_plant_in"],
    n_pv        = ckpt["n_pv"],
    hidden      = ckpt["hidden"],
    layers      = ckpt["layers"],
    n_scenarios = ckpt["n_scenarios"],
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
    c = torch.load(path, map_location=DEVICE)
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
    model_name   = "GRU",
    out_dir      = OUT_DIR,
    results_path = CKPT_DIR / "results.json",
    device       = DEVICE,
    batch_size   = BATCH,
)
