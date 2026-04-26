"""
plot_transformer.py — Generate all diagnostic plots for the trained Transformer model.

Plots saved to outputs/transformer_plant/plots/:
    loss_curves.png           — train/val loss + scheduled-sampling ratio
    nrmse_per_pv.png          — closed-loop NRMSE per PV (val split)
    scenario_1_AP_no.png      — predicted vs real PV for AP_no attack windows
    scenario_2_AP_with.png    — predicted vs real PV for AP_with attack windows
    scenario_3_AE_no.png      — predicted vs real PV for AE_no attack windows

Usage:
    python 03_model/plot_transformer.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from transformer import TransformerPlant, TransformerController
from plot_utils import plot_all, CTRL_LOOPS

CKPT_DIR = ROOT / "outputs" / "transformer_plant"
OUT_DIR  = CKPT_DIR / "plots"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH    = 32   # smaller batch: Transformer uses more memory

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
data       = load_and_prepare_data()
ctrl_data  = data["ctrl"]
TARGET_LEN = data["metadata"]["target_len"]

# ── Load plant model ──────────────────────────────────────────────────────────
ckpt = torch.load(CKPT_DIR / "transformer_plant.pt", map_location=DEVICE)
plant_model = TransformerPlant(
    n_plant_in  = ckpt["n_plant_in"],
    n_pv        = ckpt["n_pv"],
    d_model     = ckpt["d_model"],
    n_heads     = ckpt["n_heads"],
    n_layers    = ckpt["n_layers"],
    n_scenarios = ckpt["n_scenarios"],
).to(DEVICE)
plant_model.load_state_dict(ckpt["model_state"])
epoch    = ckpt.get('epoch', '?')
val_loss = ckpt.get('val_loss', '?')
val_str  = f"{val_loss:.5f}" if isinstance(val_loss, float) else str(val_loss)
print(f"  Loaded TransformerPlant (epoch {epoch}, val_loss={val_str})")

# ── Load controller models ────────────────────────────────────────────────────
ctrl_models = {}
for ln in CTRL_LOOPS:
    path = CKPT_DIR / f"transformer_ctrl_{ln.lower()}.pt"
    if not path.exists():
        print(f"  WARNING: {path.name} not found, skipping {ln}")
        continue
    c = torch.load(path, map_location=DEVICE)
    ctrl_models[ln] = TransformerController(
        n_inputs   = c["n_inputs"],
        d_model    = c.get("hidden", 64),
        n_layers   = c.get("layers", 2),
        output_len = TARGET_LEN,
    ).to(DEVICE)
    ctrl_models[ln].load_state_dict(c["model_state"])

# ── Plot ──────────────────────────────────────────────────────────────────────
plot_all(
    plant_model  = plant_model,
    ctrl_models  = ctrl_models,
    data         = data,
    model_name   = "Transformer",
    out_dir      = OUT_DIR,
    results_path = CKPT_DIR / "results.json",
    device       = DEVICE,
    batch_size   = BATCH,
)
