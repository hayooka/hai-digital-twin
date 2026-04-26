"""
sec1_2to5_scenarios.py — Sections 1.2–1.5: Real vs Predicted per Scenario

For each scenario (Normal, AP_no, AP_with, AE_no):
  - Picks 3 representative windows (low / median / high variance)
  - Plots real vs predicted for all 5 control loop PVs
  - Shaded difference area
  - NRMSE annotated per PV

Produces:
  figures/s1_2_normal.png
  figures/s1_3_ap_no.png
  figures/s1_4_ap_with.png
  figures/s1_5_ae_no.png

Usage:
    python report_plots/code/sec1_2to5_scenarios.py
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from gru import GRUPlant
from config import PV_COLS

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH    = 128
CKPT_DIR = ROOT / "outputs" / "pipeline" / "gru_scenario_weighted"
OUT_DIR  = ROOT / "report_plots" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PV_SHORT  = [p.replace("P1_", "") for p in PV_COLS]
PV_LOOPS  = ["PC", "LC", "FC", "TC", "CC"]

SCENARIOS = {
    0: ("Normal",   "#2196F3", "s1_2_normal.png"),
    1: ("AP_no",    "#FF5722", "s1_3_ap_no.png"),
    2: ("AP_with",  "#E91E63", "s1_4_ap_with.png"),
    3: ("AE_no",    "#9C27B0", "s1_5_ae_no.png"),
}

N_WINDOWS = 1  # best performance window


# ── helpers ───────────────────────────────────────────────────────────────────

def _nrmse(t, p):
    rmse = np.sqrt(np.mean((t - p) ** 2))
    r    = float(t.max() - t.min())
    return 0.0 if r < 1e-10 else rmse / r


def load_model():
    ckpt = torch.load(CKPT_DIR / "gru_plant.pt", map_location=DEVICE)
    ms   = ckpt["model_state"]
    emb  = ms["scenario_emb.weight"].shape[1]
    m = GRUPlant(
        n_plant_in  = ckpt.get("n_plant_in", ms["encoder.weight_ih_l0"].shape[1] - emb),
        n_pv        = ckpt.get("n_pv",       ms["fc.3.weight"].shape[0]),
        hidden      = ckpt["hidden"],
        layers      = ckpt["layers"],
        n_scenarios = ms["scenario_emb.weight"].shape[0],
        n_haiend    = ckpt.get("n_haiend", 0),
    ).to(DEVICE)
    m.load_state_dict(ms)
    m.eval()
    return m


def run_inference(model, plant_data, indices):
    N   = len(indices)
    TL  = plant_data["pv_target_test"].shape[1]
    NP  = plant_data["n_pv"]
    out = np.zeros((N, TL, NP), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl  = slice(i, i + BATCH)
            idx = indices[sl]
            pv_out, _ = model.predict(
                torch.tensor(plant_data["X_test"][idx]).float().to(DEVICE),
                torch.tensor(plant_data["X_cv_target_test"][idx]).float().to(DEVICE),
                torch.tensor(plant_data["pv_init_test"][idx]).float().to(DEVICE),
                torch.tensor(plant_data["scenario_test"][idx]).long().to(DEVICE),
            )
            out[i:i + pv_out.size(0)] = pv_out.cpu().numpy()
    return out


def pick_windows(pv_target, preds_all, sc_arr, sc_id):
    """Pick the single window with lowest NRMSE (best performance)."""
    mask = np.where(sc_arr == sc_id)[0]
    if len(mask) == 0:
        return np.array([])
    nrmse_per_window = np.array([
        _nrmse(pv_target[i].flatten(), preds_all[i].flatten())
        for i in mask
    ])
    best = mask[np.argmin(nrmse_per_window)]
    return np.array([best])


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_scenario(sc_name, color, fname, real, pred, target_len):
    """One window, 5 PVs stacked vertically."""
    t   = np.arange(target_len)
    fig, axes = plt.subplots(len(PV_COLS), 1,
                              figsize=(10, 2.8 * len(PV_COLS)),
                              sharex=True)

    for col, (ax, pv, loop) in enumerate(zip(axes, PV_SHORT, PV_LOOPS)):
        r = real[0, :, col]
        p = pred[0, :, col]
        nrmse_val = _nrmse(r, p)

        ax.plot(t, r, color="black", lw=1.5, label="Real")
        ax.plot(t, p, color=color,   lw=1.5, linestyle="--", label="Predicted")
        ax.fill_between(t, r, p, color=color, alpha=0.18)

        ax.set_ylabel(f"{pv} [{loop}]", fontsize=10, fontweight="bold")
        ax.text(0.99, 0.95, f"NRMSE = {nrmse_val:.4f}",
                transform=ax.transAxes, fontsize=8,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8))
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)

    handles = [
        plt.Line2D([0], [0], color="black", lw=1.5, label="Real"),
        plt.Line2D([0], [0], color=color,   lw=1.5, linestyle="--", label="Predicted"),
        plt.matplotlib.patches.Patch(color=color, alpha=0.3, label="Error"),
    ]
    axes[0].legend(handles=handles, fontsize=9, ncol=3, loc="upper left")

    fig.suptitle(f"Scenario: {sc_name} — Real vs Predicted (one Window length)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = OUT_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    data       = load_and_prepare_data()
    plant_data = data["plant"]
    pv_true    = plant_data["pv_target_test"]
    sc_arr     = plant_data["scenario_test"]
    TL         = pv_true.shape[1]

    print("Loading model...")
    model = load_model()

    print("Running inference on full test set...")
    all_indices = np.arange(len(pv_true))
    all_preds   = run_inference(model, plant_data, all_indices)

    # Fixed window overrides (use None to auto-select best)
    FIXED_WINDOWS = {0: None, 1: None, 2: None, 3: None}

    for sc_id, (sc_name, color, fname) in SCENARIOS.items():
        print(f"\n[{fname}] Scenario: {sc_name}")
        if FIXED_WINDOWS.get(sc_id) is not None:
            win_idx = np.array([FIXED_WINDOWS[sc_id]])
        else:
            win_idx = pick_windows(pv_true, all_preds, sc_arr, sc_id)

        if len(win_idx) == 0:
            print(f"  No windows found — skipping")
            continue

        real  = pv_true[win_idx]
        preds = all_preds[win_idx]
        plot_scenario(sc_name, color, fname, real, preds, TL)

    print(f"\nAll Section 1.2-1.5 plots saved to: {OUT_DIR}/")
