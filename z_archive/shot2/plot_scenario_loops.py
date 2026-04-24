"""
plot_scenario_loops.py — Per-scenario real vs predicted for all 5 control loop PVs.

For each scenario (Normal, AP_no, AP_with, AE_no):
  - Picks the median-variance window as a representative sample
  - Plots real vs predicted for all 5 PVs with shaded difference
  - Saves one figure per scenario to plots/scenario_loops/

Usage:
    python 04_evaluate/plot_scenario_loops.py
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from gru import GRUPlant
from config import PV_COLS, PROCESSED_DATA_DIR

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = ROOT / "outputs" / "pipeline" / "gru_scenario_haiend"
OUT_DIR  = ROOT / "plots" / "scenario_loops"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO_NAMES  = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}
SCENARIO_COLORS = {0: "#2196F3", 1: "#FF5722", 2: "#E91E63", 3: "#9C27B0"}

PV_TO_LOOP = {
    "P1_PIT01": "PC",
    "P1_LIT01": "LC",
    "P1_FT03Z": "FC",
    "P1_TIT01": "TC",
    "P1_TIT03": "CC",
}


def load_plant(data):
    ckpt = torch.load(CKPT_DIR / "gru_plant.pt", map_location=DEVICE)
    ms = ckpt["model_state"]
    emb_dim    = ms["scenario_emb.weight"].shape[1]
    n_scenarios = ms["scenario_emb.weight"].shape[0]
    n_plant_in  = ckpt.get("n_plant_in", ms["encoder.weight_ih_l0"].shape[1] - emb_dim)
    n_pv        = ckpt.get("n_pv",       ms["fc.3.weight"].shape[0])
    model = GRUPlant(
        n_plant_in  = n_plant_in,
        n_pv        = n_pv,
        hidden      = ckpt["hidden"],
        layers      = ckpt["layers"],
        n_scenarios = n_scenarios,
        n_haiend    = ckpt.get("n_haiend", 0),
    ).to(DEVICE)
    model.load_state_dict(ms)
    model.eval()
    print(f"Loaded GRUPlant  val_loss={ckpt.get('val_loss', '?'):.5f}")
    return model, n_pv


def run_inference(model, plant_data, indices, batch_size=128):
    N  = len(indices)
    TL = plant_data["pv_target_test"].shape[1]
    NP = plant_data["n_pv"]
    preds = np.zeros((N, TL, NP), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            sl  = slice(i, i + batch_size)
            idx = indices[sl]
            xb   = torch.tensor(plant_data["X_test"][idx]).float().to(DEVICE)
            xctb = torch.tensor(plant_data["X_cv_target_test"][idx]).float().to(DEVICE)
            pvib = torch.tensor(plant_data["pv_init_test"][idx]).float().to(DEVICE)
            scb  = torch.tensor(plant_data["scenario_test"][idx]).long().to(DEVICE)
            pv_out, _ = model.predict(xb, xctb, pvib, scb)
            preds[i:i + pv_out.size(0)] = pv_out.cpu().numpy()
    return preds


def pick_windows(pv_target, scenario_arr, sc_id, n=5):
    """Pick n windows spread across low/median/high variance for the scenario."""
    mask = np.where(scenario_arr == sc_id)[0]
    if len(mask) == 0:
        return np.array([])
    variances = pv_target[mask].var(axis=(1, 2))
    sorted_idx = np.argsort(variances)
    picks = np.linspace(0, len(sorted_idx) - 1, n, dtype=int)
    return mask[sorted_idx[picks]]


def plot_scenario(sc_id, windows_idx, real, pred, target_len, color, sc_name):
    n_windows = len(windows_idx)
    n_pv = len(PV_COLS)
    t = np.arange(target_len)

    fig, axes = plt.subplots(
        n_windows, n_pv,
        figsize=(3.5 * n_pv, 2.8 * n_windows),
        squeeze=False,
    )

    for row in range(n_windows):
        for col, pv in enumerate(PV_COLS):
            ax = axes[row, col]
            r = real[row, :, col]
            p = pred[row, :, col]

            ax.plot(t, r, color="black",  lw=1.5, label="Real")
            ax.plot(t, p, color=color,    lw=1.5, linestyle="--", label="Predicted")
            ax.fill_between(t, r, p, color=color, alpha=0.20, label="Difference")

            if row == 0:
                loop = PV_TO_LOOP.get(pv, "")
                ax.set_title(f"{pv.replace('P1_', '')}  [{loop}]",
                             fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Window {row + 1}", fontsize=8)
            ax.set_xlabel("t (s)", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, linestyle="--", alpha=0.3)

    handles = [
        plt.Line2D([0], [0], color="black", lw=1.5, label="Real"),
        plt.Line2D([0], [0], color=color,   lw=1.5, linestyle="--", label="Predicted"),
        plt.matplotlib.patches.Patch(color=color, alpha=0.3, label="Difference"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9, ncol=3)
    fig.suptitle(
        f"Scenario: {sc_name} — Real vs Predicted per Control Loop PV",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    path = OUT_DIR / f"scenario_{sc_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def main():
    print("Loading data...")
    data       = load_and_prepare_data()
    plant_data = data["plant"]

    model, _ = load_plant(data)

    for sc_id, sc_name in SCENARIO_NAMES.items():
        color = SCENARIO_COLORS[sc_id]
        win_idx = pick_windows(plant_data["pv_target_test"],
                               plant_data["scenario_test"], sc_id, n=5)
        if len(win_idx) == 0:
            print(f"  {sc_name}: no test windows — skipping")
            continue

        print(f"\nScenario {sc_name} — running inference on {len(win_idx)} windows...")
        preds = run_inference(model, plant_data, win_idx)
        real  = plant_data["pv_target_test"][win_idx]
        plot_scenario(sc_id, win_idx, real, preds,
                      plant_data["pv_target_test"].shape[1], color, sc_name)

    print(f"\nAll plots saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
