"""
sec1_6_ctrl_loops.py — Section 1.6: Per-Control Loop Controller Evaluation

For each of the 5 control loops (PC, LC, FC, TC, CC):
  - Controller model predicts the CV (control valve output) from [SP, PV] history
  - Evaluates prediction accuracy (NRMSE, MAE) on the test set

Produces:
  figures/s1_6_ctrl_nrmse_per_loop.png       — NRMSE per loop (bar chart)
  figures/s1_6_ctrl_nrmse_per_scenario.png   — NRMSE per loop per scenario (grouped bar)
  figures/s1_6_ctrl_real_vs_pred.png         — Real vs predicted CV, best window, all 5 loops

Usage:
    python report_plots/code/sec1_6_ctrl_loops.py
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

import joblib
from pipeline import load_and_prepare_data
from gru import GRUController, CCSequenceModel
from config import LOOPS, PROCESSED_DATA_DIR

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH      = 128
CKPT_DIR   = ROOT / "outputs" / "pipeline" / "gru_scenario_weighted"
OUT_DIR    = ROOT / "report_plots" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CTRL_LOOPS  = ["PC", "LC", "FC", "TC", "CC"]
CTRL_H      = {"PC": 64, "LC": 64, "FC": 128, "TC": 64, "CC": 64}

# Extra causal channels added to each controller's input during training
EXTRA_CHANNELS = {
    "PC": ["P1_PCV02D", "P1_FT01",   "P1_TIT01"],
    "LC": ["P1_FT03",   "P1_FCV03D", "P1_PCV01D"],
    "FC": ["P1_PIT01",  "P1_LIT01",  "P1_TIT03"],
    "TC": ["P1_FT02",   "P1_PIT02",  "P1_TIT02"],
    "CC": ["P1_PP04D",  "P1_FCV03D", "P1_PCV02D"],
}
LOOP_COLORS = {"PC": "#2196F3", "LC": "#4CAF50", "FC": "#FF9800", "TC": "#E91E63", "CC": "#9C27B0"}

SCENARIOS     = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}
SC_COLORS     = {0: "#2196F3", 1: "#FF5722", 2: "#E91E63", 3: "#9C27B0"}


# ── augment controller inputs with causal channels (mirrors training setup) ───

def augment_ctrl_data(ctrl_data, sensor_cols):
    scaler  = joblib.load(f"{PROCESSED_DATA_DIR}/scaler.pkl")
    col_idx = {c: i for i, c in enumerate(sensor_cols)}
    for ln, extra_cols in EXTRA_CHANNELS.items():
        for ec in extra_cols:
            if ec not in col_idx:
                continue
            ei = col_idx[ec]
            mean_e, scale_e = scaler.mean_[ei], scaler.scale_[ei]
            for split in ("train", "val", "test"):
                npz = np.load(f"{PROCESSED_DATA_DIR}/{split}_data.npz")
                raw = npz["X"][:, :, [ei]].astype(np.float32)
                ctrl_data[ln][f"X_{split}"] = np.concatenate(
                    [ctrl_data[ln][f"X_{split}"], (raw - mean_e) / scale_e], axis=-1)
    return ctrl_data


# ── helpers ───────────────────────────────────────────────────────────────────

def _nrmse(t, p):
    rmse = np.sqrt(np.mean((t - p) ** 2))
    r    = float(t.max() - t.min())
    return 0.0 if r < 1e-10 else rmse / r


def _mae(t, p):
    return float(np.abs(t - p).mean())


def load_controllers(ctrl_data, TL):
    ctrls = {}
    for ln in CTRL_LOOPS:
        n_in = ctrl_data[ln]["X_train"].shape[-1]
        m = (CCSequenceModel(n_inputs=n_in, hidden=CTRL_H[ln], layers=2,
                             dropout=0.0, output_len=TL)
             if ln == "CC" else
             GRUController(n_inputs=n_in, hidden=CTRL_H[ln], layers=2,
                           dropout=0.0, output_len=TL)).to(DEVICE)
        p = CKPT_DIR / f"gru_ctrl_{ln.lower()}.pt"
        if p.exists():
            c = torch.load(p, map_location=DEVICE)
            m.load_state_dict(c["model_state"], strict=False)
        else:
            print(f"  WARNING: checkpoint not found for {ln}: {p}")
        m.eval()
        ctrls[ln] = m
    return ctrls


def run_ctrl_inference(ctrls, ctrl_data, TL):
    """Returns dict ln → pred (N, TL, 1)"""
    preds = {}
    for ln in CTRL_LOOPS:
        X = ctrl_data[ln]["X_test"]
        N = len(X)
        out = np.zeros((N, TL, 1), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, N, BATCH):
                sl = slice(i, i + BATCH)
                xb = torch.tensor(X[sl]).float().to(DEVICE)
                p  = ctrls[ln].predict(xb, target_len=TL)
                out[i:i + p.size(0)] = p.cpu().numpy()
        preds[ln] = out
    return preds


# ── Plot 1: NRMSE per loop ────────────────────────────────────────────────────

def plot_nrmse_per_loop(ctrl_data, preds, TL):
    nrmse_vals, mae_vals = [], []
    for ln in CTRL_LOOPS:
        true = ctrl_data[ln]["y_test"]          # (N, TL, 1)
        pred = preds[ln]
        nrmse_vals.append(_nrmse(true.flatten(), pred.flatten()))
        mae_vals.append(_mae(true.flatten(), pred.flatten()))

    loop_names  = [f"{ln}\n({LOOPS[ln].cv.replace('P1_','')})" for ln in CTRL_LOOPS]
    colors      = [LOOP_COLORS[ln] for ln in CTRL_LOOPS]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(loop_names, nrmse_vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, nrmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("NRMSE", fontsize=12)
    ax.set_xlabel("Control Loop  (CV signal)", fontsize=11)
    ax.set_title("Controller CV Prediction — NRMSE per Loop", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = OUT_DIR / "s1_6_ctrl_nrmse_per_loop.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")

    print(f"\n  {'Loop':<6} {'NRMSE':>8} {'MAE':>8}  CV signal")
    print(f"  {'-'*42}")
    for ln, nr, ma in zip(CTRL_LOOPS, nrmse_vals, mae_vals):
        print(f"  {ln:<6} {nr:>8.4f} {ma:>8.4f}  {LOOPS[ln].cv}")
    return nrmse_vals


# ── Plot 2: NRMSE per loop per scenario ───────────────────────────────────────

def plot_nrmse_per_scenario(ctrl_data, preds):
    sc_ids  = [0, 1, 2, 3]
    n_sc    = len(sc_ids)
    x       = np.arange(len(CTRL_LOOPS))
    w       = 0.8 / n_sc
    loop_names = [f"{ln}\n({LOOPS[ln].cv.replace('P1_','')})" for ln in CTRL_LOOPS]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, sc_id in enumerate(sc_ids):
        vals = []
        for ln in CTRL_LOOPS:
            sc_arr = ctrl_data[ln]["scenario_test"]
            mask   = sc_arr == sc_id
            if mask.sum() == 0:
                vals.append(0.0)
                continue
            true = ctrl_data[ln]["y_test"][mask].flatten()
            pred = preds[ln][mask].flatten()
            vals.append(_nrmse(true, pred))
        offset = (i - n_sc / 2 + 0.5) * w
        bars   = ax.bar(x + offset, vals, w, label=SCENARIOS[sc_id],
                        color=SC_COLORS[sc_id], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(loop_names, fontsize=10)
    ax.set_ylabel("NRMSE", fontsize=12)
    ax.set_title("Controller CV Prediction — NRMSE per Loop per Scenario",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "s1_6_ctrl_nrmse_per_scenario.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 3: Real vs Predicted CV — best window, all 5 loops ──────────────────

def plot_real_vs_pred(ctrl_data, preds, TL):
    """One subplot per loop, best-performance window for each."""
    t = np.arange(TL)
    fig, axes = plt.subplots(len(CTRL_LOOPS), 1,
                              figsize=(10, 2.8 * len(CTRL_LOOPS)),
                              sharex=True)

    for ax, ln in zip(axes, CTRL_LOOPS):
        true    = ctrl_data[ln]["y_test"][:, :, 0]   # (N, TL)
        pred    = preds[ln][:, :, 0]
        nrmse_w = np.array([_nrmse(true[i], pred[i]) for i in range(len(true))])
        best    = np.argmin(nrmse_w)

        r = true[best]
        p = pred[best]
        nrmse_val = _nrmse(r, p)
        color     = LOOP_COLORS[ln]

        ax.plot(t, r, color="black", lw=1.5, label="Real CV")
        ax.plot(t, p, color=color,   lw=1.5, linestyle="--", label="Predicted CV")
        ax.fill_between(t, r, p, color=color, alpha=0.18)
        ax.set_ylabel(f"{ln} — {LOOPS[ln].cv.replace('P1_','')}", fontsize=10, fontweight="bold")
        ax.text(0.99, 0.95, f"NRMSE = {nrmse_val:.4f}",
                transform=ax.transAxes, fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8))
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    handles = [
        plt.Line2D([0], [0], color="black", lw=1.5, label="Real CV"),
        plt.Line2D([0], [0], color="gray",  lw=1.5, linestyle="--", label="Predicted CV"),
        plt.matplotlib.patches.Patch(color="gray", alpha=0.3, label="Error"),
    ]
    axes[0].legend(handles=handles, fontsize=9, ncol=3, loc="upper left")
    fig.suptitle("Controller CV Prediction — Real vs Predicted (Best Window per Loop)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "s1_6_ctrl_real_vs_pred.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    data      = load_and_prepare_data()
    ctrl_data = data["ctrl"]
    TL        = data["metadata"]["target_len"]

    print("Augmenting controller inputs with causal channels...")
    sensor_cols = data["metadata"]["sensor_cols"]
    ctrl_data   = augment_ctrl_data(ctrl_data, sensor_cols)

    print("Loading controller models...")
    ctrls = load_controllers(ctrl_data, TL)

    print("Running inference on all 5 controllers...")
    preds = run_ctrl_inference(ctrls, ctrl_data, TL)

    print("\n[Plot 1] NRMSE per loop...")
    plot_nrmse_per_loop(ctrl_data, preds, TL)

    print("\n[Plot 2] NRMSE per loop per scenario...")
    plot_nrmse_per_scenario(ctrl_data, preds)

    print("\n[Plot 3] Real vs predicted CV (best window per loop)...")
    plot_real_vs_pred(ctrl_data, preds, TL)

    print(f"\nAll Section 1.6 plots saved to: {OUT_DIR}/")
