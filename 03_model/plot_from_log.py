"""
plot_from_log.py — Build plots from values printed in the terminal output.
No model checkpoint needed.

Edit the RUNS dict below with values from your terminal logs,
then run:  python 03_model/plot_from_log.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "outputs" / "from_log"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PV_COLS  = ["P1_PIT01", "P1_LIT01", "P1_FT03Z", "P1_TIT01", "P1_TIT03"]
PV_SHORT = ["PIT01", "LIT01", "FT03Z", "TIT01", "TIT03"]

# ── Paste values from terminal output here ────────────────────────────────────
# Format: "ModelName": { "nrmse": [PIT01, LIT01, FT03Z, TIT01, TIT03],
#                        "best_val_loss": float,
#                        "train_losses": [...],   # optional, every-10-epoch values
#                        "val_losses":   [...] }  # optional

RUNS = {
    "GRU": {
        "nrmse": [0.0042, 0.0184, 0.0179, 0.0092, 0.0258],
        "best_val_loss": 0.00051,
        # From the log — every 10 epochs
        "val_losses": [0.01243, 0.00347, 0.00165, 0.00125, 0.00079,
                       0.00062, 0.00070, 0.00059, 0.00063, 0.00066,
                       0.00052, 0.00059, 0.00054, 0.00051, 0.00054, 0.00053],
        "train_losses": [0.10723, 0.01130, 0.01049, 0.01071, 0.00965,
                         0.00962, 0.00957, 0.00953, 0.00948, 0.00949,
                         0.00945, 0.00942, 0.00936, 0.00935, 0.00934, 0.00930],
    },
    # Add more models here as they finish, e.g.:
    # "LSTM": {
    #     "nrmse": [...],
    #     "best_val_loss": ...,
    # },
}

COLORS = ["steelblue", "darkorange", "seagreen", "crimson"]


# ── Plot 1: NRMSE per PV (grouped bar) ───────────────────────────────────────

def plot_nrmse(runs: dict):
    models   = list(runs.keys())
    n_models = len(models)
    x        = np.arange(len(PV_SHORT))
    width    = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (name, color) in enumerate(zip(models, COLORS)):
        vals   = runs[name]["nrmse"]
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0003,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(PV_SHORT)
    ax.set_ylabel("NRMSE (closed-loop)")
    ax.set_title("Closed-Loop NRMSE per PV Signal")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = OUT_DIR / "nrmse_per_pv.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Plot 2: Mean NRMSE bar ────────────────────────────────────────────────────

def plot_mean_nrmse(runs: dict):
    models = list(runs.keys())
    means  = [float(np.mean(runs[m]["nrmse"])) for m in models]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, means, color=COLORS[:len(models)], alpha=0.85)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean NRMSE")
    ax.set_title("Mean Closed-Loop NRMSE — Model Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = OUT_DIR / "mean_nrmse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Plot 3: Val loss curves ───────────────────────────────────────────────────

def plot_loss_curves(runs: dict):
    has_curves = {n: d for n, d in runs.items() if "val_losses" in d}
    if not has_curves:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    for (name, data), color in zip(has_curves.items(), COLORS):
        val_losses   = data["val_losses"]
        train_losses = data.get("train_losses", [])
        # x-axis: every 10 epochs (epoch 1, 10, 20, ...)
        epochs = [1] + list(range(10, 10 * len(val_losses), 10))
        epochs = epochs[:len(val_losses)]

        ax1.plot(epochs, train_losses[:len(epochs)],
                 color=color, linestyle="--", alpha=0.6, label=f"{name} train")
        ax1.plot(epochs, val_losses,
                 color=color, linewidth=2, label=f"{name} val")
        ax2.plot(epochs, val_losses,
                 color=color, linewidth=2, label=name,
                 marker="o", markersize=4)

    for ax, title in [(ax1, "Train & Val Loss"), (ax2, "Val Loss (zoom)")]:
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss (log scale)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.suptitle("Plant Loss Curves (from log)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "loss_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not RUNS:
        print("No data in RUNS dict. Paste values from terminal output.")
        sys.exit(1)

    print(f"Building plots for: {list(RUNS.keys())}")
    plot_nrmse(RUNS)
    plot_mean_nrmse(RUNS)
    plot_loss_curves(RUNS)
    print(f"\nAll plots saved to {OUT_DIR}/")
