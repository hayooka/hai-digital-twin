"""
compare_results.py — Load results.json from each model and produce comparison plots.

Plots produced (saved to outputs/comparison/):
    1. nrmse_per_pv.png       — grouped bar chart: NRMSE per PV signal per model
    2. mean_nrmse.png         — bar chart: mean NRMSE across PV signals
    3. best_val_loss.png      — bar chart: best validation loss per model
    4. train_val_loss.png     — loss curves for models that tracked them (GRU)

Usage:
    python 03_model/compare_results.py
    (also called automatically by run_all.py)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
from config import PV_COLS

OUT_DIR = ROOT / "outputs" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATHS = {
    "GRU":            ROOT / "outputs" / "gru_plant"            / "results.json",
    "LSTM":           ROOT / "outputs" / "lstm_plant"           / "results.json",
    "Transformer":    ROOT / "outputs" / "transformer_plant"    / "results.json",
    "Transformer-SS": ROOT / "outputs" / "transformer_plant_ss" / "results.json",
}

COLORS = ["steelblue", "darkorange", "seagreen", "crimson"]


def load_results() -> dict:
    results = {}
    for name, path in RESULT_PATHS.items():
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
            print(f"  Loaded: {name}  (mean NRMSE={results[name]['mean_nrmse']:.4f})")
        else:
            print(f"  Missing: {name}  ({path})")
    return results


# ── Plot 1: NRMSE per PV (grouped bar) ────────────────────────────────────────

def plot_nrmse_per_pv(results: dict):
    models = list(results.keys())
    pv_labels = [c.replace("P1_", "") for c in PV_COLS]
    n_pv = len(pv_labels)
    n_models = len(models)

    x = np.arange(n_pv)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, color) in enumerate(zip(models, COLORS)):
        vals = [results[name]["nrmse_per_pv"].get(col, np.nan) for col in PV_COLS]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(pv_labels, fontsize=10)
    ax.set_ylabel("NRMSE (closed-loop)")
    ax.set_title("Closed-Loop NRMSE per PV Signal")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "nrmse_per_pv.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: nrmse_per_pv.png")
    plt.close(fig)


# ── Plot 2: Mean NRMSE ────────────────────────────────────────────────────────

def plot_mean_nrmse(results: dict):
    models = list(results.keys())
    means  = [results[m]["mean_nrmse"] for m in models]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(models, means, color=COLORS[:len(models)], alpha=0.85)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean NRMSE (closed-loop)")
    ax.set_title("Mean Closed-Loop NRMSE — Model Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mean_nrmse.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: mean_nrmse.png")
    plt.close(fig)


# ── Plot 3: Best validation loss ──────────────────────────────────────────────

def plot_best_val_loss(results: dict):
    models = list(results.keys())
    losses = [results[m]["best_val_loss"] for m in models]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(models, losses, color=COLORS[:len(models)], alpha=0.85)
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.5f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Best Val Loss (MSE)")
    ax.set_title("Best Validation Loss — Model Comparison")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.4, which="both")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "best_val_loss.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: best_val_loss.png")
    plt.close(fig)


# ── Plot 4: Training loss curves (models that tracked them) ───────────────────

def plot_loss_curves(results: dict):
    models_with_curves = {
        name: data for name, data in results.items()
        if "train_losses" in data and "val_losses" in data
    }
    if not models_with_curves:
        print("  No loss curves to plot (none saved).")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    for (name, data), color in zip(models_with_curves.items(), COLORS):
        epochs = range(1, len(data["train_losses"]) + 1)
        ax1.plot(epochs, data["train_losses"], color=color, label=name)
        ax2.plot(epochs, data["val_losses"],   color=color, label=name)

    for ax, title in [(ax1, "Train Loss"), (ax2, "Val Loss")]:
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss (log scale)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.suptitle("Plant Training Loss Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "train_val_loss.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: train_val_loss.png")
    plt.close(fig)


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(results: dict):
    col_w = 14
    pv_short = [c.replace("P1_", "") for c in PV_COLS]
    header = f"{'Model':<18}" + "".join(f"{c:>{col_w}}" for c in pv_short) + f"{'Mean':>{col_w}}"
    print("\n" + "=" * len(header))
    print("NRMSE COMPARISON TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, data in results.items():
        row = f"{name:<18}"
        for col in PV_COLS:
            v = data["nrmse_per_pv"].get(col, float("nan"))
            row += f"{v:>{col_w}.4f}"
        row += f"{data['mean_nrmse']:>{col_w}.4f}"
        print(row)
    print("=" * len(header))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COMPARING RESULTS")
    print("=" * 70)

    results = load_results()
    if not results:
        print("  No results found. Run training scripts first.")
        sys.exit(1)

    print_summary_table(results)

    print(f"\nGenerating plots → {OUT_DIR}/")
    plot_nrmse_per_pv(results)
    plot_mean_nrmse(results)
    plot_best_val_loss(results)
    plot_loss_curves(results)

    print("\nDone.")
