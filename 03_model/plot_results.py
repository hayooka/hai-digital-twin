"""
plot_results.py — Reusable plotting utilities for HAI Digital Twin models.

Produces the standard 5-figure diagnostic suite:
    Plot 1 : Plant training loss curves  (14×4, 2 subplots)
    Plot 2 : Closed-loop validation — 5-min  horizon (16×15, 5 PV subplots)
    Plot 3 : Closed-loop validation — 10-min horizon (16×15, 5 PV subplots)
    Plot 4 : Closed-loop validation — 15-min horizon (16×15, 5 PV subplots)
    Plot 5 : Closed-loop validation — 30-min horizon (16×15, 5 PV subplots)

Public API
----------
    plot_training_curves(train_losses, val_losses, ss_ratios, *, model_name, save_path)
    plot_horizon_validation(pv_true, pv_preds, horizon_steps, pv_cols, *,
                            horizon_label, model_name, save_path)
    plot_all_horizons(horizon_data, pv_cols, *, model_name, out_dir)

Usage example (inside a training script)
-----------------------------------------
    from plot_results import plot_training_curves, plot_all_horizons

    # After training loop:
    plot_training_curves(train_losses, val_losses, ss_ratios,
                         model_name="GRU", save_path=OUT_DIR / "loss_curves.png")

    # After closed-loop validation; horizon_data maps horizon_steps → (pv_true, pv_preds)
    # where each array is shape (horizon_steps, n_pv) — one representative sample.
    horizon_data = {
        300:  (pv_true_300,  pv_preds_300),
        600:  (pv_true_600,  pv_preds_600),
        900:  (pv_true_900,  pv_preds_900),
        1800: (pv_true_1800, pv_preds_1800),
    }
    plot_all_horizons(horizon_data, PV_COLS, model_name="GRU", out_dir=OUT_DIR)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; swap to "TkAgg" if needed
import matplotlib.pyplot as plt
import numpy as np

# Human-readable labels for each horizon length (in seconds)
_HORIZON_LABELS: Dict[int, str] = {
    300:  "5-min",
    600:  "10-min",
    900:  "15-min",
    1800: "30-min",
}


# ─── Plot 1 ───────────────────────────────────────────────────────────────────

def plot_training_curves(
    train_losses: list[float],
    val_losses:   list[float],
    ss_ratios:    list[float],
    *,
    model_name: str = "",
    save_path:  Optional[str | Path] = None,
) -> plt.Figure:
    """
    14×4 figure with two side-by-side subplots:
        Left  — train / val MSE on a log-y scale
        Right — scheduled-sampling ratio per epoch
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # ── Left: loss curves ───────────────────────────────────────────────────
    ax1.plot(epochs, train_losses, color="steelblue",   label="Train")
    ax1.plot(epochs, val_losses,   color="darkorange",  label="Val (autoregressive)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss (log scale)")
    title_prefix = f"{model_name} — " if model_name else ""
    ax1.set_title(f"{title_prefix}Plant Training Loss Curves")
    ax1.legend()
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)

    # ── Right: SS ratio ─────────────────────────────────────────────────────
    ax2.plot(epochs, ss_ratios, color="seagreen")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("SS Ratio")
    ax2.set_title(f"{title_prefix}Scheduled-Sampling Ratio")
    ax2.set_ylim(-0.05, max(ss_ratios) * 1.15 + 1e-6)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


# ─── Plot 2–5 (one horizon each) ──────────────────────────────────────────────

def plot_horizon_validation(
    pv_true:       np.ndarray,
    pv_preds:      np.ndarray,
    horizon_steps: int,
    pv_cols:       list[str],
    *,
    horizon_label: str = "",
    model_name:    str = "",
    save_path:     Optional[str | Path] = None,
) -> plt.Figure:
    """
    16×15 figure with 5 vertically-stacked subplots (one per PV signal).

    Parameters
    ----------
    pv_true       : np.ndarray, shape (horizon_steps, n_pv)
                    Ground truth PV values (normalised scale).
    pv_preds      : np.ndarray, shape (horizon_steps, n_pv)
                    Closed-loop model predictions.
    horizon_steps : number of time steps shown on the x-axis.
    pv_cols       : ordered list of PV signal names, length n_pv.
    horizon_label : human-readable horizon string, e.g. "5-min".
    model_name    : shown in the overall figure title.
    save_path     : if given, figure is saved there (PNG, 150 dpi).
    """
    n_pv = len(pv_cols)
    actual_steps = pv_true.shape[0]
    assert pv_true.shape[1]  == n_pv, \
        f"pv_true has {pv_true.shape[1]} features, expected {n_pv}"
    assert pv_preds.shape == pv_true.shape, \
        f"pv_preds shape {pv_preds.shape} != pv_true shape {pv_true.shape}"

    t = np.arange(actual_steps)

    fig, axes = plt.subplots(n_pv, 1, figsize=(16, 15), sharex=True)
    if n_pv == 1:
        axes = [axes]   # keep iterable

    label = horizon_label or _HORIZON_LABELS.get(horizon_steps, f"{horizon_steps}s")
    title_prefix = f"{model_name} — " if model_name else ""
    fig.suptitle(
        f"{title_prefix}Closed-Loop Validation — {label} Horizon",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for i, (ax, col) in enumerate(zip(axes, pv_cols)):
        ax.plot(t, pv_true[:, i],  color="steelblue",  linewidth=1.2, label="Ground truth")
        ax.plot(t, pv_preds[:, i], color="crimson",    linewidth=1.0,
                linestyle="--", label="Prediction (closed-loop)")
        ax.set_title(col, fontsize=11)
        ax.set_ylabel("Value (normalised)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("Time steps", fontsize=10)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


# ─── Convenience: produce all four horizon plots at once ──────────────────────

def plot_all_horizons(
    horizon_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    pv_cols: list[str],
    *,
    model_name: str = "",
    out_dir:    Optional[str | Path] = None,
) -> Dict[int, plt.Figure]:
    """
    Call plot_horizon_validation for every entry in *horizon_data*.

    Parameters
    ----------
    horizon_data : dict mapping horizon_steps (int) →
                   (pv_true, pv_preds), each shape (horizon_steps, n_pv).
                   Typical keys: 300, 600, 900, 1800.
    pv_cols      : ordered list of PV signal names.
    model_name   : string prefix for figure titles.
    out_dir      : if given, each figure is saved as
                   <out_dir>/<model_name_lower>_horizon_<H>s.png

    Returns
    -------
    dict mapping horizon_steps → matplotlib Figure
    """
    figures: Dict[int, plt.Figure] = {}
    out = Path(out_dir) if out_dir is not None else None

    for horizon_steps, (pv_true, pv_preds) in sorted(horizon_data.items()):
        label = _HORIZON_LABELS.get(horizon_steps, f"{horizon_steps}s")
        save_path = None
        if out is not None:
            stem = f"{model_name.lower()}_horizon_{horizon_steps}s" if model_name \
                   else f"horizon_{horizon_steps}s"
            save_path = out / f"{stem}.png"

        fig = plot_horizon_validation(
            pv_true, pv_preds, horizon_steps, pv_cols,
            horizon_label=label,
            model_name=model_name,
            save_path=save_path,
        )
        figures[horizon_steps] = fig

    return figures
