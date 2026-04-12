"""
eval_twin.py — Evaluation plots for the Boiler Digital Twin.

Combines plots from the GRU notebook and 04_evaluate/eval_plots.py,
adapted to the boiler_twin closed-loop rollout output format.

Plots generated
───────────────
  1. pv_timeseries.png          — predicted vs actual for all 5 PVs per scenario
  2. cv_timeseries.png          — controller CV outputs per loop over time
  3. loss_curves.png            — train/val loss curves (from saved .npy files)
  4. rmse_per_pv.png            — bar chart of RMSE per PV per scenario
  5. error_over_time.png        — rolling RMSE over rollout horizon
  6. error_boxplot.png          — squared error boxplot per PV × scenario
  7. scatter_true_vs_pred.png   — true vs predicted scatter per PV (normal only)
  8. residual_qq.png            — Q-Q normality plot of residuals per PV

Usage:
  python eval_twin.py --scenario 0 --horizon 1800
  python eval_twin.py --all                        # run all scenarios
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MplFigure
from scipy.stats import probplot, ks_2samp

ROOT       = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "boiler_twin"
PLOTS_DIR  = OUTPUT_DIR / "plots"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_model"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_causal"))

from parse_dcs import LOOPS_DEF, PV_COLS

SCENARIO_NAMES = {
    0: "Normal",
    1: "AP no-comb",
    2: "AP with-comb",
    3: "AE no-comb",
    4: "AE with-comb",
}
SCENARIO_COLORS = {
    0: "#2ca02c",
    1: "#ff7f0e",
    2: "#d62728",
    3: "#9467bd",
    4: "#8c564b",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_rollout(scenario: int, horizon: int) -> dict | None:
    path = OUTPUT_DIR / f"generated_sc{scenario}_h{horizon}.npz"
    if not path.exists():
        print(f"  [SKIP] {path.name} not found — run rollout.py first")
        return None
    data = dict(np.load(path, allow_pickle=True))
    data["scenario"] = int(data["scenario"])
    return data


def _xticks(ax, T: int, stride: int = 300) -> None:
    ticks = np.arange(0, T + 1, stride)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 60}m" for t in ticks], fontsize=8)


def _savefig(fig: MplFigure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out.relative_to(ROOT)}")


# ── Plot 1: PV time-series ────────────────────────────────────────────────────

def plot_pv_timeseries(rollouts: dict[int, dict]) -> None:
    """Predicted vs actual for all 5 PVs, one column per scenario."""
    n_pv  = len(PV_COLS)
    scs   = sorted(rollouts)
    n_sc  = len(scs)

    fig, axes = plt.subplots(n_pv, n_sc,
                             figsize=(6 * n_sc, 3 * n_pv),
                             sharex="col", sharey="row",
                             squeeze=False)

    for col, sc in enumerate(scs):
        d     = rollouts[sc]
        pred  = d["pred_pvs"]    # (T, n_pv)
        actual= d["actual_pvs"]  # (T, n_pv)
        T     = len(pred)
        t     = np.arange(T)
        color = SCENARIO_COLORS[sc]

        for row, pv in enumerate(PV_COLS):
            ax = axes[row][col]
            ax.plot(t, actual[:, row], color="green", lw=1.0, alpha=0.85,
                    label="Actual" if (row == 0 and col == 0) else "_nolegend_")
            ax.plot(t, pred[:, row],   color=color,   lw=1.1, linestyle="--", alpha=0.9,
                    label="Predicted" if (row == 0 and col == 0) else "_nolegend_")
            ax.axvspan(0, T, alpha=0.04, color=color)
            ax.set_ylabel(pv[:18], fontsize=8)
            ax.grid(True, alpha=0.2)
            if row == 0:
                ax.set_title(SCENARIO_NAMES[sc], fontsize=10, color=color, fontweight="bold")
            if row == n_pv - 1:
                _xticks(ax, T)
                ax.set_xlabel("Time", fontsize=8)

    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle("PV Predicted vs Actual — Closed-Loop Rollout", fontsize=13, y=1.01)
    fig.tight_layout()
    _savefig(fig, "pv_timeseries.png")


# ── Plot 2: CV time-series ────────────────────────────────────────────────────

def plot_cv_timeseries(rollouts: dict[int, dict]) -> None:
    """Controller CV outputs per loop over time."""
    loop_names = list(LOOPS_DEF.keys())
    n_loops = len(loop_names)
    scs     = sorted(rollouts)
    n_sc    = len(scs)

    fig, axes = plt.subplots(n_loops, n_sc,
                             figsize=(6 * n_sc, 2.5 * n_loops),
                             sharex="col", sharey="row",
                             squeeze=False)

    for col, sc in enumerate(scs):
        d       = rollouts[sc]
        pred_cv = d["pred_cvs"]   # (T, n_loops)
        T       = len(pred_cv)
        t       = np.arange(T)
        color   = SCENARIO_COLORS[sc]

        for row, ln in enumerate(loop_names):
            ax = axes[row][col]
            ax.plot(t, pred_cv[:, row], color=color, lw=1.0)
            ax.set_ylabel(f"{ln} CV"[:18], fontsize=8)
            ax.grid(True, alpha=0.2)
            if row == 0:
                ax.set_title(SCENARIO_NAMES[sc], fontsize=10, color=color, fontweight="bold")
            if row == n_loops - 1:
                _xticks(ax, T)
                ax.set_xlabel("Time", fontsize=8)

    fig.suptitle("Controller CV Outputs — Closed-Loop Rollout", fontsize=13, y=1.01)
    fig.tight_layout()
    _savefig(fig, "cv_timeseries.png")


# ── Plot 3: Loss curves ───────────────────────────────────────────────────────

def plot_loss_curves() -> None:
    """Train/val loss curves for plant LSTM and each controller loop."""
    loop_names = list(LOOPS_DEF.keys())

    # ── Plant ──
    train_f = OUTPUT_DIR / "plant_train_losses.npy"
    val_f   = OUTPUT_DIR / "plant_val_losses.npy"
    if train_f.exists() and val_f.exists():
        tr = np.load(train_f)
        va = np.load(val_f)
        ep = np.arange(1, len(tr) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(ep, tr, label="Train", color="steelblue", lw=1.2)
        axes[0].plot(ep, va, label="Val (autoregressive)", color="orange", lw=1.2)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (MSE)")
        axes[0].set_title("Plant LSTM — Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.2)
        axes[1].semilogy(ep, tr, color="steelblue", lw=1.2, label="Train")
        axes[1].semilogy(ep, va, color="orange",    lw=1.2, label="Val")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss (log)")
        axes[1].set_title("Plant LSTM — Loss (log)"); axes[1].legend(); axes[1].grid(True, alpha=0.2)
        fig.tight_layout()
        _savefig(fig, "loss_curves_plant.png")
    else:
        print("  [SKIP] plant loss .npy files not found")

    # ── Per-loop controllers ──
    available = [(ln, OUTPUT_DIR / f"ctrl_{ln}_train_losses.npy",
                      OUTPUT_DIR / f"ctrl_{ln}_val_losses.npy")
                 for ln in loop_names
                 if (OUTPUT_DIR / f"ctrl_{ln}_train_losses.npy").exists()]
    if not available:
        print("  [SKIP] controller loss .npy files not found")
        return

    n = len(available)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7), squeeze=False)
    for col, (ln, tf, vf) in enumerate(available):
        tr = np.load(tf)
        va = np.load(vf)
        ep = np.arange(1, len(tr) + 1)
        axes[0][col].plot(ep, tr, color="steelblue", lw=1.1, label="Train")
        axes[0][col].plot(ep, va, color="orange",    lw=1.1, label="Val")
        axes[0][col].set_title(f"{ln} Controller", fontsize=10)
        axes[0][col].set_ylabel("Loss (MSE)" if col == 0 else "")
        axes[0][col].legend(fontsize=7); axes[0][col].grid(True, alpha=0.2)
        axes[1][col].semilogy(ep, tr, color="steelblue", lw=1.1)
        axes[1][col].semilogy(ep, va, color="orange",    lw=1.1)
        axes[1][col].set_xlabel("Epoch")
        axes[1][col].set_ylabel("Loss (log)" if col == 0 else "")
        axes[1][col].grid(True, alpha=0.2)
    fig.suptitle("Controller Loss Curves", fontsize=12)
    fig.tight_layout()
    _savefig(fig, "loss_curves_controllers.png")


# ── Plot 4: RMSE per PV per scenario ─────────────────────────────────────────

def plot_rmse_per_pv(rollouts: dict[int, dict]) -> None:
    """Bar chart of RMSE per PV, grouped by scenario."""
    scs = sorted(rollouts)
    n_pv = len(PV_COLS)
    x = np.arange(n_pv)
    width = 0.8 / len(scs)

    fig, ax = plt.subplots(figsize=(max(8, 2 * n_pv), 5))

    for i, sc in enumerate(scs):
        d      = rollouts[sc]
        pred   = d["pred_pvs"]
        actual = d["actual_pvs"]
        rmse   = np.sqrt(((pred - actual) ** 2).mean(axis=0))   # (n_pv,)

        offset = (i - len(scs) / 2 + 0.5) * width
        bars = ax.bar(x + offset, rmse, width=width * 0.9,
                      color=SCENARIO_COLORS[sc], label=SCENARIO_NAMES[sc],
                      edgecolor="white", alpha=0.85)
        for bar, r in zip(bars, rmse):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{r:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([p[:14] for p in PV_COLS], fontsize=9)
    ax.set_ylabel("RMSE (normalised)")
    ax.set_title("RMSE per PV × Scenario")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    _savefig(fig, "rmse_per_pv.png")


# ── Plot 5: Rolling RMSE over time ───────────────────────────────────────────

def plot_error_over_time(rollouts: dict[int, dict], window: int = 60) -> None:
    """Rolling RMSE (averaged over PVs) over the rollout horizon."""
    scs = sorted(rollouts)
    fig, ax = plt.subplots(figsize=(14, 4))
    T = 0

    for sc in scs:
        d      = rollouts[sc]
        pred   = d["pred_pvs"]
        actual = d["actual_pvs"]
        T      = len(pred)
        t      = np.arange(T)

        mse_t  = ((pred - actual) ** 2).mean(axis=1)   # (T,)
        # rolling mean
        kernel  = np.ones(window) / window
        rolling = np.convolve(mse_t, kernel, mode="same")

        ax.plot(t, np.sqrt(rolling),
                color=SCENARIO_COLORS[sc], lw=1.3,
                label=SCENARIO_NAMES[sc], alpha=0.85)

    _xticks(ax, T)
    ax.set_xlabel("Time"); ax.set_ylabel(f"Rolling RMSE (window={window}s)")
    ax.set_title("Prediction Error Over Time")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _savefig(fig, "error_over_time.png")


# ── Plot 6: Squared error boxplot ────────────────────────────────────────────

def plot_error_boxplot(rollouts: dict[int, dict]) -> None:
    """Squared error distribution per PV, grouped by scenario."""
    scs  = sorted(rollouts)
    n_pv = len(PV_COLS)

    fig, axes = plt.subplots(1, n_pv, figsize=(4 * n_pv, 5), sharey=False)
    if n_pv == 1:
        axes = [axes]

    for ax, (row, pv) in zip(axes, enumerate(PV_COLS)):
        data, labels, colors = [], [], []
        for sc in scs:
            d   = rollouts[sc]
            err = ((d["pred_pvs"][:, row] - d["actual_pvs"][:, row]) ** 2)
            data.append(err)
            labels.append(SCENARIO_NAMES[sc])
            colors.append(SCENARIO_COLORS[sc])

        bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.5))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col); patch.set_alpha(0.7)

        ax.set_title(pv[:16], fontsize=8)
        ax.set_xticklabels(labels, fontsize=7, rotation=25)
        ax.set_ylabel("Squared error" if row == 0 else "")
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Reconstruction Error per PV × Scenario", fontsize=11)
    fig.tight_layout()
    _savefig(fig, "error_boxplot.png")


# ── Plot 7: Scatter true vs predicted ────────────────────────────────────────

def plot_scatter(rollouts: dict[int, dict]) -> None:
    """True vs predicted scatter per PV (normal scenario only)."""
    if 0 not in rollouts:
        print("  [SKIP] no normal scenario rollout for scatter plot")
        return

    d      = rollouts[0]
    pred   = d["pred_pvs"]    # (T, n_pv)
    actual = d["actual_pvs"]
    n_pv   = len(PV_COLS)

    fig, axes = plt.subplots(1, n_pv, figsize=(5 * n_pv, 5), squeeze=False)

    for col, pv in enumerate(PV_COLS):
        ax   = axes[0][col]
        true_v = actual[:, col]
        pred_v = pred[:, col]

        rmse   = float(np.sqrt(np.mean((true_v - pred_v) ** 2)))
        ss_res = float(np.sum((true_v - pred_v) ** 2))
        ss_tot = float(np.sum((true_v - true_v.mean()) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        lo = min(float(true_v.min()), float(pred_v.min()))
        hi = max(float(true_v.max()), float(pred_v.max()))

        ax.scatter(true_v, pred_v, s=2, alpha=0.35, color="steelblue", rasterized=True)
        ax.plot([lo, hi], [lo, hi], color="red", lw=1.5, label="y = x")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(pv[:16], fontsize=9)
        ax.text(0.05, 0.93, f"R²={r2:.4f}\nRMSE={rmse:.4f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("True vs Predicted — Normal Scenario", fontsize=12)
    fig.tight_layout()
    _savefig(fig, "scatter_true_vs_pred.png")


# ── Plot 8: Residual Q-Q ──────────────────────────────────────────────────────

def plot_residual_qq(rollouts: dict[int, dict]) -> None:
    """Q-Q normality plot of prediction residuals per PV (normal scenario)."""
    if 0 not in rollouts:
        print("  [SKIP] no normal scenario rollout for Q-Q plot")
        return

    d      = rollouts[0]
    pred   = d["pred_pvs"]
    actual = d["actual_pvs"]
    n_pv   = len(PV_COLS)

    fig, axes = plt.subplots(1, n_pv, figsize=(5 * n_pv, 5), squeeze=False)

    for col, pv in enumerate(PV_COLS):
        ax    = axes[0][col]
        resid = (actual[:, col] - pred[:, col])
        resid -= resid.mean()
        probplot(resid, dist="norm", plot=ax)
        ax.set_title(pv[:16], fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Residual Q-Q Plot — Normal Scenario", fontsize=12)
    fig.tight_layout()
    _savefig(fig, "residual_qq.png")


# ── Plot 9: Multi-horizon NRMSE ──────────────────────────────────────────────

def _nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse      = np.sqrt(np.mean((y_true - y_pred) ** 2))
    sig_range = float(y_true.max() - y_true.min())
    return float(rmse / sig_range) if sig_range > 1e-10 else 0.0


def plot_multi_horizon_nrmse(horizons: list[int], scenario: int = 0) -> None:
    """
    For each horizon, load the rollout .npz and compute per-PV NRMSE.
    Prints a pass/fail table (threshold 0.1) and saves a bar chart.
    Mirrors the notebook's multi-horizon gating section.
    """
    NRMSE_THRESHOLD = 0.1
    results: dict[int, dict[str, float]] = {}

    print(f"\n{'='*60}")
    print("MULTI-HORIZON NRMSE VALIDATION (scenario={})".format(SCENARIO_NAMES[scenario]))
    print(f"{'='*60}")
    print(f"  {'Signal':<14} " + "  ".join(f"{h:>8}s" for h in horizons))

    for h in horizons:
        d = _load_rollout(scenario, h)
        if d is None:
            continue
        pred, actual = d["pred_pvs"], d["actual_pvs"]
        results[h] = {pv: _nrmse(actual[:, i], pred[:, i])
                      for i, pv in enumerate(PV_COLS)}

    if not results:
        print("  No rollout files found for multi-horizon plot.")
        return

    for pv in PV_COLS:
        row = f"  {pv:<14}"
        for h in horizons:
            if h not in results:
                row += f"  {'N/A':>8}"
            else:
                v = results[h][pv]
                flag = "OK" if v < NRMSE_THRESHOLD else "!!"
                row += f"  {flag}{v:>6.4f}"
        print(row)

    # Bar chart
    avail_h = [h for h in horizons if h in results]
    n_pv    = len(PV_COLS)
    x       = np.arange(n_pv)
    width   = 0.8 / len(avail_h)

    fig, ax = plt.subplots(figsize=(max(8, 2 * n_pv), 5))
    cmap = plt.cm.get_cmap("Blues", len(avail_h) + 2)

    for i, h in enumerate(avail_h):
        vals   = [results[h][pv] for pv in PV_COLS]
        offset = (i - len(avail_h) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width=width * 0.9,
                        color=cmap(i + 2), label=f"{h}s", edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6)

    ax.axhline(NRMSE_THRESHOLD, color="red", linestyle="--", lw=1.2,
               label=f"Threshold ({NRMSE_THRESHOLD})")
    ax.set_xticks(x)
    ax.set_xticklabels([p[:14] for p in PV_COLS], fontsize=9)
    ax.set_ylabel("NRMSE")
    ax.set_title(f"Multi-Horizon NRMSE — {SCENARIO_NAMES[scenario]}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    _savefig(fig, f"multi_horizon_nrmse_sc{scenario}.png")


# ── Plot 10: Level 1 — error distributions + Q-Q ─────────────────────────────

def plot_level1_validation(rollouts: dict[int, dict]) -> None:
    """
    For each scenario: histogram of (actual - predicted) per PV (top row)
    and Q-Q normality plot (bottom row). Mirrors GRU notebook Level 1 cell.
    """
    for sc, d in sorted(rollouts.items()):
        pred   = d["pred_pvs"]    # (T, n_pv)
        actual = d["actual_pvs"]
        n_pv   = len(PV_COLS)

        fig, axes = plt.subplots(2, n_pv, figsize=(4 * n_pv, 8), squeeze=False)
        fig.suptitle(f"Level 1 — PV Error Distributions  ({SCENARIO_NAMES[sc]})",
                     fontsize=12)

        for i, pv in enumerate(PV_COLS):
            err = actual[:, i] - pred[:, i]
            ln  = next((k for k, v in LOOPS_DEF.items() if v["pv"] == pv), pv)

            # Histogram
            ax = axes[0][i]
            ax.hist(err, bins=80, density=True, alpha=0.75,
                    color=SCENARIO_COLORS[sc], edgecolor="none")
            ax.axvline(0, color="red", linestyle="--", lw=1.2)
            ax.axvline(err.mean(), color="black", linestyle=":", lw=1.0,
                       label=f"μ={err.mean():.4f}")
            ax.set_title(ln, fontsize=9)
            ax.set_xlabel("Error"); ax.set_ylabel("Density" if i == 0 else "")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

            # Q-Q
            ax = axes[1][i]
            probplot(err, dist="norm", plot=ax)
            ax.set_title(f"{ln} Q-Q", fontsize=9)
            ax.grid(True, alpha=0.2)

        fig.tight_layout()
        _savefig(fig, f"level1_error_dist_sc{sc}.png")


# ── Level 2: KS test on tracking error ───────────────────────────────────────

def print_level2_ks_test(rollouts: dict[int, dict], df_source_path: Path) -> None:
    """
    KS test comparing SP-tracking error distribution between actual and
    predicted PVs. Mirrors GRU notebook Level 2 cell.
    Requires path to the raw CSV (train4) for SP values.
    """
    import pandas as pd

    if not df_source_path.exists():
        print(f"  [SKIP] Level 2 KS test — {df_source_path.name} not found")
        return

    df = pd.read_csv(df_source_path, low_memory=False)

    print(f"\n{'='*60}")
    print("LEVEL 2 — KS TEST ON SP TRACKING ERROR")
    print(f"{'='*60}")
    print(f"  {'Loop':<6} {'Scenario':<14} {'KS':>8}  {'p':>10}  Result")

    for sc, d in sorted(rollouts.items()):
        pred   = d["pred_pvs"]
        actual = d["actual_pvs"]
        T      = len(pred)

        for ln, defn in LOOPS_DEF.items():
            pv  = defn["pv"]
            sp  = defn["sp"]
            if pv not in PV_COLS or sp not in df.columns:
                continue
            pi = PV_COLS.index(pv)

            sp_vals = df[sp].iloc[:T].values[:T]
            if len(sp_vals) < T:
                continue

            real_err  = sp_vals - actual[:, pi]
            synth_err = sp_vals - pred[:, pi]
            n         = min(5000, len(real_err))
            _ks       = ks_2samp(real_err[:n], synth_err[:n])  # type: ignore
            ks: float = float(_ks[0])  # type: ignore
            p:  float = float(_ks[1])  # type: ignore
            flag      = "PASS" if ks < 0.1 else "FAIL"
            print(f"  {ln:<6} {SCENARIO_NAMES[sc]:<14} {ks:>8.4f}  {p:>10.4e}  {flag}")


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_all(scenarios: list[int], horizon: int) -> None:
    rollouts: dict[int, dict] = {}
    for sc in scenarios:
        d = _load_rollout(sc, horizon)
        if d is not None:
            rollouts[sc] = d

    if not rollouts:
        print("No rollout files found. Run rollout.py first.")
        return

    print(f"\nGenerating plots for scenarios {sorted(rollouts)} horizon={horizon}s")

    plot_loss_curves()
    plot_pv_timeseries(rollouts)
    plot_cv_timeseries(rollouts)
    plot_rmse_per_pv(rollouts)
    plot_error_over_time(rollouts)
    plot_error_boxplot(rollouts)
    plot_scatter(rollouts)
    plot_residual_qq(rollouts)

    print(f"\nDone. All plots saved to {PLOTS_DIR.relative_to(ROOT)}")

# =================================================================
# ADDITION: Multi-horizon validation with dynamic rollout (LSTM)
# =================================================================

def load_models_and_scalers(loop_names: list, device):
    """Load trained LSTM controllers, plant, and scalers from OUTPUT_DIR."""
    import torch
    from loops import ControllerLSTM
    from plant import PlantLSTM

    models = {}
    scalers = {}
    # Load per-loop controllers
    for ln in loop_names:
        ckpt = OUTPUT_DIR / f"ctrl_{ln}_best.pt"
        if not ckpt.exists():
            print(f"  [WARN] Controller {ln} not found at {ckpt}")
            continue
        # Assuming input dim = 3 (SP, PV, CV_fb) + scenario_emb_dim (set to 0 for now)
        model = ControllerLSTM(input_dim=3, hidden_dim=CONFIG.ctrl_hidden,
                               num_layers=CONFIG.ctrl_layers, scenario_dim=0).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        models[ln] = model
        # Load scaler
        scaler_path = OUTPUT_DIR / f"ctrl_{ln}_scaler.npz"
        if scaler_path.exists():
            scalers[ln] = np.load(scaler_path, allow_pickle=True)['scaler'].item()
    # Load plant
    plant_ckpt = OUTPUT_DIR / "plant_best.pt"
    if plant_ckpt.exists():
        plant_model = PlantLSTM(input_dim=24, hidden_dim=CONFIG.plant_hidden,
                                num_layers=CONFIG.plant_layers, output_dim=len(PV_COLS),
                                scenario_dim=0).to(device)
        plant_model.load_state_dict(torch.load(plant_ckpt, map_location=device))
        plant_model.eval()
        models['plant'] = plant_model
    else:
        print("  [WARN] Plant model not found")
        models['plant'] = None
    return models, scalers

def closed_loop_rollout_lstm(df, start_idx, horizon, models, scalers, device, seq_len=300):
    """
    LSTM-based closed-loop rollout similar to GRU version.
    Returns (pred_pvs, actual_pvs) or (None,None) on failure.
    """
    loop_names = ['PC', 'LC', 'FC', 'TC', 'CC']
    warmup = seq_len
    if start_idx + warmup + horizon + 1 >= len(df):
        return None, None

    # ----- Warmup controllers -----
    ctrl_hiddens = {}
    for ln in loop_names:
        if ln not in models: continue
        model = models[ln]
        scaler = scalers[ln]
        cols = scaler['cols']
        # Build warmup data: SP, PV, CV_fb (assuming cols order)
        warmup_raw = df[cols].iloc[start_idx:start_idx+warmup].values
        warmup_scaled = (warmup_raw - scaler['mean'][:-1]) / scaler['scale'][:-1]
        warmup_x = torch.tensor(warmup_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, h = model.lstm(warmup_x)  # ControllerLSTM should have .lstm attribute
        ctrl_hiddens[ln] = h

    # ----- Warmup plant -----
    plant_model = models['plant']
    plant_scaler = scalers['plant']
    plant_cols = plant_scaler['cols']
    n_plant_in = len([c for c in plant_cols if c not in PV_COLS])  # approximate
    warmup_plant = (df[plant_cols].iloc[start_idx:start_idx+warmup].values -
                    plant_scaler['mean']) / plant_scaler['scale']
    # Assume plant input = CVs + aux (first n_plant_in columns), output = PVs (last len(PV_COLS))
    warmup_cv = torch.tensor(warmup_plant[:, :n_plant_in], dtype=torch.float32).unsqueeze(0).to(device)
    warmup_pv = torch.tensor(warmup_plant[:, n_plant_in:], dtype=torch.float32).unsqueeze(0).to(device)
    plant_h = None
    with torch.no_grad():
        for t in range(warmup):
            x_t = torch.cat([warmup_cv[:, t, :], warmup_pv[:, t, :]], dim=-1).unsqueeze(1)
            _, plant_h = plant_model.lstm(x_t, plant_h)

    # Initial PV (scaled)
    pv_offset = n_plant_in
    current_pvs_scaled = torch.tensor(warmup_plant[-1, pv_offset:], dtype=torch.float32).unsqueeze(0).to(device)

    # ----- Rollout -----
    pred_pvs, actual_pvs = [], []
    for t in range(horizon):
        abs_t = start_idx + warmup + t
        if abs_t + 1 >= len(df):
            break

        # Predict CVs for each loop
        cvs_raw = {}
        for ln in loop_names:
            if ln not in models: continue
            model = models[ln]
            scaler = scalers[ln]
            cols = scaler['cols']
            # Get SP from real data
            sp_raw = df[cols[0]].iloc[abs_t]
            # Get PV from current_pvs_scaled (denormalize)
            pv_idx = PV_COLS.index(LOOPS_DEF[ln]['pv'])
            pv_raw = current_pvs_scaled[0, pv_idx].item() * plant_scaler['scale'][pv_offset+pv_idx] + plant_scaler['mean'][pv_offset+pv_idx]
            # CV_fb (real data)
            cv_fb_raw = df[cols[2]].iloc[abs_t] if len(cols) > 2 else 0.0
            input_raw = np.array([sp_raw, pv_raw, cv_fb_raw])
            input_scaled = (input_raw - scaler['mean'][:-1]) / scaler['scale'][:-1]
            x_t = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                cv_scaled, ctrl_hiddens[ln] = model.step(x_t, ctrl_hiddens[ln])
            cv_raw = cv_scaled.item() * scaler['scale'][-1] + scaler['mean'][-1]
            cv_raw = np.clip(cv_raw, *CV_LIMITS[ln])
            cvs_raw[ln] = cv_raw

        # Build plant input (scaled)
        plant_row_raw = df[plant_cols].iloc[abs_t].values.copy()
        for ln in loop_names:
            cv_col = LOOPS_DEF[ln]['cv']
            if cv_col in plant_cols:
                idx = plant_cols.index(cv_col)
                plant_row_raw[idx] = cvs_raw[ln]
        plant_row_scaled = (plant_row_raw - plant_scaler['mean']) / plant_scaler['scale']
        cv_aux_scaled = torch.tensor(plant_row_scaled[:n_plant_in], dtype=torch.float32).unsqueeze(0).to(device)

        # Plant step
        with torch.no_grad():
            pv_pred_scaled, plant_h = plant_model.step(cv_aux_scaled, current_pvs_scaled, plant_h)
        pv_pred_raw = pv_pred_scaled.cpu().numpy()[0] * plant_scaler['scale'][pv_offset:] + plant_scaler['mean'][pv_offset:]
        for i, col in enumerate(PV_COLS):
            pv_pred_raw[i] = np.clip(pv_pred_raw[i], *PV_LIMITS[col])
        # Rescale for next step
        pv_pred_rescaled = (pv_pred_raw - plant_scaler['mean'][pv_offset:]) / plant_scaler['scale'][pv_offset:]
        current_pvs_scaled = torch.tensor(pv_pred_rescaled, dtype=torch.float32).unsqueeze(0).to(device)

        pred_pvs.append(pv_pred_raw)
        actual_pvs.append(df[PV_COLS].iloc[abs_t + 1].values)

    if len(pred_pvs) == 0:
        return None, None
    return np.array(pred_pvs), np.array(actual_pvs)

def run_multi_horizon_validation(df_val, horizons, config, device):
    """Multi-horizon validation with multiple windows, prints NRMSE table and saves plot."""
    print("\n" + "=" * 60)
    print("MULTI-HORIZON VALIDATION (LSTM) — Level 1 & 2")
    print("=" * 60)

    # Load models and scalers (assumed saved during training)
    models, scalers = load_models_and_scalers(['PC','LC','FC','TC','CC'], device)
    if not models or models.get('plant') is None:
        print("  Models not found. Skipping dynamic validation.")
        return

    gate_results = {}
    for H in horizons:
        n_windows = min(8, (len(df_val) - config.seq_len - H - 1) // H)
        if n_windows < 1:
            continue
        print(f"\n--- Horizon: {H}s ({H/60:.0f} min), {n_windows} windows ---")
        nrmse_per_pv = {c: [] for c in PV_COLS}
        first_pred, first_act = None, None
        for wi in range(n_windows):
            start = config.seq_len + wi * H
            pred, act = closed_loop_rollout_lstm(df_val, start, H, models, scalers, device, config.seq_len)
            if pred is None:
                continue
            if wi == 0:
                first_pred, first_act = pred, act
            for i, pv in enumerate(PV_COLS):
                # Use documented PV range for NRMSE
                pv_range = LOOPS_DEF[[k for k,v in LOOPS_DEF.items() if v['pv']==pv][0]]['pv_range']
                sig_range = pv_range[1] - pv_range[0]
                nrmse = compute_nrmse(act[:, i], pred[:, i], sig_range)
                nrmse_per_pv[pv].append(nrmse)
        # Print results for this horizon
        print(f"\n  {'Signal':<14} {'Mean NRMSE':<12} {'Std':<10} {'Pass? (thresh=0.1)'}")
        horizon_ok = True
        for pv in PV_COLS:
            vals = nrmse_per_pv[pv]
            if not vals:
                continue
            m, s = np.mean(vals), np.std(vals)
            passed = m < 0.1
            if not passed:
                horizon_ok = False
            print(f"  {'OK' if passed else '!!'} {pv:<12} {m:<12.4f} {s:<10.4f} {'PASS' if passed else 'FAIL'}")
        gate_results[H] = {'passed': horizon_ok, 'nrmse': nrmse_per_pv}
        # Plot first window if available
        if first_pred is not None:
            fig, axes = plt.subplots(len(PV_COLS), 1, figsize=(14, 2*len(PV_COLS)), sharex=True)
            t = np.arange(len(first_pred)) / 60
            for i, pv in enumerate(PV_COLS):
                axes[i].plot(t, first_act[:, i], label='Actual', color='green')
                axes[i].plot(t, first_pred[:, i], label='Predicted', linestyle='--', color='red')
                axes[i].fill_between(t, first_act[:, i], first_pred[:, i], alpha=0.2, color='red')
                axes[i].set_ylabel(pv[:12])
                axes[i].legend(loc='upper right')
                axes[i].grid(True, alpha=0.3)
            axes[-1].set_xlabel('Minutes')
            fig.suptitle(f'Closed-Loop Rollout (LSTM) — Horizon {H}s')
            fig.tight_layout()
            _savefig(fig, f"multi_horizon_rollout_h{H}.png")
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Best passing horizon:")
    best_h = 0
    for H, res in sorted(gate_results.items()):
        if res['passed']:
            best_h = H
            print(f"  Horizon {H}s: ALL PVs PASS")
        else:
            print(f"  Horizon {H}s: FAIL (some PVs exceed threshold)")
    if best_h == 0:
        print("  No horizon passed all PVs. Consider retraining or reducing threshold.")
    else:
        print(f"  Recommended rollout horizon: {best_h}s")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, nargs="+", default=None,
                        help="Scenario id(s) to evaluate (default: all found)")
    parser.add_argument("--all",      action="store_true",
                        help="Try all 5 scenarios")
    parser.add_argument("--horizon",  type=int, default=1800)
    args = parser.parse_args()

    if args.all or args.scenario is None:
        scenarios = list(range(5))
    else:
        scenarios = args.scenario

    run_all(scenarios, args.horizon)
