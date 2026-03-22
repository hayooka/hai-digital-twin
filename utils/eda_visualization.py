"""
EDA Visualization utilities — Steps 2-9 of the EDA pipeline.

All functions save to reports/figures/ and return the output path.
matplotlib uses Agg backend (no display required).
"""
from __future__ import annotations

import sys
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.schema import META_COLS, select_critical_sensors

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Step 2 — Missing Values ────────────────────────────────────────────────────

def plot_missing_values_heatmap(df: pd.DataFrame,
                                max_cols: int = 50,
                                sample_rows: int = 1000,
                                ) -> Path:
    """
    Bar chart of top columns by missing % + heatmap of missing pattern.
    Output: reports/figures/missing_values.png
    """
    null_pct = df.isnull().mean() * 100
    top_cols = null_pct[null_pct > 0].nlargest(max_cols)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Missing Values Analysis", fontsize=14, fontweight="bold")

    # Bar chart
    ax = axes[0]
    if len(top_cols) > 0:
        top_cols.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(f"Top {len(top_cols)} Columns by Missing %")
        ax.set_ylabel("Missing %")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.axhline(y=5, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="5% threshold")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No missing values found", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("Missing Values — Bar Chart")

    # Heatmap on random sample
    ax2 = axes[1]
    sample = df.sample(min(sample_rows, len(df)), random_state=42)
    cols_with_nulls = sample.columns[sample.isnull().any()].tolist()[:max_cols]
    if cols_with_nulls:
        missing_matrix = sample[cols_with_nulls].isnull().astype(int)
        im = ax2.imshow(missing_matrix.T, aspect="auto", cmap="RdYlGn_r",
                        interpolation="none", vmin=0, vmax=1)
        ax2.set_yticks(range(len(cols_with_nulls)))
        ax2.set_yticklabels(cols_with_nulls, fontsize=6)
        ax2.set_xlabel(f"Sample rows (n={len(sample)})")
        ax2.set_title("Missing Pattern Heatmap")
        plt.colorbar(im, ax=ax2, label="Missing (1=yes)")
    else:
        ax2.text(0.5, 0.5, "No missing values in sample", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Missing Pattern Heatmap")

    plt.tight_layout()
    out = FIGURES_DIR / "missing_values.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


# ── Step 3 — Label Distribution ───────────────────────────────────────────────

def plot_label_distribution(df: pd.DataFrame) -> Path:
    """
    Bar chart with counts & percentages + pie chart with attack ratio.
    Output: reports/figures/label_distribution.png
    """
    if "attack" not in df.columns:
        raise ValueError("DataFrame has no 'attack' column.")

    vc = df["attack"].value_counts().sort_index()
    labels_map = {0: "Normal", 1: "Attack"}
    labels = [labels_map.get(k, str(k)) for k in vc.index]
    counts = vc.values
    pcts   = counts / counts.sum() * 100
    colors = ["steelblue", "tomato"]

    imbalance_ratio = counts[0] / counts[1] if len(counts) > 1 and counts[1] > 0 else float("inf")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Label Distribution", fontsize=14, fontweight="bold")

    # Bar chart
    ax = axes[0]
    bars = ax.bar(labels, counts, color=colors[:len(labels)], edgecolor="white", width=0.5)
    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.01,
                f"{cnt:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_title(f"Class Counts  |  Imbalance ratio {imbalance_ratio:.1f}:1")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="y", alpha=0.3)

    # Pie chart
    ax2 = axes[1]
    ax2.pie(counts, labels=labels, colors=colors[:len(labels)],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax2.set_title(f"Attack Rate: {pcts[1]:.2f}%" if len(pcts) > 1 else "All Normal")

    plt.tight_layout()
    out = FIGURES_DIR / "label_distribution.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


# ── Step 6 — Correlation Heatmap ──────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, top_n: int = 40) -> Path:
    """
    Pearson correlation matrix on top N columns by variance.
    Lower-triangle heatmap + top 10 features correlated with attack label.
    Output: reports/figures/correlation_heatmap.png
    """
    sensor_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in META_COLS]

    # Select top columns by variance
    variances = df[sensor_cols].var().nlargest(top_n)
    cols = variances.index.tolist()

    corr = df[cols].corr()

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(np.where(mask, np.nan, corr.values),
                   cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=6)
    ax.set_yticklabels(cols, fontsize=6)
    ax.set_title(f"Correlation Heatmap — Top {top_n} sensors by variance\n"
                 f"(lower triangle, Pearson r)", fontsize=11)

    plt.tight_layout()
    out = FIGURES_DIR / "correlation_heatmap.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()

    # Top 10 features correlated with attack label
    if "attack" in df.columns:
        atk_corr = df[cols + ["attack"]].corr()["attack"].drop("attack").abs()
        top10 = atk_corr.nlargest(10)
        print("\n  Top 10 features by |correlation| with attack label:")
        for feat, val in top10.items():
            print(f"    {feat:<35s}  {val:.4f}")

    return out


# ── Step 7 — Sensor Distribution by Class ─────────────────────────────────────

def plot_sensor_distributions(df: pd.DataFrame,
                              max_sensors: int = 20,
                              n_cols: int = 4,
                              ) -> Path:
    """
    Overlaid histograms (Normal vs Attack) for up to 20 sensors.
    Output: reports/figures/sensor_distributions.png
    """
    if "attack" not in df.columns:
        raise ValueError("DataFrame has no 'attack' column.")

    sensor_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in META_COLS]

    # Pick sensors with highest variance (most informative)
    variances = df[sensor_cols].var().nlargest(max_sensors)
    cols = variances.index.tolist()

    n_rows_fig = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows_fig, n_cols,
                             figsize=(5 * n_cols, 3.5 * n_rows_fig))
    axes = np.array(axes).flatten()
    fig.suptitle("Sensor Distribution — Normal vs Attack", fontsize=13, fontweight="bold")

    normal = df[df["attack"] == 0]
    attack = df[df["attack"] == 1]

    for i, col in enumerate(cols):
        ax = axes[i]
        ax.hist(normal[col].dropna(), bins=50, alpha=0.6, color="steelblue",
                label="Normal", density=True)
        ax.hist(attack[col].dropna(), bins=50, alpha=0.6, color="tomato",
                label="Attack", density=True)
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_ylabel("Density", fontsize=6)
        if i == 0:
            ax.legend(fontsize=7)

    # Hide unused axes
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "sensor_distributions.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


# ── Step 8 — Attack Timeline ───────────────────────────────────────────────────

def plot_attack_timeline(df: pd.DataFrame) -> Path:
    """
    Top subplot: attack label as filled area over time.
    Bottom subplot: one critical sensor with shaded attack windows.
    Output: reports/figures/attack_timeline.png
    """
    if "attack" not in df.columns:
        raise ValueError("DataFrame has no 'attack' column.")

    critical = select_critical_sensors(df)
    sensor = critical[0] if critical else None

    # Use timestamp index if available
    if "timestamp" in df.columns:
        x = pd.to_datetime(df["timestamp"])
    else:
        x = pd.RangeIndex(len(df))

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.suptitle("Attack Timeline", fontsize=13, fontweight="bold")

    # Top: attack label
    ax1 = axes[0]
    ax1.fill_between(x, df["attack"], color="tomato", alpha=0.7, step="post")
    ax1.set_ylabel("Attack Label")
    ax1.set_ylim(-0.05, 1.3)
    ax1.set_yticks([0, 1])
    ax1.set_title("Attack Label (1 = Attack)")
    ax1.grid(axis="y", alpha=0.3)

    # Bottom: critical sensor signal with shaded attack windows
    ax2 = axes[1]
    if sensor and sensor in df.columns:
        ax2.plot(x, df[sensor], color="steelblue", linewidth=0.5, label=sensor)
        # Shade attack windows
        attack_mask = df["attack"].values.astype(bool)
        ax2.fill_between(x, df[sensor].min(), df[sensor].max(),
                         where=attack_mask, color="tomato", alpha=0.25,
                         label="Attack window")
        ax2.set_ylabel(sensor)
        ax2.set_title(f"Sensor: {sensor}")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No critical sensor found in dataset",
                 ha="center", va="center", transform=ax2.transAxes)

    ax2.set_xlabel("Time")
    plt.tight_layout()
    out = FIGURES_DIR / "attack_timeline.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


# ── Step 9 — Rolling Statistics ───────────────────────────────────────────────

def plot_rolling_statistics(df: pd.DataFrame,
                            window: int = 300,
                            max_rows: int = 10_000,
                            max_sensors: int = 4,
                            ) -> Path:
    """
    Rolling mean (steelblue) and rolling std (coral) for up to 4 critical sensors.
    Window: 300 seconds (5 minutes). Computed on first max_rows rows.
    Output: reports/figures/rolling_statistics.png
    """
    critical = select_critical_sensors(df)
    sensors  = [s for s in critical if s in df.columns][:max_sensors]

    if not sensors:
        # Fall back to highest-variance numeric sensors
        sensor_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c not in META_COLS]
        sensors = df[sensor_cols].var().nlargest(max_sensors).index.tolist()

    sub = df.iloc[:max_rows].copy()

    n = len(sensors)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Rolling Statistics (window={window}s, first {max_rows:,} rows)",
                 fontsize=12, fontweight="bold")

    for i, sensor in enumerate(sensors):
        series = sub[sensor].ffill()
        roll_mean = series.rolling(window, min_periods=1).mean()
        roll_std  = series.rolling(window, min_periods=1).std().fillna(0)

        # Rolling mean
        ax_mean = axes[i, 0]
        ax_mean.plot(series.values, color="lightgray", linewidth=0.4, label="raw")
        ax_mean.plot(roll_mean.values, color="steelblue", linewidth=1.2, label=f"mean(w={window})")
        ax_mean.set_title(f"{sensor} — Rolling Mean", fontsize=9)
        ax_mean.set_ylabel("Value", fontsize=8)
        ax_mean.legend(fontsize=7)
        ax_mean.grid(alpha=0.3)

        # Rolling std
        ax_std = axes[i, 1]
        ax_std.plot(roll_std.values, color="coral", linewidth=1.0, label=f"std(w={window})")
        ax_std.set_title(f"{sensor} — Rolling Std", fontsize=9)
        ax_std.set_ylabel("Std", fontsize=8)
        ax_std.legend(fontsize=7)
        ax_std.grid(alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "rolling_statistics.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out
