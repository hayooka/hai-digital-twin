"""
EDA Pipeline — 10 Steps
Entry point: python run_eda.py

Steps:
  1.  Schema Summary
  2.  Missing Values Analysis
  3.  Label Distribution
  4.  Descriptive Statistics
  5.  Outlier Detection
  6.  Correlation Analysis
  7.  Sensor Distribution by Class
  8.  Attack Timeline
  9.  Rolling Statistics
  10. Attack Segment Analysis
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import load_merged
from utils.eda_visualization import (
    plot_missing_values_heatmap,
    plot_label_distribution,
    plot_correlation_heatmap,
    plot_sensor_distributions,
    plot_attack_timeline,
    plot_rolling_statistics,
)
from utils.schema import META_COLS


def load_test2() -> "pd.DataFrame":
    return load_merged("test", 2)


def get_schema_summary(df: "pd.DataFrame") -> dict:
    import numpy as np
    n_rows, n_cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    numeric_cols    = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols   = df.select_dtypes(include=["datetime64"]).columns.tolist()
    col_type_counts = {"numeric": len(numeric_cols), "categorical": len(categorical_cols),
                       "datetime": len(datetime_cols)}
    null_counts = df.isnull().sum().to_dict()
    null_pct    = (df.isnull().mean() * 100).round(2).to_dict()
    duplicate_rows = int(df.duplicated().sum())
    label_value_counts: dict = {}
    attack_rate_pct: float = 0.0
    if "attack" in df.columns:
        vc = df["attack"].value_counts().sort_index()
        label_value_counts = vc.to_dict()
        attack_rate_pct = round(int(vc.get(1, 0)) / n_rows * 100, 4)
    time_range: dict = {}
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        time_range = {"min": str(ts.min()), "max": str(ts.max()),
                      "duration_hours": round((ts.max() - ts.min()).total_seconds() / 3600, 2)}
    return {"shape": {"rows": n_rows, "cols": n_cols}, "memory_mb": round(memory_mb, 2),
            "col_type_counts": col_type_counts, "null_counts": null_counts, "null_pct": null_pct,
            "duplicate_rows": duplicate_rows, "label_value_counts": label_value_counts,
            "attack_rate_pct": attack_rate_pct, "time_range": time_range}

Path("reports/figures").mkdir(parents=True, exist_ok=True)

SEP = "=" * 60


# ── Load Data ─────────────────────────────────────────────────────────────────

print(SEP)
print("  HAI DIGITAL TWIN — EDA PIPELINE")
print(SEP)
print("\nLoading test2 (held-out evaluation split)...")
df = load_test2()
print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")


# ── Step 1 — Schema Summary ───────────────────────────────────────────────────

print(f"\n{SEP}")
print("  STEP 1 — SCHEMA SUMMARY")
print(SEP)

summary = get_schema_summary(df)

print(f"  Shape:          {summary['shape']['rows']:,} rows x {summary['shape']['cols']} cols")
print(f"  Memory:         {summary['memory_mb']:.1f} MB")
print(f"  Column types:   {summary['col_type_counts']}")
print(f"  Duplicate rows: {summary['duplicate_rows']:,}")
print(f"  Attack rate:    {summary['attack_rate_pct']:.3f}%")

if summary["label_value_counts"]:
    print(f"  Label counts:   {summary['label_value_counts']}")

if summary["time_range"]:
    tr = summary["time_range"]
    print(f"  Time range:     {tr['min']}  →  {tr['max']}")
    print(f"  Duration:       {tr['duration_hours']:.1f} hours")

total_nulls = sum(v for v in summary["null_counts"].values() if v > 0)
print(f"  Total nulls:    {total_nulls:,}")


# ── Step 2 — Missing Values ───────────────────────────────────────────────────

print(f"\n{SEP}")
print("  STEP 2 — MISSING VALUES ANALYSIS")
print(SEP)

out = plot_missing_values_heatmap(df)
print(f"  Saved: {out}")


# ── Step 3 — Label Distribution ───────────────────────────────────────────────

print(f"\n{SEP}")
print("  STEP 3 — LABEL DISTRIBUTION")
print(SEP)

out = plot_label_distribution(df)
vc = df["attack"].value_counts().sort_index()
if 1 in vc.index and vc[1] > 0:
    ratio = vc.get(0, 0) / vc[1]
    print(f"  Normal : Attack = {ratio:.1f} : 1")
print(f"  Saved: {out}")


# ── Step 4 — Descriptive Statistics ──────────────────────────────────────────

print(f"\n{SEP}")
print("  STEP 4 — DESCRIPTIVE STATISTICS")
print(SEP)

sensor_cols = [c for c in df.select_dtypes(include=[np.number]).columns
               if c not in META_COLS][:50]

desc = df[sensor_cols].describe().T
desc["cv"] = desc["std"] / (desc["mean"].abs() + 1e-9)   # coefficient of variation

out_csv = Path("reports/descriptive_stats.csv")
desc.to_csv(out_csv)
print(f"  Computed stats for {len(sensor_cols)} sensors")
print(f"  Saved: {out_csv}")
print(f"\n  Top 5 by std:\n{desc['std'].nlargest(5).to_string()}")


# ── Step 5 — Outlier Detection ────────────────────────────────────────────────

print(f"\n{SEP}")
print("  STEP 5 — OUTLIER DETECTION  (IQR x3)")
print(SEP)

outlier_counts: dict[str, int] = {}
for col in sensor_cols:
    q1  = df[col].quantile(0.25)
    q3  = df[col].quantile(0.75)
    iqr = q3 - q1
    lo  = q1 - 3 * iqr
    hi  = q3 + 3 * iqr
    n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
    if n_out > 0:
        outlier_counts[col] = n_out

print(f"  Columns with outliers: {len(outlier_counts)} / {len(sensor_cols)}")
top10 = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:10]
print("  Top 10 offenders:")
for col, cnt in top10:
    pct = cnt / len(df) * 100
    print(f"    {col:<35s}  {cnt:6,}  ({pct:.2f}%)")


# ── Step 6 — Correlation Analysis ────────────────────────────────────────────

print(f"\n{SEP}")
print("  STEP 6 — CORRELATION ANALYSIS")
print(SEP)

out = plot_correlation_heatmap(df, top_n=40)
print(f"  Saved: {out}")


# ── Step 7 — Sensor Distribution by Class ────────────────────────────────────

print(f"\n{SEP}")
print("  STEP 7 — SENSOR DISTRIBUTION BY CLASS")
print(SEP)

out = plot_sensor_distributions(df, max_sensors=20)
print(f"  Saved: {out}")


# ── Step 8 — Attack Timeline ──────────────────────────────────────────────────

print(f"\n{SEP}")
print("  STEP 8 — ATTACK TIMELINE")
print(SEP)

from utils.schema import select_critical_sensors
critical = select_critical_sensors(df, n=10)
print(f"  Critical sensors (data-driven, top variance + |corr with attack|):")
for s in critical:
    print(f"    {s}")

out = plot_attack_timeline(df)
print(f"  Saved: {out}")


# ── Step 9 — Rolling Statistics ───────────────────────────────────────────────

print(f"\n{SEP}")
print("  STEP 9 — ROLLING STATISTICS  (window=300s)")
print(SEP)

out = plot_rolling_statistics(df, window=300, max_rows=10_000, max_sensors=4)
print(f"  Saved: {out}")


# ── Step 10 — Attack Segment Analysis ────────────────────────────────────────

print(f"\n{SEP}")
print("  STEP 10 — ATTACK SEGMENT ANALYSIS")
print(SEP)

if "attack" in df.columns:
    labels = df["attack"].values
    segments: list[dict] = []
    in_seg = False
    start  = 0

    for i, v in enumerate(labels):
        if v == 1 and not in_seg:
            in_seg = True
            start  = i
        elif v == 0 and in_seg:
            in_seg = False
            segments.append({"start": start, "end": i - 1, "duration": i - start})
    if in_seg:
        segments.append({"start": start, "end": len(labels) - 1,
                         "duration": len(labels) - start})

    if segments:
        durations = [s["duration"] for s in segments]
        print(f"  Total attack segments: {len(segments)}")
        print(f"  Min duration:          {min(durations)}s  "
              f"({min(durations)//60}:{min(durations)%60:02d})")
        print(f"  Max duration:          {max(durations)}s  "
              f"({max(durations)//60}:{max(durations)%60:02d})")
        mean_d = int(np.mean(durations))
        print(f"  Mean duration:         {mean_d}s  "
              f"({mean_d//60}:{mean_d%60:02d})")
    else:
        print("  No attack segments found in this split.")
else:
    print("  No attack column — skipped.")


# ── Done ──────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  EDA PIPELINE COMPLETE")
print(f"  Figures → reports/figures/")
print(f"  Stats   → reports/descriptive_stats.csv")
print(SEP)
