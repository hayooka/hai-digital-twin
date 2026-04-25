"""
calibrate_thresholds.py — per-PV threshold sweep, read-only.

Runs the existing frozen twin across a test CSV, collects the 180s-rolling
residual for each of the 5 PVs, sweeps thresholds, reports F1-max per PV
and compares against the single global threshold currently in results.json.

Does NOT modify any app file. Pure evaluation.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from twin_core import (
    INPUT_LEN, TARGET_LEN, PV_COLS, default_paths, load_bundle, load_replay,
)
from twin_runtime import TwinRuntime, SCORE_WINDOW_SEC


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling mean along axis 0."""
    c = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0)
    out = (c[w:] - c[:-w]) / w
    pad = np.full((w - 1,) + x.shape[1:], np.nan)
    return np.concatenate([pad, out], axis=0)


def run_on_csv(bundle, src, step_size: int = 60) -> dict:
    rt = TwinRuntime(bundle, src)
    # Remove buffer cap so we capture the ENTIRE residual stream.
    rt.per_pv_residual = deque()
    rt.cursor_times = deque()
    rt.twin_pv = deque()
    rt.actual_pv = deque()
    rt.sim_times = deque()
    rt.warm_up(cursor=INPUT_LEN, scenario=0)
    # warm_up clears deques — restore unbounded
    rt.per_pv_residual = deque()
    rt.cursor_times = deque()
    rt.twin_pv = deque()
    rt.actual_pv = deque()
    rt.sim_times = deque()

    total_steps = len(src) - INPUT_LEN - TARGET_LEN
    print(f"  stepping through {total_steps:,} simulated seconds "
          f"(chunks of {step_size})...")
    done = 0
    while done < total_steps:
        n = rt.step(min(step_size, total_steps - done))
        done += n
        if n == 0:
            break
        if done % 6000 == 0:
            print(f"    progress {done:,}/{total_steps:,}")

    per_pv = np.stack(list(rt.per_pv_residual), axis=0)   # (T, 5)
    cursor_times = np.fromiter(rt.cursor_times, dtype=np.int64)
    labels = src.df_raw["label"].to_numpy()[cursor_times]
    return {
        "per_pv": per_pv,
        "labels": (labels > 0).astype(int),
        "alerts_live": len(rt.alerts),
    }


def sweep(residual_pv: np.ndarray, y: np.ndarray, n: int = 400) -> tuple:
    mask = ~np.isnan(residual_pv)
    r, yy = residual_pv[mask], y[mask]
    if r.size == 0 or yy.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan
    lo, hi = float(np.percentile(r, 1)), float(np.percentile(r, 99.9))
    best = (-1.0, np.nan, np.nan, np.nan)
    for tau in np.linspace(lo, hi, n):
        pred = (r >= tau).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            yy, pred, average="binary", zero_division=0,
        )
        if f1 > best[0]:
            best = (f1, float(tau), float(prec), float(rec))
    try:
        auc = roc_auc_score(yy, r)
    except ValueError:
        auc = np.nan
    return best[0], best[1], best[2], best[3], auc


def main() -> None:
    paths = default_paths()
    bundle = load_bundle(paths["ckpt_dir"], paths["split_dir"])
    print(f"Global threshold in checkpoint: {bundle.threshold:.5f}\n")

    csv = paths["test_csvs"][0]
    print(f"CSV: {csv.name}")
    src = load_replay(csv, bundle.scalers)
    result = run_on_csv(bundle, src)
    per_pv = result["per_pv"]                      # (T, 5)
    y = result["labels"]                           # (T,)

    # Apply the same 180s rolling mean the live detector uses
    smoothed = rolling_mean(per_pv, SCORE_WINDOW_SEC)   # (T, 5)
    global_score = rolling_mean(per_pv.mean(axis=1, keepdims=True),
                                SCORE_WINDOW_SEC).ravel()

    print(f"\nLabels in buffer: total={y.size:,}  "
          f"attack-seconds={int(y.sum()):,}  "
          f"attack-ratio={y.mean():.3f}")
    print(f"Alerts fired under current rule (global τ={bundle.threshold:.3f}): "
          f"{result['alerts_live']}")

    # ── global-threshold evaluation for comparison ────────────────────────
    mask = ~np.isnan(global_score)
    g = global_score[mask]; yy = y[mask]
    pred_g = (g >= bundle.threshold).astype(int)
    prec_g, rec_g, f1_g, _ = precision_recall_fscore_support(
        yy, pred_g, average="binary", zero_division=0,
    )
    try:
        auc_g = roc_auc_score(yy, g)
    except ValueError:
        auc_g = float("nan")
    print(f"\n=== CURRENT (single global τ={bundle.threshold:.4f}) ===")
    print(f"  AUROC = {auc_g:.4f}")
    print(f"  F1    = {f1_g:.4f}   precision = {prec_g:.4f}   "
          f"recall = {rec_g:.4f}")

    # Sweep global
    f1s, tau_s, p_s, r_s, auc_s = sweep(g, yy)
    print(f"\n=== RE-CALIBRATED global τ ===")
    print(f"  best τ = {tau_s:.4f}   F1 = {f1s:.4f}   "
          f"precision = {p_s:.4f}   recall = {r_s:.4f}   AUROC = {auc_s:.4f}")

    # ── per-PV sweep ─────────────────────────────────────────────────────
    print(f"\n=== PER-PV best thresholds (sweep on val residual) ===")
    print(f"{'PV':<12}{'τ*':>10}{'F1':>10}{'prec':>10}{'recall':>10}{'AUROC':>10}")
    taus = {}
    for i, pv in enumerate(PV_COLS):
        f1, tau, p, r, auc = sweep(smoothed[:, i], y)
        taus[pv] = tau
        print(f"{pv:<12}{tau:>10.4f}{f1:>10.4f}{p:>10.4f}{r:>10.4f}{auc:>10.4f}")

    # OR fusion: alert if ANY PV exceeds its own τ
    preds_per_pv = np.zeros_like(smoothed, dtype=int)
    for i, pv in enumerate(PV_COLS):
        tau = taus[pv]
        if np.isnan(tau):
            continue
        col = smoothed[:, i]
        preds_per_pv[:, i] = (col >= tau).astype(int) * (~np.isnan(col))
    pred_any = preds_per_pv.any(axis=1).astype(int)
    valid = ~np.isnan(smoothed).any(axis=1)
    pa, ya = pred_any[valid], y[valid]
    prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(
        ya, pa, average="binary", zero_division=0,
    )
    print(f"\n=== Per-PV OR-fusion (fire if ANY PV exceeds its own τ*) ===")
    print(f"  F1 = {f1_a:.4f}   precision = {prec_a:.4f}   recall = {rec_a:.4f}")

    # 2-of-5 fusion
    pred_2of5 = (preds_per_pv.sum(axis=1) >= 2).astype(int)
    p2, y2 = pred_2of5[valid], y[valid]
    prec2, rec2, f12, _ = precision_recall_fscore_support(
        y2, p2, average="binary", zero_division=0,
    )
    print(f"\n=== Per-PV 2-of-5 fusion ===")
    print(f"  F1 = {f12:.4f}   precision = {prec2:.4f}   recall = {rec2:.4f}")


if __name__ == "__main__":
    main()
