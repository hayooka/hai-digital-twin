"""Calibrate the plant-GRU detector threshold on normal-only training data.

Runs the plant GRU on random windows from train*.csv (all labels are 0 there),
takes the 99th percentile of window MSE as the alert threshold → FPR ~ 1 %.
No test-set leakage.

Output: outputs/classifier/plant_threshold.json
  {"threshold": 0.012, "n_windows": 500, "p95": ..., "p99": ..., "max": ...}
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from twin_core import (
    INPUT_LEN, TARGET_LEN,
    build_plant_window, default_paths,
    load_bundle, load_replay, predict_plant, window_mse,
)

OUT = ROOT / "outputs" / "classifier" / "plant_threshold.json"
N_SAMPLES_PER_FILE = 150   # → ~600 normal windows total
SEED = 0


def main() -> int:
    paths = default_paths()
    bundle = load_bundle(paths["ckpt_dir"], paths["split_dir"])
    rng = random.Random(SEED)
    scores: list[float] = []

    for fname in ["train1.csv", "train2.csv", "train3.csv", "train4.csv"]:
        csv_path = ROOT / "data" / "processed" / fname
        print(f"[{fname}] loading …", flush=True)
        src = load_replay(csv_path, bundle.scalers)
        n = len(src)
        lo, hi = INPUT_LEN, n - TARGET_LEN
        if hi <= lo:
            print(f"  too short, skipping")
            continue
        anchors = [rng.randrange(lo, hi) for _ in range(N_SAMPLES_PER_FILE)]
        for i, t in enumerate(anchors):
            win = build_plant_window(bundle, src, int(t))
            if win is None:
                continue
            pv_pred = predict_plant(bundle, win)
            pv_true = win["pv_target"].squeeze(0).cpu().numpy()
            scores.append(window_mse(pv_pred, pv_true))
            if (i + 1) % 25 == 0:
                print(f"  [{fname}] {i+1}/{len(anchors)}", flush=True)

    arr = np.asarray(scores, dtype=np.float64)
    print(f"\nCollected {arr.size} normal-window MSEs")
    print(f"  mean={arr.mean():.6f}  median={np.median(arr):.6f}")
    print(f"  p95={np.percentile(arr, 95):.6f}  p99={np.percentile(arr, 99):.6f}")
    print(f"  max={arr.max():.6f}")

    thr = float(np.percentile(arr, 99))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": thr,
        "calibration": "train-99th-percentile",
        "n_windows": int(arr.size),
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
        "p99": thr,
        "max": float(arr.max()),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {OUT}")
    print(f"Threshold = {thr:.6f}  (FPR target ≈ 1 %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
