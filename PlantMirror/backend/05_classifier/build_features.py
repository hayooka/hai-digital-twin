"""Build the (X_train, X_test, y_test) feature matrices used by the bench.

Reads data/processed/train{1..4}.csv and test{1,2}.csv, slides 180 s windows
with stride 60 s over the 5 PV columns, extracts 50 features per window, and
saves outputs/classifier/features.npz.

Training set = all train*.csv (normal only). Test set = labeled test*.csv.
Feature z-scoring is computed on X_train only and applied to both.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "05_classifier"))
from features import PV_COLS, N_FEATS, slide_windows  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "classifier"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILES = [f"train{i}.csv" for i in range(1, 5)]
TEST_FILES = ["test1.csv", "test2.csv"]


def _load_pv_and_label(path: Path) -> tuple[np.ndarray, np.ndarray]:
    usecols = PV_COLS + (["label"] if "label" else [])
    df = pd.read_csv(path, usecols=PV_COLS + ["label"])
    return df[PV_COLS].to_numpy(dtype=np.float32), df["label"].to_numpy()


def _process(files: list[str], label_aware: bool) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_meta: list[str] = []
    for fname in files:
        path = DATA_DIR / fname
        print(f"  [{fname}] loading…", flush=True)
        pv, lab = _load_pv_and_label(path)
        X, y, anchors = slide_windows(pv, lab if label_aware else None)
        print(f"  [{fname}] {X.shape[0]} windows", flush=True)
        all_X.append(X)
        if label_aware and y is not None:
            all_y.append(y)
        all_meta.extend([f"{fname}@{int(a)}" for a in anchors])
    X = np.concatenate(all_X, axis=0) if all_X else np.zeros((0, N_FEATS), dtype=np.float32)
    y = np.concatenate(all_y, axis=0) if (label_aware and all_y) else None
    return X, y, all_meta


def main() -> int:
    print("Building TRAIN features (normal only)…")
    X_train, _, meta_train = _process(TRAIN_FILES, label_aware=False)
    print(f"  -> X_train {X_train.shape}")

    print("Building TEST features (labeled)...")
    X_test, y_test, meta_test = _process(TEST_FILES, label_aware=True)
    print(f"  -> X_test {X_test.shape} | attack windows {int((y_test == 1).sum())}")

    # Z-score using train-only stats
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd < 1e-9] = 1.0
    X_train_z = (X_train - mu) / sd
    X_test_z = (X_test - mu) / sd

    out = OUT_DIR / "features.npz"
    np.savez_compressed(
        out,
        X_train=X_train_z.astype(np.float32),
        X_test=X_test_z.astype(np.float32),
        y_test=y_test.astype(np.int8),
        mu=mu.astype(np.float32),
        sd=sd.astype(np.float32),
        meta_train=np.array(meta_train),
        meta_test=np.array(meta_test),
    )
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
