"""
Data Loader for HAI + HAIEnd 23.05
====================================
Merges HAI (86 tags) and HAIEnd (225 tags) by row alignment.
HAI timestamps are 1 second ahead of HAIEnd — same rows, dropped timestamp.

Output per split:
  X: (N, 277) float32  — merged features (no duplicates)
  y: (N,)     int      — labels (0=normal, 1=attack) — test only

Usage:
    from utils.data_loader import load_split, load_all

    X_train, _  = load_split('train', [1,2,3], HAI_DIR, HAIEND_DIR)
    X_val,   _  = load_split('train', [4],     HAI_DIR, HAIEND_DIR)
    X_test,  y  = load_split('test',  [1,2],   HAI_DIR, HAIEND_DIR)
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# Paths — edit these to match your machine
# ─────────────────────────────────────────────────────────────────
HAI_DIR    = Path("C:/Users/farah/OneDrive/Desktop/AI_project/hai-23.05")
HAIEND_DIR = Path("C:/Users/farah/OneDrive/Desktop/AI_project/haiend-23.05")

# Columns to drop (labels + timestamps)
DROP_COLS  = ['timestamp', 'attack', 'Attack']


# ─────────────────────────────────────────────────────────────────
# Single file loader
# ─────────────────────────────────────────────────────────────────

def load_file(hai_path: Path, haiend_path: Path,
              return_labels: bool = False):
    """
    Load and merge one HAI + HAIEnd file pair.

    Args:
        hai_path:      path to hai CSV file
        haiend_path:   path to haiend CSV file
        return_labels: if True, also return attack labels

    Returns:
        X: (N, 277) float32 numpy array
        y: (N,) int numpy array — only if return_labels=True
    """
    print(f"Loading {hai_path.name} + {haiend_path.name} ...")

    hai    = pd.read_csv(hai_path)
    haiend = pd.read_csv(haiend_path)

    # Sanity check — must have same number of rows
    assert len(hai) == len(haiend), \
        f"Row mismatch: {hai_path.name}={len(hai)}, {haiend_path.name}={len(haiend)}"

    # Extract labels before dropping
    y = None
    if return_labels:
        label_col = next((c for c in ['attack', 'Attack'] if c in hai.columns), None)
        if label_col:
            y = hai[label_col].values.astype(int)
        else:
            label_col = next((c for c in ['attack', 'Attack'] if c in haiend.columns), None)
            if label_col:
                y = haiend[label_col].values.astype(int)

    # Drop timestamp + label columns
    for col in DROP_COLS:
        if col in hai.columns:    hai    = hai.drop(columns=[col])
        if col in haiend.columns: haiend = haiend.drop(columns=[col])

    # Merge — drop HAIEnd columns that already exist in HAI (avoid duplicates)
    duplicate_cols = [c for c in haiend.columns if c in hai.columns]
    if duplicate_cols:
        haiend = haiend.drop(columns=duplicate_cols)

    merged = pd.concat([hai, haiend], axis=1)
    X = merged.values.astype(np.float32)

    if return_labels:
        return X, y
    return X, None


# ─────────────────────────────────────────────────────────────────
# Split loader  (multiple numbered files)
# ─────────────────────────────────────────────────────────────────

def load_split(
    split: str,
    file_nums: list[int],
    hai_dir: Path = HAI_DIR,
    haiend_dir: Path = HAIEND_DIR,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load and concatenate multiple files for one split.

    Args:
        split:     'train' or 'test'
        file_nums: list of file indices, e.g. [1, 2, 3]
        hai_dir:   root directory for HAI CSVs
        haiend_dir: root directory for HAIEnd CSVs

    Returns:
        X: (N_total, 277) float32
        y: (N_total,) int  — None for train split (no labels)
    """
    is_test = split == 'test'
    Xs, ys = [], []

    for n in file_nums:
        hai_path    = hai_dir    / split / f"hai_{split}{n:02d}.csv"
        haiend_path = haiend_dir / split / f"haiend_{split}{n:02d}.csv"

        X, y = load_file(hai_path, haiend_path, return_labels=is_test)
        Xs.append(X)
        if y is not None:
            ys.append(y)

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0) if ys else None
    return X_all, y_all


# ─────────────────────────────────────────────────────────────────
# Convenience: load everything
# ─────────────────────────────────────────────────────────────────

def load_all(
    train_nums: list[int],
    test_nums:  list[int],
    hai_dir:    Path = HAI_DIR,
    haiend_dir: Path = HAIEND_DIR,
) -> dict[str, np.ndarray | None]:
    """
    Load train and test splits in one call.

    Returns:
        dict with keys: 'X_train', 'X_test', 'y_test'
        (y_train is None — train files have no attack labels)
    """
    X_train, _      = load_split('train', train_nums, hai_dir, haiend_dir)
    X_test,  y_test = load_split('test',  test_nums,  hai_dir, haiend_dir)

    return {
        'X_train': X_train,
        'X_test':  X_test,
        'y_test':  y_test,
    }
