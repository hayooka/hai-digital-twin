"""
Sliding-window utilities for the merged HAI + HAIEnd dataset.
=============================================================
Converts a flat time-series array (N, 277) into overlapping windows
suitable for sequence models (LSTM, Transformer, VAE, Diffusion).

The HAI dataset is sampled at 1 Hz (one row = 1 second), so
window_size=60 gives 1-minute context windows.

Key functions
-------------
make_windows(X, window_size, stride)
    → windows  (M, window_size, 277) float32

label_windows(y, window_size, stride, policy)
    → labels   (M,) int  — one label per window

make_windows_with_labels(X, y, window_size, stride, policy)
    → windows, labels      (convenience wrapper)

Usage
-----
    from utils.windows import make_windows, make_windows_with_labels

    # train — no labels
    W_train = make_windows(X_train, window_size=60, stride=1)

    # test — with labels
    W_test, y_win = make_windows_with_labels(X_test, y_test,
                                              window_size=60, stride=1)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────
# Core windowing
# ─────────────────────────────────────────────────────────────────

def make_windows(
    X: np.ndarray,
    window_size: int = 300,
    stride: int = 1,
) -> np.ndarray:
    """
    Slice a (N, F) time-series into overlapping windows.

    Args:
        X:           (N, F) float32 array — output of load_split / normalize
        window_size: number of time steps per window (seconds at 1 Hz)
        stride:      step between window starts (1 = fully overlapping)

    Returns:
        windows: (M, window_size, F) float32
                 where M = (N - window_size) // stride + 1
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D array (N, F), got shape {X.shape}")
    if window_size > len(X):
        raise ValueError(
            f"window_size={window_size} exceeds number of rows={len(X)}"
        )

    N, F = X.shape
    starts = range(0, N - window_size + 1, stride)
    M = len(starts)

    windows = np.empty((M, window_size, F), dtype=np.float32)
    for i, s in enumerate(starts):
        windows[i] = X[s : s + window_size]

    return windows


# ─────────────────────────────────────────────────────────────────
# Label assignment per window
# ─────────────────────────────────────────────────────────────────

def label_windows(
    y: np.ndarray,
    window_size: int = 300,
    stride: int = 1,
    policy: str = "last",
) -> np.ndarray:
    """
    Assign one integer label per window from a flat label array.

    Args:
        y:           (N,) int array — row-level labels from load_split
        window_size: must match the value used in make_windows
        stride:      must match the value used in make_windows
        policy:      how to collapse window labels into one value
                       "last"  — label of the final row in the window
                                 (standard for next-step prediction)
                       "any"   — 1 if any row in window is an attack
                                 (recommended for anomaly detection)
                       "majority" — majority vote (>50% attack → 1)

    Returns:
        labels: (M,) int array, same M as make_windows output
    """
    if y.ndim != 1:
        raise ValueError(f"Expected 1-D label array, got shape {y.shape}")
    if policy not in ("last", "any", "majority"):
        raise ValueError(f"policy must be 'last', 'any', or 'majority', got '{policy}'")

    N = len(y)
    starts = range(0, N - window_size + 1, stride)
    M = len(starts)
    labels = np.empty(M, dtype=np.int32)

    for i, s in enumerate(starts):
        window_y = y[s : s + window_size]
        if policy == "last":
            labels[i] = window_y[-1]
        elif policy == "any":
            labels[i] = int(window_y.any())
        else:  # majority
            labels[i] = int(window_y.mean() > 0.5)

    return labels


# ─────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────

def make_windows_with_labels(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 300,
    stride: int = 1,
    policy: str = "any",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build windows and per-window labels in one call.

    Args:
        X:           (N, 277) float32
        y:           (N,) int
        window_size: seconds of context per window
        stride:      step between windows
        policy:      label-collapse policy — see label_windows()

    Returns:
        windows: (M, window_size, 277) float32
        labels:  (M,) int
    """
    windows = make_windows(X, window_size, stride)
    labels  = label_windows(y, window_size, stride, policy)
    return windows, labels


# ─────────────────────────────────────────────────────────────────
# Utility: count windows without allocating memory
# ─────────────────────────────────────────────────────────────────

def n_windows(n_rows: int, window_size: int, stride: int = 1) -> int:
    """Return the number of windows that make_windows would produce."""
    if n_rows < window_size:
        return 0
    return (n_rows - window_size) // stride + 1
