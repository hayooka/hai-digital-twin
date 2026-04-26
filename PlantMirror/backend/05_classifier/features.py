"""Shared feature extractor — used by build_features.py and the dashboard tab.

Extracts 10 engineered features per PV from a 180-second window, flattened
into a fixed 50-dim vector (5 PVs x 10 stats). The same function must be used
at train time and at inference time, otherwise the classifier will see a
distribution shift.
"""
from __future__ import annotations

import numpy as np

PV_COLS = ["P1_PIT01", "P1_LIT01", "P1_FT03Z", "P1_TIT01", "P1_TIT03"]
WIN_SEC = 180
STRIDE_SEC = 60
N_FEATS_PER_PV = 10
N_FEATS = len(PV_COLS) * N_FEATS_PER_PV  # 50


def _pv_features(x: np.ndarray) -> np.ndarray:
    """10 features for one PV channel of length WIN_SEC."""
    x = x.astype(np.float32)
    mean = float(x.mean())
    std = float(x.std())
    mn = float(x.min())
    mx = float(x.max())
    p05 = float(np.percentile(x, 5))
    p95 = float(np.percentile(x, 95))
    # Linear slope
    t = np.arange(x.size, dtype=np.float32)
    slope = float(np.polyfit(t, x, 1)[0]) if std > 1e-9 else 0.0
    # Lag-1 autocorrelation
    if std > 1e-9:
        xc = x - mean
        denom = float((xc ** 2).sum())
        lag1 = float((xc[:-1] * xc[1:]).sum() / denom) if denom > 0 else 0.0
    else:
        lag1 = 0.0
    # FFT peak (skip DC bin)
    spec = np.abs(np.fft.rfft(x - mean))
    if spec.size > 1:
        k = int(np.argmax(spec[1:])) + 1
        peak_freq = float(k / x.size)
        peak_power = float(spec[k])
    else:
        peak_freq = 0.0
        peak_power = 0.0
    return np.array(
        [mean, std, mn, mx, p05, p95, slope, lag1, peak_freq, peak_power],
        dtype=np.float32,
    )


def window_features(window: np.ndarray) -> np.ndarray:
    """window: (WIN_SEC, 5) → (50,) feature vector."""
    feats = [_pv_features(window[:, i]) for i in range(window.shape[1])]
    return np.concatenate(feats, axis=0)


def slide_windows(
    pv_array: np.ndarray,
    label_array: np.ndarray | None,
    win: int = WIN_SEC,
    stride: int = STRIDE_SEC,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Returns (X_feats, y, anchors).
    - pv_array: (T, 5) float
    - label_array: (T,) int or None. y[i]=1 if any second in window has label>0.
    - anchors: (N,) int — start index of each window.
    """
    T = pv_array.shape[0]
    if T < win:
        return np.zeros((0, N_FEATS), dtype=np.float32), None, np.zeros((0,), dtype=np.int64)
    anchors = np.arange(0, T - win + 1, stride, dtype=np.int64)
    X = np.zeros((anchors.size, N_FEATS), dtype=np.float32)
    y = None
    if label_array is not None:
        y = np.zeros(anchors.size, dtype=np.int8)
    for i, a in enumerate(anchors):
        w = pv_array[a : a + win]
        X[i] = window_features(w)
        if y is not None:
            y[i] = int((label_array[a : a + win] > 0).any())
    return X, y, anchors
