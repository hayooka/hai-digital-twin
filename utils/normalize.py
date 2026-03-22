"""
Normalization utilities for the merged HAI + HAIEnd dataset (277 features).

Provides per-feature z-score and min-max normalization with a std clamp fix:
  std values below MIN_STD are clamped to MIN_STD to prevent division by zero
  or exploding values for near-constant sensor channels.

The normalizer works with both:
  - numpy arrays  (N, 277) — output of data_loader.load_split
  - pandas DataFrames      — for column-aware usage

Typical usage with numpy arrays:
    norm = HAISensorNormalizer()
    X_train_scaled            = norm.fit_transform_array(X_train)
    X_test_scaled             = norm.transform_array(X_test)
    norm.save("scaler.json")
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

# Merged HAI + HAIEnd feature count (86 HAI + 225 HAIEnd − duplicates = 277)
N_FEATURES = 277
N_SENSORS  = N_FEATURES   # backward-compat alias

# Minimum allowed std before clamping — prevents blow-up on constant channels
# 1e-6 is too small: sensors constant in training but active in test → (val/1e-6) = millions
MIN_STD = 0.01


class HAISensorNormalizer:
    """
    Fit-transform normalizer for HAI 225-sensor data.

    Supports z-score (default) and min-max normalization.
    Fitted statistics can be saved/loaded as JSON for reproducibility.

    Usage
    -----
    norm = HAISensorNormalizer(method="zscore")
    norm.fit(train_df)                  # learn stats from training split only
    train_scaled = norm.transform(train_df)
    test_scaled  = norm.transform(test_df)
    original     = norm.inverse_transform(train_scaled)
    norm.save("scaler_stats.json")
    """

    def __init__(self, method: str = "zscore", sensor_cols: list[str] | None = None):
        """
        Parameters
        ----------
        method      : "zscore" or "minmax"
        sensor_cols : explicit list of column names to normalize (DataFrame mode).
                      If None, all numeric columns (up to N_FEATURES) are used.
                      Ignored when using the array API (fit_array / transform_array).
        """
        if method not in ("zscore", "minmax"):
            raise ValueError(f"method must be 'zscore' or 'minmax', got '{method}'")
        self.method = method
        self.sensor_cols: list[str] = sensor_cols or []
        self._stats: dict[str, dict] = {}   # col -> {mean, std} or {min, max}
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_cols(self, df: pd.DataFrame) -> list[str]:
        if self.sensor_cols:
            missing = [c for c in self.sensor_cols if c not in df.columns]
            if missing:
                raise KeyError(f"Columns not found in DataFrame: {missing}")
            return self.sensor_cols
        # Auto-detect: numeric columns, skip known meta columns
        meta = {"timestamp", "time", "attack", "label", "attack_p1",
                 "attack_p2", "attack_p3"}
        cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c.lower() not in meta]
        return cols[:N_FEATURES]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "HAISensorNormalizer":
        """Compute normalization statistics from df (training data only)."""
        cols = self._resolve_cols(df)
        self.sensor_cols = cols
        self._stats = {}

        # Detect binary columns (only 0/1 values) — skip normalization for these.
        # Normalizing a binary switch makes no physical sense and distorts the scale.
        binary_cols = {
            col for col in cols
            if df[col].dropna().isin([0, 1]).all() and df[col].nunique() <= 2
        }
        if binary_cols:
            pass  # handled per-column below with binary=True marker

        if self.method == "zscore":
            for col in cols:
                if col in binary_cols:
                    self._stats[col] = {"binary": True}
                    continue
                mean = float(df[col].mean())
                std  = float(df[col].std(ddof=0))
                # std clamp fix — prevent blow-up on constant/near-constant channels
                if std < MIN_STD:
                    std = MIN_STD
                self._stats[col] = {"mean": mean, "std": std}

        else:  # minmax
            for col in cols:
                if col in binary_cols:
                    self._stats[col] = {"binary": True}
                    continue
                lo = float(df[col].min())
                hi = float(df[col].max())
                rng = hi - lo
                # clamp tiny ranges the same way
                if rng < MIN_STD:
                    rng = MIN_STD
                self._stats[col] = {"min": lo, "max": hi, "range": rng}

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with sensor columns normalized."""
        self._check_fitted()
        out = df.copy()

        if self.method == "zscore":
            for col in self.sensor_cols:
                s = self._stats[col]
                if s.get("binary"):
                    continue   # leave 0/1 values unchanged
                out[col] = (df[col] - s["mean"]) / s["std"]
        else:
            for col in self.sensor_cols:
                s = self._stats[col]
                if s.get("binary"):
                    continue
                out[col] = (df[col] - s["min"]) / s["range"]

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df then transform it (convenience for training split)."""
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Numpy array API  — primary path when using data_loader output
    # ------------------------------------------------------------------

    def fit_array(self, X: np.ndarray) -> "HAISensorNormalizer":
        """
        Fit on a (N, F) float32 array (output of load_split).
        Stores per-feature statistics indexed by integer position.
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}")

        n_features = X.shape[1]
        self._arr_stats: dict[int, dict] = {}

        if self.method == "zscore":
            mean = X.mean(axis=0)
            std  = X.std(axis=0, ddof=0)
            std  = np.where(std < MIN_STD, MIN_STD, std)
            for i in range(n_features):
                self._arr_stats[i] = {"mean": float(mean[i]), "std": float(std[i])}

        else:  # minmax
            lo  = X.min(axis=0)
            hi  = X.max(axis=0)
            rng = hi - lo
            rng = np.where(rng < MIN_STD, MIN_STD, rng)
            for i in range(n_features):
                self._arr_stats[i] = {"min": float(lo[i]), "max": float(hi[i]),
                                      "range": float(rng[i])}

        self._n_features = n_features
        self._fitted = True
        return self

    def transform_array(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted statistics to a (N, F) array. Returns float32."""
        self._check_fitted()
        if not hasattr(self, '_arr_stats'):
            raise RuntimeError("Fitted with DataFrame API — use transform() instead.")

        X = X.astype(np.float32)
        if self.method == "zscore":
            mean = np.array([self._arr_stats[i]["mean"] for i in range(self._n_features)],
                            dtype=np.float32)
            std  = np.array([self._arr_stats[i]["std"]  for i in range(self._n_features)],
                            dtype=np.float32)
            return (X - mean) / std
        else:
            lo  = np.array([self._arr_stats[i]["min"]   for i in range(self._n_features)],
                           dtype=np.float32)
            rng = np.array([self._arr_stats[i]["range"] for i in range(self._n_features)],
                           dtype=np.float32)
            return (X - lo) / rng

    def fit_transform_array(self, X: np.ndarray) -> np.ndarray:
        """Fit on X then transform it (convenience for training split)."""
        return self.fit_array(X).transform_array(X)

    def inverse_transform_array(self, X: np.ndarray) -> np.ndarray:
        """Undo normalization on a (N, F) array. Returns float32."""
        self._check_fitted()
        if not hasattr(self, '_arr_stats'):
            raise RuntimeError("Fitted with DataFrame API — use inverse_transform() instead.")

        X = X.astype(np.float32)
        if self.method == "zscore":
            mean = np.array([self._arr_stats[i]["mean"] for i in range(self._n_features)],
                            dtype=np.float32)
            std  = np.array([self._arr_stats[i]["std"]  for i in range(self._n_features)],
                            dtype=np.float32)
            return X * std + mean
        else:
            lo  = np.array([self._arr_stats[i]["min"]   for i in range(self._n_features)],
                           dtype=np.float32)
            rng = np.array([self._arr_stats[i]["range"] for i in range(self._n_features)],
                           dtype=np.float32)
            return X * rng + lo

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Undo normalization; returns values in the original sensor scale."""
        self._check_fitted()
        out = df.copy()

        if self.method == "zscore":
            for col in self.sensor_cols:
                s = self._stats[col]
                if s.get("binary"):
                    continue   # binary cols were never scaled — nothing to undo
                out[col] = df[col] * s["std"] + s["mean"]
        else:
            for col in self.sensor_cols:
                s = self._stats[col]
                if s.get("binary"):
                    continue
                out[col] = df[col] * s["range"] + s["min"]

        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save fitted statistics to a JSON file."""
        self._check_fitted()
        payload = {
            "method": self.method,
            "sensor_cols": self.sensor_cols,
            "stats": self._stats,
        }
        if hasattr(self, '_arr_stats'):
            payload["arr_stats"]   = {str(k): v for k, v in self._arr_stats.items()}
            payload["n_features"]  = self._n_features
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "HAISensorNormalizer":
        """Load a previously saved normalizer from JSON."""
        payload = json.loads(Path(path).read_text())
        obj = cls(method=payload["method"], sensor_cols=payload["sensor_cols"])
        obj._stats = payload["stats"]
        if "arr_stats" in payload:
            obj._arr_stats   = {int(k): v for k, v in payload["arr_stats"].items()}
            obj._n_features  = payload["n_features"]
        obj._fitted = True
        return obj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Normalizer has not been fitted yet. Call fit() first.")

    def __repr__(self) -> str:
        status = f"{len(self.sensor_cols)} sensors" if self._fitted else "unfitted"
        return f"HAISensorNormalizer(method='{self.method}', {status})"


# ---------------------------------------------------------------------------
# Convenience functions for quick, stateless use
# ---------------------------------------------------------------------------

def zscore_normalize(
    arr: np.ndarray,
    mean: np.ndarray | None = None,
    std:  np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize a 2-D array (samples × sensors) with std clamp fix.

    If mean/std are supplied (e.g., computed on training data) they are used
    directly instead of being recomputed.

    Returns
    -------
    normalized : np.ndarray  shape (N, 225)
    mean       : np.ndarray  shape (225,)
    std        : np.ndarray  shape (225,)  — already clamped
    """
    if mean is None:
        mean = arr.mean(axis=0)
    if std is None:
        std = arr.std(axis=0, ddof=0)

    # std clamp fix
    std = np.where(std < MIN_STD, MIN_STD, std)

    return (arr - mean) / std, mean, std


def minmax_normalize(
    arr: np.ndarray,
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Min-max normalize a 2-D array (samples × sensors) with range clamp fix.

    Returns
    -------
    normalized : np.ndarray
    lo         : np.ndarray  per-sensor min
    hi         : np.ndarray  per-sensor max
    """
    if lo is None:
        lo = arr.min(axis=0)
    if hi is None:
        hi = arr.max(axis=0)

    rng = hi - lo
    # clamp fix — same pattern as std clamp
    rng = np.where(rng < MIN_STD, MIN_STD, rng)

    return (arr - lo) / rng, lo, hi
