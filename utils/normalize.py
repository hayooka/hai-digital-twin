"""
Normalization utilities for the HAI dataset (225 sensors).

Provides per-sensor z-score and min-max normalization with a std clamp fix:
  std values below MIN_STD are clamped to MIN_STD to prevent division by zero
  or exploding values for near-constant sensor channels.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

# HAI dataset: 225 sensor/actuator columns (excludes timestamp and attack label)
N_SENSORS = 225

# Minimum allowed std before clamping — prevents blow-up on constant channels
MIN_STD = 1e-6


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
        sensor_cols : explicit list of column names to normalize.
                      If None, all numeric columns (up to N_SENSORS) are used.
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
        return cols[:N_SENSORS]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "HAISensorNormalizer":
        """Compute normalization statistics from df (training data only)."""
        cols = self._resolve_cols(df)
        self.sensor_cols = cols
        self._stats = {}

        if self.method == "zscore":
            for col in cols:
                mean = float(df[col].mean())
                std  = float(df[col].std(ddof=0))
                # std clamp fix — prevent blow-up on constant/near-constant channels
                if std < MIN_STD:
                    std = MIN_STD
                self._stats[col] = {"mean": mean, "std": std}

        else:  # minmax
            for col in cols:
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
                out[col] = (df[col] - s["mean"]) / s["std"]
        else:
            for col in self.sensor_cols:
                s = self._stats[col]
                out[col] = (df[col] - s["min"]) / s["range"]

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df then transform it (convenience for training split)."""
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Undo normalization; returns values in the original sensor scale."""
        self._check_fitted()
        out = df.copy()

        if self.method == "zscore":
            for col in self.sensor_cols:
                s = self._stats[col]
                out[col] = df[col] * s["std"] + s["mean"]
        else:
            for col in self.sensor_cols:
                s = self._stats[col]
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
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "HAISensorNormalizer":
        """Load a previously saved normalizer from JSON."""
        payload = json.loads(Path(path).read_text())
        obj = cls(method=payload["method"], sensor_cols=payload["sensor_cols"])
        obj._stats = payload["stats"]
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
