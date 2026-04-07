"""
Schema utilities — data-driven critical sensor selection.

select_critical_sensors() picks the top N sensors by a combined score:
    score = variance_rank + |correlation_with_attack|_rank
so the chosen sensors are both highly variable AND most linked to attacks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Meta columns that are NOT sensor readings
META_COLS = {"timestamp", "attack", "label", "attack_p1", "attack_p2", "attack_p3"}


def select_critical_sensors(df: pd.DataFrame, n: int = 10) -> list[str]:
    """
    Data-driven selection of the top N most informative sensors.

    Scoring:
      1. Rank sensors by variance           (rank 1 = highest variance)
      2. Rank sensors by |corr with attack| (rank 1 = highest |correlation|)
      3. Combined rank = variance_rank + corr_rank  (lower = better)
      4. Return top N by combined rank

    If no attack column is present, falls back to top N by variance only.
    """
    sensor_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in META_COLS
    ]

    var_series = df[sensor_cols].var()
    var_rank   = var_series.rank(ascending=False)   # rank 1 = highest variance

    if "attack" in df.columns and df["attack"].nunique() > 1:
        corr_series = df[sensor_cols].corrwith(df["attack"]).abs()
        corr_rank   = corr_series.rank(ascending=False)
        combined    = var_rank + corr_rank
    else:
        combined = var_rank

    return combined.nsmallest(n).index.tolist()
