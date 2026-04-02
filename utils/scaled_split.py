"""
scaled_split.py — Split and normalize HAI + HAIEnd processed data.

Pipeline
--------
Step 1 : Split each train file 80/20 (time-ordered, no shuffle)
Step 2 : Extract attack segments from test1 + test2
           300s before attack + attack period + 180s after attack
Step 3 : Randomly split attack segments 80/20
Step 4 : Fit scaler on train_80_normal ONLY (no attacks)
Step 5 : Transform all splits with the same scaler

Outputs
-------
X_train_normal : 80% of normal data        (label = 0)
X_train_attack : 80% of attack segments    (label = 1-4)
X_val          : 20% of normal data        (label = 0)
X_test         : 20% of attack segments    (label = 1-4)
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

PROCESSED_DIR     = "data/processed"
OUTPUT_DIR        = "outputs"
TRAIN_RATIO       = 0.80
BEFORE_ATTACK_SEC = 300
AFTER_ATTACK_SEC  = 180
RANDOM_SEED       = 42

META_COLS = {
    "timestamp", "label",
    "attack_id", "scenario", "attack_type", "combination",
    "target_controller", "target_points", "duration_sec",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _sensor_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def _extract_segments(df: pd.DataFrame,
                      before: int, after: int) -> list[pd.DataFrame]:
    """
    Find contiguous attack blocks (label > 0) and return one DataFrame
    per block containing: before_sec rows + attack rows + after_sec rows.
    """
    if "label" not in df.columns or not (df["label"] > 0).any():
        return []

    mask = (df["label"] > 0).values
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            blocks.append((i, j - 1))
            i = j
        else:
            i += 1

    segments = []
    for start, end in blocks:
        lo = max(0, start - before)
        hi = min(len(df) - 1, end + after)
        segments.append(df.iloc[lo: hi + 1].copy().reset_index(drop=True))

    return segments


# ── main ──────────────────────────────────────────────────────────────────────

def load_and_prepare(
    processed_dir:     str   = PROCESSED_DIR,
    output_dir:        str   = OUTPUT_DIR,
    train_ratio:       float = TRAIN_RATIO,
    before_attack_sec: int   = BEFORE_ATTACK_SEC,
    after_attack_sec:  int   = AFTER_ATTACK_SEC,
    random_seed:       int   = RANDOM_SEED,
) -> dict:

    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: split each train file 80/20 ──────────────────────────────────
    print("\n[Step 1] Splitting train files 80/20 (time-ordered) …")
    train_parts, val_parts = [], []
    sensor_cols: list[str] | None = None

    for n in range(1, 5):
        df = _load(os.path.join(processed_dir, f"train{n}.csv"))
        if sensor_cols is None:
            sensor_cols = _sensor_cols(df)
            print(f"  sensor features: {len(sensor_cols)}")
        cut = int(len(df) * train_ratio)
        train_parts.append(df.iloc[:cut])
        val_parts.append(df.iloc[cut:])
        print(f"  train{n}: {len(df):,} rows → 80%: {cut:,}  20%: {len(df)-cut:,}")

    train_normal = pd.concat(train_parts, ignore_index=True)
    val_normal   = pd.concat(val_parts,   ignore_index=True)
    print(f"  combined train_normal: {len(train_normal):,}  val_normal: {len(val_normal):,}")

    # ── Step 2: extract attack segments from test1 + test2 ───────────────────
    print("\n[Step 2] Extracting attack segments …")
    segments: list[pd.DataFrame] = []
    for n in range(1, 3):
        df = _load(os.path.join(processed_dir, f"test{n}.csv"))
        segs = _extract_segments(df, before_attack_sec, after_attack_sec)
        print(f"  test{n}: {len(segs)} segments")
        for i, s in enumerate(segs):
            lbls = sorted(s["label"].unique())
            print(f"    segment {i}: {len(s):,} rows  labels={lbls}")
        segments.extend(segs)
    print(f"  total segments: {len(segments)}")

    # ── Step 3: split attack segments 80/20 ──────────────────────────────────
    print("\n[Step 3] Splitting attack segments 80/20 …")
    if len(segments) == 0:
        attack_train = pd.DataFrame()
        attack_test  = pd.DataFrame()
        print("  WARNING: no attack segments found")
    elif len(segments) == 1:
        attack_train = segments[0]
        attack_test  = pd.DataFrame()
        print("  Only 1 segment — all goes to train, test is empty")
    else:
        segs_train, segs_test = train_test_split(
            segments, train_size=train_ratio,
            random_state=random_seed, shuffle=True,
        )
        attack_train = pd.concat(segs_train, ignore_index=True)
        attack_test  = pd.concat(segs_test,  ignore_index=True)
        print(f"  attack_train: {len(segs_train)} segs  {len(attack_train):,} rows")
        print(f"  attack_test : {len(segs_test)} segs  {len(attack_test):,} rows")

    # ── Step 4: fit scaler on train_normal ONLY ───────────────────────────────
    print("\n[Step 4] Fitting scaler on train_normal only …")
    assert sensor_cols is not None, "No train files loaded — sensor_cols is empty"
    scaler = StandardScaler()
    scaler.fit(train_normal[sensor_cols].values.astype(np.float32))
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  fitted on {len(train_normal):,} normal samples → saved to {scaler_path}")

    # ── Step 5: transform ─────────────────────────────────────────────────────
    print("\n[Step 5] Transforming all splits …")

    def _transform(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if len(df) == 0:
            return (np.empty((0, len(sensor_cols)), dtype=np.float32),
                    np.empty(0, dtype=np.int32))
        X_raw = df[sensor_cols].to_numpy(dtype=np.float32)
        X = np.array(scaler.transform(X_raw), dtype=np.float32)
        y = np.array(df["label"].values, dtype=np.int32)
        return X, y

    X_train_normal, y_train_normal = _transform(train_normal)
    X_train_attack, y_train_attack = _transform(attack_train)
    X_val,          y_val          = _transform(val_normal)
    X_test,         y_test         = _transform(attack_test)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    for name, X, y in [
        ("X_train_normal", X_train_normal, y_train_normal),
        ("X_train_attack", X_train_attack, y_train_attack),
        ("X_val          ", X_val,          y_val),
        ("X_test         ", X_test,         y_test),
    ]:
        lbl_info = dict(zip(*np.unique(y, return_counts=True))) if len(y) else {}
        print(f"  {name}: {X.shape}  labels={lbl_info}")
    print("="*60)

    return {
        "X_train_normal": X_train_normal, "y_train_normal": y_train_normal,
        "X_train_attack": X_train_attack, "y_train_attack": y_train_attack,
        "X_val":          X_val,          "y_val":          y_val,
        "X_test":         X_test,         "y_test":         y_test,
        "sensor_cols":    sensor_cols,
        "scaler":         scaler,
    }


if __name__ == "__main__":
    load_and_prepare()
