"""
prep.py — Prepared data loaders for each model.
Each function returns arrays ready to feed directly into a model.

Pipeline order:
    data_loader → normalize → [window / filter] → model-specific output

┌─────────────────────────────────────────────────────────────────────────┐
│ MODEL → FUNCTION MAPPING                                                │
├──────────────────────────────────┬──────────────────────────────────────┤
│ Digital Twin                     │                                      │
│   Transformer Seq2Seq (primary)  │ twin()                               │
│   LSTM Seq2Seq (baseline)        │ twin()                               │
│   Mamba/S4 (future)              │ twin()                               │
│   Merging girl (Modhi)           │ twin()                               │
├──────────────────────────────────┼──────────────────────────────────────┤
│ Attack Generator                 │                                      │
│   Diffusion (primary)            │ generate()                           │
│   VAE (baseline)                 │ generate()                           │
│   LSTM-AE generator (alt)        │ generate()                           │
│   DoppelGANger (future)          │ generate()                           │
├──────────────────────────────────┼──────────────────────────────────────┤
│ Anomaly Detection                │                                      │
│   ISO Forest (primary)           │ detect_errors(errors)                │
│   Random Forest (future)         │ detect_errors(errors)                │
│   Threshold + MSE (ablation)     │ detect_errors(errors)                │
│   LSTM-AE baseline / PD2         │ detect()                             │
└──────────────────────────────────┴──────────────────────────────────────┘

NOTE: detect_errors() receives reconstruction errors from the trained
      Digital Twin — not raw sensor data. Run twin() first, train the
      model, compute errors, then pass them to detect_errors().
"""

import numpy as np
import pandas as pd

from utils.data_loader import load_merged
from utils.normalize   import HAISensorNormalizer

# Columns that are NOT sensor readings — excluded from all arrays
META_COLS = {"timestamp", "attack", "label", "attack_p1", "attack_p2", "attack_p3"}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: get sensor column names from a DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _sensor_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and df[c].dtype != object]


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: sliding seq2seq windows — no gap crossing
# ─────────────────────────────────────────────────────────────────────────────

def _seq2seq_windows(
    arr: np.ndarray,
    input_len: int = 60,
    target_len: int = 180,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice a (N, F) array into non-gap-crossing seq2seq pairs.

    Returns
    -------
    X : (M, input_len,  F)   — encoder input
    Y : (M, target_len, F)   — decoder target
    """
    span = input_len + target_len
    starts = range(0, len(arr) - span + 1, stride)
    M = len(starts)

    X = np.empty((M, input_len,  arr.shape[1]), dtype=np.float32)
    Y = np.empty((M, target_len, arr.shape[1]), dtype=np.float32)

    for i, s in enumerate(starts):
        X[i] = arr[s           : s + input_len]
        Y[i] = arr[s + input_len : s + span]

    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# 1. DIGITAL TWIN — Transformer + LSTM Seq2Seq + Mamba
#    Usage: from utils.prep import twin
# ─────────────────────────────────────────────────────────────────────────────

def twin(
    input_len: int = 60,
    target_len: int = 180,
    stride: int = 1,
) -> dict:
    """
    Prepared data for the Digital Twin group (Transformer / LSTM / Mamba Seq2Seq).

    Train:  train1 + train2 + train3  — benign rows only (attack == 0)
            windowed per file — no gap crossing between files
    Val:    train4                     — benign rows only
    Test:   test1 + test2             — all rows (benign + attack)

    Returns dict with keys:
        X_train, Y_train  : (N, input_len, 277)  /  (N, target_len, 277)
        X_val,   Y_val    : same shape
        X_test,  Y_test   : same shape
        y_test_labels     : (M,) int  — per-window attack labels for evaluation
        norm              : fitted HAISensorNormalizer (pass to detect_errors later)
    """
    # ── Load raw splits ───────────────────────────────────────────────────────
    train_dfs = [load_merged("train", i) for i in range(1, 5)]   # train1-4
    test_dfs  = [load_merged("test",  i) for i in range(1, 3)]   # test1-2

    # ── Fit normalizer on train1+2+3 ─────────────────────────────────────────
    # train files are 100% benign (attack==0 throughout) — no filtering needed.
    # train4 is held out as val — must not influence normalization stats.
    fit_df = pd.concat(train_dfs[:3], ignore_index=True)
    norm = HAISensorNormalizer(method="zscore")
    norm.fit(fit_df)

    cols = _sensor_cols(train_dfs[0])

    # ── Helper: scale → array ────────────────────────────────────────────────
    def _to_array(df: pd.DataFrame) -> np.ndarray:
        return norm.transform(df)[cols].values.astype(np.float32)

    # ── Train: window each file separately → concat (no gap crossing) ─────────
    # train1+2+3 are fully benign — no row filtering needed
    X_parts, Y_parts = [], []
    for df in train_dfs[:3]:                            # train1, train2, train3
        arr = _to_array(df)
        Xi, Yi = _seq2seq_windows(arr, input_len, target_len, stride)
        X_parts.append(Xi)
        Y_parts.append(Yi)

    X_train = np.concatenate(X_parts, axis=0)
    Y_train = np.concatenate(Y_parts, axis=0)

    # ── Val: train4 (also fully benign) ──────────────────────────────────────
    arr_val      = _to_array(train_dfs[3])
    X_val, Y_val = _seq2seq_windows(arr_val, input_len, target_len, stride)

    # ── Test: test1 + test2 (all rows, keep labels) ───────────────────────────
    test_all = pd.concat(test_dfs, ignore_index=True)
    arr_test = norm.transform(test_all)[cols].values.astype(np.float32)
    y_labels = test_all["attack"].values.astype(np.int32)

    X_test, Y_test = _seq2seq_windows(arr_test, input_len, target_len, stride)

    # align labels to windows (label = any attack in the full span)
    span = input_len + target_len
    starts = range(0, len(y_labels) - span + 1, stride)
    y_test_labels = np.array(
        [int(y_labels[s : s + span].any()) for s in starts], dtype=np.int32
    )

    print("Digital Twin data ready:")
    print(f"  X_train {X_train.shape}  Y_train {Y_train.shape}")
    print(f"  X_val   {X_val.shape}    Y_val   {Y_val.shape}")
    print(f"  X_test  {X_test.shape}   Y_test  {Y_test.shape}")

    return {
        "X_train": X_train, "Y_train": Y_train,
        "X_val":   X_val,   "Y_val":   Y_val,
        "X_test":  X_test,  "Y_test":  Y_test,
        "y_test_labels": y_test_labels,
        "norm": norm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. ATTACK GENERATOR — Diffusion + VAE + LSTM-AE generator + DoppelGANger
#    Usage: from utils.prep import generate
# ─────────────────────────────────────────────────────────────────────────────

def generate(norm: HAISensorNormalizer | None = None) -> dict:
    """
    Prepared data for the Attack Generator group (Diffusion / VAE / LSTM-AE / DoppelGANger).

    Train on:
        attack_rows  : test1 + test2  where attack == 1   shape (N, 277)
        normal_rows  : train1         where attack == 0   shape (M, 277)
                       used for class conditioning

    Eval (TSTR):
        generate synthetic attack rows → train XGBoost → test on real test1+2

    Args
    ----
    norm : fitted HAISensorNormalizer from twin().
           If None, a new one is fitted on train1 benign.

    Returns dict with keys:
        attack_rows  : (N, 277) float32  — real attack samples to learn from
        normal_rows  : (M, 277) float32  — real normal samples for conditioning
        X_test_all   : (K, 277) float32  — full test1+2 for TSTR evaluation
        y_test_all   : (K,)     int      — labels for TSTR evaluation
        norm         : the normalizer used
    """
    train1_df = load_merged("train", 1)
    test_dfs  = [load_merged("test", i) for i in range(1, 3)]
    test_all  = pd.concat(test_dfs, ignore_index=True)

    if norm is None:
        norm = HAISensorNormalizer(method="zscore")
        norm.fit(train1_df)   # train1 is fully benign — no filtering needed

    cols = _sensor_cols(train1_df)

    # Normal rows from train1 — entire file is benign
    train1_scaled = norm.transform(train1_df)
    normal_rows   = train1_scaled[cols].values.astype(np.float32)

    # Attack rows from test1+2 — attacks only exist in test files
    test_scaled = norm.transform(test_all)
    attack_mask = test_all["attack"] == 1
    attack_rows = test_scaled.loc[attack_mask, cols].values.astype(np.float32)

    # Full test set for TSTR evaluation
    X_test_all = test_scaled[cols].values.astype(np.float32)
    y_test_all = test_all["attack"].values.astype(np.int32)

    print("Attack Generator data ready:")
    print(f"  attack_rows  {attack_rows.shape}   (from test1+2, label=1)")
    print(f"  normal_rows  {normal_rows.shape}   (from train1,  label=0)")
    print(f"  X_test_all   {X_test_all.shape}    y_test_all {y_test_all.shape}")

    return {
        "attack_rows": attack_rows,
        "normal_rows": normal_rows,
        "X_test_all":  X_test_all,
        "y_test_all":  y_test_all,
        "norm":        norm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANOMALY DETECTION (errors) — ISO Forest + Random Forest + Threshold+MSE
#    Usage: from utils.prep import detect_errors
#    NOTE: call twin() first, train the model, compute errors, then call this.
# ─────────────────────────────────────────────────────────────────────────────

def detect_errors(
    train_errors: np.ndarray,
    test_errors:  np.ndarray,
    y_test:       np.ndarray,
) -> dict:
    """
    Package Digital Twin reconstruction errors for error-based detectors.

    ISO Forest / Random Forest / Threshold+MSE all train on reconstruction
    error patterns — not raw sensor data.

    Args
    ----
    train_errors : (N, 277)  per-sensor MSE errors on train4 benign windows
    test_errors  : (M, 277)  per-sensor MSE errors on test1+2 windows
    y_test       : (M,)      attack labels for test windows

    Returns dict with keys:
        X_train, X_test, y_test
    """
    assert train_errors.ndim == 2, "train_errors must be (N, 277)"
    assert test_errors.ndim  == 2, "test_errors must be (M, 277)"

    print("Anomaly Detection (errors) data ready:")
    print(f"  X_train (errors) {train_errors.shape}")
    print(f"  X_test  (errors) {test_errors.shape}   labels {y_test.shape}")

    return {
        "X_train": train_errors.astype(np.float32),
        "X_test":  test_errors.astype(np.float32),
        "y_test":  y_test.astype(np.int32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. ANOMALY DETECTION (raw windows) — LSTM-AE baseline / PD2
#    Usage: from utils.prep import detect
#    Different from twin(): reconstructs the SAME window (autoencoder, not seq2seq)
# ─────────────────────────────────────────────────────────────────────────────

def detect(
    window_size: int = 60,
    stride: int = 1,
) -> dict:
    """
    Prepared data for LSTM Autoencoder anomaly detection (PD2 baseline).

    Different from twin() Seq2Seq:
        Input  = (N, window_size, 277)
        Target = same window  ← autoencoder reconstructs its own input
        High reconstruction error at test time = anomaly

    Train:  train1 + train2 + train3  — benign only, windowed per file (no gap crossing)
    Val:    train4                     — benign only
    Test:   test1 + test2             — all rows

    Returns dict with keys:
        X_train      : (N, window_size, 277)  — input == target for autoencoder
        X_val        : (M, window_size, 277)
        X_test       : (K, window_size, 277)
        y_test_labels: (K,) int  — attack labels per window
        norm         : fitted HAISensorNormalizer
    """
    train_dfs = [load_merged("train", i) for i in range(1, 5)]
    test_dfs  = [load_merged("test",  i) for i in range(1, 3)]

    fit_df = pd.concat(train_dfs[:3], ignore_index=True)  # train files are fully benign
    norm = HAISensorNormalizer(method="zscore")
    norm.fit(fit_df)

    cols = _sensor_cols(train_dfs[0])

    def _benign_windows(df: pd.DataFrame) -> np.ndarray:
        # train files are fully benign — scale and window directly
        arr = norm.transform(df)[cols].values.astype(np.float32)
        starts = range(0, len(arr) - window_size + 1, stride)
        wins   = np.empty((len(starts), window_size, arr.shape[1]), dtype=np.float32)
        for i, s in enumerate(starts):
            wins[i] = arr[s : s + window_size]
        return wins

    # Train: window each file separately, concat (no gap crossing)
    X_train = np.concatenate(
        [_benign_windows(df) for df in train_dfs[:3]], axis=0
    )

    # Val: train4 benign
    X_val = _benign_windows(train_dfs[3])

    # Test: all rows + labels
    test_all  = pd.concat(test_dfs, ignore_index=True)
    arr_test  = norm.transform(test_all)[cols].values.astype(np.float32)
    y_raw     = test_all["attack"].values.astype(np.int32)

    starts        = range(0, len(arr_test) - window_size + 1, stride)
    X_test        = np.empty((len(starts), window_size, arr_test.shape[1]), dtype=np.float32)
    y_test_labels = np.empty(len(starts), dtype=np.int32)
    for i, s in enumerate(starts):
        X_test[i]        = arr_test[s : s + window_size]
        y_test_labels[i] = int(y_raw[s : s + window_size].any())

    print("Anomaly Detection (LSTM-AE) data ready:")
    print(f"  X_train {X_train.shape}  (input == target)")
    print(f"  X_val   {X_val.shape}")
    print(f"  X_test  {X_test.shape}   labels {y_test_labels.shape}")

    return {
        "X_train":       X_train,
        "X_val":         X_val,
        "X_test":        X_test,
        "y_test_labels": y_test_labels,
        "norm":          norm,
    }
