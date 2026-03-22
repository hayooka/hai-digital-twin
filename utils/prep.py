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
│   Transformer+lstm (mudhi)       │ twin()                               │
├──────────────────────────────────┼──────────────────────────────────────┤
│ Attack Generator                 │                                      │
│   Diffusion (primary)            │ generate()                           │
│   VAE (baseline)                 │ generate()                           │
│   LSTM-AE generator (alt)        │ generate()                           │
├──────────────────────────────────┼──────────────────────────────────────┤
│ Anomaly Detection                │                                      │
│   ISO Forest (primary)           │ detect_errors(errors)                │
│   Random Forest (future)         │ detect_errors(errors)                │
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

    Plan B split (no data leakage):
        Train:  train1+2+3 (normal) + test1 attack windows (known attacks)
        Val:    train4 (benign)
        Test:   test2 only (held-out — never seen during training)

    Attack type labels (a_train, a_test):
        0 = normal, 1 = FDI, 2 = Replay, 3 = DoS
        Heuristic assignment by thirds of attack rows (real labels not in dataset).

    Returns dict with keys:
        X_train, Y_train  : (N, input_len, 277)  /  (N, target_len, 277)
        a_train           : (N,) int  — attack type per training window
        X_val,   Y_val    : (M, input_len, 277)  /  (M, target_len, 277)
        X_test,  Y_test   : (K, input_len, 277)  /  (K, target_len, 277)
        y_test_labels     : (K,) int  — 0/1 attack labels for evaluation
        a_test            : (K,) int  — attack type per test window
        norm              : fitted HAISensorNormalizer
    """
    # ── Load raw splits ───────────────────────────────────────────────────────
    train_dfs = [load_merged("train", i) for i in range(1, 5)]   # train1-4
    test1_df  = load_merged("test", 1)                            # normal rows used for training
    test2_df  = load_merged("test", 2)                            # held-out eval only

    # ── Fit normalizer on train1+2+3 (benign only) ───────────────────────────
    fit_df = pd.concat(train_dfs[:3], ignore_index=True)
    norm = HAISensorNormalizer(method="zscore")
    norm.fit(fit_df)

    cols = _sensor_cols(train_dfs[0])
    span = input_len + target_len

    def _to_array(df: pd.DataFrame) -> np.ndarray:
        return norm.transform(df)[cols].values.astype(np.float32)

    # ── Train: train1+2+3 (normal only) ──────────────────────────────────────
    # Transformer = physical baseline of NORMAL system behavior.
    # Must NOT see attacks — otherwise it partially accepts them as normal,
    # weakening the physics constraint and anomaly separation.
    X_parts, Y_parts = [], []
    for df in train_dfs[:3]:
        arr = _to_array(df)
        Xi, Yi = _seq2seq_windows(arr, input_len, target_len, stride)
        X_parts.append(Xi)
        Y_parts.append(Yi)

    X_train = np.concatenate(X_parts, axis=0)
    Y_train = np.concatenate(Y_parts, axis=0)

    # ── Val: train4 benign ────────────────────────────────────────────────────
    arr_val      = _to_array(train_dfs[3])
    X_val, Y_val = _seq2seq_windows(arr_val, input_len, target_len, stride)

    # ── Test: test2 only (held-out) ───────────────────────────────────────────
    arr_t2        = _to_array(test2_df)
    y_t2          = test2_df["attack"].values.astype(np.int32)
    X_test, Y_test = _seq2seq_windows(arr_t2, input_len, target_len, stride)
    starts         = list(range(0, len(arr_t2) - span + 1, stride))
    y_test_labels  = np.array([int(y_t2[s : s + span].any()) for s in starts], dtype=np.int32)

    print("Digital Twin data ready:")
    print(f"  X_train {X_train.shape}  Y_train {Y_train.shape}  (normal only)")
    print(f"  X_val   {X_val.shape}    Y_val   {Y_val.shape}")
    print(f"  X_test  {X_test.shape}   Y_test  {Y_test.shape}   (test2 held-out)")

    return {
        "X_train": X_train, "Y_train": Y_train,
        "X_val":   X_val,   "Y_val":   Y_val,
        "X_test":  X_test,  "Y_test":  Y_test,
        "y_test_labels": y_test_labels,
        "norm":    norm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. ATTACK GENERATOR — Diffusion + VAE + LSTM-AE generator + DoppelGANger
#    Usage: from utils.prep import generate
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    norm:       HAISensorNormalizer | None = None,
    window_len: int = 240,   # 60 enc + 180 dec  — matches Transformer split
    stride:     int = 60,
) -> dict:
    """
    Prepared data for the Attack Generator group (Diffusion / VAE / LSTM-AE).

    Train on:
        attack_windows : test1 attack windows  shape (N, window_len, 277)
        normal_windows : train1 normal windows shape (M, window_len, 277)

    test1 is used for training (known attacks).
    test2 is the held-out evaluation set — never used here.

    Args
    ----
    norm       : fitted HAISensorNormalizer from twin().
                 If None, fitted on train1 benign.
    window_len : total window length (enc + dec).  Default 240 = 60+180.
    stride     : sliding window stride.

    Returns dict with keys:
        attack_windows : (N, window_len, 277) — test1 attack windows
        normal_windows : (M, window_len, 277) — train1 normal windows
        norm           : the normalizer used
    """
    train1_df = load_merged("train", 1)
    test1_df  = load_merged("test",  1)   # test1 only — test2 is held-out

    if norm is None:
        norm = HAISensorNormalizer(method="zscore")
        norm.fit(train1_df)

    cols = _sensor_cols(train1_df)

    def _windows(df: pd.DataFrame, mask: np.ndarray | None = None) -> np.ndarray:
        arr = norm.transform(df)[cols].values.astype(np.float32)
        if mask is not None:
            arr = arr[mask]
        starts = range(0, len(arr) - window_len + 1, stride)
        wins = np.empty((len(starts), window_len, arr.shape[1]), dtype=np.float32)
        for i, s in enumerate(starts):
            wins[i] = arr[s : s + window_len]
        return wins

    # Attack windows — test1 rows where attack==1
    attack_mask = test1_df["attack"].values == 1
    attack_windows = _windows(test1_df, mask=attack_mask)

    # Normal windows — train1 (fully benign)
    normal_windows = _windows(train1_df)

    print("Attack Generator data ready:")
    print(f"  attack_windows {attack_windows.shape}  (from test1, attack==1)")
    print(f"  normal_windows {normal_windows.shape}  (from train1, normal)")

    return {
        "attack_windows": attack_windows,
        "normal_windows": normal_windows,
        "norm":           norm,
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
