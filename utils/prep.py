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

from utils.data_loader import load_merged, identify_common_constants
from utils.normalize   import HAISensorNormalizer

# Columns that are NOT sensor readings — excluded from all arrays
META_COLS = {"timestamp", "attack", "label", "attack_p1", "attack_p2", "attack_p3"}

# Temporal train/val split fraction — last (1-TRAIN_FRAC) of each run is validation.
# Must be temporal (not random) to avoid crossing time gaps inside windows.
TRAIN_FRAC = 0.8


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sensor_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and df[c].dtype != object]


def _temporal_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC):
    """
    Split a single run DataFrame into train / val portions.

    The split is strictly temporal — last (1-train_frac) rows become val.
    Random splitting is wrong here: it can pair encoder and decoder rows from
    different sides of the real time gap that exists between HAI run files.

    Returns (train_df, val_df) — both have reset indices.
    """
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


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
    input_len: int = 300,
    target_len: int = 180,
    stride: int = 1,
) -> dict:
    """
    Layer 1 — Physical Model (Digital Twin): Transformer (main) / LSTM (baseline).

    Predicts normal sensor readings. Trained on benign data only so any attack
    causes a measurable reconstruction error in Layer 2.

    Data split (temporal, per-file — never random):
        For each of train1-4:
            first 80%  → training windows
            last  20%  → validation windows
        test2  → held-out evaluation (never seen during training)

    Windowing is done per file portion (never across file boundaries or
    train/val boundary) to prevent gap-crossing inside a window.

    Normalizer fit on full train1-4 combined (all benign, after constant deletion).
    Window size = 300s (from window_size_analysis notebook).

    Returns
    -------
    X_train, Y_train  : (N, input_len, F) / (N, target_len, F)
    X_val,   Y_val    : (M, input_len, F) / (M, target_len, F)
    X_test,  Y_test   : (K, input_len, F) / (K, target_len, F)
    y_test_labels     : (K,) int   — 0/1 attack labels for test2 evaluation
    norm              : fitted HAISensorNormalizer  (reuse in Layer 2 / 3 / 4)
    """
    # ── Compute common constants once (consistent columns across all splits) ──
    const_hai, const_hiend, hiend_dups = identify_common_constants()

    def _load(split, num):
        return load_merged(split, num, drop_constants=True, keep_hai_duplicates=True,
                           const_cols_hai=const_hai, const_cols_hiend=const_hiend,
                           hiend_dup_cols=hiend_dups)

    # ── Load raw splits ───────────────────────────────────────────────────────
    train_dfs = [_load("train", i) for i in range(1, 5)]   # train1-4 (all benign)
    test2_df  = _load("test", 2)                            # held-out eval only

    # ── Fit normalizer on ALL train splits combined (train1-4, benign only) ──
    fit_df = pd.concat(train_dfs, ignore_index=True)
    norm = HAISensorNormalizer(method="zscore")
    norm.fit(fit_df)

    cols = _sensor_cols(train_dfs[0])
    span = input_len + target_len

    def _to_array(df: pd.DataFrame) -> np.ndarray:
        return norm.transform(df)[cols].values.astype(np.float32)

    # ── Temporal 80/20 split per file → window each portion separately ────────
    # Windowing across the split boundary would mix training and validation rows
    # inside one window, which is a form of data leakage.
    X_train_parts, Y_train_parts = [], []
    X_val_parts,   Y_val_parts   = [], []

    for df in train_dfs:
        train_part, val_part = _temporal_split(df)

        arr_tr = _to_array(train_part)
        if len(arr_tr) >= span:
            Xi, Yi = _seq2seq_windows(arr_tr, input_len, target_len, stride)
            X_train_parts.append(Xi)
            Y_train_parts.append(Yi)

        arr_val = _to_array(val_part)
        if len(arr_val) >= span:
            Xi, Yi = _seq2seq_windows(arr_val, input_len, target_len, stride)
            X_val_parts.append(Xi)
            Y_val_parts.append(Yi)

    X_train = np.concatenate(X_train_parts, axis=0)
    Y_train = np.concatenate(Y_train_parts, axis=0)
    X_val   = np.concatenate(X_val_parts,   axis=0)
    Y_val   = np.concatenate(Y_val_parts,   axis=0)

    # ── Test: test2 only (held-out) ───────────────────────────────────────────
    arr_t2         = _to_array(test2_df)
    y_t2           = test2_df["attack"].values.astype(np.int32)
    X_test, Y_test = _seq2seq_windows(arr_t2, input_len, target_len, stride)
    starts         = list(range(0, len(arr_t2) - span + 1, stride))
    y_test_labels  = np.array([int(y_t2[s : s + span].any()) for s in starts], dtype=np.int32)

    print("Layer 1 (Digital Twin) data ready:")
    print(f"  X_train {X_train.shape}  Y_train {Y_train.shape}  (80% of each train file)")
    print(f"  X_val   {X_val.shape}    Y_val   {Y_val.shape}    (last 20% of each train file)")
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
    window_len: int = 300,   # from window_size_analysis notebook recommendation
    stride:     int = 60,
) -> dict:
    """
    Prepared data for the Attack Generator group (Diffusion / VAE / LSTM-AE).

    Train on:
        attack_windows : test1 attack windows  shape (N, window_len, F)
        normal_windows : train1 normal windows shape (M, window_len, F)

    test1 is used for training (known attacks).
    test2 is the held-out evaluation set — never used here.

    Args
    ----
    norm       : fitted HAISensorNormalizer from twin().
                 If None, fitted on train1-4 combined (after constant deletion).
    window_len : total window length. Default 300s (from window_size_analysis notebook).
    stride     : sliding window stride.

    Returns dict with keys:
        attack_windows : (N, window_len, F) — test1 attack windows
        normal_windows : (M, window_len, F) — train1 normal windows
        norm           : the normalizer used
    """
    const_hai, const_hiend, hiend_dups = identify_common_constants()

    def _load(split, num):
        return load_merged(split, num, drop_constants=True, keep_hai_duplicates=True,
                           const_cols_hai=const_hai, const_cols_hiend=const_hiend,
                           hiend_dup_cols=hiend_dups)

    test1_df = _load("test", 1)   # test1 only — test2 is held-out

    if norm is None:
        train_dfs = [_load("train", i) for i in range(1, 5)]
        fit_df = pd.concat(train_dfs, ignore_index=True)
        norm = HAISensorNormalizer(method="zscore")
        norm.fit(fit_df)
        train1_df = train_dfs[0]
    else:
        train1_df = _load("train", 1)

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
    window_size: int = 300,
    stride: int = 1,
) -> dict:
    """
    Layer 2 (baseline) — LSTM Autoencoder anomaly detector.

    Reconstructs the SAME window (autoencoder, not seq2seq).
    High reconstruction error = anomaly.

    The main Layer 2 detector is Isolation Forest, which receives per-window
    reconstruction errors from a trained twin() model — use detect_errors() for that.
    This function provides the LSTM-AE as a standalone detector baseline.

    Data split (temporal 80/20 per file, matching twin()):
        Train: first 80% of each train1-4 → windows (no gap crossing)
        Val:   last  20% of each train1-4 → windows
        Test:  test1 + test2 (concatenated — detection across both splits)

    Normalizer fit on train1-4 combined (after constant deletion).
    Window size = 300s (from window_size_analysis notebook).

    Returns
    -------
    X_train      : (N, window_size, F)  — input == target for autoencoder
    X_val        : (M, window_size, F)
    X_test       : (K, window_size, F)
    y_test_labels: (K,) int  — 0/1 attack label per window ("any" policy)
    norm         : fitted HAISensorNormalizer
    """
    const_hai, const_hiend, hiend_dups = identify_common_constants()

    def _load(split, num):
        return load_merged(split, num, drop_constants=True, keep_hai_duplicates=True,
                           const_cols_hai=const_hai, const_cols_hiend=const_hiend,
                           hiend_dup_cols=hiend_dups)

    train_dfs = [_load("train", i) for i in range(1, 5)]
    test_dfs  = [_load("test",  i) for i in range(1, 3)]

    # Fit normalizer on ALL 4 training splits combined (all benign data)
    fit_df = pd.concat(train_dfs, ignore_index=True)
    norm = HAISensorNormalizer(method="zscore")
    norm.fit(fit_df)

    cols = _sensor_cols(train_dfs[0])

    def _windows(arr: np.ndarray) -> np.ndarray:
        starts = range(0, len(arr) - window_size + 1, stride)
        wins   = np.empty((len(starts), window_size, arr.shape[1]), dtype=np.float32)
        for i, s in enumerate(starts):
            wins[i] = arr[s : s + window_size]
        return wins

    # Temporal 80/20 split per file — window each portion separately (no gap crossing)
    X_train_parts, X_val_parts = [], []

    for df in train_dfs:
        train_part, val_part = _temporal_split(df)

        arr_tr = norm.transform(train_part)[cols].values.astype(np.float32)
        if len(arr_tr) >= window_size:
            X_train_parts.append(_windows(arr_tr))

        arr_val = norm.transform(val_part)[cols].values.astype(np.float32)
        if len(arr_val) >= window_size:
            X_val_parts.append(_windows(arr_val))

    X_train = np.concatenate(X_train_parts, axis=0)
    X_val   = np.concatenate(X_val_parts,   axis=0)

    # Test: test1 + test2 concatenated (evaluate detection across all known attacks)
    test_all  = pd.concat(test_dfs, ignore_index=True)
    arr_test  = norm.transform(test_all)[cols].values.astype(np.float32)
    y_raw     = test_all["attack"].values.astype(np.int32)

    starts        = range(0, len(arr_test) - window_size + 1, stride)
    X_test        = np.empty((len(starts), window_size, arr_test.shape[1]), dtype=np.float32)
    y_test_labels = np.empty(len(starts), dtype=np.int32)
    for i, s in enumerate(starts):
        X_test[i]        = arr_test[s : s + window_size]
        y_test_labels[i] = int(y_raw[s : s + window_size].any())

    print("Layer 2 (LSTM-AE) data ready:")
    print(f"  X_train {X_train.shape}  (80% of each train file, input == target)")
    print(f"  X_val   {X_val.shape}    (last 20% of each train file)")
    print(f"  X_test  {X_test.shape}   labels {y_test_labels.shape}  (test1+test2)")

    return {
        "X_train":       X_train,
        "X_val":         X_val,
        "X_test":        X_test,
        "y_test_labels": y_test_labels,
        "norm":          norm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. LAYER 4 — Attack Physical Model (same architecture as twin, attack-conditioned)
#    Usage: from utils.prep import attack_twin
#    Call AFTER Layer 3 has generated validated attack sequences.
# ─────────────────────────────────────────────────────────────────────────────

def attack_twin(
    generated_attacks: np.ndarray,
    norm: "HAISensorNormalizer",
    input_len: int = 300,
    target_len: int = 180,
    stride: int = 1,
    val_frac: float = 0.2,
) -> dict:
    """
    Layer 4 — Attack Physical Model.

    Same seq2seq architecture as twin() (Layer 1), but trained on ATTACK sequences
    generated by Layer 3 (Diffusion/VAE) after Guided Generation filtering.

    The model learns to predict sensor behaviour DURING attacks — the physical
    trajectory the system follows once an attack is in progress.

    Args
    ----
    generated_attacks : (N, window_len, F) float32 — output of Layer 3 guided generation.
                        Already normalised (uses the same norm as Layer 1).
    norm              : HAISensorNormalizer fitted in twin() — passed in so that
                        Layer 4 shares the exact same feature space as Layer 1.
    input_len         : encoder input length (seconds). Default 300s.
    target_len        : decoder target length (seconds). Default 180s.
    stride            : sliding window stride over generated_attacks.
    val_frac          : fraction of generated windows held out for validation
                        (temporal split — last val_frac of the sequence).

    Returns
    -------
    X_train, Y_train  : (M, input_len, F) / (M, target_len, F)
    X_val,   Y_val    : (V, input_len, F) / (V, target_len, F)
    norm              : same normalizer (passed through unchanged)
    """
    if generated_attacks.ndim != 3:
        raise ValueError(
            f"generated_attacks must be (N, window_len, F), got shape {generated_attacks.shape}"
        )

    span = input_len + target_len
    N, window_len, F = generated_attacks.shape

    if window_len < span:
        raise ValueError(
            f"window_len={window_len} < input_len+target_len={span}. "
            f"Regenerate with a longer window or reduce input_len/target_len."
        )

    # Slice each generated attack sequence into seq2seq pairs
    all_X, all_Y = [], []
    for seq in generated_attacks:
        starts = range(0, window_len - span + 1, stride)
        for s in starts:
            all_X.append(seq[s           : s + input_len])
            all_Y.append(seq[s + input_len : s + span])

    all_X = np.array(all_X, dtype=np.float32)
    all_Y = np.array(all_Y, dtype=np.float32)

    # Temporal split — last val_frac of the windows for validation
    cut = int(len(all_X) * (1 - val_frac))
    X_train, Y_train = all_X[:cut], all_Y[:cut]
    X_val,   Y_val   = all_X[cut:], all_Y[cut:]

    print("Layer 4 (Attack Physical Model) data ready:")
    print(f"  generated_attacks input : {generated_attacks.shape}")
    print(f"  X_train {X_train.shape}  Y_train {Y_train.shape}")
    print(f"  X_val   {X_val.shape}    Y_val   {Y_val.shape}")

    return {
        "X_train": X_train, "Y_train": Y_train,
        "X_val":   X_val,   "Y_val":   Y_val,
        "norm":    norm,
    }
