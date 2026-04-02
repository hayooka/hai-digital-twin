"""
prep.py — Data pipeline for the Generative Digital Twin (Transformer Seq2Seq).

Pipeline order:
    data_loader → normalize → episode-aware windowing → twin_generative()

Usage:
    from utils.prep import twin_generative
    data = twin_generative()
"""

import numpy as np
import pandas as pd

from data_loaderold import load_merged, identify_common_constants
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


def _seq2seq_windows(
    arr: np.ndarray,
    input_len: int = 300,
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
        X[i] = arr[s            : s + input_len]
        Y[i] = arr[s + input_len : s + span]

    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# GENERATIVE DIGITAL TWIN — Transformer Seq2Seq
# ─────────────────────────────────────────────────────────────────────────────

def twin_generative(
    input_len: int = 300,
    target_len: int = 180,
    stride: int = 1,
    episode_train_frac: float = 0.80,
    seed: int = 42,
) -> dict:
    """
    Data pipeline for the Generative Digital Twin (Transformer Seq2Seq).

    Goal: train on BOTH normal and attack sensor readings so the model
    learns to predict realistic physical trajectories for any situation.

    Data split (by complete episode — never split a single attack across train/test):
        52 attack episodes (test1: A101-A114, test2: A201-A238) are shuffled
        with a fixed seed and split 80/20 by episode count:
            first ~42 episodes → training windows
            last  ~10 episodes → test windows (held-out, never seen in training)

        Normal data (train1-4): all rows → training.
        Inter-episode normal rows in test1/test2: → training.

    Windowing is done per contiguous segment — a window never crosses an
    episode boundary.

    Normalizer: fit on train1-4 combined (normal steady-state statistics only).
    Val set:    last 20% of each train1-4 file (normal only, for early stopping).

    Returns
    -------
    X_train, Y_train  : (N, input_len, F) / (N, target_len, F)
    X_val,   Y_val    : (M, input_len, F) / (M, target_len, F)
    X_test,  Y_test   : (K, input_len, F) / (K, target_len, F)
    y_test_labels     : (K,) int  — 0/1 per window ("any" policy)
    norm              : fitted HAISensorNormalizer
    train_ep_ids      : set[str]  — attack_ids used in training
    test_ep_ids       : set[str]  — attack_ids held out for test
    """
    from utils.for_notebook.label_attacks import get_attack_windows, enrich_labels

    # ── Shared setup ──────────────────────────────────────────────────────────
    const_hai, const_hiend, hiend_dups = identify_common_constants()

    def _load(split, num):
        return load_merged(split, num, drop_constants=True, keep_hai_duplicates=True,
                           const_cols_hai=const_hai, const_cols_hiend=const_hiend,
                           hiend_dup_cols=hiend_dups)

    # ── Load all raw files ────────────────────────────────────────────────────
    train_dfs = [_load("train", i) for i in range(1, 5)]   # all normal
    test1_df  = _load("test", 1)                            # 54K rows, 14 episodes
    test2_df  = _load("test", 2)                            # 230K rows, 38 episodes

    # ── Fit normalizer on train1-4 (normal only) ─────────────────────────────
    fit_df = pd.concat(train_dfs, ignore_index=True)
    norm   = HAISensorNormalizer(method="zscore")
    norm.fit(fit_df)
    cols   = _sensor_cols(train_dfs[0])
    span   = input_len + target_len

    def _to_array(df: pd.DataFrame) -> np.ndarray:
        return norm.transform(df)[cols].values.astype(np.float32)

    # ── Episode split (80/20 by count, seed=42) ───────────────────────────────
    all_episodes = get_attack_windows(split_num=None)   # 52 rows
    rng          = np.random.default_rng(seed)
    idx          = rng.permutation(len(all_episodes))
    n_train_ep   = int(len(all_episodes) * episode_train_frac)
    train_ep_ids = set(all_episodes.iloc[idx[:n_train_ep]]["attack_id"])
    test_ep_ids  = set(all_episodes.iloc[idx[n_train_ep:]]["attack_id"])

    # ── Normal data (train1-4): temporal 80/20 for train/val ─────────────────
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

    # ── Segment helper: split a df by contiguous attack_id runs ───────────────
    def _segment_by_episode(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
        """
        Walk df, split whenever attack_id changes.
        Returns list of (attack_id, segment_df); attack_id=="" for normal rows.
        """
        segments = []
        ids = df["attack_id"].values
        cur = ids[0]
        s   = 0
        for i in range(1, len(ids)):
            if ids[i] != cur:
                segments.append((cur, df.iloc[s:i].reset_index(drop=True)))
                cur = ids[i]
                s   = i
        segments.append((cur, df.iloc[s:].reset_index(drop=True)))
        return segments

    # ── Attack files: window per episode, route to train or test ─────────────
    X_test_parts, Y_test_parts, y_test_parts = [], [], []

    for source_df in [test1_df, test2_df]:
        enriched = enrich_labels(source_df)
        segments = _segment_by_episode(enriched)

        for ep_id, seg_df in segments:
            arr = _to_array(seg_df)
            if len(arr) < span:
                continue   # episode too short (e.g. A110=55s < 480s) — skip

            if ep_id == "" or ep_id in train_ep_ids:
                Xi, Yi = _seq2seq_windows(arr, input_len, target_len, stride)
                X_train_parts.append(Xi)
                Y_train_parts.append(Yi)

            elif ep_id in test_ep_ids:
                Xi, Yi = _seq2seq_windows(arr, input_len, target_len, stride)
                X_test_parts.append(Xi)
                Y_test_parts.append(Yi)
                y_raw  = seg_df["attack"].values.astype(np.int32)
                starts = list(range(0, len(arr) - span + 1, stride))
                y_wins = np.array(
                    [int(y_raw[s : s + span].any()) for s in starts],
                    dtype=np.int32,
                )
                y_test_parts.append(y_wins)

    # ── Concatenate ───────────────────────────────────────────────────────────
    X_train = np.concatenate(X_train_parts, axis=0)
    Y_train = np.concatenate(Y_train_parts, axis=0)
    X_val   = np.concatenate(X_val_parts,   axis=0)
    Y_val   = np.concatenate(Y_val_parts,   axis=0)
    X_test  = np.concatenate(X_test_parts,  axis=0)
    Y_test  = np.concatenate(Y_test_parts,  axis=0)
    y_test_labels = np.concatenate(y_test_parts, axis=0)

    print("Generative Digital Twin data ready:")
    print(f"  Episode split (seed={seed}): {len(train_ep_ids)} train / {len(test_ep_ids)} test")
    print(f"  Test episodes: {sorted(test_ep_ids)}")
    print(f"  X_train {X_train.shape}  Y_train {Y_train.shape}")
    print(f"  X_val   {X_val.shape}    Y_val   {Y_val.shape}")
    print(f"  X_test  {X_test.shape}   Y_test  {Y_test.shape}")
    print(f"  Attack windows in test: {y_test_labels.sum()} / {len(y_test_labels)}")

    return {
        "X_train":       X_train,     "Y_train":       Y_train,
        "X_val":         X_val,       "Y_val":         Y_val,
        "X_test":        X_test,      "Y_test":        Y_test,
        "y_test_labels": y_test_labels,
        "norm":          norm,
        "train_ep_ids":  train_ep_ids,
        "test_ep_ids":   test_ep_ids,
    }
