from __future__ import annotations
import os
import functools
import pandas as pd
import numpy as np

HAI_DIR        = "C:\\Users\\ahmma\\Desktop\\farah\\hai-23.05"
HIEND_DIR      = "C:\\Users\\ahmma\\Desktop\\farah\\haiend-23.05"
LABELED_DIR    = HAI_DIR  # labeled files live alongside the raw files
TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test_data.csv")


# HAI column names that have a known counterpart in HAIEnd.
# Used in the fallback path of load_merged (keep_hai_duplicates=False).
HAI_DUPLICATES: list[str] = [
    "x1001_15_ASSIGN_OUT", "P1_FCV01D", "P1_FCV01Z", "P1_FCV02D", "P1_FCV02Z",
    "P1_FCV03D", "P1_FCV03Z", "P1_FT01", "P1_FT01Z", "P1_FT02", "P1_FT02Z",
    "P1_FT03", "P1_FT03Z", "P1_LCV01D", "P1_LCV01Z", "P1_LIT01",
    "P1_PCV01D", "P1_PCV01Z", "P1_PCV02D", "P1_PCV02Z", "P1_PIT01_HH",
    "P1_PIT01", "P1_PIT02", "P1_PP01AD", "P1_PP01AR", "P1_PP01BD", "P1_PP01BR",
    "P1_PP02D", "P1_PP02R", "P1_SOL01D", "P1_SOL03D", "P1_STSP",
    "P1_TIT01", "P1_TIT03", "P4_ST_GOV",
]

# (HAI col, HAIEnd col) pairs to test for correlation.
# Pairs with corr > 0.99 in ALL 4 train files → drop the HAIEnd copy.
# Shared by identify_common_constants() and load_merged() — defined once here.
CANDIDATE_PAIRS: list[tuple[str, str]] = [
    ("x1001_15_ASSIGN_OUT", "1001.15-OUT"),
    ("P1_FCV01D",   "DM-FCV01-D"),  ("P1_FCV01Z",   "DM-FCV01-Z"),
    ("P1_FCV02D",   "DM-FCV02-D"),  ("P1_FCV02Z",   "DM-FCV02-Z"),
    ("P1_FCV03D",   "DM-FCV03-D"),  ("P1_FCV03Z",   "DM-FCV03-Z"),
    ("P1_FT01",     "DM-FT01"),     ("P1_FT01Z",    "DM-FT01Z"),
    ("P1_FT02",     "DM-FT02"),     ("P1_FT02Z",    "DM-FT02Z"),
    ("P1_FT03",     "DM-FT03"),     ("P1_FT03Z",    "DM-FT03Z"),
    ("P1_LCV01D",   "DM-LCV01-D"), ("P1_LCV01Z",   "DM-LCV01-Z"),
    ("P1_LIT01",    "DM-LIT01"),
    ("P1_PCV01D",   "DM-PCV01-D"), ("P1_PCV01Z",   "DM-PCV01-Z"),
    ("P1_PCV02D",   "DM-PCV02-D"), ("P1_PCV02Z",   "DM-PCV02-Z"),
    ("P1_PIT01_HH", "DM-PIT01-HH"),
    ("P1_PIT01",    "DM-PIT01"),    ("P1_PIT02",    "DM-PIT02"),
    ("P1_PP01AD",   "DM-PP01A-D"), ("P1_PP01AR",   "DM-PP01A-R"),
    ("P1_PP01BD",   "DM-PP01B-D"), ("P1_PP01BR",   "DM-PP01B-R"),
    ("P1_PP02D",    "DM-PP02-D"),  ("P1_PP02R",    "DM-PP02-R"),
    ("P1_SOL01D",   "DM-SOL01-D"), ("P1_SOL03D",   "DM-SOL03-D"),
    ("P1_STSP",     "DM-ST-SP"),
    ("P1_TIT01",    "DM-TIT01"),
    ("P1_TIT03",    "DM-TIT02"),   # HAI TIT03 = HAIEnd TIT02
    ("P4_ST_GOV",   "GATEOPEN"),
]


def get_constant_columns(df, meta_cols={'timestamp', 'attack', 'label', 'attack_p1', 'attack_p2', 'attack_p3'},
                         std_threshold=1e-6) -> list[str]:
    """
    Identify constant sensor columns (std < threshold).
    
    Based on data analysis (verified from actual data):
    - HAI: 18 universal constant columns (std < 1e-6) across all train files
    - HAIEnd: 151 universal constant columns across all train files
    
    These columns add noise without information and should be dropped from training.
    
    :param df: DataFrame to analyze
    :param meta_cols: Metadata columns to exclude from analysis
    :param std_threshold: Standard deviation threshold for constant detection
    :return: List of constant column names
    """
    sensor_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                   if c not in meta_cols]
    
    arr = df[sensor_cols].values.astype(float)
    stds = np.nanstd(arr, axis=0)
    
    const_cols = [col for col, s in zip(sensor_cols, stds) if s < std_threshold]
    return const_cols


def drop_constant_columns(df, meta_cols={'timestamp', 'attack', 'label', 'attack_p1', 'attack_p2', 'attack_p3'},
                          std_threshold=1e-6, verbose=True) -> pd.DataFrame:
    """
    Drop constant columns from DataFrame. Recommended for training.
    
    From data analysis:
    - Constant columns (std < 1e-6) contain no useful signal
    - Reduces feature space and emphasizes real sensor variation
    - For test evaluation, keep these to detect reconstruction anomalies during attacks
    
    :param df: DataFrame to filter
    :param meta_cols: Metadata columns to protect (timestamp, attack, etc.)
    :param std_threshold: Threshold for constant detection
    :param verbose: Print statistics about dropped columns
    :return: DataFrame with constant columns removed
    """
    const_cols = get_constant_columns(df, meta_cols, std_threshold)
    
    if verbose and const_cols:
        print(f"  Dropping {len(const_cols)} constant columns (std < {std_threshold})")
    
    df_filtered = df.drop(columns=const_cols)
    return df_filtered


def handle_missing_values(df, method='forward_fill') -> pd.DataFrame:
    """
    Handle missing values in sensor data.
    
    Data analysis shows currently NO missing values in HAI+HAIEnd merged data,
    but this handles edge cases and future data variations.
    
    :param df: DataFrame to clean
    :param method: 'forward_fill' (ffill), 'backward_fill' (bfill), or 'drop'
    :return: DataFrame with missing values handled
    """
    n_missing = df.isnull().sum().sum()
    
    if n_missing == 0:
        return df
    
    df_clean = df.copy()
    
    if method == 'forward_fill':
        df_clean = df_clean.fillna(method='ffill')
        # Backward fill remaining NaNs at the start
        df_clean = df_clean.fillna(method='bfill')
    elif method == 'backward_fill':
        df_clean = df_clean.fillna(method='bfill')
        df_clean = df_clean.fillna(method='ffill')
    elif method == 'drop':
        df_clean = df_clean.dropna()
    
    n_remaining = df_clean.isnull().sum().sum()
    if n_remaining == 0:
        print(f"  Handled {n_missing} missing values using '{method}' (now clean)")
    else:
        print(f"  Handled {n_missing} missing values; {n_remaining} remain")
    
    return df_clean


def identify_universal_constants(split: str, std_threshold=1e-6) -> tuple[list[str], list[str]]:
    """
    Identify columns that are CONSTANT across ALL train or ALL test files.
    
    Instead of dropping different constants per file, this finds the intersection
    of constant columns across all files in the split, ensuring consistent columns.
    
    :param split: 'train' or 'test'
    :param std_threshold: Standard deviation threshold
    :return: (hai_universal_constants, hiend_universal_constants)
    """
    meta_cols = {'timestamp', 'attack', 'label', 'attack_p1', 'attack_p2', 'attack_p3'}
    
    # Determine file range
    file_range = range(1, 5) if split == 'train' else range(1, 3)
    
    # Track constants across all files
    hai_const_sets = []
    hiend_const_sets = []
    
    for num in file_range:
        # Load HAI
        hai_path = os.path.join(HAI_DIR, f"hai-{split}{num}.csv")
        hai = pd.read_csv(hai_path)
        hai.rename(columns={hai.columns[0]: "timestamp"}, inplace=True)
        hai["timestamp"] = pd.to_datetime(hai["timestamp"])
        
        # Load HAIEnd
        hiend_path = os.path.join(HIEND_DIR, f"end-{split}{num}.csv")
        hiend = pd.read_csv(hiend_path)
        hiend.rename(columns={hiend.columns[0]: "timestamp"}, inplace=True)
        hiend["timestamp"] = pd.to_datetime(hiend["timestamp"])
        
        # Find constants in this file
        hai_const = get_constant_columns(hai, meta_cols, std_threshold)
        hiend_const = get_constant_columns(hiend, meta_cols, std_threshold)
        
        hai_const_sets.append(set(hai_const))
        hiend_const_sets.append(set(hiend_const))
    
    # Find INTERSECTION (columns constant in ALL files)
    hai_universal = list(set.intersection(*hai_const_sets)) if hai_const_sets else []
    hiend_universal = list(set.intersection(*hiend_const_sets)) if hiend_const_sets else []
    
    return sorted(hai_universal), sorted(hiend_universal)


@functools.lru_cache(maxsize=1)
def identify_common_constants(std_threshold: float = 1e-6) -> tuple[list[str], list[str], list[str]]:
    """
    Compute — once per process — the three drop-lists needed for a consistent
    feature space across every train and test file.

    Returns
    -------
    hai_common        : HAI cols constant in BOTH train & test          → DROP (18)
    hiend_common      : HAIEnd cols constant in BOTH train & test       → DROP (151)
    hiend_consistent_dups : HAIEnd cols with corr > 0.99 vs HAI partner
                            in ALL 4 train files                        → DROP (21)

    Rule:
        constant in BOTH splits  → no signal anywhere      → DROP
        constant in train ONLY   → varies in test attacks  → KEEP
        high-corr dup in all 4 trains → truly redundant    → DROP HAIEnd copy

    Verified: 68 HAI + 53 HAIEnd + 2 metadata = 123 cols → N_FEAT = 121.

    Cached with lru_cache — calling from twin(), generate(), detect() is free
    after the first call; files are only read once per process.
    """
    meta_cols = {"timestamp", "attack", "label", "attack_p1", "attack_p2", "attack_p3"}

    hai_train_sets:   list[set] = []
    hiend_train_sets: list[set] = []
    pair_high_count = {(h, he): 0 for h, he in CANDIDATE_PAIRS}

    # ── Single pass over train files: compute constants + correlations ─────────
    for num in range(1, 5):
        h_df  = pd.read_csv(os.path.join(HAI_DIR,   f"hai-train{num}.csv"))
        he_df = pd.read_csv(os.path.join(HIEND_DIR, f"end-train{num}.csv"))
        h_df.rename(columns={h_df.columns[0]:   "timestamp"}, inplace=True)
        he_df.rename(columns={he_df.columns[0]: "timestamp"}, inplace=True)
        h_df["timestamp"]  = pd.to_datetime(h_df["timestamp"])
        he_df["timestamp"] = pd.to_datetime(he_df["timestamp"])

        hai_train_sets.append(set(get_constant_columns(h_df,  meta_cols, std_threshold)))
        hiend_train_sets.append(set(get_constant_columns(he_df, meta_cols, std_threshold)))

        for hai_col, hiend_col in CANDIDATE_PAIRS:
            if hai_col not in h_df.columns or hiend_col not in he_df.columns:
                continue
            tmp = pd.merge(
                h_df[["timestamp", hai_col]],
                he_df[["timestamp", hiend_col]],
                on="timestamp", how="inner",
            )
            if len(tmp) < 2:
                continue
            hai_v   = tmp[hai_col].values[1:].astype(float)
            hiend_v = tmp[hiend_col].values[:-1].astype(float)
            if hai_v.std() < 1e-9 or hiend_v.std() < 1e-9:
                continue
            if float(np.corrcoef(hai_v, hiend_v)[0, 1]) > 0.99:
                pair_high_count[(hai_col, hiend_col)] += 1

    # ── Single pass over test files: constants only ────────────────────────────
    hai_test_sets:   list[set] = []
    hiend_test_sets: list[set] = []
    for num in range(1, 3):
        h_df  = pd.read_csv(os.path.join(HAI_DIR,   f"hai-test{num}.csv"))
        he_df = pd.read_csv(os.path.join(HIEND_DIR, f"end-test{num}.csv"))
        h_df.rename(columns={h_df.columns[0]:   "timestamp"}, inplace=True)
        he_df.rename(columns={he_df.columns[0]: "timestamp"}, inplace=True)
        hai_test_sets.append(set(get_constant_columns(h_df,  meta_cols, std_threshold)))
        hiend_test_sets.append(set(get_constant_columns(he_df, meta_cols, std_threshold)))

    # ── Compute drop lists ─────────────────────────────────────────────────────
    hai_train_univ   = set.intersection(*hai_train_sets)
    hiend_train_univ = set.intersection(*hiend_train_sets)
    hai_test_univ    = set.intersection(*hai_test_sets)
    hiend_test_univ  = set.intersection(*hiend_test_sets)

    hai_common   = sorted(hai_train_univ   & hai_test_univ)
    hiend_common = sorted(hiend_train_univ & hiend_test_univ)
    hiend_consistent_dups = sorted(
        hiend_col for (_, hiend_col), cnt in pair_high_count.items() if cnt == 4
    )

    print(f"  identify_common_constants → "
          f"HAI drop {len(hai_common)}, HAIEnd drop {len(hiend_common)}, "
          f"HIEND dups drop {len(hiend_consistent_dups)}")

    return hai_common, hiend_common, hiend_consistent_dups


def load_merged(split: str, num: int, drop_constants: bool = True,
                keep_hai_duplicates: bool = True,
                const_cols_hai: list | None = None,
                const_cols_hiend: list | None = None,
                hiend_dup_cols: list | None = None) -> "pd.DataFrame":
    """
    Load and merge HAI + HIEND for a given split ('train' or 'test') and number.

    **Processing pipeline (verified from actual data):**
    
    1. Drop UNIVERSAL constant columns (std < 1e-6) from BOTH datasets
       - HAI universal constants: 18 columns (constant across ALL train files)
       - HAIEnd universal constants: 151 columns (constant across ALL train files)
       - Ensures consistent column structure across train1-4
    
    2. **Dynamic correlation-based duplicate handling**:
       - Identify potential duplicate pairs between HAI and HAIEnd
       - Calculate Pearson correlation for each pair (accounting for 1s offset)
       - DROP HIEND duplicate if correlation > 0.99 (true duplicates)
       - KEEP BOTH if correlation ≤ 0.99 (borderline or different signals)
       - Verified: 21 high-correlation duplicates across train files
    
    3. Merge on exact timestamp (inner join — unmatched boundary rows dropped)
    
    **Final column count per train file (verified from actual data):**
       - HAI sensors: 68 columns (86 - 18 common constants)
       - HAIEnd sensors: 53 columns (225 - 151 common constants - 21 consistent duplicates)
       - Metadata: 2 columns (timestamp, attack)
       - TOTAL: 123 columns  →  N_FEAT = 121 sensor features
    
    :param split: 'train' or 'test'
    :param num: Split number (1-4 for train, 1-2 for test)
    :param drop_constants: If True, remove UNIVERSAL constant columns
    :param keep_hai_duplicates: If True, prefer HAI names over HIEND (with correlation check)
    :param const_cols_hai: List of HAI columns to drop (computed once per split)
    :param const_cols_hiend: List of HAIEnd columns to drop (computed once per split)
    :return: Merged DataFrame with consistent columns across split
    """
    if split == "test":
        hai_path   = os.path.join(HAI_DIR,   f"hai-{split}{num}-labeled.csv")
        hiend_path = os.path.join(HIEND_DIR, f"end-{split}{num}-labeled.csv")
    else:
        hai_path   = os.path.join(HAI_DIR,   f"hai-{split}{num}.csv")
        hiend_path = os.path.join(HIEND_DIR, f"end-{split}{num}.csv")

    hai   = pd.read_csv(hai_path)
    hiend = pd.read_csv(hiend_path)

    hai.rename(columns={hai.columns[0]: "timestamp"}, inplace=True)
    hiend.rename(columns={hiend.columns[0]: "timestamp"}, inplace=True)

    hai["timestamp"]   = pd.to_datetime(hai["timestamp"])
    hiend["timestamp"] = pd.to_datetime(hiend["timestamp"])
    
    # Handle any missing values before processing
    hai = handle_missing_values(hai, method='forward_fill')
    hiend = handle_missing_values(hiend, method='forward_fill')
    
    # ---- STEP 1: Drop UNIVERSAL constant columns ----
    if drop_constants:
        # Use pre-computed constant columns (same across all files in this split)
        cols_to_drop_hai = const_cols_hai if const_cols_hai is not None else []
        cols_to_drop_hiend = const_cols_hiend if const_cols_hiend is not None else []
        
        # Filter to only columns that exist in this specific file
        cols_to_drop_hai = [c for c in cols_to_drop_hai if c in hai.columns]
        cols_to_drop_hiend = [c for c in cols_to_drop_hiend if c in hiend.columns]
        
        if cols_to_drop_hai:
            hai = hai.drop(columns=cols_to_drop_hai)
        if cols_to_drop_hiend:
            hiend = hiend.drop(columns=cols_to_drop_hiend)
        
        # Only print on first file to avoid spam
        if num == 1:
            print(f"  [UNIVERSAL] Dropping {len(cols_to_drop_hai)} HAI + {len(cols_to_drop_hiend)} HAIEnd constants (same across all {split} files)")
    
    
    # ---- STEP 2: Handle duplicates ----
    # When hiend_dup_cols is provided (pre-computed by identify_common_constants),
    # use the fixed list — guarantees identical N_FEAT across all files.
    # Otherwise fall back to per-call dynamic correlation (slower, may vary per file).
    if keep_hai_duplicates and hiend_dup_cols is not None:
        cols_to_drop = [c for c in hiend_dup_cols if c in hiend.columns]
        if cols_to_drop:
            hiend = hiend.drop(columns=cols_to_drop)
        if num == 1:
            print(f"  Duplicate handling: removed {len(cols_to_drop)} HIEND cols (fixed consistent list)")
    elif keep_hai_duplicates:
        # Fallback: compute correlation per-call using module-level CANDIDATE_PAIRS.
        # Prefer passing hiend_dup_cols (from identify_common_constants) to avoid
        # this path — it re-reads the data and may give inconsistent counts per file.
        hiend_cols_to_drop = []
        dropped_high_corr = 0
        kept_low_corr = 0

        for hai_col, hiend_col in CANDIDATE_PAIRS:
            # Skip if either column doesn't exist
            if hai_col not in hai.columns or hiend_col not in hiend.columns:
                continue
            
            # Merge to compute correlation (HAIEnd is offset by 1s)
            temp_merge = pd.merge(
                hai[['timestamp', hai_col]],
                hiend[['timestamp', hiend_col]],
                on='timestamp',
                how='inner'
            )
            
            if len(temp_merge) == 0:
                continue
            
            # Get correlation (account for 1s offset: HAI[t] vs HAIEnd[t-1])
            hai_vals = temp_merge[hai_col].values[1:].astype(float)
            hiend_vals = temp_merge[hiend_col].values[:-1].astype(float)
            
            # Handle constant case
            if hai_vals.std() < 1e-9 and hiend_vals.std() < 1e-9:
                # Both constant: will be removed by constant filter anyway
                continue
            elif hai_vals.std() < 1e-9 or hiend_vals.std() < 1e-9:
                # One constant: skip (keep both)
                kept_low_corr += 1
                continue
            
            # Calculate correlation
            corr = float(np.corrcoef(hai_vals, hiend_vals)[0, 1])
            
            # Decision: drop HIEnd if correlation > 0.99, otherwise keep both
            if corr > 0.99:
                hiend_cols_to_drop.append(hiend_col)
                dropped_high_corr += 1
            else:
                kept_low_corr += 1
        
        if hiend_cols_to_drop:
            hiend = hiend.drop(columns=hiend_cols_to_drop)
        
        print(f"  Duplicate handling: removed {dropped_high_corr} HIEND cols (corr > 0.99), "
              f"kept {kept_low_corr} pairs (corr ≤ 0.99)")
    else:
        # Old behavior: drop HAI duplicates, keep HIEND
        cols_to_drop = [c for c in HAI_DUPLICATES if c in hai.columns]
        hai = hai.drop(columns=cols_to_drop)
        print(f"  Dropped {len(cols_to_drop)} duplicate HAI columns (kept HAIEnd names)")

    # ---- STEP 3: Merge on exact timestamp ----
    merged = pd.merge(
        hai,
        hiend,
        on="timestamp",
        how="inner",
        suffixes=("", "_hiend"),
    )
    
    # Final missing value check post-merge
    merged = handle_missing_values(merged, method='forward_fill')

    # Drop any stray label/attack columns from raw files; label added later
    _meta_to_drop = ["attack", "attack_p1", "attack_p2", "attack_p3",
                     "label", "attack_type", "combination"]
    merged = merged.drop(columns=[c for c in _meta_to_drop if c in merged.columns])
    merged["label"] = 0

    print(
        f"[{split}{num}] Merged: {len(merged)} rows, {len(merged.columns)} cols "
    )
    return merged

    # --- HAI-only mode (commented out) ---
    # hai_path = os.path.join(HAI_DIR, f"hai-{split}{num}.csv")
    # df = pd.read_csv(hai_path)
    # df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    # df["timestamp"] = pd.to_datetime(df["timestamp"])
    # if split == "test" and "attack" not in df.columns:
    #     label_path = os.path.join(HAI_DIR, f"label-{split}{num}.csv")
    #     if os.path.exists(label_path):
    #         labels = pd.read_csv(label_path)
    #         labels.rename(columns={labels.columns[0]: "timestamp"}, inplace=True)
    #         labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    #         labels = labels.drop_duplicates(subset="timestamp", keep="first")
    #         label_col = next((c for c in labels.columns if c != "timestamp"), None)
    #         if label_col:
    #             df = pd.merge(df, labels[["timestamp", label_col]],
    #                           on="timestamp", how="left")
    #             df.rename(columns={label_col: "attack"}, inplace=True)
    #             df["attack"] = df["attack"].fillna(0).astype(int)
    # if "attack" not in df.columns:
    #     df["attack"] = 0
    # return df  # 86 sensor features


def load_all_train() -> dict[str, pd.DataFrame]:
    """
    Return a dict of all merged train splits keyed by 'train1'..'train4'.
    
    🎯 Strategy: Drop ONLY common constants (in both train & test), KEEP train-only constants
    
    Each train split has (verified final structure):
    - COMMON constant columns DROPPED (18 HAI + 151 HAIEnd - no variance anywhere)
    - TRAIN-ONLY constant columns KEPT (2 HAI + ? HAIEnd - may be early anomaly signals)
    - HAI + HAIEnd merged on timestamp with dynamic correlation filtering
    - High-correlation duplicates removed (21 consistent pairs, corr > 0.99)
    - Kept as SEPARATE files (not concatenated)
    - All train files have CONSISTENT columns:
      * 68-70 HAI sensor columns (86 - 18 common = 68, PLUS 2 train-only = 70)
      * ~48-50 HAIEnd sensor columns (225 - 151 common, PLUS train-only, MINUS 21 duplicates)
      * 2 metadata columns (timestamp, attack)
    """
    print("\n  [Step A: Identifying COMMON constants (in both train & test)]")
    const_hai, const_hiend, hiend_dups = identify_common_constants()
    print(f"  ✓ Will DROP these constants: HAI {len(const_hai)}, HAIEnd {len(const_hiend)}")
    print(f"  ✓ KEEPING train-only constants (may be early anomaly signals)")

    return {f"train{i}": load_merged("train", i, drop_constants=True, keep_hai_duplicates=True,
                                     const_cols_hai=const_hai, const_cols_hiend=const_hiend,
                                     hiend_dup_cols=hiend_dups)
            for i in range(1, 5)}


def load_all_test() -> dict[str, pd.DataFrame]:
    """
    Return a dict of all merged test splits keyed by 'test1'..'test2'.
    
    Each test split has:
    - UNIVERSAL constant columns removed (constant across ALL test files — may differ from train)
    - HAI + HAIEnd merged on timestamp
    - Attack labels loaded and included
    - High-correlation duplicates removed (corr > 0.99)
    - Kept as SEPARATE files (not concatenated)
    - All test files have CONSISTENT column counts (may differ from train counts)
    
    Note: Test files may have different dimensions than train due to different constant distributions
    """
    # Use the same drop lists as train → identical columns across all 6 files
    const_hai, const_hiend, hiend_dups = identify_common_constants()

    splits = {}
    for i in range(1, 3):
        df = load_merged("test", i, drop_constants=True, keep_hai_duplicates=True,
                         const_cols_hai=const_hai, const_cols_hiend=const_hiend,
                         hiend_dup_cols=hiend_dups)
        df = add_condition(df, split_num=i)
        splits[f"test{i}"] = df
    return splits


def assign_label(attack_type, combination) -> int:
    """
    Map attack_type + combination columns to a numeric label.

    0 = normal
    1 = AP simple   (AP, no combination)
    2 = AP combined (AP, with combination)
    3 = AE simple   (AE, no combination)
    4 = AE combined (AE, with combination)
    """
    if pd.isna(attack_type) or attack_type == "normal":
        return 0
    if attack_type == "AP":
        return 1 if str(combination).strip().lower() == "no" else 2
    if attack_type == "AE":
        return 3 if str(combination).strip().lower() == "no" else 4
    return 0


ATTACK_META_COLS = ["attack_id", "scenario", "attack_type", "combination",
                    "duration_sec", "target_controller", "target_points"]


def add_condition(df: pd.DataFrame, split_num: int,
                  test_data_path: str = TEST_DATA_PATH) -> pd.DataFrame:
    """
    Add 'condition' + all test_data.csv metadata columns to a merged test DataFrame.

    condition encoding
    ------------------
    0 = normal
    1 = AP simple    (attack_type=AP,  combination=no)
    2 = AP combined  (attack_type=AP,  combination=yes)
    3 = AE simple    (attack_type=AE,  combination=no)
    4 = AE combined  (attack_type=AE,  combination=yes)

    Added columns (NaN outside attack windows)
    -------------------------------------------
    attack_id, scenario, attack_type, combination,
    duration_sec, target_controller, target_points

    :param df:             Merged test DataFrame (must have 'timestamp' column).
    :param split_num:      1 or 2 — matches the 'split' column in test_data.csv.
    :param test_data_path: Path to test_data.csv.
    :return:               df with condition + metadata columns added.
    """
    attacks = pd.read_csv(test_data_path)
    attacks["start"] = pd.to_datetime(attacks["start"])
    attacks["end"]   = pd.to_datetime(attacks["end"])
    attacks = attacks[attacks["split"] == split_num].reset_index(drop=True)

    df = df.copy()

    for _, row in attacks.iterrows():
        mask = (df["timestamp"] >= row["start"]) & (df["timestamp"] <= row["end"])
        df.loc[mask, "label"] = assign_label(row["attack_type"], row["combination"])

    n_attack = (df["label"] > 0).sum()
    print(f"  [label] split={split_num}: {n_attack} attack rows labelled "
          f"({attacks.shape[0]} windows from test_data.csv)")
    return df


def build_labeled_datasets(output_dir: str = "data/processed") -> None:
    """
    Build and save 12 labeled CSV files ready for model training/evaluation.

    Sources
    -------
    Test  → LABELED_DIR  (hai-test{1,2}-labeled.csv / end-test{1,2}-labeled.csv)
    Train → HAI_DIR / HIEND_DIR  (hai-train{1-4}.csv / end-train{1-4}.csv)

    Outputs (saved to output_dir)
    ------------------------------
    test_combined_hai.csv       test_combined_haiend.csv
    test2_combined_hai.csv      test2_combined_haiend.csv
    train{1-4}_combined_hai.csv train{1-4}_combined_haiend.csv

    Label column
    ------------
    0 = normal | 1 = AP simple | 2 = AP combined | 3 = AE simple | 4 = AE combined
    Train files are 100% normal → label = 0 throughout.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _load_labeled(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["label"] = df.apply(
            lambda r: assign_label(r.get("attack_type"), r.get("combination")), axis=1
        )
        return df

    def _load_train(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["label"] = 0
        return df

    def _save(df: pd.DataFrame, name: str) -> None:
        out_path = os.path.join(output_dir, name)
        df.to_csv(out_path, index=False)
        label_counts = df["label"].value_counts().sort_index().to_dict()
        print(f"  Saved {name:45s} | {len(df):>7,} rows  labels={label_counts}")

    print("\n[build_labeled_datasets] Starting ...")
    print(f"  Output dir : {output_dir}\n")

    # -- TEST DATA -------------------------------------------------------------
    print("Test splits (from LABELED_DIR)")
    for split_num, out_prefix in [(1, "test"), (2, "test2")]:
        hai_path  = os.path.join(LABELED_DIR, f"hai-test{split_num}-labeled.csv")
        end_path  = os.path.join(LABELED_DIR, f"end-test{split_num}-labeled.csv")

        _save(_load_labeled(hai_path), f"{out_prefix}_combined_hai.csv")
        _save(_load_labeled(end_path), f"{out_prefix}_combined_haiend.csv")

    # -- TRAIN DATA ------------------------------------------------------------
    print("\nTrain splits (from HAI_DIR / HIEND_DIR)")
    for n in range(1, 5):
        hai_path  = os.path.join(HAI_DIR,   f"hai-train{n}.csv")
        end_path  = os.path.join(HIEND_DIR,  f"end-train{n}.csv")

        _save(_load_train(hai_path),  f"train{n}_combined_hai.csv")
        _save(_load_train(end_path),  f"train{n}_combined_haiend.csv")

    print("\n[build_labeled_datasets] Done. 12 files written.")


def save_processed(output_dir: str = "data/processed") -> None:
    """
    Merge HAI + HAIEnd for each split on timestamp, add label, save.

    train1-4 : hai-trainN.csv + end-trainN.csv        → label = 0
    test1-2  : hai-testN-labeled.csv + end-testN-labeled.csv → label from test_data.csv

    No columns are dropped — all sensor columns kept as-is.
    """
    os.makedirs(output_dir, exist_ok=True)

    _meta_drop = ["attack", "attack_p1", "attack_p2", "attack_p3"]

    def _merge(hai_path, hiend_path):
        hai   = pd.read_csv(hai_path,   low_memory=False)
        hiend = pd.read_csv(hiend_path, low_memory=False)
        hai.rename(columns={hai.columns[0]: "timestamp"},   inplace=True)
        hiend.rename(columns={hiend.columns[0]: "timestamp"}, inplace=True)
        hai["timestamp"]   = pd.to_datetime(hai["timestamp"])
        hiend["timestamp"] = pd.to_datetime(hiend["timestamp"])
        merged = pd.merge(hai, hiend, on="timestamp", how="inner", suffixes=("", "_hiend"))
        merged = merged.drop(columns=[c for c in _meta_drop if c in merged.columns])
        return merged

    print("\n[save_processed] Building train files …")
    for n in range(1, 5):
        df = _merge(
            os.path.join(HAI_DIR,   f"hai-train{n}.csv"),
            os.path.join(HIEND_DIR, f"end-train{n}.csv"),
        )
        df["label"] = 0
        path = os.path.join(output_dir, f"train{n}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved {path}  ({len(df):,} rows, {len(df.columns)} cols)")

    print("\n[save_processed] Building test files …")
    attacks = pd.read_csv(TEST_DATA_PATH)
    attacks["start"] = pd.to_datetime(attacks["start"])
    attacks["end"]   = pd.to_datetime(attacks["end"])

    for n in range(1, 3):
        df = _merge(
            os.path.join(HAI_DIR,   f"hai-test{n}-labeled.csv"),
            os.path.join(HIEND_DIR, f"end-test{n}-labeled.csv"),
        )
        _labeled_meta = ["attack_id", "scenario", "attack_type", "combination",
                         "target_controller", "target_points", "duration_sec"]
        dup_cols = [f"{c}_hiend" for c in _labeled_meta if f"{c}_hiend" in df.columns]
        if dup_cols:
            df = df.drop(columns=dup_cols)
        df["label"] = 0
        for _, row in attacks[attacks["split"] == n].iterrows():
            mask = (df["timestamp"] >= row["start"]) & (df["timestamp"] <= row["end"])
            df.loc[mask, "label"] = assign_label(row["attack_type"], row["combination"])
        path = os.path.join(output_dir, f"test{n}.csv")
        df.to_csv(path, index=False)
        label_counts = df["label"].value_counts().sort_index().to_dict()
        print(f"  Saved {path}  ({len(df):,} rows, {len(df.columns)} cols)  label={label_counts}")

    print("\n[save_processed] Done — 6 files written to", output_dir)


if __name__ == "__main__":
    save_processed()

    # ============================================================================
    # (legacy analysis kept below for reference — not executed)
    # ============================================================================
    import sys; sys.exit(0)

    meta_cols = {'timestamp', 'attack', 'label', 'attack_p1', 'attack_p2', 'attack_p3'}

    # ============================================================================
    # STEP 1: Total sensors in HAI and HAIEnd
    # ============================================================================
    print("\n[STEP 1] Total sensor counts")
    print("-" * 80)
    
    hai_sample = pd.read_csv(os.path.join(HAI_DIR, "hai-train1.csv"))
    hiend_sample = pd.read_csv(os.path.join(HIEND_DIR, "end-train1.csv"))
    
    hai_sensor_cols = [c for c in hai_sample.columns if c != hai_sample.columns[0]]
    hiend_sensor_cols = [c for c in hiend_sample.columns if c != hiend_sample.columns[0]]
    
    print(f"  HAI raw sensors:      {len(hai_sensor_cols)}")
    print(f"  HAIEnd raw sensors:   {len(hiend_sensor_cols)}")
    print(f"  Total (combined):     {len(hai_sensor_cols) + len(hiend_sensor_cols)}")
    
    # ============================================================================
    # STEP 2: Constants in HAI (TRAIN) - Which are ALSO constant in TEST
    # ============================================================================
    print("\n[STEP 2] Constants in HAI (TRAIN) - Which are ALSO constant in TEST")
    print("-" * 80)
    
    # Load train constants
    print("  Loading HAI train constants...")
    hai_train_constants_list = []
    for num in range(1, 5):
        hai_path = os.path.join(HAI_DIR, f"hai-train{num}.csv")
        hai = pd.read_csv(hai_path)
        hai.rename(columns={hai.columns[0]: "timestamp"}, inplace=True)
        const_cols = get_constant_columns(hai, meta_cols, std_threshold=1e-6)
        hai_train_constants_list.append(set(const_cols))
        print(f"    train{num}: {len(const_cols)} constant columns")
    
    hai_train_universal = set.intersection(*hai_train_constants_list) if hai_train_constants_list else set()
    print(f"    → Universal in ALL train files: {len(hai_train_universal)} columns")
    
    # Load test constants
    print("  Loading HAI test constants...")
    hai_test_constants_list = []
    for num in range(1, 3):
        hai_path = os.path.join(HAI_DIR, f"hai-test{num}.csv")
        hai = pd.read_csv(hai_path)
        hai.rename(columns={hai.columns[0]: "timestamp"}, inplace=True)
        const_cols = get_constant_columns(hai, meta_cols, std_threshold=1e-6)
        hai_test_constants_list.append(set(const_cols))
        print(f"    test{num}: {len(const_cols)} constant columns")
    
    hai_test_universal = set.intersection(*hai_test_constants_list) if hai_test_constants_list else set()
    print(f"    → Universal in ALL test files: {len(hai_test_universal)} columns")
    
    # Overlap analysis
    hai_common = hai_train_universal & hai_test_universal
    hai_train_only = hai_train_universal - hai_test_universal
    hai_test_only = hai_test_universal - hai_train_universal
    
    print(f"\n  Comparison:")
    print(f"    ✓ BOTH train and test: {len(hai_common)} columns")
    if hai_common:
        print(f"      Examples: {sorted(list(hai_common))[:5]}")
    print(f"    • Train ONLY: {len(hai_train_only)} columns")
    if hai_train_only:
        print(f"      Examples: {sorted(list(hai_train_only))[:5]}")
    print(f"    • Test ONLY: {len(hai_test_only)} columns")
    if hai_test_only:
        print(f"      Examples: {sorted(list(hai_test_only))[:5]}")
    
    # ============================================================================
    # STEP 3: Constants in HAIEnd (TRAIN) - Which are ALSO constant in TEST
    # ============================================================================
    print("\n[STEP 3] Constants in HAIEnd (TRAIN) - Which are ALSO constant in TEST")
    print("-" * 80)
    
    # Load train constants
    print("  Loading HAIEnd train constants...")
    hiend_train_constants_list = []
    for num in range(1, 5):
        hiend_path = os.path.join(HIEND_DIR, f"end-train{num}.csv")
        hiend = pd.read_csv(hiend_path)
        hiend.rename(columns={hiend.columns[0]: "timestamp"}, inplace=True)
        const_cols = get_constant_columns(hiend, meta_cols, std_threshold=1e-6)
        hiend_train_constants_list.append(set(const_cols))
        print(f"    train{num}: {len(const_cols)} constant columns")
    
    hiend_train_universal = set.intersection(*hiend_train_constants_list) if hiend_train_constants_list else set()
    print(f"    → Universal in ALL train files: {len(hiend_train_universal)} columns")
    
    # Load test constants
    print("  Loading HAIEnd test constants...")
    hiend_test_constants_list = []
    for num in range(1, 3):
        hiend_path = os.path.join(HIEND_DIR, f"end-test{num}.csv")
        hiend = pd.read_csv(hiend_path)
        hiend.rename(columns={hiend.columns[0]: "timestamp"}, inplace=True)
        const_cols = get_constant_columns(hiend, meta_cols, std_threshold=1e-6)
        hiend_test_constants_list.append(set(const_cols))
        print(f"    test{num}: {len(const_cols)} constant columns")
    
    hiend_test_universal = set.intersection(*hiend_test_constants_list) if hiend_test_constants_list else set()
    print(f"    → Universal in ALL test files: {len(hiend_test_universal)} columns")
    
    # Overlap analysis
    hiend_common = hiend_train_universal & hiend_test_universal
    hiend_train_only = hiend_train_universal - hiend_test_universal
    hiend_test_only = hiend_test_universal - hiend_train_universal
    
    print(f"\n  Comparison:")
    print(f"    ✓ BOTH train and test: {len(hiend_common)} columns")
    if hiend_common:
        print(f"      Examples: {sorted(list(hiend_common))[:5]}")
    print(f"    • Train ONLY: {len(hiend_train_only)} columns")
    if hiend_train_only:
        print(f"      Examples: {sorted(list(hiend_train_only))[:5]}")
    print(f"    • Test ONLY: {len(hiend_test_only)} columns")
    if hiend_test_only:
        print(f"      Examples: {sorted(list(hiend_test_only))[:5]}")
    
    # ============================================================================
    # STEP 4: Duplicates with correlation > 0.99
    # ============================================================================
    print("\n[STEP 4] Duplicates with correlation > 0.99 (checking across TRAIN files)")
    print("-" * 80)
    
    candidate_pairs = [
        ('x1001_15_ASSIGN_OUT',    '1001.15-OUT'), 
        ('P1_FCV01D',   'DM-FCV01-D'),
        ('P1_FCV01Z',   'DM-FCV01-Z'),
        ('P1_FCV02D',   'DM-FCV02-D'),
        ('P1_FCV02Z',   'DM-FCV02-Z'),
        ('P1_FCV03D',   'DM-FCV03-D'),
        ('P1_FCV03Z',   'DM-FCV03-Z'),
        ('P1_FT01',     'DM-FT01'),
        ('P1_FT01Z',    'DM-FT01Z'),
        ('P1_FT02',     'DM-FT02'),
        ('P1_FT02Z',    'DM-FT02Z'),
        ('P1_FT03',     'DM-FT03'),
        ('P1_FT03Z',    'DM-FT03Z'),
        ('P1_LCV01D',   'DM-LCV01-D'),
        ('P1_LCV01Z',   'DM-LCV01-Z'),
        ('P1_LIT01',    'DM-LIT01'),
        ('P1_PCV01D',   'DM-PCV01-D'),
        ('P1_PCV01Z',   'DM-PCV01-Z'),
        ('P1_PCV02D',   'DM-PCV02-D'),
        ('P1_PCV02Z',   'DM-PCV02-Z'),
        ('P1_PIT01_HH', 'DM-PIT01-HH'),
        ('P1_PIT01',    'DM-PIT01'),
        ('P1_PIT02',    'DM-PIT02'),
        ('P1_PP01AD',   'DM-PP01A-D'),
        ('P1_PP01AR',   'DM-PP01A-R'),
        ('P1_PP01BD',   'DM-PP01B-D'),
        ('P1_PP01BR',   'DM-PP01B-R'),
        ('P1_PP02D',    'DM-PP02-D'),
        ('P1_PP02R',    'DM-PP02-R'),
        ('P1_SOL01D',   'DM-SOL01-D'),
        ('P1_SOL03D',   'DM-SOL03-D'),
        ('P1_STSP',     'DM-ST-SP'),
        ('P1_TIT01',    'DM-TIT01'),
        ('P1_TIT03',    'DM-TIT02'),   # naming mismatch: HAI TIT03 = HAIEnd TIT02
        ('P4_ST_GOV',   'GATEOPEN'),
    ]
    
    # Check each pair across all train files
    print(f"  Total candidate pairs to check: {len(candidate_pairs)}\n")
    
    correlation_results = {}
    
    for train_num in range(1, 5):
        print(f"  [train{train_num}] Analyzing correlations...")
        
        hai_train = pd.read_csv(os.path.join(HAI_DIR, f"hai-train{train_num}.csv"))
        hiend_train = pd.read_csv(os.path.join(HIEND_DIR, f"end-train{train_num}.csv"))
        
        hai_train.rename(columns={hai_train.columns[0]: "timestamp"}, inplace=True)
        hiend_train.rename(columns={hiend_train.columns[0]: "timestamp"}, inplace=True)
        
        hai_train["timestamp"] = pd.to_datetime(hai_train["timestamp"])
        hiend_train["timestamp"] = pd.to_datetime(hiend_train["timestamp"])
        
        high_corr_count = 0
        low_corr_count = 0
        missing_count = 0
        
        train_high_pairs = []
        train_low_pairs = []
        
        for hai_col, hiend_col in candidate_pairs:
            if hai_col not in hai_train.columns or hiend_col not in hiend_train.columns:
                missing_count += 1
                continue
            
            temp_merge = pd.merge(
                hai_train[['timestamp', hai_col]],
                hiend_train[['timestamp', hiend_col]],
                on='timestamp',
                how='inner'
            )
            
            if len(temp_merge) == 0:
                missing_count += 1
                continue
            
            hai_vals = temp_merge[hai_col].values[1:].astype(float)
            hiend_vals = temp_merge[hiend_col].values[:-1].astype(float)
            
            if hai_vals.std() < 1e-9 or hiend_vals.std() < 1e-9:
                continue
            
            corr = float(np.corrcoef(hai_vals, hiend_vals)[0, 1])
            
            if corr > 0.99:
                train_high_pairs.append((hai_col, hiend_col, corr))
                high_corr_count += 1
            else:
                train_low_pairs.append((hai_col, hiend_col, corr))
                low_corr_count += 1
        
        correlation_results[f"train{train_num}"] = {
            'high': train_high_pairs,
            'low': train_low_pairs,
            'missing': missing_count
        }
        
        print(f"    → High (corr > 0.99): {high_corr_count}")
        print(f"    → Low (corr ≤ 0.99):  {low_corr_count}")
        if missing_count > 0:
            print(f"    → Missing/skipped:   {missing_count}")
    
    # ============================================================================
    # Summary of correlations across all train files
    # ============================================================================
    print(f"\n  ANALYSIS ACROSS ALL TRAIN FILES:")
    print("  " + "-" * 76)
    
    # Collect all high correlation pairs from each file
    all_high_by_file = {}
    for train_num in range(1, 5):
        all_high_by_file[f"train{train_num}"] = set([(p[0], p[1]) for p in correlation_results[f"train{train_num}"]['high']])
    
    # Find intersection (pairs that are high correlation in ALL files)
    if all_high_by_file:
        consistent_high = set.intersection(*all_high_by_file.values()) if all_high_by_file['train1'] else set()
    else:
        consistent_high = set()
    
    print(f"  High correlation pairs (corr > 0.99):")
    all_high_pairs = set()
    for train_num in range(1, 5):
        for pair in all_high_by_file[f"train{train_num}"]:
            all_high_pairs.add(pair)
    
    for hai_col, hiend_col in sorted(all_high_pairs)[:20]:
        statuses = []
        for train_num in range(1, 5):
            if (hai_col, hiend_col) in all_high_by_file[f"train{train_num}"]:
                statuses.append("✓")
            else:
                statuses.append("✗")
        status_str = " ".join(statuses)
        print(f"    {hai_col:15} ↔ {hiend_col:20} [{status_str}]")
    
    if len(all_high_pairs) > 20:
        print(f"    ... and {len(all_high_pairs) - 20} more pairs")
    
    print(f"\n  Pairs marked HIGH in:")
    print(f"    All 4 train files (consistent): {len(consistent_high)}")
    for pair in sorted(consistent_high)[:10]:
        print(f"      {pair[0]:15} ↔ {pair[1]:20}")
    if len(consistent_high) > 10:
        print(f"      ... and {len(consistent_high) - 10} more")
    
    variable_high = len(all_high_pairs) - len(consistent_high)
    print(f"    Variable across train files: {variable_high}")
    
    total_to_drop = len(all_high_pairs)
    total_to_keep = len(candidate_pairs) - total_to_drop
    
    print(f"\n  FINAL DECISION (using consistent pairs from all trains):")
    print(f"    → Pairs to DROP (corr > 0.99): {len(consistent_high)}")
    print(f"    → Pairs to KEEP (corr ≤ 0.99): {len(candidate_pairs) - len(consistent_high)}")
    
    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY")
    print("="*80)
    print(f"\nRAW SENSORS:")
    print(f"  HAI:          {len(hai_sensor_cols)} columns")
    print(f"  HAIEnd:       {len(hiend_sensor_cols)} columns")
    print(f"  Total:        {len(hai_sensor_cols) + len(hiend_sensor_cols)} columns")
    
    print(f"\nHAI CONSTANTS:")
    print(f"  Train universal (in ALL train files): {len(hai_train_universal)} columns")
    print(f"  Test universal (in ALL test files):   {len(hai_test_universal)} columns")
    print(f"  Overlap (same in both):               {len(hai_common)} columns")
    
    print(f"\nHAIEnd CONSTANTS:")
    print(f"  Train universal (in ALL train files): {len(hiend_train_universal)} columns")
    print(f"  Test universal (in ALL test files):   {len(hiend_test_universal)} columns")
    print(f"  Overlap (same in both):               {len(hiend_common)} columns")
    
    print(f"\nDUPLICATES ANALYSIS (correlation > 0.99):")
    print(f"  Total candidate pairs: {len(candidate_pairs)}")
    print(f"  High correlation (to DROP):   {len(consistent_high)} pairs")
    print(f"  Low/borderline (to KEEP):     {len(candidate_pairs) - len(consistent_high)} pairs")
    
    final_hai = len(hai_sensor_cols) - len(hai_common)
    final_hiend = len(hiend_sensor_cols) - len(hiend_common) - len(consistent_high)
    final_total = final_hai + final_hiend + 2  # +2 for timestamp and attack
    
    print(f"\nFINAL MERGED COLUMNS (train files, per verified analysis):")
    print(f"  HAI sensors:      {final_hai} columns ({len(hai_sensor_cols)} - {len(hai_common)} constants)")
    print(f"  HAIEnd sensors:   {final_hiend} columns ({len(hiend_sensor_cols)} - {len(hiend_common)} constants - {len(consistent_high)} duplicates)")
    print(f"  Metadata:         2 columns (timestamp, attack)")
    print(f"  " + "-" * 50)
    print(f"  TOTAL:            {final_total} columns per train file")
    print("="*80)
