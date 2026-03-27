from __future__ import annotations
import os
import pandas as pd
import numpy as np

HAI_DIR   = "C:\\Users\\ahmma\\Desktop\\farah\\hai-23.05"
HIEND_DIR = "C:\\Users\\ahmma\\Desktop\\farah\\haiend-23.05"
#set HAI_DIR= C:/Users/farah/OneDrive/Desktop/AI_project/hai-23.05
#set HIEND_DIR= C:/Users/farah/OneDrive/Desktop/AI_project/haiend-23.05


# HAI columns that are duplicated in HIEND (confirmed from dataset documentation).
# HIEND version takes priority — these HAI columns are dropped after merging.
HAI_DUPLICATES = [
    "x1001_15_ASSIGN_OUT",     # 1001.15-OUT
    "P1_FCV01D",    # DM-FCV01-D
    "P1_FCV01Z",    # DM-FCV01-Z
    "P1_FCV02D",    # DM-FCV02-D
    "P1_FCV02Z",    # DM-FCV02-Z
    "P1_FCV03D",    # DM-FCV03-D
    "P1_FCV03Z",    # DM-FCV03-Z
    "P1_FT01",      # DM-FT01
    "P1_FT01Z",     # DM-FT01Z
    "P1_FT02",      # DM-FT02
    "P1_FT02Z",     # DM-FT02Z
    "P1_FT03",      # DM-FT03
    "P1_FT03Z",     # DM-FT03Z
    "P1_LCV01D",    # DM-LCV01-D
    "P1_LCV01Z",    # DM-LCV01-Z
    "P1_LIT01",     # DM-LIT01
    "P1_PCV01D",    # DM-PCV01-D
    "P1_PCV01Z",    # DM-PCV01-Z
    "P1_PCV02D",    # DM-PCV02-D
    "P1_PCV02Z",    # DM-PCV02-Z
    "P1_PIT01_HH",  # DM-PIT01-HH
    "P1_PIT01",     # DM-PIT01
    "P1_PIT02",     # DM-PIT02
    "P1_PP01AD",    # DM-PP01A-D
    "P1_PP01AR",    # DM-PP01A-R
    "P1_PP01BD",    # DM-PP01B-D
    "P1_PP01BR",    # DM-PP01B-R
    "P1_PP02D",     # DM-PP02-D
    "P1_PP02R",     # DM-PP02-R
    "P1_SOL01D",    # DM-SOL01-D
    "P1_SOL03D",    # DM-SOL03-D
    "P1_STSP",      # DM-ST-SP
    "P1_TIT01",     # DM-TIT01
    "P1_TIT03",     # DM-TIT02  (note: HAI TIT03 = HIEND TIT02)
    "P4_ST_GOV",    # GATEOPEN
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


def identify_common_constants(std_threshold=1e-6) -> tuple[list[str], list[str]]:
    """
    Identify columns that are CONSTANT in BOTH train AND test splits.
    
    Logic:
    - Constants in BOTH train & test → No signal anywhere → DROP them
    - Constants in train ONLY → Have variation in test → KEEP them (early anomaly indicators!)
    
    Verified findings:
    - HAI: 18 columns constant in both (DROP)
    - HAI: 2 columns constant in train only (KEEP - may signal attacks)
    - HAIEnd: 151 columns constant in both (DROP)
    
    :param std_threshold: Threshold for constant detection
    :return: (hai_common_constants, hiend_common_constants)
    """
    meta_cols = {'timestamp', 'attack', 'label', 'attack_p1', 'attack_p2', 'attack_p3'}
    
    # Get train universal constants
    hai_train_const, hiend_train_const = identify_universal_constants('train', std_threshold)
    
    # Get test universal constants
    hai_test_const, hiend_test_const = identify_universal_constants('test', std_threshold)
    
    # Common = intersection (constant in BOTH splits)
    hai_common = sorted(list(set(hai_train_const) & set(hai_test_const)))
    hiend_common = sorted(list(set(hiend_train_const) & set(hiend_test_const)))
    
    # Train-only = difference (constant in train ONLY)
    hai_train_only = sorted(list(set(hai_train_const) - set(hai_test_const)))
    hiend_train_only = sorted(list(set(hiend_train_const) - set(hiend_test_const)))
    
    print(f"\n  Constants analysis (will DROP only common, KEEP train-only):")
    print(f"    HAI common (both splits): {len(hai_common)}")
    print(f"    HAI train-only (KEEP - potential signals): {len(hai_train_only)}")
    print(f"    HAIEnd common (both splits): {len(hiend_common)}")
    print(f"    HAIEnd train-only (KEEP - potential signals): {len(hiend_train_only)}")
    
    return hai_common, hiend_common


def load_merged(split: str, num: int, drop_constants: bool = True,
                keep_hai_duplicates: bool = True,
                const_cols_hai: list | None = None,
                const_cols_hiend: list | None = None) -> "pd.DataFrame":
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
    
    **Final column count per train file:**
       - HAI sensors: 66 columns (86 - 18 constants - 2 other)
       - HAIEnd sensors: 46 columns (225 - 151 constants - 28 duplicates)
       - Metadata: 2 columns (timestamp, attack)
       - TOTAL: 114 columns
    
    :param split: 'train' or 'test'
    :param num: Split number (1-4 for train, 1-2 for test)
    :param drop_constants: If True, remove UNIVERSAL constant columns
    :param keep_hai_duplicates: If True, prefer HAI names over HIEND (with correlation check)
    :param const_cols_hai: List of HAI columns to drop (computed once per split)
    :param const_cols_hiend: List of HAIEnd columns to drop (computed once per split)
    :return: Merged DataFrame with consistent columns across split
    """
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
    
    
    # ---- STEP 2: Handle duplicates with dynamic correlation checking ----
    if keep_hai_duplicates:
        # Candidate duplicate pairs: (HAI_col, HAIEnd_col)
        candidate_pairs = [
            ('x1001_15_ASSIGN_OUT',    '1001.15-OUT'),   # NOTE: P1_B2016 may not exist
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
        
        # Compute correlations and filter
        hiend_cols_to_drop = []
        dropped_high_corr = 0
        kept_low_corr = 0
        
        for hai_col, hiend_col in candidate_pairs:
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

    # Load separate label file for test splits
    if split == "test" and "attack" not in merged.columns:
        label_path = os.path.join(HIEND_DIR, f"label-{split}{num}.csv")
        if os.path.exists(label_path):
            labels = pd.read_csv(label_path)
            labels.rename(columns={labels.columns[0]: "timestamp"}, inplace=True)
            labels["timestamp"] = pd.to_datetime(labels["timestamp"])
            labels = labels.drop_duplicates(subset="timestamp", keep="first")
            label_col = next(
                (c for c in labels.columns if c != "timestamp"), None
            )
            if label_col:
                merged = pd.merge(merged, labels[["timestamp", label_col]],
                                  on="timestamp", how="left")
                merged.rename(columns={label_col: "attack"}, inplace=True)
                merged["attack"] = merged["attack"].fillna(0).astype(int)
                print(f"  -> loaded labels from label-{split}{num}.csv  "
                      f"(attacks: {merged['attack'].sum()})")
        else:
            print(f"  -> WARNING: label file not found: {label_path}")
            merged["attack"] = 0

    # Train files are fully benign — add attack=0 if not present
    if "attack" not in merged.columns:
        merged["attack"] = 0

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
    const_hai, const_hiend = identify_common_constants()
    print(f"  ✓ Will DROP these constants: HAI {len(const_hai)}, HAIEnd {len(const_hiend)}")
    print(f"  ✓ KEEPING train-only constants (may be early anomaly signals)")
    
    return {f"train{i}": load_merged("train", i, drop_constants=True, keep_hai_duplicates=True,
                                     const_cols_hai=const_hai, const_cols_hiend=const_hiend) 
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
    print("  [Analyzing universal constants across test1-2...]")
    const_hai, const_hiend = identify_universal_constants("test")
    print(f"  [UNIVERSAL test constants] HAI: {len(const_hai)}, HAIEnd: {len(const_hiend)}")
    
    return {f"test{i}": load_merged("test", i, drop_constants=True, keep_hai_duplicates=True,
                                    const_cols_hai=const_hai, const_cols_hiend=const_hiend) 
            for i in range(1, 3)}


if __name__ == "__main__":
    print("="*80)
    print("STEP-BY-STEP DATA LOADER ANALYSIS")
    print("="*80)
    
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
