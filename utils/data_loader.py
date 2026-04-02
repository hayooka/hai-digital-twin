"""
data_loader.py — Simple HAI + HAIEnd merger with automatic file name detection.
"""

from __future__ import annotations
import os
import glob
import pandas as pd
import numpy as np

HAI_DIR = "C:\\Users\\ahmma\\Desktop\\farah\\hai-23.05"
HIEND_DIR = "C:\\Users\\ahmma\\Desktop\\farah\\haiend-23.05"

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test_data.csv")

# Metadata columns that appear in both HAI and HAIEnd (duplicates to remove)
DUPLICATE_META_COLS = ["attack_id", "scenario", "attack_type", "combination", 
                       "target_controller", "target_points"]


def find_file(base_dir: str, patterns: list[str]) -> str:
    """Find a file matching any of the patterns."""
    for pattern in patterns:
        full_pattern = os.path.join(base_dir, pattern)
        matches = glob.glob(full_pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No file found matching patterns: {patterns}")


def handle_missing_values(df, method='forward_fill') -> pd.DataFrame:
    """Handle missing values in sensor data."""
    n_missing = df.isnull().sum().sum()
    
    if n_missing == 0:
        return df
    
    df_clean = df.copy()
    
    if method == 'forward_fill':
        df_clean = df_clean.ffill().bfill()
    elif method == 'backward_fill':
        df_clean = df_clean.bfill().ffill()
    elif method == 'drop':
        df_clean = df_clean.dropna()
    
    n_remaining = df_clean.isnull().sum().sum()
    if n_remaining == 0:
        print(f"  Handled {n_missing} missing values using '{method}' (now clean)")
    else:
        print(f"  Handled {n_missing} missing values; {n_remaining} remain")
    
    return df_clean


def load_merged(split: str, num: int) -> pd.DataFrame:
    """
    Load and merge HAI + HIEND for a given split and number.
    Automatically detects file naming patterns.
    """
    # Define possible file name patterns
    if split == "train":
        hai_patterns = [f"hai-train{num}.csv", f"train{num}.csv"]
        hiend_patterns = [f"end-train{num}.csv", f"haiend-train{num}.csv", f"train{num}_hiend.csv"]
    else:  # test
        hai_patterns = [
            f"hai-test{num}-labeled.csv",
            f"hai-test{num}.csv", 
            f"test{num}.csv",
            f"test{num}_labeled.csv"
        ]
        hiend_patterns = [
            f"end-test{num}-labeled.csv",
            f"end-test{num}.csv",
            f"haiend-test{num}.csv",
            f"test{num}_hiend.csv"
        ]
    
    try:
        hai_path = find_file(HAI_DIR, hai_patterns)
        hiend_path = find_file(HIEND_DIR, hiend_patterns)
        print(f"  Found HAI: {os.path.basename(hai_path)}")
        print(f"  Found HAIEnd: {os.path.basename(hiend_path)}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        raise
    
    # Load files
    hai = pd.read_csv(hai_path, low_memory=False)
    hiend = pd.read_csv(hiend_path, low_memory=False)
    
    # Rename first column to timestamp if it's not already
    if hai.columns[0] != "timestamp":
        hai.rename(columns={hai.columns[0]: "timestamp"}, inplace=True)
    if hiend.columns[0] != "timestamp":
        hiend.rename(columns={hiend.columns[0]: "timestamp"}, inplace=True)
    
    # Convert to datetime
    hai["timestamp"] = pd.to_datetime(hai["timestamp"])
    hiend["timestamp"] = pd.to_datetime(hiend["timestamp"])
    
    # Handle missing values
    hai = handle_missing_values(hai, method='forward_fill')
    hiend = handle_missing_values(hiend, method='forward_fill')
    
    # Merge on timestamp (inner join)
    merged = pd.merge(
        hai,
        hiend,
        on="timestamp",
        how="inner",
        suffixes=("", "_hiend")
    )
    
    # Remove duplicate metadata columns that came from HAIEnd (suffix _hiend)
    dup_cols = [f"{col}_hiend" for col in DUPLICATE_META_COLS if f"{col}_hiend" in merged.columns]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)
    
    # Derive labels from per-row attack metadata if available (train files have these)
    if "attack_type" in merged.columns and "combination" in merged.columns:
        merged["label"] = merged.apply(
            lambda row: assign_label(row["attack_type"], row["combination"]), axis=1
        )
        n_attack = (merged["label"] > 0).sum()
        print(f"  [label] Derived from attack_type/combination: {n_attack} attack rows")
    else:
        merged["label"] = 0

    # Remove any raw attack columns that might interfere
    raw_attack_cols = ["attack", "attack_p1", "attack_p2", "attack_p3"]
    merged = merged.drop(columns=[c for c in raw_attack_cols if c in merged.columns])
    
    print(f"[{split}{num}] Merged: {len(merged)} rows, {len(merged.columns)} cols")
    return merged


def assign_label(attack_type, combination) -> int:
    """Map attack_type + combination to numeric label."""
    if pd.isna(attack_type) or attack_type == "normal":
        return 0
    if attack_type == "AP":
        return 1 if str(combination).strip().lower() == "no" else 2
    if attack_type == "AE":
        return 3 if str(combination).strip().lower() == "no" else 4
    return 0


def add_condition(df: pd.DataFrame, split_num: int) -> pd.DataFrame:
    """Add condition labels to test DataFrames using test_data.csv."""
    # Check if test_data.csv exists
    if not os.path.exists(TEST_DATA_PATH):
        print(f"  WARNING: {TEST_DATA_PATH} not found. No labels applied.")
        return df
    
    attacks = pd.read_csv(TEST_DATA_PATH)
    attacks["start"] = pd.to_datetime(attacks["start"])
    attacks["end"] = pd.to_datetime(attacks["end"])
    attacks = attacks[attacks["split"] == split_num].reset_index(drop=True)
    
    df = df.copy()
    
    for _, row in attacks.iterrows():
        mask = (df["timestamp"] >= row["start"]) & (df["timestamp"] <= row["end"])
        df.loc[mask, "label"] = assign_label(row["attack_type"], row["combination"])
    
    n_attack = (df["label"] > 0).sum()
    print(f"  [label] split={split_num}: {n_attack} attack rows labelled "
          f"({attacks.shape[0]} windows from test_data.csv)")
    return df


def save_processed(output_dir: str = "data/processed") -> None:
    """Merge HAI + HAIEnd for each split and save as CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Processing HAI + HAIEnd files")
    print("="*60)
    
    # Process train files (1-4)
    print("\n[save_processed] Building train files …")
    for n in range(1, 5):
        try:
            df = load_merged("train", n)
            path = os.path.join(output_dir, f"train{n}.csv")
            df.to_csv(path, index=False)
            print(f"  Saved {path}  ({len(df):,} rows, {len(df.columns)} cols)\n")
        except Exception as e:
            print(f"  ERROR processing train{n}: {e}\n")
    
    # Process test files (1-2)
    print("\n[save_processed] Building test files …")
    for n in range(1, 3):
        try:
            df = load_merged("test", n)
            df = add_condition(df, n)
            path = os.path.join(output_dir, f"test{n}.csv")
            df.to_csv(path, index=False)
            label_counts = df["label"].value_counts().sort_index().to_dict()
            print(f"  Saved {path}  ({len(df):,} rows, {len(df.columns)} cols)  labels={label_counts}\n")
        except Exception as e:
            print(f"  ERROR processing test{n}: {e}\n")
    
    print("\n[save_processed] Done — 6 files written to", output_dir)
    print("="*60)


def list_available_files():
    """List all available files in HAI and HAIEnd directories."""
    print("\nAvailable files in HAI_DIR:")
    print("-" * 40)
    for f in sorted(os.listdir(HAI_DIR)):
        if f.endswith('.csv'):
            print(f"  {f}")
    
    print("\nAvailable files in HIEND_DIR:")
    print("-" * 40)
    for f in sorted(os.listdir(HIEND_DIR)):
        if f.endswith('.csv'):
            print(f"  {f}")


if __name__ == "__main__":
    # First, list available files to debug
    list_available_files()
    
    # Then process
    save_processed()