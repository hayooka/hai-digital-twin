"""
data_loader_hai_only.py — Load HAI data only, no merging with HAIEnd.
All preprocessing steps (missing values, labels, saving) are kept.
"""

from __future__ import annotations
import os
import glob
import pandas as pd

HAI_DIR = "C:\\Users\\ahmma\\Desktop\\farah\\hai-23.05"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test_data.csv")


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


def load_hai(split: str, num: int) -> pd.DataFrame:
    """Load HAI file for a given split and number, apply preprocessing."""
    if split == "train":
        patterns = [f"hai-train{num}.csv", f"train{num}.csv"]
    else:  # test
        patterns = [
            f"hai-test{num}-labeled.csv",
            f"hai-test{num}.csv",
            f"test{num}.csv",
            f"test{num}_labeled.csv"
        ]
    
    hai_path = find_file(HAI_DIR, patterns)
    print(f"  Found HAI: {os.path.basename(hai_path)}")
    
    df = pd.read_csv(hai_path, low_memory=False)
    
    # Rename first column to timestamp if needed
    if df.columns[0] != "timestamp":
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Handle missing values
    df = handle_missing_values(df, method='forward_fill')
    
    # Remove raw attack columns if present
    raw_attack_cols = ["attack", "attack_p1", "attack_p2", "attack_p3"]
    df = df.drop(columns=[c for c in raw_attack_cols if c in df.columns])
    
    # Assign labels
    if split == "train":
        if "attack_type" in df.columns and "combination" in df.columns:
            df["label"] = df.apply(
                lambda row: assign_label(row["attack_type"], row["combination"]), axis=1
            )
            n_attack = (df["label"] > 0).sum()
            print(f"  Derived labels from attack_type/combination: {n_attack} attack rows")
        else:
            df["label"] = 0
            print("  No attack_type/combination columns -> all labels = 0")
    else:  # test
        df = add_condition(df, num)
    
    print(f"[{split}{num}] Loaded: {len(df)} rows, {len(df.columns)} cols")
    return df


def save_processed(output_dir: str = "data/processed") -> None:
    """Load HAI only for each split and save as CSV (no merging)."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Processing HAI data only (no HAIEnd)")
    print("="*60)
    
    # Process train files (1-4)
    print("\n[save_processed] Building train files …")
    for n in range(1, 5):
        try:
            df = load_hai("train", n)
            path = os.path.join(output_dir, f"trainhai{n}.csv")
            df.to_csv(path, index=False)
            print(f"  Saved {path}  ({len(df):,} rows, {len(df.columns)} cols)\n")
        except Exception as e:
            print(f"  ERROR processing train{n}: {e}\n")
    
    # Process test files (1-2)
    print("\n[save_processed] Building test files …")
    for n in range(1, 3):
        try:
            df = load_hai("test", n)
            path = os.path.join(output_dir, f"testhai{n}.csv")
            df.to_csv(path, index=False)
            label_counts = df["label"].value_counts().sort_index().to_dict()
            print(f"  Saved {path}  ({len(df):,} rows, {len(df.columns)} cols)  labels={label_counts}\n")
        except Exception as e:
            print(f"  ERROR processing test{n}: {e}\n")
    
    print("\n[save_processed] Done — 6 files written to", output_dir)
    print("="*60)


def list_available_files():
    """List all available files in HAI directory."""
    print("\nAvailable files in HAI_DIR:")
    print("-" * 40)
    for f in sorted(os.listdir(HAI_DIR)):
        if f.endswith('.csv'):
            print(f"  {f}")


if __name__ == "__main__":
    list_available_files()
    save_processed()