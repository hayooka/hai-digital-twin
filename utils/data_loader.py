from __future__ import annotations
import os
import pandas as pd

HAI_DIR   = os.environ.get("HAI_DIR",   "C:/Users/ahmma/Desktop/hai-23.05")
HIEND_DIR = os.environ.get("HIEND_DIR", "C:/Users/ahmma/Desktop/haiend-23.05")
#set HAI_DIR= C:/Users/farah/OneDrive/Desktop/AI_project/hai-23.05
#set HIEND_DIR= C:/Users/farah/OneDrive/Desktop/AI_project/haiend-23.05


# HAI columns that are duplicated in HIEND (confirmed from dataset documentation).
# HIEND version takes priority — these HAI columns are dropped after merging.
HAI_DUPLICATES = [
    "P1_B2016",     # 1001.15-OUT
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


def load_merged(split: str, num: int) -> pd.DataFrame:
    """
    Load and merge HAI + HIEND for a given split ('train' or 'test') and number.

    Merges on exact timestamp (inner join — unmatched boundary rows are dropped).
    HAI columns that are duplicated in HIEND are removed; HIEND names take priority.

    # --- HAI-only mode (commented out) ---
    # To use HAI-23.05 only (86 features instead of 277), comment out the merge
    # block below and use the HAI-only block instead.
    """
    hai_path   = os.path.join(HAI_DIR,   f"hai-{split}{num}.csv")
    hiend_path = os.path.join(HIEND_DIR, f"end-{split}{num}.csv")

    hai   = pd.read_csv(hai_path)
    hiend = pd.read_csv(hiend_path)

    hai.rename(columns={hai.columns[0]: "timestamp"}, inplace=True)
    hiend.rename(columns={hiend.columns[0]: "timestamp"}, inplace=True)

    hai["timestamp"]   = pd.to_datetime(hai["timestamp"])
    hiend["timestamp"] = pd.to_datetime(hiend["timestamp"])

    # Drop HAI columns that have an authoritative HIEND counterpart
    cols_to_drop = [c for c in HAI_DUPLICATES if c in hai.columns]
    hai = hai.drop(columns=cols_to_drop)

    # Merge on exact timestamp — rows with no match are dropped (no manipulation)
    merged = pd.merge(
        hai,
        hiend,
        on="timestamp",
        how="inner",
        suffixes=("", "_hiend"),
    )

    # Load separate label file for test splits (labels are not in the sensor CSVs)
    if split == "test" and "attack" not in merged.columns:
        label_path = os.path.join(HIEND_DIR, f"label-{split}{num}.csv")
        if os.path.exists(label_path):
            labels = pd.read_csv(label_path)
            labels.rename(columns={labels.columns[0]: "timestamp"}, inplace=True)
            labels["timestamp"] = pd.to_datetime(labels["timestamp"])
            labels = labels.drop_duplicates(subset="timestamp", keep="first")
            # find the label column (binary 0/1)
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
        f"[{split}{num}] merged: {len(merged)} rows, {len(merged.columns)} cols "
        f"(dropped {len(cols_to_drop)} duplicate HAI cols)"
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
    """Return a dict of all merged train splits keyed by 'train1'..'train4'."""
    return {f"train{i}": load_merged("train", i) for i in range(1, 5)}


def load_all_test() -> dict[str, pd.DataFrame]:
    """Return a dict of all merged test splits keyed by 'test1'..'test2'."""
    return {f"test{i}": load_merged("test", i) for i in range(1, 3)}


if __name__ == "__main__":
    print("=== Loading train files ===")
    train_data = load_all_train()

    print("\n=== Loading test files ===")
    test_data = load_all_test()

    print("\nColumns in merged train1:")
    print(list(train_data["train1"].columns))
