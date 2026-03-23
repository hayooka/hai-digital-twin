from __future__ import annotations
import os
import pandas as pd

_CSV_PATH = os.path.join(os.path.dirname(__file__), "test_data.csv")

_timetable: pd.DataFrame | None = None


def _load() -> pd.DataFrame:
    global _timetable
    if _timetable is None:
        df = pd.read_csv(_CSV_PATH, parse_dates=["start", "end"])
        _timetable = df
    return _timetable


def enrich_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add typed attack columns to a merged DataFrame.

    New columns added:
      attack_id   : "A101" … "A238", or "" for normal rows
      scenario    : "AP", "AE01", "AE03", … or "" for normal rows
      attack_type : "AP", "AE", or "normal"

    Requires df to have a 'timestamp' column (datetime).
    """
    timetable = _load()

    df = df.copy()
    df["attack_id"]   = ""
    df["scenario"]    = ""
    df["attack_type"] = "normal"

    ts = df["timestamp"]
    for _, row in timetable.iterrows():
        mask = (ts >= row["start"]) & (ts <= row["end"])
        df.loc[mask, "attack_id"]   = row["attack_id"]
        df.loc[mask, "scenario"]    = row["scenario"]
        df.loc[mask, "attack_type"] = row["attack_type"]

    return df


def get_attack_windows(split_num: int | None = None) -> pd.DataFrame:
    """
    Return the timetable as a DataFrame, optionally filtered by test split.

    split_num=1  → A101-A114 (test1)
    split_num=2  → A201-A238 (test2)
    split_num=None → all 52 attacks
    """
    timetable = _load()
    if split_num == 1:
        return timetable[timetable["attack_id"].str.startswith("A1")].reset_index(drop=True)
    if split_num == 2:
        return timetable[timetable["attack_id"].str.startswith("A2")].reset_index(drop=True)
    return timetable.copy()


if __name__ == "__main__":
    timetable = _load()
    print(f"Loaded {len(timetable)} attacks from {_CSV_PATH}\n")

    print("=== test1 (A101-A114) — all AP ===")
    print(get_attack_windows(1).to_string(index=False))

    print("\n=== test2 (A201-A238) — AP + AE ===")
    t2 = get_attack_windows(2)
    print(t2.to_string(index=False))

    print(f"\nAE attacks in test2: {(t2['attack_type'] == 'AE').sum()}")
    print(f"AP attacks in test2: {(t2['attack_type'] == 'AP').sum()}")
