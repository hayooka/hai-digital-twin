"""
Adds attack label columns to the 4 original test CSV files:
  attack_id, scenario, attack_type, combination, target_controller, target_points

Rows inside an attack window get the values from test_data.csv.
All other rows get empty strings (normal rows).

Overwrites the original files in-place.
Run: python utils/enrich_test_files.py
"""
import os
import pandas as pd

HAI_DIR   = os.environ.get('HAI_DIR',   'C:/Users/farah/OneDrive/Desktop/AI_project/hai-23.05')
HIEND_DIR = os.environ.get('HIEND_DIR', 'C:/Users/farah/OneDrive/Desktop/AI_project/haiend-23.05')
OUT_DIR   = os.environ.get('OUT_DIR',   'C:/Users/farah/OneDrive/Desktop/AI_project/labeled')

TIMETABLE_PATH = os.path.join(os.path.dirname(__file__), 'test_data.csv')

FILES = [
    (os.path.join(HAI_DIR,   'hai-test1.csv'), 'timestamp',  'hai-test1-labeled.csv'),
    (os.path.join(HAI_DIR,   'hai-test2.csv'), 'timestamp',  'hai-test2-labeled.csv'),
    (os.path.join(HIEND_DIR, 'end-test1.csv'), 'Timestamp',  'end-test1-labeled.csv'),
    (os.path.join(HIEND_DIR, 'end-test2.csv'), 'Timestamp',  'end-test2-labeled.csv'),
]

NEW_COLS = ['attack_id', 'scenario', 'attack_type', 'combination',
            'target_controller', 'target_points']


def enrich(df: pd.DataFrame, ts_col: str, timetable: pd.DataFrame) -> pd.DataFrame:
    for col in NEW_COLS:
        df[col] = ''
    df['attack_type'] = 'normal'
    df['combination'] = 'no'

    ts = pd.to_datetime(df[ts_col])
    for _, row in timetable.iterrows():
        mask = (ts >= row['start']) & (ts <= row['end'])
        for col in NEW_COLS:
            df.loc[mask, col] = row[col]

    # Move new columns to right after timestamp (column 0)
    other_cols = [c for c in df.columns if c not in NEW_COLS and c != ts_col]
    df = df[[ts_col] + NEW_COLS + other_cols]

    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    timetable = pd.read_csv(TIMETABLE_PATH, parse_dates=['start', 'end'])
    print(f'Loaded timetable: {len(timetable)} attacks')
    print(f'Saving copies to: {OUT_DIR}\n')

    for path, ts_col, out_name in FILES:
        print(f'Processing {os.path.basename(path)} ...', end=' ', flush=True)
        df = pd.read_csv(path, low_memory=False)
        df = enrich(df, ts_col, timetable)

        attack_rows = (df['attack_id'] != '').sum()
        out_path = os.path.join(OUT_DIR, out_name)
        df.to_csv(out_path, index=False)
        print(f'done  ({len(df):,} rows, {attack_rows:,} attack rows labeled)')
        print(f'  saved to {out_path}')

    print('\nDone. Originals untouched.')


if __name__ == '__main__':
    main()
