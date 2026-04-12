"""
data_pipeline.py — Per-loop sequence builder for Boiler Digital Twin.

Reuses scaled_split.py logic:
  - Scaler fitted on normal data from train1-3 ONLY
  - train4 completely held out for testing
  - 80% attack segments → train, 20% → test
  - scenario labels: 0=normal, 1=AP_no, 2=AP_with, 3=AE_no, 4=AE_with

Outputs two types of datasets:
  1. Per-loop controller sequences: [SP, PV, CV_fb] → CV
  2. Plant MIMO sequences:          [CVs + aux]     → PVs  (with teacher forcing)
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_causal"))

from parse_dcs import (
    LOOPS_DEF, PV_COLS, CV_COLS, PLANT_AUX_COLS,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_DIR    = Path(__file__).resolve().parents[2] / "outputs" / "boiler_twin"

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN            = 300    # 5 minutes at 1Hz
STRIDE             = 10     # step between sequences
TRAIN_RATIO        = 0.80
ATTACK_SPLIT_RATIO = 0.80
BEFORE_ATTACK_SEC  = 300
AFTER_ATTACK_SEC   = 180
RANDOM_SEED        = 42

META_COLS = {
    "timestamp", "label", "attack_id", "scenario",
    "attack_type", "combination", "target_controller",
    "target_points", "duration_sec", "episode_id",
    "source_file", "scenario_label",
}

PLANT_IN_COLS  = CV_COLS + PLANT_AUX_COLS   # ~19 total inputs
PLANT_OUT_COLS = PV_COLS                     # 5 outputs


# ── Helpers (same as scaled_split.py) ─────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _sensor_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def get_scenario_label(attack_type: str, combination: str = "no") -> int:
    if pd.isna(attack_type) or attack_type == "normal":
        return 0
    if attack_type == "AP":
        return 1 if str(combination).strip().lower() == "no" else 2
    if attack_type == "AE":
        return 3 if str(combination).strip().lower() == "no" else 4
    return 0


def _extract_attack_segments(df: pd.DataFrame,
                              before_sec: int = BEFORE_ATTACK_SEC,
                              after_sec:  int = AFTER_ATTACK_SEC) -> list[pd.DataFrame]:
    if "label" not in df.columns or not (df["label"] > 0).any():
        return []
    mask = (df["label"] > 0).values
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            blocks.append((i, j - 1))
            i = j
        else:
            i += 1
    segments = []
    for start, end in blocks:
        lo = max(0, start - before_sec)
        hi = min(len(df) - 1, end + after_sec)
        segment = df.iloc[lo:hi + 1].copy().reset_index(drop=True)
        attack_rows = segment[segment["label"] > 0]
        if len(attack_rows) > 0:
            at   = attack_rows.iloc[0]["attack_type"]
            comb = attack_rows.iloc[0].get("combination", "no")
            segment["scenario_label"] = get_scenario_label(at, comb)
        else:
            segment["scenario_label"] = 0
        segments.append(segment)
    return segments


# ── Per-loop sequence builder ─────────────────────────────────────────────────

def _make_ctrl_sequences(
    df: pd.DataFrame,
    loop_name: str,
    scaler: StandardScaler,
    all_sensor_cols: list[str],
    seq_len: int = SEQ_LEN,
    stride:  int = STRIDE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build controller sequences for one loop.

    Returns:
      inputs  : (N, seq_len, n_ctrl_in)  — [SP, PV, CV_fb] scaled
      targets : (N, seq_len)             — CV scaled
      scenarios: (N,)                    — scenario label per sequence
    """
    defn   = LOOPS_DEF[loop_name]
    sp_col = defn["sp"]
    pv_col = defn["pv"]
    cv_col = defn["cv"]
    fb_col = defn["cv_fb"]

    ctrl_cols = [sp_col, pv_col] + ([fb_col] if fb_col else []) + [cv_col]
    missing   = [c for c in ctrl_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{loop_name}] Missing columns in DataFrame: {missing}")

    # Scale using the global sensor scaler (subset of columns)
    col_idx  = {c: all_sensor_cols.index(c) for c in ctrl_cols}
    data_raw = df[ctrl_cols].values.astype(np.float32)

    # Build per-column sub-scaler indices
    full_data = np.zeros((len(df), len(all_sensor_cols)), dtype=np.float32)
    for i, c in enumerate(ctrl_cols):
        full_data[:, col_idx[c]] = data_raw[:, i]
    scaled_full = scaler.transform(full_data)
    scaled      = np.column_stack([scaled_full[:, col_idx[c]] for c in ctrl_cols])

    n_input = len(ctrl_cols) - 1   # all except last (CV)
    n_seq   = (len(df) - seq_len) // stride

    inputs    = np.zeros((n_seq, seq_len, n_input), dtype=np.float32)
    targets   = np.zeros((n_seq, seq_len),           dtype=np.float32)
    scenarios = np.zeros(n_seq, dtype=np.int32)

    has_sc = "scenario_label" in df.columns
    for i in range(n_seq):
        s = i * stride
        inputs[i]  = scaled[s:s + seq_len, :-1]
        targets[i] = scaled[s:s + seq_len, -1]
        if has_sc:
            window_sc = df["scenario_label"].iloc[s:s + seq_len]
            scenarios[i] = int(window_sc[window_sc > 0].mode()[0]) if (window_sc > 0).any() else 0

    return inputs, targets, scenarios


# ── Plant sequence builder ─────────────────────────────────────────────────────

def _make_plant_sequences(
    df: pd.DataFrame,
    scaler: StandardScaler,
    all_sensor_cols: list[str],
    seq_len: int = SEQ_LEN,
    stride:  int = STRIDE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build plant MIMO sequences.

    Returns:
      cv_seqs     : (N, seq_len, n_plant_in)  — CV + aux inputs
      pv_init     : (N, n_pv)                 — initial PV state
      pv_teacher  : (N, seq_len, n_pv)        — ground-truth PVs (teacher forcing)
      pv_target   : (N, seq_len, n_pv)        — next-step PVs (prediction targets)
      scenarios   : (N,)
    """
    plant_cols = PLANT_IN_COLS + PLANT_OUT_COLS
    missing    = [c for c in plant_cols if c not in df.columns]
    if missing:
        # Drop missing aux columns silently (some may not exist in all CSV files)
        available_in  = [c for c in PLANT_IN_COLS  if c in df.columns]
        available_out = [c for c in PLANT_OUT_COLS if c in df.columns]
        plant_cols    = available_in + available_out
        print(f"  [Plant] Using {len(available_in)} inputs, {len(available_out)} outputs "
              f"(dropped {len(missing)} missing cols)")
    else:
        available_in  = PLANT_IN_COLS
        available_out = PLANT_OUT_COLS

    n_in  = len(available_in)
    n_out = len(available_out)

    col_idx  = {c: all_sensor_cols.index(c) for c in plant_cols if c in all_sensor_cols}
    data_raw = df[[c for c in plant_cols if c in df.columns]].values.astype(np.float32)

    full_data = np.zeros((len(df), len(all_sensor_cols)), dtype=np.float32)
    for i, c in enumerate([c for c in plant_cols if c in df.columns]):
        if c in col_idx:
            full_data[:, col_idx[c]] = data_raw[:, i]
    scaled_full = scaler.transform(full_data)
    scaled      = np.column_stack([
        scaled_full[:, col_idx[c]] for c in plant_cols if c in col_idx
    ])

    n_seq = (len(df) - seq_len - 1) // stride

    cv_seqs    = np.zeros((n_seq, seq_len, n_in),  dtype=np.float32)
    pv_init    = np.zeros((n_seq, n_out),           dtype=np.float32)
    pv_teacher = np.zeros((n_seq, seq_len, n_out),  dtype=np.float32)
    pv_target  = np.zeros((n_seq, seq_len, n_out),  dtype=np.float32)
    scenarios  = np.zeros(n_seq, dtype=np.int32)

    has_sc = "scenario_label" in df.columns
    for i in range(n_seq):
        s = i * stride
        cv_seqs[i]    = scaled[s:s + seq_len, :n_in]
        pv_init[i]    = scaled[s, n_in:]
        pv_teacher[i] = scaled[s:s + seq_len, n_in:]
        pv_target[i]  = scaled[s + 1:s + seq_len + 1, n_in:]
        if has_sc:
            window_sc = df["scenario_label"].iloc[s:s + seq_len]
            scenarios[i] = int(window_sc[window_sc > 0].mode()[0]) if (window_sc > 0).any() else 0

    return cv_seqs, pv_init, pv_teacher, pv_target, scenarios


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_datasets(
    processed_dir: Path  = PROCESSED_DIR,
    output_dir:    Path  = OUTPUT_DIR,
    seq_len:       int   = SEQ_LEN,
    stride:        int   = STRIDE,
    train_ratio:   float = TRAIN_RATIO,
    attack_split:  float = ATTACK_SPLIT_RATIO,
    random_seed:   int   = RANDOM_SEED,
    save:          bool  = True,
) -> Dict:
    """
    Full data pipeline — mirrors scaled_split.py structure.

    Returns dict with keys:
      ctrl_{loop}_train / ctrl_{loop}_val / ctrl_{loop}_test
      plant_train / plant_val / plant_test
      scaler, sensor_cols, n_scenarios
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(random_seed)

    # ── Step 1: Load train1-3 ─────────────────────────────────────────────────
    print("\n[Step 1] Loading train1, train2, train3")
    train_eps, val_eps, sensor_cols = [], [], []
    for n in range(1, 4):
        df = _load(processed_dir / f"train{n}.csv")
        if not sensor_cols:
            sensor_cols = _sensor_cols(df)
            print(f"  Sensor features: {len(sensor_cols)}")
        df["scenario_label"] = 0
        cut = int(len(df) * train_ratio)
        tr, va = df.iloc[:cut].copy(), df.iloc[cut:].copy()
        tr["episode_id"] = va["episode_id"] = f"train{n}"
        train_eps.append(tr)
        val_eps.append(va)
        print(f"  train{n}: {len(df):,} rows → train {len(tr):,} / val {len(va):,}")

    # ── Step 2: Hold out train4 ───────────────────────────────────────────────
    print("\n[Step 2] Loading train4 (test — completely held out)")
    train4 = _load(processed_dir / "train4.csv")
    train4["scenario_label"] = 0
    train4["episode_id"]     = "train4_TEST"
    print(f"  train4: {len(train4):,} rows")

    # ── Step 3: Extract attack segments from test1 + test2 ───────────────────
    print("\n[Step 3] Extracting attack segments from test1, test2")
    all_attack_segs: list[pd.DataFrame] = []
    for n in range(1, 3):
        path = processed_dir / f"test{n}.csv"
        if not path.exists():
            continue
        df   = _load(path)
        segs = _extract_attack_segments(df)
        for idx, seg in enumerate(segs):
            seg["episode_id"]  = f"test{n}_seg{idx}"
            seg["source_file"] = f"test{n}"
            all_attack_segs.append(seg)
        print(f"  test{n}: {len(segs)} segments")
    print(f"  Total attack segments: {len(all_attack_segs)}")

    # ── Step 4: Split attacks 80/20 ───────────────────────────────────────────
    print("\n[Step 4] Splitting attack segments")
    if len(all_attack_segs) >= 2:
        atk_train, atk_test = train_test_split(
            all_attack_segs, train_size=attack_split,
            random_state=random_seed, shuffle=True,
        )
    else:
        atk_train, atk_test = all_attack_segs, []
    print(f"  Attack train: {len(atk_train)}  Attack test: {len(atk_test)}")

    # ── Step 5: Fit scaler on normal train1-3 ─────────────────────────────────
    print("\n[Step 5] Fitting scaler on normal data (train1-3, label==0)")
    normal_data = pd.concat(
        [ep[ep["label"] == 0] if "label" in ep.columns else ep for ep in train_eps],
        ignore_index=True,
    )
    scaler = StandardScaler()
    scaler.fit(normal_data[sensor_cols].values.astype(np.float32))
    if save:
        joblib.dump(scaler, output_dir / "scaler.pkl")
        np.save(output_dir / "scaler_cols.npy", np.array(sensor_cols))
        print(f"  Scaler saved → {output_dir / 'scaler.pkl'}")

    # ── Step 6: Build sequences ───────────────────────────────────────────────
    print("\n[Step 6] Building sequences")

    def _concat_ctrl(episodes, loop):
        parts = [_make_ctrl_sequences(ep, loop, scaler, sensor_cols, seq_len, stride)
                 for ep in episodes]
        inp = np.concatenate([p[0] for p in parts], axis=0)
        tgt = np.concatenate([p[1] for p in parts], axis=0)
        sc  = np.concatenate([p[2] for p in parts], axis=0)
        return inp, tgt, sc

    def _concat_plant(episodes):
        parts = [_make_plant_sequences(ep, scaler, sensor_cols, seq_len, stride)
                 for ep in episodes]
        cv  = np.concatenate([p[0] for p in parts], axis=0)
        pi  = np.concatenate([p[1] for p in parts], axis=0)
        pte = np.concatenate([p[2] for p in parts], axis=0)
        ptr = np.concatenate([p[3] for p in parts], axis=0)
        sc  = np.concatenate([p[4] for p in parts], axis=0)
        return cv, pi, pte, ptr, sc

    train_all = train_eps + atk_train
    test_all  = [train4] + atk_test

    result: Dict = {
        "scaler":      scaler,
        "sensor_cols": sensor_cols,
        "n_scenarios": 5,           # 0=normal, 1-4 attack types
        "seq_len":     seq_len,
        "n_plant_in":  len([c for c in PLANT_IN_COLS  if c in sensor_cols]),
        "n_pv":        len([c for c in PLANT_OUT_COLS if c in sensor_cols]),
    }

    # Controller datasets
    for ln in LOOPS_DEF:
        try:
            tr_in, tr_tg, tr_sc = _concat_ctrl(train_all, ln)
            va_in, va_tg, va_sc = _concat_ctrl(val_eps,   ln)
            te_in, te_tg, te_sc = _concat_ctrl(test_all,  ln)
            result[f"ctrl_{ln}_train"] = (tr_in, tr_tg, tr_sc)
            result[f"ctrl_{ln}_val"]   = (va_in, va_tg, va_sc)
            result[f"ctrl_{ln}_test"]  = (te_in, te_tg, te_sc)
            print(f"  {ln}: train={len(tr_in):,}  val={len(va_in):,}  test={len(te_in):,}  "
                  f"shape={tr_in.shape[1:]}")
        except Exception as e:
            print(f"  [WARN] {ln}: {e}")

    # Plant datasets
    cv_tr, pi_tr, pte_tr, ptr_tr, sc_tr = _concat_plant(train_all)
    cv_va, pi_va, pte_va, ptr_va, sc_va = _concat_plant(val_eps)
    cv_te, pi_te, pte_te, ptr_te, sc_te = _concat_plant(test_all)
    result["plant_train"] = (cv_tr, pi_tr, pte_tr, ptr_tr, sc_tr)
    result["plant_val"]   = (cv_va, pi_va, pte_va, ptr_va, sc_va)
    result["plant_test"]  = (cv_te, pi_te, pte_te, ptr_te, sc_te)
    print(f"  Plant: train={len(cv_tr):,}  val={len(cv_va):,}  test={len(cv_te):,}  "
          f"cv_shape={cv_tr.shape[1:]}")

    # ── Save ─────────────────────────────────────────────────────────────────
    if save:
        np.savez_compressed(output_dir / "ctrl_train.npz",
                            **{f"{ln}_inputs":  result[f"ctrl_{ln}_train"][0] for ln in LOOPS_DEF if f"ctrl_{ln}_train" in result},
                            **{f"{ln}_targets": result[f"ctrl_{ln}_train"][1] for ln in LOOPS_DEF if f"ctrl_{ln}_train" in result},
                            **{f"{ln}_sc":      result[f"ctrl_{ln}_train"][2] for ln in LOOPS_DEF if f"ctrl_{ln}_train" in result})
        np.savez_compressed(output_dir / "plant_train.npz",
                            cv_seqs=cv_tr, pv_init=pi_tr,
                            pv_teacher=pte_tr, pv_target=ptr_tr, scenarios=sc_tr)
        np.savez_compressed(output_dir / "plant_val.npz",
                            cv_seqs=cv_va, pv_init=pi_va,
                            pv_teacher=pte_va, pv_target=ptr_va, scenarios=sc_va)
        print(f"\n  Saved datasets → {output_dir}")

    return result


if __name__ == "__main__":
    data = build_datasets()
    print("\n✓ Pipeline complete.")
