"""
scaled_split_episodic.py — Split and normalize HAI + HAIEnd processed data.
Handles each file as a separate episode for Transformer seq2seq.

CRITICAL CHANGE: train4 is COMPLETELY HELD OUT for testing causal generalization.
ADDED: Multi-class scenario labels (0=normal, 1-4=attack types)

Pipeline
--------
Step 1 : train1, train2, train3 only for training/validation
         Split each episode 80/20 (time-ordered, no shuffle)
Step 2 : train4 is COMPLETELY HELD OUT (never seen during training)
Step 3 : Extract attack segments from test1 + test2
Step 4 : Split attack segments 80/20 (by segment, not by row)
         80% → training (with train1-3)
         20% → testing (with train4)
Step 5 : Fit scaler on train1-3 normal data ONLY
Step 6 : Create sliding windows with scenario labels

Test Sets (truly unseen):
├── train4 (100%) → Tests cross-period generalization of normal behavior
└── 20% of attack segments → Tests attack trajectory prediction
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional

PROCESSED_DIR      = "data/processed"
OUTPUT_DIR         = "outputs"
TRAIN_RATIO        = 0.80          # For temporal split within train1-3
ATTACK_SPLIT_RATIO = 0.80          # 80% attacks to train, 20% to test
BEFORE_ATTACK_SEC  = 300
AFTER_ATTACK_SEC   = 180
RANDOM_SEED        = 42
INPUT_LEN          = 300           # Input sequence length (past timesteps)
TARGET_LEN         = 180           # Target sequence length (to predict)
STRIDE             = 60            # Sliding window stride

META_COLS = {
    "timestamp", "label",
    "attack_id", "scenario", "attack_type", "combination",
    "target_controller", "target_points", "duration_sec",
}

# ── Attack type to numeric label mapping ──────────────────────────────────────
# Based on assign_label() from data_loader.py
ATTACK_TYPE_TO_LABEL = {
    "normal": 0,
    "AP": 1,      # AP with combination="no"
    "AP_comb": 2, # AP with combination="yes"  
    "AE": 3,      # AE with combination="no"
    "AE_comb": 4, # AE with combination="yes"
}

def get_attack_label(attack_type: str, combination: str = "no") -> int:
    """Map attack_type + combination to numeric label (0-4)."""
    if pd.isna(attack_type) or attack_type == "normal":
        return 0
    if attack_type == "AP":
        return 1 if str(combination).strip().lower() == "no" else 2
    if attack_type == "AE":
        return 3 if str(combination).strip().lower() == "no" else 4
    return 0


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    """Load CSV and prepare timestamp."""
    df = pd.read_csv(path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _sensor_cols(df: pd.DataFrame) -> list[str]:
    """Identify numeric sensor columns (exclude metadata)."""
    return [c for c in df.columns
            if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def _extract_attack_segments(df: pd.DataFrame,
                              before_sec: int = 300, 
                              after_sec: int = 180) -> list[pd.DataFrame]:
    """
    Extract attack segments with context windows.
    Returns list of DataFrames, each containing one attack + context.
    """
    if "label" not in df.columns or not (df["label"] > 0).any():
        return []

    # Find contiguous attack blocks
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

    # Extract segments with context
    segments = []
    for start, end in blocks:
        lo = max(0, start - before_sec)
        hi = min(len(df) - 1, end + after_sec)
        segment = df.iloc[lo:hi + 1].copy().reset_index(drop=True)
        
        # Add scenario label for this segment (based on attack type)
        attack_rows = segment[segment['label'] > 0]
        if len(attack_rows) > 0:
            # Use the first attack row's type (all attacks in segment are same type)
            attack_type = attack_rows.iloc[0]['attack_type']
            combination = attack_rows.iloc[0].get('combination', 'no')
            segment['scenario_label'] = get_attack_label(attack_type, combination)
        else:
            segment['scenario_label'] = 0
        
        segments.append(segment)

    return segments


def create_forecasting_windows(
    df: pd.DataFrame,
    sensor_cols: list[str],
    input_len: int,
    target_len: int,
    stride: int,
    scaler: Optional[StandardScaler] = None,
    episode_id: str = ""
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Create sliding windows for time series forecasting.
    
    For each window:
    - X: past input_len timesteps (input to model)
    - y: next target_len timesteps (what to predict)
    - attack_flags: 1 if ANY attack occurs in the target window (binary)
    - scenario_labels: scenario class for conditioning (0=normal, 1-4=attack types)
    
    Returns:
        X: (n_windows, input_len, n_features)
        y: (n_windows, target_len, n_features)
        attack_flags: (n_windows,) - binary
        scenario_labels: (n_windows,) - multi-class (0-4)
        metadata: List of window identifiers
    """
    total_length = input_len + target_len
    
    if len(df) < total_length:
        return (
            np.empty((0, input_len, len(sensor_cols)), dtype=np.float32),
            np.empty((0, target_len, len(sensor_cols)), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            []
        )
    
    # Extract sensor data
    if scaler:
        X_raw = scaler.transform(df[sensor_cols].values.astype(np.float32))
    else:
        X_raw = df[sensor_cols].values.astype(np.float32)
    
    windows_X = []
    windows_y = []
    attack_flags = []
    scenario_labels = []
    metadata = []
    
    for i in range(0, len(df) - total_length + 1, stride):
        # Input: past input_len steps
        X_window = X_raw[i:i + input_len]
        
        # Target: next target_len steps
        y_window = X_raw[i + input_len:i + input_len + target_len]
        
        # Check if attack occurs in the target window
        attack_in_target = (df['label'].iloc[i + input_len:i + input_len + target_len] > 0).any()
        
        # Get scenario label for this window (from the attack segment, or 0 for normal)
        if 'scenario_label' in df.columns:
            # Use the scenario label from the segment (same for all rows in attack segment)
            window_scenario = df['scenario_label'].iloc[i + input_len] if attack_in_target else 0
        else:
            window_scenario = 0
        
        windows_X.append(X_window)
        windows_y.append(y_window)
        attack_flags.append(1 if attack_in_target else 0)
        scenario_labels.append(window_scenario)
        metadata.append(f"{episode_id}_pos{i}")
    
    return (
        np.array(windows_X, dtype=np.float32),
        np.array(windows_y, dtype=np.float32),
        np.array(attack_flags, dtype=np.int32),
        np.array(scenario_labels, dtype=np.int32),
        metadata
    )


# ── main ──────────────────────────────────────────────────────────────────────

def load_and_prepare_episodic(
    processed_dir: str = PROCESSED_DIR,
    output_dir: str = OUTPUT_DIR,
    train_ratio: float = TRAIN_RATIO,
    attack_split_ratio: float = ATTACK_SPLIT_RATIO,
    before_attack_sec: int = BEFORE_ATTACK_SEC,
    after_attack_sec: int = AFTER_ATTACK_SEC,
    random_seed: int = RANDOM_SEED,
    input_len: int = INPUT_LEN,
    target_len: int = TARGET_LEN,
    stride: int = STRIDE,
    save_windows: bool = True
) -> Dict:
    """
    Load data with train4 COMPLETELY HELD OUT for testing causal generalization.
    
    TRAIN (never sees train4 or test attacks):
    ├── train1, train2, train3 (80% train / 20% val)
    └── 80% of attack segments (from test1/test2)
    
    TEST (truly unseen):
    ├── train4 (100%) → Tests cross-period generalization of normal behavior
    └── 20% of attack segments → Tests attack trajectory prediction
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(random_seed)

    # ── Step 1: Load train1, train2, train3 ONLY ──────────────────────────────
    print("\n" + "="*70)
    print("[Step 1] Loading train1, train2, train3 (train4 is HELD OUT for testing)")
    print("="*70)
    
    train_episodes = []      # 80% of train1-3
    val_episodes = []        # 20% of train1-3
    sensor_cols: list[str] = []
    
    for n in range(1, 4):  # ONLY train1, train2, train3
        df = _load(os.path.join(processed_dir, f"train{n}.csv"))
        
        if not sensor_cols:
            sensor_cols = _sensor_cols(df)
            print(f"  Sensor features: {len(sensor_cols)}")
            print(f"  Sample sensors: {sensor_cols[:5]}")
        
        # Add scenario_label column (all 0 for normal data)
        df['scenario_label'] = 0
        
        # Temporal split within episode
        cut = int(len(df) * train_ratio)
        train_part = df.iloc[:cut].copy()
        val_part = df.iloc[cut:].copy()
        
        # Add episode metadata
        train_part['episode_id'] = f"train{n}"
        val_part['episode_id'] = f"train{n}"
        
        train_episodes.append(train_part)
        val_episodes.append(val_part)
        
        print(f"  train{n}: {len(df):,} rows -> train: {len(train_part):,} ({train_ratio:.0%}), val: {len(val_part):,} ({(1-train_ratio):.0%})")
    
    # ── Step 2: Load train4 as TEST (completely unseen) ───────────────────────
    print("\n" + "="*70)
    print("[Step 2] Loading train4 as TEST (completely held out for causal generalization)")
    print("="*70)
    
    train4_df = _load(os.path.join(processed_dir, f"train4.csv"))
    train4_df['episode_id'] = "train4_TEST"
    train4_df['scenario_label'] = 0  # Normal data
    print(f"  train4: {len(train4_df):,} rows (COMPLETELY UNSEEN during training)")
    
    # ── Step 3: Extract attack segments from test1 + test2 ────────────────────
    print("\n" + "="*70)
    print("[Step 3] Extracting attack segments from test1, test2")
    print("="*70)
    
    all_attack_segments: list[pd.DataFrame] = []
    
    for n in range(1, 3):
        df = _load(os.path.join(processed_dir, f"test{n}.csv"))
        segments = _extract_attack_segments(df, before_attack_sec, after_attack_sec)
        
        for seg_idx, segment in enumerate(segments):
            segment['episode_id'] = f"test{n}_segment{seg_idx}"
            segment['source_file'] = f"test{n}"
            all_attack_segments.append(segment)
            
            attack_types = segment[segment['label'] > 0]['attack_type'].unique()
            scenario_label = segment['scenario_label'].iloc[0] if len(segment) > 0 else 0
            print(f"  test{n}_segment{seg_idx}: {len(segment):,} rows, attack_types={attack_types}, scenario_label={scenario_label}")
        
        print(f"  test{n}: total {len(segments)} segments")
    
    print(f"  Total attack segments: {len(all_attack_segments)}")
    
    # ── Step 4: Split attack segments (80% train, 20% test) ───────────────────
    print("\n" + "="*70)
    print(f"[Step 4] Splitting attack segments {attack_split_ratio:.0%} train / {1-attack_split_ratio:.0%} test")
    print("="*70)
    
    if len(all_attack_segments) == 0:
        attack_train_segments = []
        attack_test_segments = []
        print("  WARNING: No attack segments found")
    elif len(all_attack_segments) == 1:
        attack_train_segments = all_attack_segments
        attack_test_segments = []
        print("  Only 1 segment — all to train")
    else:
        attack_train_segments, attack_test_segments = train_test_split(
            all_attack_segments,
            train_size=attack_split_ratio,
            random_state=random_seed,
            shuffle=True
        )
        print(f"  Attack train: {len(attack_train_segments)} segments (will be used with train1-3)")
        print(f"  Attack test:  {len(attack_test_segments)} segments (will be used with train4)")
    
    # ── Step 5: Fit scaler on NORMAL data from train1-3 ONLY ──────────────────
    print("\n" + "="*70)
    print("[Step 5] Fitting scaler on normal data from train1-3 ONLY")
    print("="*70)
    
    all_normal_data = pd.concat(
        [ep[ep['label'] == 0] for ep in train_episodes],
        ignore_index=True
    )
    print(f"  Total normal samples for fitting: {len(all_normal_data):,} (from train1, train2, train3)")
    
    scaler = StandardScaler()
    scaler.fit(all_normal_data[sensor_cols].values.astype(np.float32))
    
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to {scaler_path}")
    print(f"  NOTE: train4 was NOT used for fitting scaler!")
    
    # ── Step 6: Create forecasting windows for each split ─────────────────────
    print("\n" + "="*70)
    print(f"[Step 6] Creating forecasting windows (input={input_len}s, target={target_len}s, stride={stride}s)")
    print("="*70)
    
    # Process training normal episodes (train1-3, 80% portions)
    print("\n  TRAIN NORMAL (from train1-3, 80% portions):")
    train_normal_windows = []
    for episode in train_episodes:
        ep_id = episode['episode_id'].iloc[0]
        X, y, attack_flags, scenario_labels, meta = create_forecasting_windows(
            episode, sensor_cols, input_len, target_len, stride, scaler, ep_id
        )
        if len(X) > 0:
            train_normal_windows.append({
                'X': X, 'y': y, 'attack_flags': attack_flags,
                'scenario_labels': scenario_labels,
                'metadata': meta, 'episode_id': ep_id
            })
            n_attack_windows = attack_flags.sum()
            unique_scenarios = np.unique(scenario_labels)
            print(f"    {ep_id}: {len(X)} windows (attack_in_target={n_attack_windows}, scenarios={unique_scenarios})")
    
    # Process validation normal episodes (train1-3, 20% portions)
    print("\n  VALIDATION NORMAL (from train1-3, 20% portions):")
    val_normal_windows = []
    for episode in val_episodes:
        ep_id = episode['episode_id'].iloc[0]
        X, y, attack_flags, scenario_labels, meta = create_forecasting_windows(
            episode, sensor_cols, input_len, target_len, stride, scaler, ep_id
        )
        if len(X) > 0:
            val_normal_windows.append({
                'X': X, 'y': y, 'attack_flags': attack_flags,
                'scenario_labels': scenario_labels,
                'metadata': meta, 'episode_id': ep_id
            })
            n_attack_windows = attack_flags.sum()
            print(f"    {ep_id}: {len(X)} windows (attack_in_target={n_attack_windows})")
    
    # Process attack training segments
    print("\n  TRAIN ATTACK (80% of attack segments):")
    attack_train_windows = []
    for episode in attack_train_segments:
        ep_id = episode['episode_id'].iloc[0]
        X, y, attack_flags, scenario_labels, meta = create_forecasting_windows(
            episode, sensor_cols, input_len, target_len, stride, scaler, ep_id
        )
        if len(X) > 0:
            attack_train_windows.append({
                'X': X, 'y': y, 'attack_flags': attack_flags,
                'scenario_labels': scenario_labels,
                'metadata': meta, 'episode_id': ep_id
            })
            n_attack_windows = attack_flags.sum()
            unique_scenarios = np.unique(scenario_labels)
            print(f"    {ep_id}: {len(X)} windows (attack_in_target={n_attack_windows}, scenarios={unique_scenarios})")
    
    # Process TEST normal (train4 - completely unseen!)
    print("\n  TEST NORMAL (train4 - COMPLETELY UNSEEN during training):")
    test_normal_windows = []
    X_test_normal, y_test_normal, attack_flags_normal, scenario_labels_normal, meta_test_normal = create_forecasting_windows(
        train4_df, sensor_cols, input_len, target_len, stride, scaler, "train4_TEST"
    )
    if len(X_test_normal) > 0:
        test_normal_windows.append({
            'X': X_test_normal, 'y': y_test_normal, 
            'attack_flags': attack_flags_normal,
            'scenario_labels': scenario_labels_normal,
            'metadata': meta_test_normal, 'episode_id': "train4_TEST"
        })
        n_attack_windows = attack_flags_normal.sum()
        print(f"    train4_TEST: {len(X_test_normal)} windows (attack_in_target={n_attack_windows}) - CAUSAL GENERALIZATION TEST")
    
    # Process TEST attack (20% of attack segments)
    print("\n  TEST ATTACK (20% of attack segments):")
    attack_test_windows = []
    for episode in attack_test_segments:
        ep_id = episode['episode_id'].iloc[0]
        X, y, attack_flags, scenario_labels, meta = create_forecasting_windows(
            episode, sensor_cols, input_len, target_len, stride, scaler, ep_id
        )
        if len(X) > 0:
            attack_test_windows.append({
                'X': X, 'y': y, 'attack_flags': attack_flags,
                'scenario_labels': scenario_labels,
                'metadata': meta, 'episode_id': ep_id
            })
            n_attack_windows = attack_flags.sum()
            unique_scenarios = np.unique(scenario_labels)
            print(f"    {ep_id}: {len(X)} windows (attack_in_target={n_attack_windows}, scenarios={unique_scenarios})")
    
    # ── Step 7: Combine all windows (always create arrays) ────────────────────
    print("\n" + "="*70)
    print("[Step 7] Combining windows into final arrays")
    print("="*70)
    
    # Combine training windows (normal + attack)
    if train_normal_windows or attack_train_windows:
        all_train_windows = train_normal_windows + attack_train_windows
        X_train = np.concatenate([w['X'] for w in all_train_windows], axis=0)
        y_train = np.concatenate([w['y'] for w in all_train_windows], axis=0)
        attack_train_labels = np.concatenate([w['attack_flags'] for w in all_train_windows], axis=0)
        scenario_train = np.concatenate([w['scenario_labels'] for w in all_train_windows], axis=0)
    else:
        X_train = np.empty((0, input_len, len(sensor_cols)), dtype=np.float32)
        y_train = np.empty((0, target_len, len(sensor_cols)), dtype=np.float32)
        attack_train_labels = np.empty(0, dtype=np.int32)
        scenario_train = np.empty(0, dtype=np.int32)
    
    # Validation windows (normal only)
    if val_normal_windows:
        X_val = np.concatenate([w['X'] for w in val_normal_windows], axis=0)
        y_val = np.concatenate([w['y'] for w in val_normal_windows], axis=0)
        attack_val_labels = np.concatenate([w['attack_flags'] for w in val_normal_windows], axis=0)
        scenario_val = np.concatenate([w['scenario_labels'] for w in val_normal_windows], axis=0)
    else:
        X_val = np.empty((0, input_len, len(sensor_cols)), dtype=np.float32)
        y_val = np.empty((0, target_len, len(sensor_cols)), dtype=np.float32)
        attack_val_labels = np.empty(0, dtype=np.int32)
        scenario_val = np.empty(0, dtype=np.int32)
    
    # Test windows (normal from train4 + attack segments)
    all_test_windows = test_normal_windows + attack_test_windows
    if all_test_windows:
        X_test = np.concatenate([w['X'] for w in all_test_windows], axis=0)
        y_test = np.concatenate([w['y'] for w in all_test_windows], axis=0)
        attack_test_labels = np.concatenate([w['attack_flags'] for w in all_test_windows], axis=0)
        scenario_test = np.concatenate([w['scenario_labels'] for w in all_test_windows], axis=0)
    else:
        X_test = np.empty((0, input_len, len(sensor_cols)), dtype=np.float32)
        y_test = np.empty((0, target_len, len(sensor_cols)), dtype=np.float32)
        attack_test_labels = np.empty(0, dtype=np.int32)
        scenario_test = np.empty(0, dtype=np.int32)
    
    # Get number of unique scenarios
    all_scenarios = np.concatenate([scenario_train, scenario_val, scenario_test])
    n_scenarios = int(all_scenarios.max()) + 1 if len(all_scenarios) > 0 else 1
    
    # ── Save to disk if requested ────────────────────────────────────────────
    if save_windows:
        print("\n" + "="*70)
        print("Saving processed data")
        print("="*70)
        
        np.savez_compressed(
            os.path.join(output_dir, "train_data.npz"),
            X=X_train, y=y_train, 
            attack_labels=attack_train_labels,
            scenario_labels=scenario_train
        )
        np.savez_compressed(
            os.path.join(output_dir, "val_data.npz"),
            X=X_val, y=y_val,
            attack_labels=attack_val_labels,
            scenario_labels=scenario_val
        )
        np.savez_compressed(
            os.path.join(output_dir, "test_data.npz"),
            X=X_test, y=y_test,
            attack_labels=attack_test_labels,
            scenario_labels=scenario_test
        )
        
        # Also save metadata
        metadata = {
            'sensor_cols': sensor_cols,
            'input_len': input_len,
            'target_len': target_len,
            'stride': stride,
            'n_features': len(sensor_cols),
            'n_scenarios': n_scenarios,
            'train_files_used': ['train1', 'train2', 'train3'],
            'test_files_used': ['train4'],
            'n_attack_segments_train': len(attack_train_segments),
            'n_attack_segments_test': len(attack_test_segments),
            'scenario_mapping': {
                0: 'normal',
                1: 'AP_no_combination',
                2: 'AP_with_combination',
                3: 'AE_no_combination',
                4: 'AE_with_combination'
            }
        }
        joblib.dump(metadata, os.path.join(output_dir, "metadata.pkl"))
        
        print(f"  Saved to {output_dir}/")
        print(f"    train_data.npz: X={X_train.shape}, y={y_train.shape}")
        print(f"    val_data.npz:   X={X_val.shape}, y={y_val.shape}")
        print(f"    test_data.npz:  X={X_test.shape}, y={y_test.shape}")
        print(f"    metadata.pkl:   saved (n_scenarios={n_scenarios})")
    
    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("FINAL SUMMARY - train4 COMPLETELY HELD OUT")
    print("="*70)
    
    print("\n  TRAIN SET (model sees this):")
    print(f"    - train1, train2, train3 (80% portions each)")
    print(f"    - {len(attack_train_segments)} attack segments (80% of total)")
    print(f"    - X_train shape: {X_train.shape}")
    print(f"    - Scenario distribution: {dict(zip(*np.unique(scenario_train, return_counts=True)))}")
    
    print("\n  VALIDATION SET (for early stopping/hyperparameters):")
    print(f"    - train1, train2, train3 (20% portions each)")
    print(f"    - X_val shape: {X_val.shape}")
    
    print("\n  TEST SET (truly unseen - FINAL EVALUATION ONLY):")
    print(f"    - train4 (100%) → Tests causal generalization across periods")
    print(f"    - {len(attack_test_segments)} attack segments (20% of total) → Tests attack prediction")
    print(f"    - X_test shape: {X_test.shape}")
    print(f"    - Scenario distribution: {dict(zip(*np.unique(scenario_test, return_counts=True)))}")
    
    print(f"\n  Number of scenario classes: {n_scenarios}")
    print("\n  ✓ train4 was NEVER seen during training")
    print("  ✓ train4 was NOT used for fitting the scaler")
    print("  ✓ If RMSE on train4 ≈ RMSE on val → model learned causal relationships")
    print("="*70)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'attack_train_labels': attack_train_labels,
        'scenario_train': scenario_train,
        'X_val': X_val,
        'y_val': y_val,
        'attack_val_labels': attack_val_labels,
        'scenario_val': scenario_val,
        'X_test': X_test,
        'y_test': y_test,
        'attack_test_labels': attack_test_labels,
        'scenario_test': scenario_test,
        'n_scenarios': n_scenarios,
        'sensor_cols': sensor_cols,
        'scaler': scaler,
        'input_len': input_len,
        'target_len': target_len,
        'stride': stride,
        'n_features': len(sensor_cols),
    }


if __name__ == "__main__":
    data = load_and_prepare_episodic(
        input_len=300,
        target_len=180,
        stride=60,
        save_windows=True
    )
    
    print("\n✓ Data ready for Transformer training!")
    print(f"  Train on: {data['X_train'].shape[0]:,} windows")
    print(f"  Validate on: {data['X_val'].shape[0]:,} windows")
    print(f"  Test on: {data['X_test'].shape[0]:,} windows")
    print(f"  Scenario classes: {data['n_scenarios']}")