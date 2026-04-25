import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add the generator directory to sys.path to import core
sys.path.append(str(Path(r"C:\Users\PC GAMING\Desktop\new_ai\hai-digital-twin\generator")))
from core import (
    load_bundle,
    load_replay,
    closed_loop_rollout,
    INPUT_LEN,
    TARGET_LEN,
    PV_COLS,
    default_paths
)

def main():
    print("Initializing Generator...")
    paths = default_paths()
    # Ensure paths are correct for this environment
    ckpt_dir = Path(r"C:\Users\PC GAMING\Desktop\new_ai\hai-digital-twin\generator\weights")
    split_dir = Path(r"C:\Users\PC GAMING\Desktop\new_ai\hai-digital-twin\generator\scalers")
    test_csv = Path(r"C:\Users\PC GAMING\Desktop\new_ai\processed\test1.csv")

    bundle = load_bundle(ckpt_dir, split_dir)
    src = load_replay(test_csv, bundle.scalers)
    
    # We will generate N samples for each attack scenario
    samples_per_scenario = 50 
    scenarios = [1, 2, 3] # AP_no, AP_with, AE_no
    
    all_synthetic_rows = []
    
    # Feature columns used by the classifier (we'll need to match these)
    # For now, we generate the full 133-column state
    sensor_cols = bundle.scalers.sensor_cols
    
    print(f"Generating synthetic data (approx {samples_per_scenario * len(scenarios) * TARGET_LEN} rows)...")
    
    for scen in scenarios:
        print(f"  Simulating Scenario {scen}...")
        for i in range(samples_per_scenario):
            # Pick a random cursor that allows for 300s history and 180s future
            cursor = np.random.randint(INPUT_LEN, len(src) - TARGET_LEN - 1)
            
            rollout = closed_loop_rollout(bundle, src, cursor, scenario=scen)
            if rollout is None:
                continue
                
            # rollout['pv_physical'] is (180, 5)
            # rollout['x_cv_target_used'] is (180, 128) - this is already scaled
            # We need to unscale x_cv_target_used to match the CSV format
            
            # Reconstruct the 133-column row
            scaled_full = rollout['x_cv_target_used'].copy()
            # The indices for PVs in the 133 columns are s.pv_idx
            # The indices for the others are s.plant_in_idx
            # Wait, x_cv_target_used IS the full 133 columns with PVs at their positions?
            # No, looking at closed_loop_rollout code:
            # x_cv_tgt = scaled_local[INPUT_LEN:, bundle.scalers.plant_in_idx].copy()
            # It only has 128 columns.
            
            # Let's create the full 133 scaled matrix
            full_133_scaled = np.zeros((TARGET_LEN, 133), dtype=np.float32)
            full_133_scaled[:, bundle.scalers.plant_in_idx] = rollout['x_cv_target_used']
            full_133_scaled[:, bundle.scalers.pv_idx] = rollout['pv_scaled']
            
            # Inverse scale everything to physical units
            # plant_mean and plant_scale are (133,)
            physical_rows = full_133_scaled * bundle.scalers.plant_scale + bundle.scalers.plant_mean
            
            df_batch = pd.DataFrame(physical_rows, columns=sensor_cols)
            df_batch['label'] = 1
            df_batch['scenario'] = scen
            
            all_synthetic_rows.append(df_batch)
            
    # Also generate some Normal data (Scenario 0) to keep balance
    print("  Simulating Scenario 0 (Normal)...")
    for i in range(samples_per_scenario):
        cursor = np.random.randint(INPUT_LEN, len(src) - TARGET_LEN - 1)
        rollout = closed_loop_rollout(bundle, src, cursor, scenario=0)
        if rollout is None: continue
        
        full_133_scaled = np.zeros((TARGET_LEN, 133), dtype=np.float32)
        full_133_scaled[:, bundle.scalers.plant_in_idx] = rollout['x_cv_target_used']
        full_133_scaled[:, bundle.scalers.pv_idx] = rollout['pv_scaled']
        physical_rows = full_133_scaled * bundle.scalers.plant_scale + bundle.scalers.plant_mean
        
        df_batch = pd.DataFrame(physical_rows, columns=sensor_cols)
        df_batch['label'] = 0
        df_batch['scenario'] = 0
        all_synthetic_rows.append(df_batch)

    final_df = pd.concat(all_synthetic_rows, ignore_index=True)
    save_path = r"C:\Users\PC GAMING\Desktop\AI\HAI\synthetic_attacks.csv"
    final_df.to_csv(save_path, index=False)
    print(f"Success! Synthetic dataset saved to {save_path}")

if __name__ == "__main__":
    main()
