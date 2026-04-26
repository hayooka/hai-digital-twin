"""
Data Pipeline - loads scaled/split windows and prepares them for model training.
"""

import numpy as np
import joblib
import os
from typing import Dict, Tuple, List, Optional
from config import LOOPS, PV_COLS, HAIEND_COLS, PROCESSED_DATA_DIR


def load_preprocessed_data(data_dir: Optional[str] = None) -> Dict:
    """Load saved .npz windows and metadata written by scaled_split.py."""
    if data_dir is None:
        data_dir = PROCESSED_DATA_DIR

    train = np.load(os.path.join(data_dir, 'train_data.npz'))
    val   = np.load(os.path.join(data_dir, 'val_data.npz'))
    test  = np.load(os.path.join(data_dir, 'test_data.npz'))
    meta  = joblib.load(os.path.join(data_dir, 'metadata.pkl'))

    plant_scaler = joblib.load(os.path.join(data_dir, 'scaler.pkl'))

    ctrl_scalers = {}
    for ln in LOOPS:
        path = os.path.join(data_dir, f'ctrl_scaler_{ln}.pkl')
        if os.path.exists(path):
            ctrl_scalers[ln] = joblib.load(path)

    return {
        'X_train': train['X'],       'y_train': train['y'],
        'scenario_train': train['scenario_labels'],
        'attack_train':   train['attack_labels'],
        'X_val':   val['X'],         'y_val':   val['y'],
        'scenario_val':   val['scenario_labels'],
        'attack_val':     val['attack_labels'],
        'X_test':  test['X'],        'y_test':  test['y'],
        'scenario_test':  test['scenario_labels'],
        'attack_test':    test['attack_labels'],
        'sensor_cols':  meta['sensor_cols'],
        'n_scenarios':  meta['n_scenarios'],
        'input_len':    meta['input_len'],
        'target_len':   meta['target_len'],
        'plant_scaler': plant_scaler,
        'ctrl_scalers': ctrl_scalers,
    }


def _feature_indices(sensor_cols: List[str], target_cols: List[str]) -> List[int]:
    """Return positional indices of target_cols within sensor_cols (skip missing)."""
    col_idx = {c: i for i, c in enumerate(sensor_cols)}
    return [col_idx[c] for c in target_cols if c in col_idx]


def prepare_plant_data(raw: Dict, split: str = 'train') -> Tuple:
    """
    Split windows into plant inputs (all non-PV features) and plant outputs.

    Teacher forcing: pv_teacher == pv_target — both carry the true PV trajectory
    so the training loop can choose its own scheduled-sampling ratio.

    Returns:
        x_cv          : (N, input_len,  n_plant_in)   non-PV input window
        x_cv_target   : (N, target_len, n_plant_in)   non-PV target window
        pv_init       : (N, n_pv)                     PV at the last input step
        pv_teacher    : (N, target_len, n_pv)         true PV for teacher forcing
        pv_target     : (N, target_len, n_pv)         PV loss target
        haiend_target : (N, target_len, n_haiend)     internal PLC signals (AE attack footprint)
        scenario      : (N,)
    """
    sensor_cols  = raw['sensor_cols']
    pv_idx       = _feature_indices(sensor_cols, PV_COLS)
    haiend_idx   = _feature_indices(sensor_cols, HAIEND_COLS)
    non_pv_idx   = [i for i in range(len(sensor_cols)) if i not in pv_idx]

    X        = raw[f'X_{split}']
    y        = raw[f'y_{split}']
    scenario = raw[f'scenario_{split}']

    x_cv          = X[:, :, non_pv_idx]
    x_cv_target   = y[:, :, non_pv_idx]
    pv_init       = X[:, -1, :][:, pv_idx]
    pv_teacher    = y[:, :, pv_idx]
    pv_target     = y[:, :, pv_idx]
    haiend_target = y[:, :, haiend_idx] if haiend_idx else np.zeros(
        (y.shape[0], y.shape[1], 0), dtype=np.float32)

    return x_cv, x_cv_target, pv_init, pv_teacher, pv_target, haiend_target, scenario


def prepare_controller_data(raw: Dict, loop_name: str, split: str = 'train') -> Tuple:
    """
    Extract [SP, PV] input and CV output windows for one control loop.

    Returns:
        X_ctrl   : (N, input_len, 2)   [SP, PV] history
        y_ctrl   : (N, target_len, 1)  CV future trajectory
        scenario : (N,)
    """
    sensor_cols = raw['sensor_cols']
    loop = LOOPS[loop_name]

    sp_idx = _feature_indices(sensor_cols, [loop.sp])
    pv_idx = _feature_indices(sensor_cols, [loop.pv])
    cv_idx = _feature_indices(sensor_cols, [loop.cv])

    X        = raw[f'X_{split}']
    y        = raw[f'y_{split}']
    scenario = raw[f'scenario_{split}']

    if not (sp_idx and pv_idx and cv_idx):
        return (
            np.empty((0, X.shape[1], 3), dtype=np.float32),
            np.empty((0, y.shape[1], 1), dtype=np.float32),
            scenario[:0],
        )

    # Extract from plant-scaled windows: [SP, PV, CV] (3 channels)
    X_ctrl = X[:, :, sp_idx + pv_idx + cv_idx]   # (N, input_len, 3)
    y_ctrl = y[:, :, cv_idx]                      # (N, target_len, 1)

    # Re-scale to per-loop controller space:
    #   1. Inverse plant scaler  →  raw sensor values
    #   2. Apply controller scaler (fitted on [SP, PV, CV] in that order)
    plant_scaler = raw.get('plant_scaler')
    ctrl_info    = raw.get('ctrl_scalers', {}).get(loop_name)
    if plant_scaler is not None and ctrl_info is not None:
        cs             = ctrl_info['scaler']
        pm, ps         = plant_scaler.mean_, plant_scaler.scale_
        sp_pv_cv_idx   = sp_idx + pv_idx + cv_idx
        X_ctrl = (X_ctrl * ps[sp_pv_cv_idx] + pm[sp_pv_cv_idx] - cs.mean_[:3]) / cs.scale_[:3]
        y_ctrl = (y_ctrl * ps[cv_idx]        + pm[cv_idx]        - cs.mean_[2:]) / cs.scale_[2:]

    return X_ctrl, y_ctrl, scenario


def get_plant_input_output_dims(sensor_cols: List[str]) -> Tuple[int, int, int]:
    """Return (n_plant_in, n_pv, n_haiend)."""
    pv_idx     = _feature_indices(sensor_cols, PV_COLS)
    haiend_idx = _feature_indices(sensor_cols, HAIEND_COLS)
    n_pv       = len(pv_idx)
    n_haiend   = len(haiend_idx)
    n_plant_in = len(sensor_cols) - n_pv
    return n_plant_in, n_pv, n_haiend


def load_and_prepare_data(data_dir: Optional[str] = None) -> Dict:
    """
    Load scaled/split windows and organize into plant + controller dicts.

    Returns:
        plant    — train/val/test arrays ready for the plant (seq2seq) model
        ctrl     — per-loop train + val arrays for each controller model
        metadata — sensor list, sequence lengths, scenario count
    """
    raw         = load_preprocessed_data(data_dir)
    sensor_cols = raw['sensor_cols']

    # Plant splits
    x_cv,      x_cv_tgt,      pv_init,      pv_teacher,      pv_target,      haiend_tr,   sc_train = prepare_plant_data(raw, 'train')
    x_cv_val,  x_cv_tgt_val,  pv_init_val,  pv_teacher_val,  pv_target_val,  haiend_val,  sc_val   = prepare_plant_data(raw, 'val')
    x_cv_test, x_cv_tgt_test, pv_init_test, pv_teacher_test, pv_target_test, haiend_test, sc_test  = prepare_plant_data(raw, 'test')

    n_plant_in, n_pv, n_haiend = get_plant_input_output_dims(sensor_cols)

    # Controller splits (all five loops, train + val + test)
    ctrl_data = {}
    for ln in LOOPS:
        X_tr, y_tr, sc_tr   = prepare_controller_data(raw, ln, 'train')
        X_vl, y_vl, sc_vl   = prepare_controller_data(raw, ln, 'val')
        X_te, y_te, sc_te   = prepare_controller_data(raw, ln, 'test')
        ctrl_data[ln] = {
            'X_train': X_tr, 'y_train': y_tr, 'scenario_train': sc_tr,
            'X_val':   X_vl, 'y_val':   y_vl, 'scenario_val':   sc_vl,
            'X_test':  X_te, 'y_test':  y_te, 'scenario_test':  sc_te,
        }

    return {
        'plant': {
            'X_train': x_cv,           'X_cv_target_train': x_cv_tgt,
            'pv_init_train': pv_init,  'pv_teacher_train': pv_teacher,
            'pv_target_train': pv_target, 'haiend_target_train': haiend_tr,
            'scenario_train': sc_train,
            'attack_train': raw['attack_train'],
            'X_val':   x_cv_val,       'X_cv_target_val': x_cv_tgt_val,
            'pv_init_val': pv_init_val,'pv_teacher_val': pv_teacher_val,
            'pv_target_val': pv_target_val, 'haiend_target_val': haiend_val,
            'scenario_val': sc_val,
            'attack_val': raw['attack_val'],
            'X_test':  x_cv_test,      'X_cv_target_test': x_cv_tgt_test,
            'pv_init_test': pv_init_test,'pv_teacher_test': pv_teacher_test,
            'pv_target_test': pv_target_test, 'haiend_target_test': haiend_test,
            'scenario_test': sc_test,
            'attack_test': raw['attack_test'],
            'n_plant_in': n_plant_in,
            'n_pv':       n_pv,
            'n_haiend':   n_haiend,
        },
        'ctrl': ctrl_data,
        'metadata': {
            'sensor_cols': sensor_cols,
            'n_scenarios': raw['n_scenarios'],
            'input_len':   raw['input_len'],
            'target_len':  raw['target_len'],
        },
    }


def print_preprocessing_summary(data: Dict):
    """Print a summary of all prepared splits and controller data."""
    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)

    p = data['plant']
    print("\n  PLANT DATA:")
    print(f"    Train : X={p['X_train'].shape},  target={p['pv_target_train'].shape}")
    print(f"    Val   : X={p['X_val'].shape},    target={p['pv_target_val'].shape}")
    print(f"    Test  : X={p['X_test'].shape},   target={p['pv_target_test'].shape}")
    print(f"    Input dim : {p['n_plant_in']}  |  Output dim (PV): {p['n_pv']}  |  HAIEND targets: {p['n_haiend']}")

    print("\n  SCENARIO DISTRIBUTION:")
    print(f"    Train : {np.bincount(p['scenario_train'])}")
    print(f"    Val   : {np.bincount(p['scenario_val'])}")
    print(f"    Test  : {np.bincount(p['scenario_test'])}")

    print("\n  CONTROLLER DATA (train / val windows):")
    for ln, cd in data['ctrl'].items():
        print(f"    {ln}: X_train={cd['X_train'].shape}, X_val={cd['X_val'].shape}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    data = load_and_prepare_data()
    print_preprocessing_summary(data)