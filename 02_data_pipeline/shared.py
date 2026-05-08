"""
shared.py — Constants and utilities shared across training, evaluation, and detection scripts.

Import from here instead of redefining in each script:
    from shared import SCENARIO_NAMES, CTRL_LOOPS, EXTRA_CHANNELS, CTRL_HIDDEN_PER_LOOP
    from shared import augment_ctrl_data
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from config import LOOPS, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

# ── Scenario Labels ────────────────────────────────────────────────────────────

SCENARIO_NAMES: Dict[int, str] = {
    0: "Normal",
    1: "AP_no",
    2: "AP_with",
    3: "AE_no",
}

# ── Control Loop Constants ─────────────────────────────────────────────────────

CTRL_LOOPS: List[str] = ["PC", "LC", "FC", "TC", "CC"]

# Per-loop GRU hidden sizes (FC=128 for extra capacity: 6 inputs vs 3)
CTRL_HIDDEN_PER_LOOP: Dict[str, int] = {
    "PC": 64, "LC": 64, "FC": 128, "TC": 64, "CC": 64,
}

# Three causal channels per loop derived from the HAI causal graph:
#   L0 = direct sensor→actuator channel
#   L1 = actuator→sensor direct path
#   L2 = 2-hop physical path
EXTRA_CHANNELS: Dict[str, List[str]] = {
    "PC": ["P1_PCV02D", "P1_FT01",   "P1_TIT01"],
    "LC": ["P1_FT03",   "P1_FCV03D", "P1_PCV01D"],
    "FC": ["P1_PIT01",  "P1_LIT01",  "P1_TIT03"],
    "TC": ["P1_FT02",   "P1_PIT02",  "P1_TIT02"],
    "CC": ["P1_PP04D",  "P1_FCV03D", "P1_PCV02D"],
}


# ── Augmentation Utility ───────────────────────────────────────────────────────

def augment_ctrl_data(
    ctrl_data: Dict,
    sensor_cols: List[str],
    extra_channels: Optional[Dict[str, List[str]]] = None,
    data_dir: Optional[str] = None,
) -> None:
    """
    Append causal channels to each loop's X arrays in-place.

    Loads raw plant-scaled windows from disk and appends each extra column
    (re-normalised to the per-loop controller space) to X_{train,val,test}.

    Args:
        ctrl_data:      per-loop data dict (modified in-place)
        sensor_cols:    ordered list of sensor column names
        extra_channels: override EXTRA_CHANNELS if provided
        data_dir:       override PROCESSED_DATA_DIR if provided
    """
    if extra_channels is None:
        extra_channels = EXTRA_CHANNELS
    if data_dir is None:
        data_dir = PROCESSED_DATA_DIR

    plant_scaler = joblib.load(Path(data_dir) / "scaler.pkl")
    npz = {s: np.load(Path(data_dir) / f"{s}_data.npz")
           for s in ("train", "val", "test")}
    col_idx = {c: i for i, c in enumerate(sensor_cols)}

    for ln, extra_cols in extra_channels.items():
        added: List[str] = []
        for ec in extra_cols:
            if ec not in col_idx:
                logger.warning("Column %s not found for loop %s — skipping", ec, ln)
                continue
            ei = col_idx[ec]
            mean_e, scale_e = plant_scaler.mean_[ei], plant_scaler.scale_[ei]
            for split, arr in npz.items():
                raw = arr["X"][:, :, [ei]].astype(np.float32)
                ctrl_data[ln][f"X_{split}"] = np.concatenate(
                    [ctrl_data[ln][f"X_{split}"], (raw - mean_e) / scale_e], axis=-1
                )
            added.append(ec)
        layer_labels = ["L0", "L1", "L2"][: len(added)]
        logger.info(
            "%s: added %s → n_inputs=%d",
            ln,
            " + ".join(f"{l}={c}" for l, c in zip(layer_labels, added)),
            ctrl_data[ln]["X_train"].shape[-1],
        )
