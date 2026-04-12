"""
Configuration file for the HAI Digital Twin
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import os

# ============================================================
# Loop Definitions
# ============================================================

@dataclass
class LoopConfig:
    name: str
    sp: str
    cv: str
    cv_fb: Optional[str]
    pv: str
    sp_range: Tuple[float, float]
    pv_range: Tuple[float, float]
    description: str = ""

LOOPS: Dict[str, LoopConfig] = {
    'PC': LoopConfig('Pressure Control',
        sp='x1001_05_SETPOINT_OUT', cv='P1_PCV01D', cv_fb='P1_PCV01Z', pv='P1_PIT01',
        sp_range=(0.1,0.3), pv_range=(0,10)),
    'LC': LoopConfig('Level Control',
        sp='x1002_07_SETPOINT_OUT', cv='P1_LCV01D', cv_fb='P1_LCV01Z', pv='P1_LIT01',
        sp_range=(300,500), pv_range=(0,720)),
    'FC': LoopConfig('Flow Control',
        sp='x1002_08_SETPOINT_OUT', cv='P1_FCV03D', cv_fb='P1_FCV03Z', pv='P1_FT03Z',
        sp_range=(900,1100), pv_range=(0,2500)),
    'TC': LoopConfig('Temperature Control',
        sp='x1003_18_SETPOINT_OUT', cv='P1_FCV01D', cv_fb='P1_FCV01Z', pv='P1_TIT01',
        sp_range=(25,30), pv_range=(0,50)),
    'CC': LoopConfig('Cooling Control',
        sp='P1_PP04SP', cv='P1_PP04', cv_fb=None, pv='P1_TIT03',
        sp_range=(26,30), pv_range=(0,50)),
}

SP_COLS = [LOOPS[k].sp for k in LOOPS]
CV_COLS = [LOOPS[k].cv for k in LOOPS]
PV_COLS = [LOOPS[k].pv for k in LOOPS]

# ============================================================
# HAI Signals Definitions
# ============================================================

HAI_AUX = [
    'P1_FCV02D', 'P1_FCV02Z', 'P1_PCV02D', 'P1_PCV02Z',
    'P1_FT01', 'P1_FT01Z', 'P1_FT02', 'P1_FT02Z', 'P1_FT03',
    'P1_PIT02', 'P1_TIT02', 'P1_PP04D',
    'x1001_15_ASSIGN_OUT',
    'x1003_10_SETPOINT_OUT', 'x1003_24_SUM_OUT',
]

HAIEND_COLS = [
    '1001.13-OUT', '1001.14-OUT', '1001.15-OUT',
    '1001.16-OUT', '1001.17-OUT', '1001.20-OUT',
    '1002.9-OUT', '1002.20-OUT', '1002.21-OUT', '1002.30-OUT', '1002.31-OUT',
    '1003.5-OUT', '1003.10-OUT', '1003.11-OUT', '1003.17-OUT',
    '1003.23-OUT', '1003.24-OUT', '1003.25-OUT', '1003.26-OUT',
    '1003.29-OUT', '1003.30-OUT',
    '1020.13-OUT', '1020.14-OUT', '1020.15-OUT',
    '1020.18-OUT', '1020.20-OUT',
    'DM-PP04-D', 'DM-PP04-AO',
    'DM-TWIT-04', 'DM-TWIT-05',
    'DM-AIT-DO', 'DM-AIT-PH',
    'GATEOPEN',
    'DM-FT01Z', 'DM-FT02Z', 'DM-FT03Z',
]

# ============================================================
# Data Split Configuration (from episodic data loader)
# ============================================================

# Data timeline (HAI 23.05)
DATA_TIMELINE = {
    'train1': 'August 12, 2022 (starting 16:25)',
    'train2': 'August 13, 2022 (starting 00:25)',
    'train3': 'August 17, 2022 (starting 08:36)',
    'train4': 'HELD OUT - Causal generalization test',
}

# Split ratios
TRAIN3_RATIO = 0.30  # First 30% of train3 to train, 70% to validation
ATTACK_SPLIT_RATIO = 0.80  # 80% attacks to train, 20% to test

# Sequence lengths
INPUT_LEN = 300   # Input sequence length (5 minutes at 1Hz)
TARGET_LEN = 180  # Target sequence length (3 minutes)
STRIDE = 60       # Sliding window stride (1 minute)

# Attack scenario labels (from get_scenario_label function)
SCENARIO_MAPPING = {
    0: 'normal',
    1: 'AP_no_combination',   # Actuator Pollution without combustion
    2: 'AP_with_combination', # Actuator Pollution with combustion
    3: 'AE_no_combination',   # Actuator Emulation without combustion
}

# Attack distribution (HAI 23.05 documentation)
ATTACK_DISTRIBUTION = {
    'AP_no': {'total': 34, 'train': 27, 'test': 7},
    'AP_with': {'total': 10, 'train': 8, 'test': 2},
    'AE_no': {'total': 8, 'train': 6, 'test': 2},
}

N_SCENARIOS = 4  # 0=normal, 1=AP_no, 2=AP_with, 3=AE_no

# ============================================================
# Model Architecture Configuration
# ============================================================

@dataclass
class ModelConfig:
    # GRU architecture
    ctrl_hidden: int = 64
    ctrl_layers: int = 2
    plant_hidden: int = 256
    plant_layers: int = 2
    
    # Transformer architecture
    transformer_d_model: int = 128
    transformer_n_heads: int = 8
    transformer_n_layers: int = 3
    transformer_emb_dim: int = 32
    
    # Training
    seq_len: int = INPUT_LEN
    batch_size: int = 256
    lr: float = 1e-3
    plant_lr: float = 1e-4
    max_epochs: int = 150
    patience: int = 20
    
    # Scheduled sampling
    ss_start_epoch: int = 10
    ss_end_epoch: int = 100
    ss_max_ratio: float = 0.5
    
    # CC classifier
    cc_threshold: float = 0.5
    
    # Validation
    rollout_horizons: List[int] = field(default_factory=lambda: [300, 600, 900, 1800])
    nrmse_threshold: float = 0.10

CONFIG = ModelConfig()

# ============================================================
# Paths
# ============================================================

PROCESSED_DATA_DIR = './outputs/scaled_split/'  # Output from episodic data loader
MODELS_DIR = './models/'
RESULTS_DIR = './results/'
FIGURES_DIR = './figures/'

# Create directories
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("✅ Config loaded successfully!")
print(f"   N_SCENARIOS: {N_SCENARIOS}")
print(f"   SCENARIO_MAPPING: {SCENARIO_MAPPING}")