"""
Conditional DDPM — Attack Generator (primary)

Data:   attack windows from test1+2 where attack==1
        Conditioned on attack_type (FDI / Replay / DoS) and physics bounds
Input:  (window_len, 277) — same shape as Transformer input
Output: novel attack windows clipped to physics-valid sensor bounds

Eval:   TSTR — generate synthetic attacks → train XGBoost → test on real test1+2

TIP: Pass norm from twin() if Digital Twin was already run,
     to keep normalization consistent across all models.
"""
from __future__ import annotations

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # add project root to path
from utils.prep import generate

# ── Constants ─────────────────────────────────────────────────────────────────

N_FEAT       = None           # set dynamically from generate() data shape
WINDOW_LEN   = 300            # from window_size_analysis notebook recommendation
N_ATTACK_TYPES = 3            # FDI=0, Replay=1, DoS=2

# DDPM schedule
T_STEPS      = 1000           # diffusion timesteps
BETA_START   = 1e-4
BETA_END     = 0.02

# Model hyperparameters
D_MODEL      = 256
N_HEADS      = 8
N_LAYERS     = 4
FFN_DIM      = 512
DROPOUT      = 0.1

# Training hyperparameters
EPOCHS       = 100
BATCH        = 32
LR           = 2e-4

Path("outputs").mkdir(exist_ok=True)


