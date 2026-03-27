"""
VAE — Layer 3: Attack Generator (baseline)

Trains on real attack windows from test1, generates synthetic attacks.
These feed into the Guided Generation pipeline (physics + detectability checks).

Eval: TSTR — generate synthetic attacks → train XGBoost → test on real test2.

Window size = 300s  (from window_size_analysis notebook).
Pass norm from twin() if the Digital Twin was already run — keeps the
normalisation consistent across all layers.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.prep import generate

Path("outputs").mkdir(exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
# norm = twin_data["norm"]   # optional: pass from twin() for cross-layer consistency
data = generate(norm=None, window_len=300, stride=60)

attack_windows = data["attack_windows"]   # (N, 300, F)  test1 attack windows
normal_windows = data["normal_windows"]   # (M, 300, F)  train1 normal windows
norm           = data["norm"]

F = attack_windows.shape[2]   # dynamic sensor count after constant deletion

# ── Flat rows for VAE (row-level generation) ──────────────────────────────────
# The VAE operates on individual timestep rows, not full windows.
# Flatten (N, 300, F) → (N*300, F) to get per-timestep attack rows.
attack_rows = attack_windows.reshape(-1, F)   # (N*300, F)
normal_rows = normal_windows.reshape(-1, F)   # (M*300, F)

print(f"VAE data ready:")
print(f"  attack_rows {attack_rows.shape}  (test1, attack==1, flattened)")
print(f"  normal_rows {normal_rows.shape}  (train1, normal, flattened)")
print(f"  F = {F}  (sensor count after constant deletion)")

# ── Build and train VAE here ──────────────────────────────────────────────────
