"""
Diffusion Model — Attack Generator (primary)

Data:   attack_rows: test1+2 where attack==1  shape (N, 277)
        normal_rows: train1 (all benign)       shape (M, 277)  ← class conditioning
Eval:   TSTR — generate synthetic attacks → train XGBoost → test on real test1+2

TIP: Pass norm from twin() if Digital Twin was already run,
     to keep normalization consistent across all models.
"""

from utils.prep import generate

# norm = data_twin["norm"]   # optional: pass from twin() for consistency
data = generate(norm=None)   # replace None with norm if available

attack_rows = data["attack_rows"]   # (N, 277)  real attacks — Diffusion trains on this
normal_rows = data["normal_rows"]   # (M, 277)  normal rows  — for class conditioning
X_test_tstr = data["X_test_all"]   # (K, 277)  full test1+2  — for TSTR evaluation
y_test_tstr = data["y_test_all"]   # (K,)       labels        — for TSTR evaluation
norm        = data["norm"]

# ── Build and train Diffusion Model here ──────────────────────────────────────
