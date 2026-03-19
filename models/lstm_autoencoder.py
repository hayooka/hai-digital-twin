"""
LSTM Autoencoder — used in TWO roles:

  A) Anomaly Detection baseline (PD2)
         trains on benign windows, detects anomalies by reconstruction error
         Input == Target  (autoencoder reconstructs its own input)
         Shape: (N, 60, 277) → (N, 60, 277)

  B) Attack Generator alternative
         trains on real attack rows, generates synthetic attacks
         Shape: (N, 277) flat rows

Load the correct data depending on which role you are building.
"""

# ─────────────────────────────────────────────────────────────────────────────
# ROLE A — Anomaly Detection (PD2 baseline)
# Uncomment this block if building the anomaly detector
# ─────────────────────────────────────────────────────────────────────────────

from utils.prep import detect

data    = detect(window_size=60, stride=1)

X_train = data["X_train"]        # (N, 60, 277)  benign windows — input == target
X_val   = data["X_val"]          # (M, 60, 277)  benign val windows
X_test  = data["X_test"]         # (K, 60, 277)  test1+2 all rows
y_test  = data["y_test_labels"]  # (K,)  0=normal 1=attack  — for evaluation
norm    = data["norm"]

# ── Build and train LSTM Autoencoder (anomaly detection) here ─────────────────


# ─────────────────────────────────────────────────────────────────────────────
# ROLE B — Attack Generator (alternative)
# Uncomment this block if building the generative model
# ─────────────────────────────────────────────────────────────────────────────

# from utils.prep import generate
#
# data = generate(norm=None)   # replace None with norm from twin() if available
#
# attack_rows = data["attack_rows"]   # (N, 277)  real attacks — train on this
# normal_rows = data["normal_rows"]   # (M, 277)  normal rows  — for conditioning
# X_test_tstr = data["X_test_all"]   # (K, 277)  for TSTR evaluation
# y_test_tstr = data["y_test_all"]   # (K,)       labels
# norm        = data["norm"]
#
# ── Build and train LSTM Autoencoder (generative) here ────────────────────────
