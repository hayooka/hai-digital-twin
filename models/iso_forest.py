"""
Isolation Forest — Anomaly Detection (primary)

IMPORTANT: ISO Forest does NOT use raw sensor data.
           It trains on reconstruction ERRORS from the trained Digital Twin.

Step 1: Run transformer_model.py or lstm_model.py first and compute errors:
            train_errors = model.reconstruction_errors(X_val)   # (M, 277)
            test_errors  = model.reconstruction_errors(X_test)  # (K, 277)
            y_test       = data["y_test_labels"]                 # (K,)

Step 2: Pass those errors here.

Shape:  X_train (N, 277) ← per-sensor MSE errors on train4 benign
        X_test  (M, 277) ← per-sensor MSE errors on test1+2
"""

from utils.prep import detect_errors

# ── Get errors from the trained Digital Twin first ────────────────────────────
# train_errors = ...   # compute from your trained model on X_val
# test_errors  = ...   # compute from your trained model on X_test
# y_test       = ...   # from twin() → data["y_test_labels"]

data = detect_errors(
    train_errors = train_errors,   # (N, 277)
    test_errors  = test_errors,    # (M, 277)
    y_test       = y_test,         # (M,)
)

X_train = data["X_train"]   # (N, 277)  errors on train4 — ISO Forest trains on this
X_test  = data["X_test"]    # (M, 277)  errors on test1+2
y_test  = data["y_test"]    # (M,)      labels for evaluation

# ── Build and train ISO Forest here ──────────────────────────────────────────
