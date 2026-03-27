"""
LSTM Autoencoder — two roles:

ROLE A — Layer 2: Anomaly Detector (baseline)
    Reconstructs the SAME window (autoencoder, not seq2seq).
    High reconstruction error = anomaly.
    Shape: (N, 300, F) → (N, 300, F)

    How to evaluate:
        from utils.eval import evaluate_detector
        scores  = model.anomaly_scores(X_test)   # (K,) per-window MSE
        results = evaluate_detector(y_test, scores,
                                    label="LSTM Autoencoder",
                                    save_path="outputs/lstm_ae_metrics.json")

ROLE B — Layer 3: Attack Generator (alternative)
    Trains on real attack windows from test1.
    Generates synthetic attacks → feed into Guided Generation pipeline.
    Shape: (N, 300, F)

Window size = 300s  (from window_size_analysis notebook).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# ROLE A — Anomaly Detector (Layer 2 baseline)
# ─────────────────────────────────────────────────────────────────────────────

from utils.prep import detect

data    = detect(window_size=300, stride=60)

X_train = data["X_train"]        # (N, 300, F)  benign — input == target
X_val   = data["X_val"]          # (M, 300, F)  benign val
X_test  = data["X_test"]         # (K, 300, F)  test1+test2 all rows
y_test  = data["y_test_labels"]  # (K,)  0=normal  1=attack
norm    = data["norm"]

N_FEAT      = X_train.shape[2]   # actual sensor count
WINDOW_SIZE = X_train.shape[1]   # 300

print(f"Role A data — N_FEAT={N_FEAT}, WINDOW_SIZE={WINDOW_SIZE}")
print(f"  X_train {X_train.shape}  (input == target)")
print(f"  X_val   {X_val.shape}")
print(f"  X_test  {X_test.shape}   attacks={y_test.sum()} / {len(y_test)}")

# ── Build and train LSTM Autoencoder (anomaly detection) here ─────────────────


# ─────────────────────────────────────────────────────────────────────────────
# ROLE B — Attack Generator (Layer 3 alternative)
# Uncomment this block if building the generative model
# ─────────────────────────────────────────────────────────────────────────────

# from utils.prep import generate
#
# gen_data       = generate(norm=None, window_len=300, stride=60)
# attack_windows = gen_data["attack_windows"]   # (N, 300, F)  test1 attack windows
# normal_windows = gen_data["normal_windows"]   # (M, 300, F)  train1 normal windows
# gen_norm       = gen_data["norm"]
# F              = attack_windows.shape[2]
#
# # Flatten to rows if the model operates timestep-by-timestep
# attack_rows = attack_windows.reshape(-1, F)   # (N*300, F)
# normal_rows = normal_windows.reshape(-1, F)   # (M*300, F)
#
# print(f"Role B data — attack_windows {attack_windows.shape}, F={F}")
#
# # ── Build and train LSTM-AE generator here ────────────────────────────────
