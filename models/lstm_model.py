"""
LSTM Seq2Seq — Digital Twin (baseline)

Data:   train1+2+3 (benign) → windowed per file, no gap crossing
        Val:  train4 (benign)
        Test: test1+2 (all rows)
Shape:  X (N, 60, 277) → Y (N, 180, 277)
"""

from utils.prep import twin

data    = twin(input_len=60, target_len=180, stride=1)

X_train = data["X_train"]        # (N, 60,  277)  encoder input
Y_train = data["Y_train"]        # (N, 180, 277)  decoder target
X_val   = data["X_val"]          # (M, 60,  277)
Y_val   = data["Y_val"]          # (M, 180, 277)
X_test  = data["X_test"]         # (K, 60,  277)
Y_test  = data["Y_test"]         # (K, 180, 277)
y_test  = data["y_test_labels"]  # (K,)  0=normal 1=attack
norm    = data["norm"]           # save this — pass to ISO Forest later

# ── Build and train LSTM here ─────────────────────────────────────────────────
