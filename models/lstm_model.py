"""
LSTM Seq2Seq — Layer 1: Physical Model / Digital Twin (baseline)

Same role as transformer_model.py — predicts normal sensor readings.
Trained on benign data only (80% of each train1-4, temporal split per file).
High reconstruction error on test2 = anomaly signal → passed to Layer 2.

Shape:  X (N, 300, F) → Y (N, 180, F)
        F = sensor count after constant deletion (determined at runtime)

Window size = 300s  (from window_size_analysis notebook).

How to evaluate:
    from utils.eval import evaluate_twin
    results = evaluate_twin(predict_fn, X_val, Y_val, X_test, Y_test, y_test,
                            label="LSTM Seq2Seq",
                            save_path="outputs/lstm_metrics.json")
    # predict_fn(X, Y) → Y_pred  (numpy, same shape as Y)

How to feed Layer 2:
    train_errors = model.reconstruction_errors(X_train, Y_train)  # (N, F)
    test_errors  = model.reconstruction_errors(X_test,  Y_test)   # (K, F)
    torch.save({"model_state": ..., "train_errors": train_errors,
                "test_errors": test_errors, "y_test": y_test},
               "outputs/lstm_twin.pt")
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.prep import twin

# ── Data ──────────────────────────────────────────────────────────────────────

data    = twin(input_len=300, target_len=180, stride=60)

X_train = data["X_train"]        # (N, 300, F)  encoder input — normal only
Y_train = data["Y_train"]        # (N, 180, F)  decoder target
X_val   = data["X_val"]          # (M, 300, F)  last 20% of each train file
Y_val   = data["Y_val"]          # (M, 180, F)
X_test  = data["X_test"]         # (K, 300, F)  test2 held-out
Y_test  = data["Y_test"]         # (K, 180, F)
y_test  = data["y_test_labels"]  # (K,)  0=normal  1=attack
norm    = data["norm"]           # fitted HAISensorNormalizer — pass to iso_forest.py

N_FEAT     = X_train.shape[2]   # actual sensor count (not hardcoded)
INPUT_LEN  = X_train.shape[1]   # 300
TARGET_LEN = Y_train.shape[1]   # 180

print(f"Data loaded — N_FEAT={N_FEAT}, INPUT_LEN={INPUT_LEN}, TARGET_LEN={TARGET_LEN}")
print(f"  X_train {X_train.shape}  Y_train {Y_train.shape}")
print(f"  X_val   {X_val.shape}    Y_val   {Y_val.shape}")
print(f"  X_test  {X_test.shape}   Y_test  {Y_test.shape}")
print(f"  y_test  attacks={y_test.sum()} / {len(y_test)}")

# ── Build and train LSTM Seq2Seq here ─────────────────────────────────────────
