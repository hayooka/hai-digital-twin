"""
models/train.py — Single training entry point for the HAI Digital Twin pipeline.

Run:
    python models/train.py

Steps:
    1. Load data via twin() pipeline
    2. Train Transformer Seq2Seq (Layer 1)
    3. Evaluate Transformer (RMSE)
    4. Extract reconstruction errors + save checkpoint
    5. Plot val predictions
    6. Load test1 (known attacks) + extract ISO Forest features
    7. Train Isolation Forest (Layer 2)
    8. Evaluate ISO Forest (eTaPR)
"""
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer_model import (
    TransformerSeq2Seq,
    D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, DROPOUT, EPOCHS, BATCH, LR,
    train as train_transformer,
    make_predict_fn,
    plot_val_predictions,
)
from models.iso_forest import extract_features, train_iso_forest
from evaluate.eval import evaluate_twin, evaluate_detector
from utils.prep import twin
from utils.data_loader import load_merged, identify_common_constants

Path("outputs").mkdir(exist_ok=True)
Path("outputs/plots").mkdir(parents=True, exist_ok=True)

ENC_LEN = 300
DEC_LEN = 180

# ── 1. Load data ──────────────────────────────────────────────────────────────

print("Loading data via twin() pipeline...")
data    = twin(input_len=ENC_LEN, target_len=DEC_LEN, stride=60)
X_train = data["X_train"]        # (N, 300, F)  encoder input — normal only
Y_train = data["Y_train"]        # (N, 180, F)  decoder target
X_val   = data["X_val"]          # (M, 300, F)  last 20% of each train file
Y_val   = data["Y_val"]          # (M, 180, F)
X_test  = data["X_test"]         # (K, 300, F)  test2 held-out
Y_test  = data["Y_test"]         # (K, 180, F)
y_test  = data["y_test_labels"]  # (K,)  0=normal 1=attack
norm    = data["norm"]           # fitted normalizer — reused for ISO Forest
N_FEAT  = X_train.shape[2]       # dynamic: actual sensor count after constant deletion

print(f"  train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")
print(f"  N_FEAT = {N_FEAT}  attacks in test2: {y_test.sum()} / {len(y_test)}")

# ── 2. Build + train Transformer ──────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = TransformerSeq2Seq(
    n_features=N_FEAT, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT,
).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

model = train_transformer(model, X_train, Y_train, X_val, Y_val)

# ── 3. Evaluate Transformer (RMSE) ────────────────────────────────────────────

metrics = evaluate_twin(
    predict_fn=make_predict_fn(model),
    X_val=X_val, Y_val=Y_val,
    X_test=X_test, Y_test=Y_test, y_test=y_test,
    label="Transformer Seq2Seq",
    save_path="outputs/transformer_metrics.json",
)

# ── 4. Compute per-sensor reconstruction errors ───────────────────────────────

print("\nComputing per-sensor reconstruction errors for ISO Forest...")
train_errors = model.reconstruction_errors(X_train, Y_train)   # (N, F)
test_errors  = model.reconstruction_errors(X_test,  Y_test)    # (K, F)
print(f"  train_errors {train_errors.shape}")
print(f"  test_errors  {test_errors.shape}")

torch.save({
    "model_state":  model.state_dict(),
    "train_errors": train_errors,
    "test_errors":  test_errors,
    "y_test":       y_test,
    "metrics":      metrics,
    "n_feat":       N_FEAT,
}, "outputs/transformer_twin.pt")

print("\nSaved: outputs/transformer_twin.pt")
print("       outputs/transformer_metrics.json")

# ── 5. Plot val predictions ───────────────────────────────────────────────────

print("\nGenerating val prediction plots...")
plot_val_predictions(model, norm, device, enc_len=ENC_LEN, dec_len=DEC_LEN)

# ── 6. Load test1 (known attacks) for ISO Forest ──────────────────────────────

print("\nLoading test1 (known attacks)...")
const_hai, const_hiend, hiend_dups = identify_common_constants()
test1_df = load_merged("test", 1, drop_constants=True, keep_hai_duplicates=True,
                        const_cols_hai=const_hai, const_cols_hiend=const_hiend,
                        hiend_dup_cols=hiend_dups)

META_COLS   = {"timestamp", "attack", "label", "attack_p1", "attack_p2", "attack_p3"}
cols        = [c for c in test1_df.columns if c not in META_COLS and test1_df[c].dtype != object]
arr_test1   = norm.transform(test1_df)[cols].values.astype(np.float32)
y_test1_raw = test1_df["attack"].values.astype(np.int32)

span      = ENC_LEN + DEC_LEN
starts_t1 = list(range(0, len(arr_test1) - span + 1, 60))
X_t1 = np.empty((len(starts_t1), ENC_LEN, len(cols)), dtype=np.float32)
Y_t1 = np.empty((len(starts_t1), DEC_LEN, len(cols)), dtype=np.float32)
y_t1 = np.array([int(y_test1_raw[s:s+span].any()) for s in starts_t1], dtype=np.int32)
for i, s in enumerate(starts_t1):
    X_t1[i] = arr_test1[s           : s + ENC_LEN]
    Y_t1[i] = arr_test1[s + ENC_LEN : s + span]
print(f"  test1 windows: {X_t1.shape[0]}  (attacks: {y_t1.sum()})")

# ── 7. Extract features + train Isolation Forest ──────────────────────────────

print(f"\nExtracting features ({N_FEAT}+184-dim)...")
feat_val   = extract_features(model, device, X_val,  Y_val,  batch_size=BATCH)
feat_test1 = extract_features(model, device, X_t1,   Y_t1,   batch_size=BATCH)
feat_test2 = extract_features(model, device, X_test, Y_test, batch_size=BATCH)
print(f"  val={feat_val.shape}  test1={feat_test1.shape}  test2={feat_test2.shape}")

attack_rate = float(y_test.sum()) / max(1, len(y_test))
print(f"\nTraining Isolation Forest on {len(feat_val) + len(feat_test1)} windows "
      f"(contamination={attack_rate:.3f})...")
iso, pca, _, _ = train_iso_forest(feat_val, feat_test1, attack_rate)

# ── 8. Evaluate ISO Forest (eTaPR) ────────────────────────────────────────────

test2_pca = pca.transform(feat_test2)
scores    = -iso.score_samples(test2_pca)   # higher = more anomalous

evaluate_detector(
    y_true=y_test,
    scores=scores,
    label="Isolation Forest (Layer 2)",
    theta_p=0.5,
    theta_r=0.1,
    save_path="outputs/iso_forest_metrics.json",
)

print("\nSaved: outputs/iso_forest_metrics.json")
