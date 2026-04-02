"""
models/train_generative.py — Training entry point for the Generative Digital Twin.

Goal: learn to predict realistic physical sensor readings for BOTH normal
      operation AND attack behavior.

Difference from train.py:
    train.py        → trained on normal only → high error = anomaly signal
    train_generative → trained on normal + attacks → low error on both = success

Run:
    python models/train_generative.py

Steps:
    1. Load data via twin_generative() — episode-aware 80/20 split
    2. Train Transformer Seq2Seq
    3. Evaluate: rmse_test_attack should be LOW (model tracks attack trajectories)
    4. Save checkpoint + metrics
    5. Plot val predictions
"""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.Transformer.transformer_model import (
    TransformerSeq2Seq,
    D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, DROPOUT, EPOCHS, BATCH, LR,
    train as train_transformer,
    make_predict_fn,
    plot_val_predictions,
)
from evaluate.Transformer.eval import evaluate_twin
from models.Transformer.prep_transformer import twin_generative

Path("outputs").mkdir(exist_ok=True)
Path("outputs/plots").mkdir(parents=True, exist_ok=True)

ENC_LEN = 300
DEC_LEN = 180

# ── 1. Load data ──────────────────────────────────────────────────────────────

print("Loading data via twin_generative() pipeline...")
data  = twin_generative(input_len=ENC_LEN, target_len=DEC_LEN, stride=60)

X_train = data["X_train"]        # (N, 300, F)  normal + train attack episodes
Y_train = data["Y_train"]        # (N, 180, F)
X_val   = data["X_val"]          # (M, 300, F)  last 20% of train1-4 (normal only)
Y_val   = data["Y_val"]          # (M, 180, F)
X_test  = data["X_test"]         # (K, 300, F)  held-out attack episodes
Y_test  = data["Y_test"]         # (K, 180, F)
y_test  = data["y_test_labels"]  # (K,)  0=normal  1=attack
norm    = data["norm"]
N_FEAT  = X_train.shape[2]

print(f"  N_FEAT = {N_FEAT}")
print(f"  Attack windows in test: {y_test.sum()} / {len(y_test)}")

# ── 2. Build + train Transformer ──────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = TransformerSeq2Seq(
    n_features=N_FEAT, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT,
).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

model = train_transformer(model, X_train, Y_train, X_val, Y_val)

# ── 3. Evaluate ───────────────────────────────────────────────────────────────

metrics = evaluate_twin(
    predict_fn=make_predict_fn(model),
    X_val=X_val, Y_val=Y_val,
    X_test=X_test, Y_test=Y_test, y_test=y_test,
    label="Transformer Seq2Seq (Generative)",
    save_path="outputs/transformer_generative_metrics.json",
)

# Generative goal interpretation (opposite of anomaly detection)
ratio = metrics["separation_ratio"]
print(f"\n  Generative fidelity check:")
if ratio <= 1.2:
    print(f"  separation_ratio={ratio:.2f} — model tracks attack trajectories accurately.")
else:
    print(f"  separation_ratio={ratio:.2f} — high ratio: model may not have learned attack dynamics.")

# ── 4. Save checkpoint ────────────────────────────────────────────────────────

torch.save({
    "model_state":   model.state_dict(),
    "train_ep_ids":  list(data["train_ep_ids"]),
    "test_ep_ids":   list(data["test_ep_ids"]),
    "metrics":       metrics,
    "n_feat":        N_FEAT,
}, "outputs/transformer_generative.pt")

print("\nSaved: outputs/transformer_generative.pt")
print("       outputs/transformer_generative_metrics.json")

# ── 5. Plot val predictions ───────────────────────────────────────────────────

print("\nGenerating val prediction plots...")
plot_val_predictions(model, norm, device, enc_len=ENC_LEN, dec_len=DEC_LEN,
                     model_name="Transformer Seq2Seq (Generative)")
