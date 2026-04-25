"""
run_eval.py — Evaluate Transformer (no-causal) checkpoint.

Loads:
    outputs/transformer/no_causal/models/transformer.pt
    outputs/scaled_split/val_data.npz
    outputs/scaled_split/test_data.npz

Run:
    python 04_evaluate/run_eval.py
"""

import sys
import numpy as np
import torch
from pathlib import Path

ROOT     = Path(__file__).parent.parent
EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT / "03_model"))
sys.path.insert(0, str(EVAL_DIR))

from transformer import TransformerSeq2Seq
from eval        import evaluate_twin
from eval_plots  import run_all_plots

# ── Config ────────────────────────────────────────────────────────────────────
BATCH      = 64
CHECKPOINT = ROOT / "outputs/transformer/no_causal/models/transformer.pt"
VAL_DATA   = ROOT / "outputs/scaled_split/val_data.npz"
TEST_DATA  = ROOT / "outputs/scaled_split/test_data.npz"
OUT_DIR    = ROOT / "outputs/transformer/no_causal"
MODEL_NAME = "Transformer (no-causal)"

# ── Load checkpoint ───────────────────────────────────────────────────────────
print(f"Loading checkpoint: {CHECKPOINT}")
ckpt        = torch.load(CHECKPOINT, map_location="cpu")
n_feat      = ckpt["n_feat"]
n_scenarios = ckpt["n_scenarios"]
sensor_cols = ckpt["sensor_cols"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}  |  features: {n_feat}  |  scenarios: {n_scenarios}")

model = TransformerSeq2Seq(
    n_features=n_feat, d_model=256, n_heads=8,
    n_layers=4, ffn_dim=1024, dropout=0.1,
    n_scenarios=n_scenarios,
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\nLoading val data:  {VAL_DATA}")
val          = np.load(VAL_DATA)
X_val, Y_val = val["X"], val["y"]
print(f"  X_val : {X_val.shape}   Y_val : {Y_val.shape}")

print(f"Loading test data: {TEST_DATA}")
test             = np.load(TEST_DATA)
X_test, Y_test   = test["X"], test["y"]
y_test           = test["attack_labels"]
test_scen_labels = test["scenario_labels"]
print(f"  X_test: {X_test.shape}   Y_test: {Y_test.shape}")

# ── predict_fn ────────────────────────────────────────────────────────────────
def predict_fn(X: np.ndarray, Y: np.ndarray,
               scenario: np.ndarray | None = None) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            src    = torch.tensor(X[i:i+BATCH]).float().to(device)
            tgt    = torch.tensor(Y[i:i+BATCH]).float().to(device)
            sc     = (torch.tensor(scenario[i:i+BATCH]).long().to(device)
                      if scenario is not None else None)
            dec_in = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
            preds.append(model(src, dec_in, sc).cpu().numpy())
    return np.concatenate(preds, axis=0)

# ── Standard metrics ──────────────────────────────────────────────────────────
print("\nEvaluating...")
evaluate_twin(
    predict_fn=lambda X, Y: predict_fn(X, Y, np.zeros(len(X), dtype=np.int32)),
    X_val=X_val, Y_val=Y_val,
    X_test=X_test, Y_test=Y_test,
    y_test=y_test,
    label=MODEL_NAME,
    save_path=str(OUT_DIR / "transformer_metrics.json"),
)

# ── All diagnostic plots ──────────────────────────────────────────────────────
run_all_plots(
    model                = model,
    predict_fn           = predict_fn,
    sensor_cols          = sensor_cols,
    device               = device,
    X_test               = X_test,
    Y_test               = Y_test,
    test_scenario_labels = test_scen_labels,
    output_dir           = OUT_DIR,
    model_name           = MODEL_NAME,
)

print("\nDone.")
