"""
train.py — Training entry point for the HAI Digital Twin Transformer (no causal loss).

Pipeline:
    scaled_split  →  window  →  TransformerSeq2Seq  →  checkpoint

Run:
    python 03_model/train.py
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from scaled_split  import load_and_prepare_episodic
from transformer   import TransformerSeq2Seq
# CausalLoss import removed

# ── Config ────────────────────────────────────────────────────────────────────
D_MODEL  = 256
N_HEADS  = 8
N_LAYERS = 4
FFN_DIM  = 1024
DROPOUT  = 0.1
EPOCHS   = 50
BATCH    = 64
LR       = 1e-4
# LAMBDA removed

ENC_LEN  = 300
DEC_LEN  = 180
STRIDE   = 60

Path("outputs/transformer/no_causal/models").mkdir(parents=True, exist_ok=True)
Path("outputs/transformer/no_causal/plots").mkdir(parents=True,  exist_ok=True)

# ── 1. Load & window data ─────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading data...")
data = load_and_prepare_episodic(input_len=300, target_len=180, stride=60)

# Extract data
X_train        = data["X_train"]            # (N, 300, F)
Y_train        = data["y_train"]            # (N, 180, F)
scenario_train = data["scenario_train"]      # (N,) 0=normal, 1-4=attack class

X_val          = data["X_val"]              # (M, 300, F)
Y_val          = data["y_val"]              # (M, 180, F)

X_test         = data["X_test"]             # (K, 300, F)
Y_test         = data["y_test"]             # (K, 180, F)
y_test         = data["attack_test_labels"] # (K,) binary: 0=normal, 1=attack
scenario_test  = data["scenario_test"]      # (K,) 0-4 for model conditioning

N_SCENARIOS    = data["n_scenarios"]        # 4: normal, AP_no, AP_comb, AE_no
sensor_cols    = data["sensor_cols"]
N_FEAT         = X_train.shape[2]

print(f"  Features    : {N_FEAT}")
print(f"  N_scenarios : {N_SCENARIOS}")
print(f"  X_train     : {X_train.shape}")
print(f"  X_val       : {X_val.shape}")
print(f"  X_test      : {X_test.shape}")

# ── 2. Build model ────────────────────────────────────────────────────────────
print("\nStep 2: Building model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

model = TransformerSeq2Seq(
    n_features=N_FEAT, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT,
    n_scenarios=N_SCENARIOS,
).to(device)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── 3. Causal loss removed ────────────────────────────────────────────────────
# No causal loss instantiation

# ── 4. Train ──────────────────────────────────────────────────────────────────
print(f"\nStep 3: Training for {EPOCHS} epochs (MSE only)...")
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
criterion = nn.MSELoss()   # only MSE
N         = len(X_train)
best_val   = float("inf")
best_state = model.state_dict()

for epoch in range(1, EPOCHS + 1):
    # train
    model.train()
    idx = np.random.permutation(N)
    total = 0.0
    for i in range(0, N, BATCH):
        b        = idx[i:i+BATCH]
        src      = torch.tensor(X_train[b]).float().to(device)
        tgt      = torch.tensor(Y_train[b]).float().to(device)
        scenario = torch.tensor(scenario_train[b]).long().to(device)
        dec_in   = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
        pred     = model(src, dec_in, scenario)
        loss     = criterion(pred, tgt)   # MSE only
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()

    # validate
    model.eval()
    val_loss, n_b = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X_val), BATCH):
            src      = torch.tensor(X_val[i:i+BATCH]).float().to(device)
            tgt      = torch.tensor(Y_val[i:i+BATCH]).float().to(device)
            scenario = torch.zeros(len(src), dtype=torch.long, device=device)
            dec_in   = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
            val_loss += criterion(model(src, dec_in, scenario), tgt).item()
            n_b += 1
    val_loss /= max(1, n_b)
    scheduler.step(val_loss)

    if val_loss < best_val:
        best_val   = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  train={total/max(1,N//BATCH):.5f}  val={val_loss:.5f}")

model.load_state_dict(best_state)
print(f"  Best val loss: {best_val:.5f}")

# ── 5. Save ───────────────────────────────────────────────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "n_feat":      N_FEAT,
    "n_scenarios": N_SCENARIOS,
    "sensor_cols": sensor_cols,
}, "outputs/transformer/no_causal/models/transformer.pt")
print("\nSaved: outputs/transformer/no_causal/models/transformer.pt")
print("Run `python 04_evaluate/run_eval.py` to evaluate and plot.")