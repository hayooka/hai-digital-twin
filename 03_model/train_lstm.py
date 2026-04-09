"""
train_lstm.py — Train LSTM Seq2Seq with scenario conditioning and causal loss.

Matches the Transformer causal training pipeline exactly:
    - Same data loading (load_and_prepare_episodic)
    - Same teacher forcing: dec_in = [last_enc_step, target[:, :-1, :]]
    - Same loss: MSE + λ * causal_loss
    - Same optimizer, scheduler, epochs, batch size
    - Same checkpoint format

Run:
    python 03_model/train_lstm.py
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from scaled_split import load_and_prepare_episodic
from lstm import LSTMSeq2Seq
from causal_loss import CausalLoss

# ── Config (identical to Transformer causal) ─────────────────────────────────
HIDDEN_SIZE = 256
NUM_LAYERS  = 4
DROPOUT     = 0.1
EPOCHS      = 50
BATCH       = 64
LR          = 1e-4
LAMBDA      = 0.1          # causal loss weight — lower = better rmse, slightly higher violations

ENC_LEN     = 300
DEC_LEN     = 180
STRIDE      = 60

OUT_DIR = Path("outputs/lstm/causal")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "models").mkdir(exist_ok=True)
(OUT_DIR / "plots").mkdir(exist_ok=True)

# ── 1. Load data (same as Transformer) ───────────────────────────────────────
print("=" * 60)
print("Step 1: Loading data...")
data = load_and_prepare_episodic(input_len=ENC_LEN, target_len=DEC_LEN, stride=STRIDE)

X_train        = data["X_train"]            # (N, 300, F)
Y_train        = data["y_train"]            # (N, 180, F)
scenario_train = data["scenario_train"]      # (N,)

X_val          = data["X_val"]
Y_val          = data["y_val"]

X_test         = data["X_test"]
Y_test         = data["y_test"]
y_test         = data["attack_test_labels"]
scenario_test  = data["scenario_test"]

N_SCENARIOS    = data["n_scenarios"]        # 4
sensor_cols    = data["sensor_cols"]
N_FEAT         = X_train.shape[2]

print(f"  Features    : {N_FEAT}")
print(f"  N_scenarios : {N_SCENARIOS}")
print(f"  X_train     : {X_train.shape}")
print(f"  X_val       : {X_val.shape}")
print(f"  X_test      : {X_test.shape}")

# ── 2. Build LSTM model (with scenario conditioning) ─────────────────────────
print("\nStep 2: Building LSTMSeq2SeqCausal...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

model = LSTMSeq2Seq(
    n_features   = N_FEAT,
    n_scenarios  = N_SCENARIOS,
    hidden_size  = HIDDEN_SIZE,
    num_layers   = NUM_LAYERS,
    dropout      = DROPOUT,
    output_len   = DEC_LEN,
).to(device)

print(f"  Parameters: {model.count_parameters():,}")

# ── 3. Causal loss ───────────────────────────────────────────────────────────
causal = CausalLoss("outputs/causal_graph/parents_full.json", sensor_cols)

# ── 4. Training (MSE + λ * causal_loss) ──────────────────────────────────────
print(f"\nStep 3: Training for {EPOCHS} epochs (MSE + {LAMBDA} * causal_loss)...")
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
mse_loss = nn.MSELoss()
N = len(X_train)
best_val = float("inf")
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
        # Teacher forcing: same as Transformer
        dec_in   = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
        pred     = model(src, dec_in, scenario)
        loss     = mse_loss(pred, tgt) + LAMBDA * causal(pred)
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
            # Use scenario = 0 for validation (normal)
            scenario = torch.zeros(len(src), dtype=torch.long, device=device)
            dec_in   = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
            val_loss += mse_loss(model(src, dec_in, scenario), tgt).item()
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

# ── 5. Save checkpoint (compatible with run_eval.py) ─────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "n_feat":      N_FEAT,
    "n_scenarios": N_SCENARIOS,
    "sensor_cols": sensor_cols,
}, OUT_DIR / "models/lstm.pt")
print(f"\nSaved: {OUT_DIR / 'models/lstm.pt'}")
print("Run evaluation with: python 04_evaluate/run_eval_lstm_causal.py (adapt paths)")