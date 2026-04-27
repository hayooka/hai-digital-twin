"""
train.py — Training entry point for the HAI Digital Twin Transformer.

Pipeline:
    scaled_split  →  window  →  TransformerSeq2Seq  →  checkpoint

Run:
    python 03_model/train.py
"""

import sys
import random
import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from scaled_split import load_and_prepare_episodic
from transformer  import TransformerSeq2Seq
from causal_loss  import CausalLoss

# ── Load config ───────────────────────────────────────────────────────────────
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
_d  = cfg["data"]
_m  = cfg["model"]
_t  = cfg["training"]

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = _d["random_seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Random seed: {SEED}")

# ── Config ────────────────────────────────────────────────────────────────────
D_MODEL  = _m["d_model"]
N_HEADS  = _m["n_heads"]
N_LAYERS = _m["n_layers"]
FFN_DIM  = _m["ffn_dim"]
DROPOUT  = _m["dropout"]
EPOCHS   = _t["epochs"]
BATCH    = _t["batch_size"]
LR       = _t["lr"]
WD       = _t["weight_decay"]
GRAD_CLIP= _t["grad_clip"]
SCH_PAT  = _t["scheduler_patience"]
SCH_FAC  = _t["scheduler_factor"]
LAMBDA   = _t["lambda_causal"]

ENC_LEN  = _d["input_len"]
DEC_LEN  = _d["target_len"]
STRIDE   = _d["stride"]

Path("outputs/transformer/causal/models").mkdir(parents=True, exist_ok=True)
Path("outputs/transformer/causal/plots").mkdir(parents=True,  exist_ok=True)

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

# ── 3. Causal loss ────────────────────────────────────────────────────────────
causal = CausalLoss("outputs/causal_graph/parents_full.json", sensor_cols)

# ── 4. Train ──────────────────────────────────────────────────────────────────
print(f"\nStep 3: Training for {EPOCHS} epochs...")
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=SCH_PAT, factor=SCH_FAC)
mse       = nn.MSELoss()
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
        normal_mask = (scenario == 0)
        c_loss   = causal(pred[normal_mask]) if normal_mask.any() else torch.tensor(0.0, device=device)
        loss     = mse(pred, tgt) + LAMBDA * c_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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
            val_loss += mse(model(src, dec_in, scenario), tgt).item()
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
}, "outputs/transformer/causal/models/transformer.pt")
print("\nSaved: outputs/transformer/causal/models/transformer.pt")
print("Run `python 04_evaluate/run_eval.py` to evaluate and plot.")

