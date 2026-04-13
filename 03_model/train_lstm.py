"""
train_lstm.py — Train the complete HAI Digital Twin (LSTM plant variant).

Per-epoch training order:
    1. Controllers (PC, LC, FC, TC) — GRUController × 4, MSE loss on CV sequence
    2. CC Classifier-Regressor       — BCE (pump on/off) + MSE (speed) combined loss
    3. LSTMPlant                     — scheduled-sampling MSE on PV

Validation (per epoch):
    Open-loop, ss_ratio=0 for all models.

Post-training:
    Closed-loop rollout over val windows (target_len horizon):
        1. Run each GRUController on input window  →  predicted CV sequence
        2. Patch x_cv_target with predicted CVs
        3. Run LSTMPlant autoregressively          →  PV sequence
    Compute NRMSE per PV channel.

Run:
    python 03_model/train_lstm.py
"""

import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from lstm import LSTMPlant
from gru import GRUController, CCClassifierRegressor
from config import LOOPS, PV_COLS

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
# Plant
HIDDEN      = 256
LAYERS      = 2
DROPOUT     = 0.1
# Controller
CTRL_HIDDEN = 64
CTRL_LAYERS = 2
# CC
CC_HIDDEN   = 32
# Training
EPOCHS      = 150
BATCH       = 128
LR          = 1e-3
CTRL_LR     = 1e-3
WD          = 1e-5
GRAD_CLIP   = 1.0
SCH_PAT     = 5
SCH_FAC     = 0.5
PATIENCE    = 15      # early stopping: epochs without val improvement
# Scheduled sampling
SS_START    = 10
SS_END      = 40
SS_MAX      = 0.5

OUT_DIR = Path("outputs/lstm_plant")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading data via pipeline...")
data        = load_and_prepare_data()
plant_data  = data['plant']
ctrl_data   = data['ctrl']
sensor_cols = data['metadata']['sensor_cols']
TARGET_LEN  = data['metadata']['target_len']

# Plant arrays
X_train        = plant_data['X_train']            # (N, input_len, n_plant_in)
X_cv_tgt_train = plant_data['X_cv_target_train']  # (N, target_len, n_plant_in)
pv_init_train  = plant_data['pv_init_train']       # (N, n_pv)
pv_teacher_tr  = plant_data['pv_teacher_train']    # (N, target_len, n_pv)
pv_target_tr   = plant_data['pv_target_train']     # (N, target_len, n_pv)
scenario_train = plant_data['scenario_train']      # (N,)

X_val          = plant_data['X_val']
X_cv_tgt_val   = plant_data['X_cv_target_val']
pv_init_val    = plant_data['pv_init_val']
pv_teacher_val = plant_data['pv_teacher_val']
pv_target_val  = plant_data['pv_target_val']
scenario_val   = plant_data['scenario_val']

N_PLANT_IN  = plant_data['n_plant_in']
N_PV        = plant_data['n_pv']
N_SCENARIOS = data['metadata']['n_scenarios']
N           = len(X_train)

print(f"  n_plant_in  : {N_PLANT_IN}")
print(f"  n_pv        : {N_PV}")
print(f"  n_scenarios : {N_SCENARIOS}")
print(f"  Train       : {X_train.shape}   Val: {X_val.shape}")

# ── CV column indices in the non-PV feature space (used for closed-loop) ─────
pv_set      = set(PV_COLS)
non_pv_cols = [c for c in sensor_cols if c not in pv_set]
col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}
pv_col_to_pv_idx = {c: i for i, c in enumerate(PV_COLS)}

CTRL_LOOPS = ['PC', 'LC', 'FC', 'TC']
# Index of each loop's CV column within x_cv / x_cv_target
ctrl_cv_col_idx = {ln: col_to_idx[LOOPS[ln].cv]
                   for ln in CTRL_LOOPS if LOOPS[ln].cv in col_to_idx}

# ── 2. Build models ───────────────────────────────────────────────────────────
print(f"\nStep 2: Building models... (device: {device})")

plant_model = LSTMPlant(
    n_plant_in=N_PLANT_IN, n_pv=N_PV,
    hidden=HIDDEN, layers=LAYERS,
    n_scenarios=N_SCENARIOS, dropout=DROPOUT,
).to(device)

ctrl_models = {
    ln: GRUController(
        n_inputs    = ctrl_data[ln]['X_train'].shape[-1],  # 2 = [SP, PV]
        hidden      = CTRL_HIDDEN,
        layers      = CTRL_LAYERS,
        dropout     = DROPOUT,
        output_len  = TARGET_LEN,
    ).to(device)
    for ln in CTRL_LOOPS
}

cc_model = CCClassifierRegressor(
    n_inputs=2, hidden=CC_HIDDEN, dropout=DROPOUT
).to(device)

print(f"  LSTMPlant   : {sum(p.numel() for p in plant_model.parameters()):,}")
print(f"  Controllers : {sum(p.numel() for m in ctrl_models.values() for p in m.parameters()):,}")
print(f"  CC model    : {sum(p.numel() for p in cc_model.parameters()):,}")

# ── 3. Optimizers & losses ────────────────────────────────────────────────────
plant_opt = torch.optim.Adam(plant_model.parameters(), lr=LR, weight_decay=WD)
plant_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
    plant_opt, patience=SCH_PAT, factor=SCH_FAC)
ctrl_opts = {ln: torch.optim.Adam(m.parameters(), lr=CTRL_LR, weight_decay=WD)
             for ln, m in ctrl_models.items()}
cc_opt    = torch.optim.Adam(cc_model.parameters(), lr=CTRL_LR, weight_decay=WD)
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()


# ── Helpers ───────────────────────────────────────────────────────────────────
def ss_ratio_for(epoch: int) -> float:
    """Linear ramp 0 → SS_MAX from SS_START to SS_END."""
    if epoch < SS_START:  return 0.0
    if epoch >= SS_END:   return SS_MAX
    return SS_MAX * (epoch - SS_START) / (SS_END - SS_START)


def train_controllers(ss: float) -> dict:
    """One shuffled pass over training data for all four GRUControllers."""
    for m in ctrl_models.values(): m.train()
    N_ctrl = len(ctrl_data['PC']['X_train'])
    idx    = np.random.permutation(N_ctrl)
    totals = {ln: 0.0 for ln in CTRL_LOOPS}

    for i in range(0, N_ctrl, BATCH):
        b = idx[i:i + BATCH]
        for ln in CTRL_LOOPS:
            Xc = torch.tensor(ctrl_data[ln]['X_train'][b]).float().to(device)
            yc = torch.tensor(ctrl_data[ln]['y_train'][b]).float().to(device)
            pred = ctrl_models[ln](Xc, y_cv_teacher=yc, ss_ratio=ss)
            loss = mse(pred, yc)
            ctrl_opts[ln].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ctrl_models[ln].parameters(), GRAD_CLIP)
            ctrl_opts[ln].step()
            totals[ln] += loss.item()

    n_b = max(1, N_ctrl // BATCH)
    return {ln: totals[ln] / n_b for ln in CTRL_LOOPS}


def val_controllers() -> dict:
    """Open-loop val MSE (ss_ratio=0) for each controller."""
    for m in ctrl_models.values(): m.eval()
    losses = {}
    with torch.no_grad():
        for ln in CTRL_LOOPS:
            Xv = torch.tensor(ctrl_data[ln]['X_val']).float().to(device)
            yv = torch.tensor(ctrl_data[ln]['y_val']).float().to(device)
            losses[ln] = mse(ctrl_models[ln](Xv, y_cv_teacher=yv, ss_ratio=0.0), yv).item()
    return losses


def train_cc() -> float:
    """
    One pass over CC training data.
    Loss = BCE(pump on/off) + MSE(pump speed, on-samples only).
    Input:  last timestep of [SP, PV] window → (B, 2)
    Labels: first target-step CV value (pump speed; 0 = off)
    """
    cc_model.train()
    X_cc  = ctrl_data['CC']['X_train']   # (N, input_len, 2)
    y_cc  = ctrl_data['CC']['y_train']   # (N, target_len, 1)
    N_cc  = len(X_cc)
    idx   = np.random.permutation(N_cc)
    total = 0.0

    for i in range(0, N_cc, BATCH):
        b         = idx[i:i + BATCH]
        x_in      = torch.tensor(X_cc[b]).float().to(device)         # (B, input_len, 2)
        cv_target = torch.tensor(y_cc[b, 0, 0]).float().to(device)   # first target step (B,)

        pump_on_lbl = (cv_target > 0.0).float().unsqueeze(-1)        # (B, 1)
        pump_speed  = cv_target.unsqueeze(-1)                        # (B, 1)

        logit, speed = cc_model(x_in)                                # each (B, 1)
        loss_cls = bce(logit, pump_on_lbl)
        on_mask  = pump_on_lbl.squeeze(-1).bool()
        loss_reg = (mse(speed[on_mask], pump_speed[on_mask])
                    if on_mask.any() else torch.tensor(0.0, device=device))
        loss = loss_cls + loss_reg

        cc_opt.zero_grad()
        loss.backward()
        cc_opt.step()
        total += loss.item()

    return total / max(1, N_cc // BATCH)


def val_cc() -> float:
    """Open-loop val loss for CC model."""
    cc_model.eval()
    X_cc = ctrl_data['CC']['X_val']
    y_cc = ctrl_data['CC']['y_val']
    x_in = torch.tensor(X_cc).float().to(device)
    cv_t = torch.tensor(y_cc[:, 0, 0]).float().to(device)
    pump_on_lbl = (cv_t > 0.0).float().unsqueeze(-1)
    pump_speed  = cv_t.unsqueeze(-1)
    with torch.no_grad():
        logit, speed = cc_model(x_in)
        loss_cls = bce(logit, pump_on_lbl)
        on_mask  = pump_on_lbl.squeeze(-1).bool()
        loss_reg = (mse(speed[on_mask], pump_speed[on_mask])
                    if on_mask.any() else torch.tensor(0.0, device=device))
    return (loss_cls + loss_reg).item()


# ── 4. Training loop ──────────────────────────────────────────────────────────
print(f"\nStep 3: Training for {EPOCHS} epochs...")
best_plant_val   = float("inf")
best_plant_state = plant_model.state_dict()
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    ss = ss_ratio_for(epoch)

    # ── a) Controllers ──────────────────────────────────────────────────────
    ctrl_train = train_controllers(ss)

    # ── b) CC Classifier-Regressor ──────────────────────────────────────────
    cc_train = train_cc()

    # ── c) Plant ────────────────────────────────────────────────────────────
    plant_model.train()
    idx         = np.random.permutation(N)
    plant_total = 0.0

    for i in range(0, N, BATCH):
        b = idx[i:i + BATCH]
        x_cv    = torch.tensor(X_train[b]).float().to(device)
        x_cv_t  = torch.tensor(X_cv_tgt_train[b]).float().to(device)
        pv_init = torch.tensor(pv_init_train[b]).float().to(device)
        pv_tf   = torch.tensor(pv_teacher_tr[b]).float().to(device)
        pv_tgt  = torch.tensor(pv_target_tr[b]).float().to(device)
        sc      = torch.tensor(scenario_train[b]).long().to(device)

        pred = plant_model(x_cv, x_cv_t, pv_init, sc, pv_teacher=pv_tf, ss_ratio=ss)
        loss = mse(pred, pv_tgt)
        plant_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(plant_model.parameters(), GRAD_CLIP)
        plant_opt.step()
        plant_total += loss.item()

    # ── d) Validation (open-loop, ss_ratio=0) ───────────────────────────────
    plant_model.eval()
    ctrl_val   = val_controllers()
    cc_val     = val_cc()
    pval, n_b  = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X_val), BATCH):
            x_cv    = torch.tensor(X_val[i:i+BATCH]).float().to(device)
            x_cv_t  = torch.tensor(X_cv_tgt_val[i:i+BATCH]).float().to(device)
            pv_init = torch.tensor(pv_init_val[i:i+BATCH]).float().to(device)
            pv_tf   = torch.tensor(pv_teacher_val[i:i+BATCH]).float().to(device)
            pv_tgt  = torch.tensor(pv_target_val[i:i+BATCH]).float().to(device)
            sc      = torch.tensor(scenario_val[i:i+BATCH]).long().to(device)
            pval   += mse(plant_model(
                x_cv, x_cv_t, pv_init, sc, pv_teacher=pv_tf, ss_ratio=0.0
            ), pv_tgt).item()
            n_b += 1

    pval /= max(1, n_b)
    plant_sch.step(pval)

    if pval < best_plant_val:
        best_plant_val   = pval
        best_plant_state = {k: v.clone() for k, v in plant_model.state_dict().items()}
        patience_counter = 0
        torch.save({
            "model_state": best_plant_state,
            "n_plant_in":  N_PLANT_IN,
            "n_pv":        N_PV,
            "n_scenarios": N_SCENARIOS,
            "hidden":      HIDDEN,
            "layers":      LAYERS,
            "epoch":       epoch,
            "val_loss":    best_plant_val,
        }, OUT_DIR / "lstm_plant.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    if epoch % 10 == 0 or epoch == 1:
        ctrl_str = "  ".join(f"{ln}={v:.4f}" for ln, v in ctrl_train.items())
        print(f"  Epoch {epoch:3d}/{EPOCHS} | plant: train={plant_total/max(1,N//BATCH):.5f}"
              f"  val={pval:.5f}  ss={ss:.3f}")
        print(f"             | ctrl:  {ctrl_str}")
        print(f"             | cc:    train={cc_train:.4f}  val={cc_val:.4f}")

plant_model.load_state_dict(best_plant_state)
print(f"\n  Best plant val loss: {best_plant_val:.5f}")

# ── 5. Closed-loop validation (target_len-step horizon) ───────────────────────
# Strategy:
#   1. Run each GRUController on its val input window → predicted CV sequence
#   2. Patch x_cv_target with predicted CVs (replacing true CVs)
#   3. Run LSTMPlant autoregressively → predicted PV sequence
#   4. Compute NRMSE vs true PV target
#
# Note: this is a one-shot controller rollout (not step-by-step PV feedback).
# Full step-by-step closed-loop (with PV feedback to controllers each step)
# is implemented in the separate inference script.
print(f"\nStep 4: Closed-loop validation ({TARGET_LEN}-step / ~{TARGET_LEN//60}-min horizon)...")
for m in ctrl_models.values(): m.eval()
cc_model.eval()
plant_model.eval()

N_val    = len(X_val)
pv_preds = np.zeros((N_val, TARGET_LEN, N_PV), dtype=np.float32)

with torch.no_grad():
    for i in range(0, N_val, BATCH):
        sl       = slice(i, i + BATCH)
        x_cv_b   = torch.tensor(X_val[sl]).float().to(device)
        xct_b    = torch.tensor(X_cv_tgt_val[sl]).float().to(device).clone()
        pv_init_b = torch.tensor(pv_init_val[sl]).float().to(device)
        sc_b     = torch.tensor(scenario_val[sl]).long().to(device)

        # Predict CVs from controllers and patch x_cv_target
        for ln in CTRL_LOOPS:
            if ln not in ctrl_cv_col_idx:
                continue
            Xc      = torch.tensor(ctrl_data[ln]['X_val'][sl]).float().to(device)
            cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)  # (B, T, 1)
            cv_i    = ctrl_cv_col_idx[ln]
            xct_b[:, :, cv_i:cv_i + 1] = cv_pred

        # Autoregressive plant rollout with patched CV targets
        pv_seq = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)  # (B, T, n_pv)
        B_actual = pv_seq.size(0)
        pv_preds[i:i + B_actual] = pv_seq.cpu().numpy()

# NRMSE per PV channel
pv_true = pv_target_val   # (N_val, target_len, n_pv)
nrmse   = []
for k in range(N_PV):
    true_k = pv_true[:, :, k]
    pred_k = pv_preds[:, :, k]
    rmse   = np.sqrt(np.mean((pred_k - true_k) ** 2))
    rng    = max(float(true_k.max() - true_k.min()), 1e-6)
    nrmse.append(rmse / rng)

pv_names = PV_COLS
print("  Closed-loop NRMSE per PV:")
for name, v in zip(pv_names, nrmse):
    print(f"    {name:<30s}: {v:.4f}")
print(f"  Mean NRMSE: {np.mean(nrmse):.4f}")

results = {
    "model": "LSTM",
    "nrmse_per_pv": {name: float(v) for name, v in zip(PV_COLS, nrmse)},
    "mean_nrmse": float(np.mean(nrmse)),
    "best_val_loss": float(best_plant_val),
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  Saved: results.json")

# ── 6. Save checkpoints ───────────────────────────────────────────────────────
print(f"\nStep 5: Saving checkpoints to {OUT_DIR}/")

torch.save({
    "model_state": plant_model.state_dict(),
    "n_plant_in":  N_PLANT_IN,
    "n_pv":        N_PV,
    "n_scenarios": N_SCENARIOS,
    "hidden":      HIDDEN,
    "layers":      LAYERS,
}, OUT_DIR / "lstm_plant.pt")
print(f"  Saved: lstm_plant.pt")

for ln, m in ctrl_models.items():
    torch.save({
        "model_state": m.state_dict(),
        "n_inputs":    ctrl_data[ln]['X_train'].shape[-1],
        "hidden":      CTRL_HIDDEN,
        "layers":      CTRL_LAYERS,
    }, OUT_DIR / f"gru_ctrl_{ln.lower()}.pt")
    print(f"  Saved: gru_ctrl_{ln.lower()}.pt")

torch.save({
    "model_state": cc_model.state_dict(),
    "n_inputs":    2,
    "hidden":      CC_HIDDEN,
}, OUT_DIR / "cc_model.pt")
print(f"  Saved: cc_model.pt")
