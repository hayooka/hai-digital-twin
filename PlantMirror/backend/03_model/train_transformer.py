"""
train_transformer.py — Train the complete HAI Digital Twin (Transformer plant variant).

Per-epoch training order:
    1. Controllers (PC, LC, FC, TC) — TransformerController × 4, MSE loss on CV sequence
    2. CC Classifier-Regressor       — BCE (pump on/off) + MSE (speed) combined loss
    3. TransformerPlant              — scheduled-sampling MSE on PV

Validation (per epoch):
    Open-loop, ss_ratio=0 for all models.

Post-training:
    Closed-loop rollout over val windows (target_len horizon):
        1. Run each TransformerController on input window  →  predicted CV sequence
        2. Patch x_cv_target with predicted CVs
        3. Run TransformerPlant autoregressively   →  PV sequence
    Compute NRMSE per PV channel.

Run:
    python 03_model/train_transformer.py
"""

import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from transformer import TransformerPlant, TransformerController, TransformerCCClassifierRegressor
from config import LOOPS, PV_COLS
from plot_results import plot_training_curves, plot_all_horizons

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
# Plant (Transformer-specific)
D_MODEL     = 128
N_HEADS     = 8
N_LAYERS    = 3
EMB_DIM     = 32
DROPOUT     = 0.1
# Controller
CTRL_HIDDEN = 64
CTRL_LAYERS = 2
# CC (now a real Transformer encoder — d_model must be divisible by n_heads)
CC_HIDDEN   = 32   # d_model for CC encoder
CC_HEADS    = 4    # attention heads  (32 / 4 = 8 per head)
CC_LAYERS   = 2
# Training
EPOCHS      = 150
BATCH       = 32
LR          = 1e-3
CTRL_LR     = 1e-3
WD          = 1e-5
GRAD_CLIP   = 1.0
SCH_PAT     = 5
SCH_FAC     = 0.5
PATIENCE    = 25      # early stopping: epochs without val improvement
CTRL_FREEZE = 10      # freeze controllers + CC after this epoch, only train plant
# Scheduled sampling — ramps from 0 → SS_MAX over [SS_START, SS_END] epochs
SS_START    = 10
SS_END      = 40
SS_MAX      = 0.2

OUT_DIR  = Path("outputs/transformer_plant")
BEST_DIR = Path("outputs/transformer_plant_best")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_DIR.mkdir(parents=True, exist_ok=True)

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

X_train        = plant_data['X_train']
X_cv_tgt_train = plant_data['X_cv_target_train']
pv_init_train  = plant_data['pv_init_train']
pv_teacher_tr  = plant_data['pv_teacher_train']
pv_target_tr   = plant_data['pv_target_train']
scenario_train = plant_data['scenario_train']

X_val          = plant_data['X_val']
X_cv_tgt_val   = plant_data['X_cv_target_val']
pv_init_val    = plant_data['pv_init_val']
pv_teacher_val = plant_data['pv_teacher_val']
pv_target_val  = plant_data['pv_target_val']
scenario_val   = plant_data['scenario_val']

X_test         = plant_data['X_test']
X_cv_tgt_test  = plant_data['X_cv_target_test']
pv_init_test   = plant_data['pv_init_test']
pv_target_test = plant_data['pv_target_test']
scenario_test  = plant_data['scenario_test']
attack_test    = plant_data['attack_test']      # binary: 1 = attack in target window

N_PLANT_IN  = plant_data['n_plant_in']
N_PV        = plant_data['n_pv']
N_SCENARIOS = data['metadata']['n_scenarios']
N           = len(X_train)

print(f"  n_plant_in  : {N_PLANT_IN}")
print(f"  n_pv        : {N_PV}")
print(f"  n_scenarios : {N_SCENARIOS}")
print(f"  Train       : {X_train.shape}   Val: {X_val.shape}")

# ── Pre-convert all numpy arrays to tensors once (avoid per-batch conversion) ─
print("  Pre-loading tensors...")
X_train_t        = torch.from_numpy(X_train).float()
X_cv_tgt_train_t = torch.from_numpy(X_cv_tgt_train).float()
pv_init_train_t  = torch.from_numpy(pv_init_train).float()
pv_teacher_tr_t  = torch.from_numpy(pv_teacher_tr).float()
pv_target_tr_t   = torch.from_numpy(pv_target_tr).float()
scenario_train_t = torch.from_numpy(scenario_train).long()

X_val_t          = torch.from_numpy(X_val).float()
X_cv_tgt_val_t   = torch.from_numpy(X_cv_tgt_val).float()
pv_init_val_t    = torch.from_numpy(pv_init_val).float()
pv_teacher_val_t = torch.from_numpy(pv_teacher_val).float()
pv_target_val_t  = torch.from_numpy(pv_target_val).float()
scenario_val_t   = torch.from_numpy(scenario_val).long()

ctrl_tensors = {}
for ln in ['PC', 'LC', 'FC', 'TC', 'CC']:
    ctrl_tensors[ln] = {
        'X_train': torch.from_numpy(ctrl_data[ln]['X_train']).float(),
        'y_train': torch.from_numpy(ctrl_data[ln]['y_train']).float(),
        'X_val':   torch.from_numpy(ctrl_data[ln]['X_val']).float(),
        'y_val':   torch.from_numpy(ctrl_data[ln]['y_val']).float(),
    }
    if 'X_test' in ctrl_data[ln]:
        ctrl_tensors[ln]['X_test'] = torch.from_numpy(ctrl_data[ln]['X_test']).float()

X_test_t        = torch.from_numpy(X_test).float()
X_cv_tgt_test_t = torch.from_numpy(X_cv_tgt_test).float()
pv_init_test_t  = torch.from_numpy(pv_init_test).float()
scenario_test_t = torch.from_numpy(scenario_test).long()

# ── CV column indices for closed-loop ─────────────────────────────────────────
pv_set      = set(PV_COLS)
non_pv_cols = [c for c in sensor_cols if c not in pv_set]
col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}

CTRL_LOOPS      = ['PC', 'LC', 'FC', 'TC']
ctrl_cv_col_idx = {ln: col_to_idx[LOOPS[ln].cv]
                   for ln in CTRL_LOOPS if LOOPS[ln].cv in col_to_idx}

# ── 2. Build models ───────────────────────────────────────────────────────────
print(f"\nStep 2: Building models... (device: {device})")

plant_model = TransformerPlant(
    n_plant_in  = N_PLANT_IN,
    n_pv        = N_PV,
    d_model     = D_MODEL,
    n_heads     = N_HEADS,
    n_layers    = N_LAYERS,
    n_scenarios = N_SCENARIOS,
    emb_dim     = EMB_DIM,
    dropout     = DROPOUT,
).to(device)

ctrl_models = {
    ln: TransformerController(
        n_inputs   = ctrl_data[ln]['X_train'].shape[-1],
        d_model    = CTRL_HIDDEN,
        n_layers   = CTRL_LAYERS,
        dropout    = DROPOUT,
        output_len = TARGET_LEN,
    ).to(device)
    for ln in CTRL_LOOPS
}

cc_model = TransformerCCClassifierRegressor(
    n_inputs=2, d_model=CC_HIDDEN, n_heads=CC_HEADS,
    n_layers=CC_LAYERS, dropout=DROPOUT,
).to(device)

print(f"  TransformerPlant: {sum(p.numel() for p in plant_model.parameters()):,}")
print(f"  Controllers     : {sum(p.numel() for m in ctrl_models.values() for p in m.parameters()):,}")
print(f"  CC model        : {sum(p.numel() for p in cc_model.parameters()):,}")

# ── 3. Optimizers & losses ────────────────────────────────────────────────────
plant_opt = torch.optim.Adam(plant_model.parameters(), lr=LR, weight_decay=WD)
plant_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
    plant_opt, patience=SCH_PAT, factor=SCH_FAC)
ctrl_opts = {ln: torch.optim.Adam(m.parameters(), lr=CTRL_LR, weight_decay=WD)
             for ln, m in ctrl_models.items()}
cc_opt    = torch.optim.Adam(cc_model.parameters(), lr=CTRL_LR, weight_decay=WD)
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
ATTACK_WEIGHT = 3.0


def weighted_mse(pred, target, scenario):
    """MSE with higher weight for attack scenario windows (scenario > 0)."""
    loss = ((pred - target) ** 2).mean(dim=(1, 2))
    weights = torch.where(scenario > 0,
                          torch.full_like(loss, ATTACK_WEIGHT),
                          torch.ones_like(loss))
    return (loss * weights).mean()


def ss_ratio_for(epoch: int) -> float:
    if epoch < SS_START:  return 0.0
    if epoch >= SS_END:   return SS_MAX
    return SS_MAX * (epoch - SS_START) / (SS_END - SS_START)


def train_controllers(ss: float) -> dict:
    for m in ctrl_models.values(): m.train()
    N_ctrl = len(ctrl_data['PC']['X_train'])
    idx    = np.random.permutation(N_ctrl)
    totals = {ln: 0.0 for ln in CTRL_LOOPS}
    for i in range(0, N_ctrl, BATCH):
        b = idx[i:i + BATCH]
        for ln in CTRL_LOOPS:
            Xc = ctrl_tensors[ln]['X_train'][b].to(device)
            yc = ctrl_tensors[ln]['y_train'][b].to(device)
            pred = ctrl_models[ln](Xc, y_cv_teacher=yc, ss_ratio=ss)
            loss = mse(pred, yc)
            ctrl_opts[ln].zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ctrl_models[ln].parameters(), GRAD_CLIP)
            ctrl_opts[ln].step()
            totals[ln] += loss.item()
    n_b = max(1, N_ctrl // BATCH)
    return {ln: totals[ln] / n_b for ln in CTRL_LOOPS}


def val_controllers() -> dict:
    for m in ctrl_models.values(): m.eval()
    with torch.no_grad():
        return {ln: mse(
            ctrl_models[ln](
                ctrl_tensors[ln]['X_val'].to(device),
                y_cv_teacher=ctrl_tensors[ln]['y_val'].to(device),
                ss_ratio=0.0,
            ),
            ctrl_tensors[ln]['y_val'].to(device),
        ).item() for ln in CTRL_LOOPS}


def train_cc() -> float:
    cc_model.train()
    N_cc = len(ctrl_tensors['CC']['X_train'])
    idx  = np.random.permutation(N_cc)
    total = 0.0
    for i in range(0, N_cc, BATCH):
        b         = idx[i:i + BATCH]
        x_in      = ctrl_tensors['CC']['X_train'][b].to(device)
        cv_target = ctrl_tensors['CC']['y_train'][b, 0, 0].to(device)
        pump_on   = (cv_target > 0.0).float().unsqueeze(-1)
        speed     = cv_target.unsqueeze(-1)
        logit, spd = cc_model(x_in)
        loss_cls   = bce(logit, pump_on)
        on_mask    = pump_on.squeeze(-1).bool()
        loss_reg   = (mse(spd[on_mask], speed[on_mask])
                      if on_mask.any() else torch.tensor(0.0, device=device))
        loss = loss_cls + loss_reg
        cc_opt.zero_grad(); loss.backward(); cc_opt.step()
        total += loss.item()
    return total / max(1, N_cc // BATCH)


def val_cc() -> float:
    cc_model.eval()
    x_in = ctrl_tensors['CC']['X_val'].to(device)
    cv_t = ctrl_tensors['CC']['y_val'][:, 0, 0].to(device)
    pump_on = (cv_t > 0.0).float().unsqueeze(-1)
    speed   = cv_t.unsqueeze(-1)
    with torch.no_grad():
        logit, spd = cc_model(x_in)
        loss_cls   = bce(logit, pump_on)
        on_mask    = pump_on.squeeze(-1).bool()
        loss_reg   = (mse(spd[on_mask], speed[on_mask])
                      if on_mask.any() else torch.tensor(0.0, device=device))
    return (loss_cls + loss_reg).item()


# ── 4. Training loop ──────────────────────────────────────────────────────────
import time
print(f"\nStep 3: Training for {EPOCHS} epochs...")
best_plant_val   = float("inf")
best_plant_state = plant_model.state_dict()
patience_counter = 0
epoch_start = time.time()

train_losses: list[float] = []
val_losses:   list[float] = []
ss_ratios:    list[float] = []

for epoch in range(1, EPOCHS + 1):
    ss = ss_ratio_for(epoch)

    if epoch == CTRL_FREEZE + 1:
        for m in ctrl_models.values():
            m.eval()
            for p in m.parameters(): p.requires_grad = False
        cc_model.eval()
        for p in cc_model.parameters(): p.requires_grad = False
        print(f"  Epoch {epoch}: controllers + CC frozen — plant-only training from here")

    if epoch <= CTRL_FREEZE:
        ctrl_train = train_controllers(ss)
        cc_train   = train_cc()
    else:
        ctrl_train = {ln: 0.0 for ln in CTRL_LOOPS}
        cc_train   = 0.0

    plant_model.train()
    idx         = np.random.permutation(N)
    plant_total = 0.0
    for i in range(0, N, BATCH):
        b = idx[i:i + BATCH]
        x_cv    = X_train_t[b].to(device)
        x_cv_t  = X_cv_tgt_train_t[b].to(device)
        pv_init = pv_init_train_t[b].to(device)
        pv_tf   = pv_teacher_tr_t[b].to(device)
        pv_tgt  = pv_target_tr_t[b].to(device)
        sc      = scenario_train_t[b].to(device)
        pred    = plant_model(x_cv, x_cv_t, pv_init, sc, pv_teacher=pv_tf, ss_ratio=ss)
        loss    = weighted_mse(pred, pv_tgt, sc)
        plant_opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(plant_model.parameters(), GRAD_CLIP)
        plant_opt.step()
        plant_total += mse(pred, pv_tgt).item()  # track unweighted for logging

    plant_model.eval()
    ctrl_val  = val_controllers()
    cc_val    = val_cc()
    pval, n_b = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X_val), BATCH):
            x_cv    = X_val_t[i:i+BATCH].to(device)
            x_cv_t  = X_cv_tgt_val_t[i:i+BATCH].to(device)
            pv_init = pv_init_val_t[i:i+BATCH].to(device)
            pv_tf   = pv_teacher_val_t[i:i+BATCH].to(device)
            pv_tgt  = pv_target_val_t[i:i+BATCH].to(device)
            sc      = scenario_val_t[i:i+BATCH].to(device)
            pval   += mse(plant_model(
                x_cv, x_cv_t, pv_init, sc, pv_teacher=pv_tf, ss_ratio=0.0
            ), pv_tgt).item()
            n_b += 1

    pval /= max(1, n_b)
    plant_sch.step(pval)

    train_losses.append(plant_total / max(1, N // BATCH))
    val_losses.append(pval)
    ss_ratios.append(ss)

    if pval < best_plant_val:
        best_plant_val   = pval
        best_plant_state = {k: v.clone() for k, v in plant_model.state_dict().items()}
        patience_counter = 0
        ckpt = {
            "model_state": best_plant_state,
            "n_plant_in":  N_PLANT_IN,
            "n_pv":        N_PV,
            "n_scenarios": N_SCENARIOS,
            "d_model":     D_MODEL,
            "n_heads":     N_HEADS,
            "n_layers":    N_LAYERS,
            "epoch":       epoch,
            "val_loss":    best_plant_val,
        }
        torch.save(ckpt, OUT_DIR  / "transformer_plant.pt")
        torch.save(ckpt, BEST_DIR / "transformer_plant.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    epoch_time = time.time() - epoch_start
    epoch_start = time.time()
    ctrl_str = "  ".join(f"{ln}={v:.4f}" for ln, v in ctrl_train.items())
    print(f"  Epoch {epoch:3d}/{EPOCHS} | {epoch_time:.0f}s | plant: train={plant_total/max(1,N//BATCH):.5f}"
          f"  val={pval:.5f}  ss={ss:.3f}")
    if epoch % 10 == 0 or epoch == 1:
        print(f"             | ctrl:  {ctrl_str}")
        print(f"             | cc:    train={cc_train:.4f}  val={cc_val:.4f}")

plant_model.load_state_dict(best_plant_state)
print(f"\n  Best plant val loss: {best_plant_val:.5f}")

# ── Plot 1: Training loss curves ──────────────────────────────────────────────
plot_training_curves(
    train_losses, val_losses, ss_ratios,
    model_name="Transformer",
    save_path=OUT_DIR / "transformer_loss_curves.png",
)

# ── 5. Closed-loop validation ─────────────────────────────────────────────────
print(f"\nStep 4: Closed-loop validation ({TARGET_LEN}-step / ~{TARGET_LEN//60}-min horizon)...")
for m in ctrl_models.values(): m.eval()
cc_model.eval()
plant_model.eval()

N_val    = len(X_val)
pv_preds = np.zeros((N_val, TARGET_LEN, N_PV), dtype=np.float32)

with torch.no_grad():
    for i in range(0, N_val, BATCH):
        sl        = slice(i, i + BATCH)
        x_cv_b    = torch.tensor(X_val[sl]).float().to(device)
        xct_b     = torch.tensor(X_cv_tgt_val[sl]).float().to(device).clone()
        pv_init_b = torch.tensor(pv_init_val[sl]).float().to(device)
        sc_b      = torch.tensor(scenario_val[sl]).long().to(device)

        for ln in CTRL_LOOPS:
            if ln not in ctrl_cv_col_idx:
                continue
            Xc      = torch.tensor(ctrl_data[ln]['X_val'][sl]).float().to(device)
            cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
            xct_b[:, :, ctrl_cv_col_idx[ln]:ctrl_cv_col_idx[ln] + 1] = cv_pred

        pv_seq   = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
        B_actual = pv_seq.size(0)
        pv_preds[i:i + B_actual] = pv_seq.cpu().numpy()

pv_true = pv_target_val
nrmse   = []
for k in range(N_PV):
    true_k = pv_true[:, :, k]
    pred_k = pv_preds[:, :, k]
    rmse   = np.sqrt(np.mean((pred_k - true_k) ** 2))
    rng    = max(float(true_k.max() - true_k.min()), 1e-6)
    nrmse.append(rmse / rng)

print("  Closed-loop NRMSE per PV:")
for name, v in zip(PV_COLS, nrmse):
    print(f"    {name:<30s}: {v:.4f}")
print(f"  Mean NRMSE: {np.mean(nrmse):.4f}")

results = {
    "model": "Transformer",
    "nrmse_per_pv": {name: float(v) for name, v in zip(PV_COLS, nrmse)},
    "mean_nrmse": float(np.mean(nrmse)),
    "best_val_loss": float(best_plant_val),
    "train_losses": [float(x) for x in train_losses],
    "val_losses":   [float(x) for x in val_losses],
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  Saved: results.json")

# ── 6. Save checkpoints (before plotting so a crash in plots doesn't lose the model) ──
print(f"\nStep 5: Saving checkpoints to {OUT_DIR}/")

torch.save({
    "model_state": plant_model.state_dict(),
    "n_plant_in":  N_PLANT_IN,
    "n_pv":        N_PV,
    "n_scenarios": N_SCENARIOS,
    "d_model":     D_MODEL,
    "n_heads":     N_HEADS,
    "n_layers":    N_LAYERS,
}, OUT_DIR / "transformer_plant.pt")
print(f"  Saved: transformer_plant.pt")

for ln, m in ctrl_models.items():
    torch.save({
        "model_state": m.state_dict(),
        "n_inputs":    ctrl_data[ln]['X_train'].shape[-1],
        "hidden":      CTRL_HIDDEN,
        "layers":      CTRL_LAYERS,
    }, OUT_DIR / f"transformer_ctrl_{ln.lower()}.pt")
    print(f"  Saved: transformer_ctrl_{ln.lower()}.pt")

torch.save({
    "model_state": cc_model.state_dict(),
    "n_inputs":    2,
    "d_model":     CC_HIDDEN,
    "n_heads":     CC_HEADS,
    "n_layers":    CC_LAYERS,
}, OUT_DIR / "cc_model.pt")
print(f"  Saved: cc_model.pt")

# ── Plots 2–5: closed-loop time series for each horizon ───────────────────────
print("\nStep 6: Generating closed-loop horizon plots...")

HORIZONS = [300, 600, 900, 1800]
horizon_data: dict = {}
sample_idx = int(np.var(pv_true[:, :, :], axis=(1, 2)).argmax())  # most informative window

for H in HORIZONS:
    steps = min(H, TARGET_LEN)
    horizon_data[H] = (
        pv_true[sample_idx, :steps, :],
        pv_preds[sample_idx, :steps, :],
    )

plot_all_horizons(
    horizon_data, PV_COLS,
    model_name="Transformer",
    out_dir=OUT_DIR,
)

# ── 7. Test-set evaluation: NRMSE + Attack detection metrics ──────────────────
print(f"\nStep 7: Test-set evaluation ({len(X_test)} windows)...")
for m in ctrl_models.values(): m.eval()
cc_model.eval()
plant_model.eval()

N_test      = len(X_test)
pv_preds_te = np.zeros((N_test, TARGET_LEN, N_PV), dtype=np.float32)

with torch.no_grad():
    for i in range(0, N_test, BATCH):
        sl          = slice(i, i + BATCH)
        x_cv_b      = torch.tensor(X_test[sl]).float().to(device)
        xct_b       = torch.tensor(X_cv_tgt_test[sl]).float().to(device).clone()
        pv_init_b   = torch.tensor(pv_init_test[sl]).float().to(device)
        sc_b        = torch.tensor(scenario_test[sl]).long().to(device)

        for ln in CTRL_LOOPS:
            if ln not in ctrl_cv_col_idx:
                continue
            Xc      = torch.tensor(ctrl_data[ln]['X_test'][sl]).float().to(device)
            cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
            xct_b[:, :, ctrl_cv_col_idx[ln]:ctrl_cv_col_idx[ln] + 1] = cv_pred

        pv_seq   = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
        B_actual = pv_seq.size(0)
        pv_preds_te[i:i + B_actual] = pv_seq.cpu().numpy()

# ── NRMSE on test set (normal windows only, i.e. train4) ─────────────────────
normal_mask  = (attack_test == 0)
nrmse_test   = []
for k in range(N_PV):
    true_k = pv_target_test[normal_mask, :, k]
    pred_k = pv_preds_te[normal_mask, :, k]
    rmse   = np.sqrt(np.mean((pred_k - true_k) ** 2))
    rng    = max(float(true_k.max() - true_k.min()), 1e-6)
    nrmse_test.append(rmse / rng)

print("  Test NRMSE (normal windows — causal generalisation):")
for name, v in zip(PV_COLS, nrmse_test):
    print(f"    {name:<30s}: {v:.4f}")
print(f"  Test Mean NRMSE: {np.mean(nrmse_test):.4f}")

# ── Attack detection: use per-window PV reconstruction error as anomaly score ─
# Higher MSE → prediction error → likely attack in target window
anomaly_scores = np.mean((pv_preds_te - pv_target_test) ** 2, axis=(1, 2))  # (N_test,)

attack_metrics: dict = {}
if attack_test.sum() > 0:
    auroc = roc_auc_score(attack_test, anomaly_scores)

    # F1 at best threshold (sweep over percentiles of the score distribution)
    thresholds    = np.percentile(anomaly_scores, np.linspace(50, 99, 100))
    best_f1, best_thresh = 0.0, thresholds[0]
    for t in thresholds:
        f1 = f1_score(attack_test, anomaly_scores > t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    # Precision / recall at best-F1 threshold
    prec_arr, rec_arr, _ = precision_recall_curve(attack_test, anomaly_scores)
    avg_prec = float(np.mean(prec_arr))

    attack_metrics = {
        "auroc":             float(auroc),
        "best_f1":           float(best_f1),
        "best_threshold":    float(best_thresh),
        "avg_precision":     avg_prec,
        "n_attack_windows":  int(attack_test.sum()),
        "n_normal_windows":  int((attack_test == 0).sum()),
    }
    print(f"\n  Attack detection metrics (anomaly score = PV reconstruction MSE):")
    print(f"    AUROC          : {auroc:.4f}")
    print(f"    Best F1        : {best_f1:.4f}  (threshold={best_thresh:.5f})")
    print(f"    Avg Precision  : {avg_prec:.4f}")
    print(f"    Attack windows : {attack_metrics['n_attack_windows']} / {N_test}")
else:
    print("  WARNING: No attack windows in test set — skipping attack metrics.")

# ── Save updated results ──────────────────────────────────────────────────────
results.update({
    "test_nrmse_per_pv":  {name: float(v) for name, v in zip(PV_COLS, nrmse_test)},
    "test_mean_nrmse":    float(np.mean(nrmse_test)),
    "attack_detection":   attack_metrics,
})
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results updated: results.json")