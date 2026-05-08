"""
train_gru_causal_plus.py — GRU-Causal + attack weighting + warm-start from Re_gru_afterweight.

Recipe (designed to beat Re_gru_afterweight's 0.87% mean val NRMSE):
  - Warm-start: plant loaded from outputs/Re_gru_afterweight/gru_plant.pt
  - LR=0.001 (gentle fine-tune for warm-started plant), CTRL_LR=0.003 (controllers from scratch)
  - BATCH=64                  (more gradient steps per epoch, same as Re_gru_afterweight)
  - PATIENCE=25               (allow longer convergence)
  - Attack weighting: ATTACK_WEIGHT=3.0 on plant loss (attack windows penalized 3×)
  - TIT03_LOSS_WEIGHT=2.0 (modest upweight; effective 6× on TIT03 attack windows = 2×3)
  - 3 causal channels per loop: CCSequenceModel with 6 inputs instead of 3

Three causal layers per loop:
  PC: L0=P1_PCV02D, L1=P1_FT01,   L2=P1_TIT01
  LC: L0=P1_FT03,   L1=P1_FCV03D, L2=P1_PCV01D
  FC: L0=P1_PIT01,  L1=P1_LIT01,  L2=P1_TIT03
  TC: L0=P1_FT02,   L1=P1_PIT02,  L2=P1_TIT02
  CC: L0=P1_PP04D,  L1=P1_FCV03D, L2=P1_PCV02D

Run:
    python 03_model/train_gru_causal_plus.py
"""

import logging
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))
sys.path.insert(0, str(ROOT / "04_evaluate"))

from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCSequenceModel
from config import LOOPS, PV_COLS
from shared import SCENARIO_NAMES, CTRL_LOOPS, CTRL_HIDDEN_PER_LOOP, EXTRA_CHANNELS, augment_ctrl_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42

# Plant loss weights
ATTACK_WEIGHT     = 3.0   # attack windows penalized 3× in plant loss
TIT03_LOSS_WEIGHT = 2.0   # upweight TIT03 (cooling PV); effective 6× on attack windows

# Architecture (matched to warm-start checkpoint)
HIDDEN  = 512
LAYERS  = 2
# DROPOUT and SS_MAX were found via Optuna hyperparameter search
DROPOUT = 0.054
CTRL_LAYERS = 2

# Training — gentle fine-tune LR; BATCH=64 for more gradient steps
EPOCHS    = 150
BATCH     = 64
LR        = 0.001   # low: fine-tunes warm-started plant without destroying learned weights
CTRL_LR   = 0.003   # higher: controllers start from scratch with 6 inputs
WD        = 1e-5
GRAD_CLIP = 1.0
SCH_PAT   = 8
SCH_FAC   = 0.5
PATIENCE  = 25

# Scheduled sampling (ratio annealed from 0 → SS_MAX over training)
SS_START = 10
SS_END   = 59
SS_MAX   = 0.483   # from Optuna search

# NOTE: The warm-start folder name contains a typo from the original training run.
# Do NOT rename the folder — it is the actual checkpoint on disk.
WARMSTART_CKPT = ROOT / "outputs/pipeline/Re__reults_of_gru_after_wight_/gru_plant.pt"

OUT_DIR = ROOT / "outputs/pipeline/gru_causal_plus/gru_causal_plus"
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. Load data ──────────────────────────────────────────────────────────────
log.info("Step 1: Loading data via pipeline...")
data        = load_and_prepare_data()
plant_data  = data['plant']
ctrl_data   = data['ctrl']
sensor_cols = data['metadata']['sensor_cols']
TARGET_LEN  = data['metadata']['target_len']

# Plant arrays
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
attack_test    = plant_data['attack_test']

N_PLANT_IN  = plant_data['n_plant_in']
N_PV        = plant_data['n_pv']
N_SCENARIOS = data['metadata']['n_scenarios']
N           = len(X_train)

assert N_PV == len(PV_COLS), f"Expected {len(PV_COLS)} PV channels, got {N_PV}"
_TIT03_IDX = PV_COLS.index('P1_TIT03')   # index of TIT03 in PV_COLS (upweighted channel)

log.info("n_plant_in=%d  n_pv=%d  n_scenarios=%d  train=%s  val=%s",
         N_PLANT_IN, N_PV, N_SCENARIOS, X_train.shape, X_val.shape)

log.info("Augmenting controller data with three causal layers...")
augment_ctrl_data(ctrl_data, sensor_cols)

# ── CV column indices in the non-PV feature space (used for closed-loop) ─────
pv_set      = set(PV_COLS)
non_pv_cols = [c for c in sensor_cols if c not in pv_set]
col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}

CTRL_LOOPS = ['PC', 'LC', 'FC', 'TC', 'CC']
ctrl_cv_col_idx = {ln: col_to_idx[LOOPS[ln].cv]
                   for ln in CTRL_LOOPS if LOOPS[ln].cv in col_to_idx}

# ── 2. Build models ───────────────────────────────────────────────────────────
log.info("Step 2: Building models (device: %s)", device)

plant_model = GRUPlant(
    n_plant_in=N_PLANT_IN, n_pv=N_PV,
    hidden=HIDDEN, layers=LAYERS,
    n_scenarios=N_SCENARIOS, dropout=DROPOUT,
).to(device)

ctrl_models = {}
for ln in CTRL_LOOPS:
    n_in = ctrl_data[ln]['X_train'].shape[-1]
    h    = CTRL_HIDDEN_PER_LOOP[ln]
    if ln == 'CC':
        ctrl_models[ln] = CCSequenceModel(
            n_inputs=n_in, hidden=h, layers=CTRL_LAYERS,
            dropout=DROPOUT, output_len=TARGET_LEN,
        ).to(device)
    else:
        ctrl_models[ln] = GRUController(
            n_inputs=n_in, hidden=h, layers=CTRL_LAYERS,
            dropout=DROPOUT, output_len=TARGET_LEN,
        ).to(device)

log.info("GRUPlant params: %s", f"{sum(p.numel() for p in plant_model.parameters()):,}")
log.info("Controller params: %s", f"{sum(p.numel() for m in ctrl_models.values() for p in m.parameters()):,}")

# ── Warm-start plant from best previous checkpoint ────────────────────────────
if WARMSTART_CKPT.exists():
    try:
        ckpt = torch.load(WARMSTART_CKPT, map_location=device, weights_only=True)
        missing, unexpected = plant_model.load_state_dict(ckpt["model_state"], strict=False)
        if missing:
            log.warning("Warm-start: missing keys: %s", missing)
        if unexpected:
            log.warning("Warm-start: unexpected keys: %s", unexpected)
        log.info("Warm-started plant from %s  (epoch=%s, val_loss=%s)",
                 WARMSTART_CKPT, ckpt.get('epoch', 'N/A'), ckpt.get('val_loss', 'N/A'))
    except Exception as exc:
        log.error("Failed to load warm-start checkpoint: %s — training from scratch", exc)
else:
    log.warning("Warm-start checkpoint not found at %s — training from scratch.", WARMSTART_CKPT)

# ── 3. Optimizers & losses ────────────────────────────────────────────────────
plant_opt = torch.optim.Adam(plant_model.parameters(), lr=LR, weight_decay=WD)
plant_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
    plant_opt, patience=SCH_PAT, factor=SCH_FAC)
ctrl_opts = {ln: torch.optim.Adam(m.parameters(), lr=CTRL_LR, weight_decay=WD)
             for ln, m in ctrl_models.items()}
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()

def weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                 scenario: torch.Tensor) -> torch.Tensor:
    """Attack-weighted MSE with TIT03 channel upweighting.

    TIT03 (cooling PV) is separated because AE attacks directly manipulate
    the cooling loop, making TIT03 residuals the earliest attack indicator.
    """
    other = [i for i in range(N_PV) if i != _TIT03_IDX]
    loss_main  = ((pred[:, :, other]        - target[:, :, other])        ** 2).mean(dim=(1, 2))
    loss_tit03 = ((pred[:, :, [_TIT03_IDX]] - target[:, :, [_TIT03_IDX]]) ** 2).mean(dim=(1, 2))
    loss = loss_main + TIT03_LOSS_WEIGHT * loss_tit03
    weights = torch.where(scenario > 0,
                          torch.full_like(loss, ATTACK_WEIGHT),
                          torch.ones_like(loss))
    return (loss * weights).mean()


# ── Helpers ───────────────────────────────────────────────────────────────────
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
            Xc = torch.tensor(ctrl_data[ln]['X_train'][b]).float().to(device)
            yc = torch.tensor(ctrl_data[ln]['y_train'][b]).float().to(device)
            if ln == 'CC':
                logits, cv_preds = ctrl_models[ln](Xc, y_teacher=yc, ss_ratio=ss)
                loss = bce(logits, (yc > 0).float()) + mse(cv_preds, yc)
            else:
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
    for m in ctrl_models.values(): m.eval()
    losses = {}
    with torch.no_grad():
        for ln in CTRL_LOOPS:
            Xv = torch.tensor(ctrl_data[ln]['X_val']).float().to(device)
            yv = torch.tensor(ctrl_data[ln]['y_val']).float().to(device)
            if ln == 'CC':
                logits, cv_preds = ctrl_models[ln](Xv, y_teacher=yv, ss_ratio=0.0)
                losses[ln] = (bce(logits, (yv > 0).float()) + mse(cv_preds, yv)).item()
            else:
                losses[ln] = mse(ctrl_models[ln](Xv, y_cv_teacher=yv, ss_ratio=0.0), yv).item()
    return losses


# ── 4. Training loop ──────────────────────────────────────────────────────────
log.info("Step 3: Training for %d epochs (LR=%s, PATIENCE=%d)", EPOCHS, LR, PATIENCE)
best_plant_val   = float("inf")
best_plant_state = plant_model.state_dict()
patience_counter = 0

train_losses: list[float] = []
val_losses:   list[float] = []
ss_ratios:    list[float] = []

for epoch in range(1, EPOCHS + 1):
    ss = ss_ratio_for(epoch)

    # ── a) Controllers ──────────────────────────────────────────────────────
    ctrl_train = train_controllers(ss)

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

        pred, _ = plant_model(x_cv, x_cv_t, pv_init, sc, pv_teacher=pv_tf, ss_ratio=ss)
        loss = weighted_mse(pred, pv_tgt, sc)
        plant_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(plant_model.parameters(), GRAD_CLIP)
        plant_opt.step()
        plant_total += loss.item()

    # ── d) Validation (open-loop, ss_ratio=0) ───────────────────────────────
    plant_model.eval()
    ctrl_val   = val_controllers()
    pval, n_b  = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X_val), BATCH):
            x_cv    = torch.tensor(X_val[i:i+BATCH]).float().to(device)
            x_cv_t  = torch.tensor(X_cv_tgt_val[i:i+BATCH]).float().to(device)
            pv_init = torch.tensor(pv_init_val[i:i+BATCH]).float().to(device)
            pv_tf   = torch.tensor(pv_teacher_val[i:i+BATCH]).float().to(device)
            pv_tgt  = torch.tensor(pv_target_val[i:i+BATCH]).float().to(device)
            sc      = torch.tensor(scenario_val[i:i+BATCH]).long().to(device)
            pred_val, _ = plant_model(
                x_cv, x_cv_t, pv_init, sc, pv_teacher=pv_tf, ss_ratio=0.0
            )
            pval += mse(pred_val, pv_tgt).item()
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
        torch.save({
            "model_state": best_plant_state,
            "n_plant_in":  N_PLANT_IN,
            "n_pv":        N_PV,
            "n_scenarios": N_SCENARIOS,
            "hidden":      HIDDEN,
            "layers":      LAYERS,
            "epoch":       epoch,
            "val_loss":    best_plant_val,
        }, OUT_DIR / "gru_plant.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            log.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, PATIENCE)
            break

    if epoch % 10 == 0 or epoch == 1:
        ctrl_str = "  ".join(f"{ln}={v:.4f}" for ln, v in ctrl_train.items())
        log.info("Epoch %3d/%d | plant train=%.5f val=%.5f ss=%.3f | ctrl: %s",
                 epoch, EPOCHS, plant_total / max(1, N // BATCH), pval, ss, ctrl_str)

plant_model.load_state_dict(best_plant_state)
log.info("Best plant val loss: %.5f", best_plant_val)

# ── Training loss curves ──────────────────────────────────────────────────────
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(train_losses, label="Train")
ax.semilogy(val_losses,   label="Val")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
ax.set_title("GRU-Causal-Plus — Training Loss"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "gru_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved: gru_loss_curves.png")

# ── 5. Closed-loop validation ─────────────────────────────────────────────────
log.info("Step 4: Closed-loop validation (%d-step / ~%d-min horizon)", TARGET_LEN, TARGET_LEN // 60)
for m in ctrl_models.values(): m.eval()
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
            cv_i    = ctrl_cv_col_idx[ln]
            xct_b[:, :, cv_i:cv_i + 1] = cv_pred

        pv_seq, _ = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
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

for name, v in zip(PV_COLS, nrmse):
    log.info("  Closed-loop NRMSE  %-30s: %.4f", name, v)
log.info("  Mean NRMSE: %.4f", np.mean(nrmse))

# ── Controller NRMSE (open-loop, val split) ───────────────────────────────────
for m in ctrl_models.values(): m.eval()
with torch.no_grad():
    for ln in CTRL_LOOPS:
        Xv = torch.tensor(ctrl_data[ln]['X_val']).float().to(device)
        yv = ctrl_data[ln]['y_val']
        if ln == 'CC':
            pred = ctrl_models[ln].predict(Xv).cpu().numpy()
        else:
            pred = ctrl_models[ln](Xv).cpu().numpy()
        rmse = np.sqrt(np.mean((pred - yv) ** 2))
        rng  = max(float(yv.max() - yv.min()), 1e-6)
        log.info("  Controller NRMSE  %s (CV→%-18s): %.4f", ln, LOOPS[ln].cv, rmse / rng)

results = {
    "model": "GRU-Causal-Plus",
    "causal_layers": {ln: cols for ln, cols in EXTRA_CHANNELS.items()},
    "hyperparams": {
        "LR": LR, "CTRL_LR": CTRL_LR, "BATCH": BATCH,
        "PATIENCE": PATIENCE, "warmstart": str(WARMSTART_CKPT),
    },
    "nrmse_per_pv": {name: float(v) for name, v in zip(PV_COLS, nrmse)},
    "mean_nrmse": float(np.mean(nrmse)),
    "best_val_loss": float(best_plant_val),
    "train_losses": [float(x) for x in train_losses],
    "val_losses":   [float(x) for x in val_losses],
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
log.info("Saved: results.json")

# ── Test-set evaluation ───────────────────────────────────────────────────────
log.info("Step 5: Test-set evaluation (%d windows)", len(X_test))
for m in ctrl_models.values(): m.eval()
plant_model.eval()

N_test      = len(X_test)
pv_preds_te = np.zeros((N_test, TARGET_LEN, N_PV), dtype=np.float32)

with torch.no_grad():
    for i in range(0, N_test, BATCH):
        sl        = slice(i, i + BATCH)
        x_cv_b    = torch.tensor(X_test[sl]).float().to(device)
        xct_b     = torch.tensor(X_cv_tgt_test[sl]).float().to(device).clone()
        pv_init_b = torch.tensor(pv_init_test[sl]).float().to(device)
        sc_b      = torch.tensor(scenario_test[sl]).long().to(device)

        for ln in CTRL_LOOPS:
            if ln not in ctrl_cv_col_idx: continue
            Xc      = torch.tensor(ctrl_data[ln]['X_test'][sl]).float().to(device)
            cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
            xct_b[:, :, ctrl_cv_col_idx[ln]:ctrl_cv_col_idx[ln] + 1] = cv_pred

        pv_seq, _ = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
        B_actual = pv_seq.size(0)
        pv_preds_te[i:i + B_actual] = pv_seq.cpu().numpy()

normal_mask = (attack_test == 0)
nrmse_test  = []
for k in range(N_PV):
    true_k = pv_target_test[normal_mask, :, k]
    pred_k = pv_preds_te[normal_mask, :, k]
    rmse   = np.sqrt(np.mean((pred_k - true_k) ** 2))
    rng    = max(float(true_k.max() - true_k.min()), 1e-6)
    nrmse_test.append(rmse / rng)

for name, v in zip(PV_COLS, nrmse_test):
    log.info("  Test NRMSE (normal)  %-30s: %.4f", name, v)
log.info("  Test Mean NRMSE: %.4f", np.mean(nrmse_test))

anomaly_scores = np.mean((pv_preds_te - pv_target_test) ** 2, axis=(1, 2))
attack_metrics: dict = {}
if attack_test.sum() > 0:
    auroc = roc_auc_score(attack_test, anomaly_scores)
    thresholds = np.percentile(anomaly_scores, np.linspace(50, 99, 100))
    best_f1, best_thresh = 0.0, thresholds[0]
    for t in thresholds:
        f1 = f1_score(attack_test, anomaly_scores > t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    prec_arr, rec_arr, _ = precision_recall_curve(attack_test, anomaly_scores)
    avg_prec = float(np.mean(prec_arr))
    attack_metrics = {
        "auroc":            float(auroc),
        "best_f1":          float(best_f1),
        "best_threshold":   float(best_thresh),
        "avg_precision":    avg_prec,
        "n_attack_windows": int(attack_test.sum()),
        "n_normal_windows": int((attack_test == 0).sum()),
    }
    log.info("Attack detection — AUROC=%.4f  F1=%.4f  AvgPrec=%.4f  (%d/%d attack windows)",
             auroc, best_f1, avg_prec, attack_metrics['n_attack_windows'], N_test)
else:
    log.warning("No attack windows in test set — skipping attack metrics.")

results.update({
    "test_nrmse_per_pv": {name: float(v) for name, v in zip(PV_COLS, nrmse_test)},
    "test_mean_nrmse":   float(np.mean(nrmse_test)),
    "attack_detection":  attack_metrics,
})
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
log.info("Results updated: results.json")

# ── 6. Save checkpoints ───────────────────────────────────────────────────────
log.info("Step 6: Saving checkpoints to %s", OUT_DIR)

torch.save({
    "model_state": plant_model.state_dict(),
    "n_plant_in":  N_PLANT_IN,
    "n_pv":        N_PV,
    "n_scenarios": N_SCENARIOS,
    "hidden":      HIDDEN,
    "layers":      LAYERS,
}, OUT_DIR / "gru_plant.pt")
log.info("Saved: gru_plant.pt")

for ln, m in ctrl_models.items():
    torch.save({
        "model_state":   m.state_dict(),
        "arch":          "CCSequenceModel" if ln == 'CC' else "GRUController",
        "n_inputs":      ctrl_data[ln]['X_train'].shape[-1],
        "hidden":        CTRL_HIDDEN_PER_LOOP[ln],
        "layers":        CTRL_LAYERS,
        "causal_layers": EXTRA_CHANNELS[ln],
    }, OUT_DIR / f"gru_ctrl_{ln.lower()}.pt")
    log.info("Saved: gru_ctrl_%s.pt", ln.lower())

log.info("Done.")
