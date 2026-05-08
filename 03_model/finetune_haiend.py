"""
finetune_haiend.py — Add HAIEND internal PLC signal output to gru_scenario_weighted.

Strategy:
  - Load gru_scenario_weighted (best generative model)
  - Freeze all weights EXCEPT the new haiend_head
  - Train haiend_head on all 4 scenarios with HAIEND targets
  - Save to outputs/pipeline/gru_scenario_haiend/

Why freeze everything else:
  - Preserves the excellent PV generation quality already achieved
  - Only teaches the model to also output 36 internal PLC signals
  - Fast: ~30-45 min on GPU

Run:
    python 03_model/finetune_haiend.py
"""

import logging
import json
import random
import shutil
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))
sys.path.insert(0, str(ROOT / "04_evaluate"))

from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCSequenceModel
from config import LOOPS, PV_COLS, HAIEND_COLS
from shared import CTRL_LOOPS, CTRL_HIDDEN_PER_LOOP, EXTRA_CHANNELS, augment_ctrl_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
SEED      = 42
EPOCHS    = 40
BATCH     = 256
LR        = 1e-4       # low LR — only HAIEND head trains
WD        = 1e-5
GRAD_CLIP = 1.0
SCH_PAT   = 5
SCH_FAC   = 0.5
PATIENCE  = 12

WARMSTART_DIR = ROOT / "outputs/pipeline/gru_scenario_weighted"
OUT_DIR       = ROOT / "outputs/pipeline/gru_scenario_haiend"
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. Load data (all scenarios) ───────────────────────────────────────────────
log.info("Step 1: Loading data (all 4 scenarios)...")
data        = load_and_prepare_data()
plant_data  = data['plant']
ctrl_data   = data['ctrl']
sensor_cols = data['metadata']['sensor_cols']
TARGET_LEN  = data['metadata']['target_len']

N_PLANT_IN  = plant_data['n_plant_in']
N_PV        = plant_data['n_pv']
N_HAIEND    = plant_data['n_haiend']
N_SCENARIOS = data['metadata']['n_scenarios']

log.info("Train windows: %d  Val windows: %d  HAIEND channels: %d",
         len(plant_data['X_train']), len(plant_data['X_val']), N_HAIEND)

log.info("Augmenting controller data...")
augment_ctrl_data(ctrl_data, sensor_cols)

pv_set      = set(PV_COLS)
non_pv_cols = [c for c in sensor_cols if c not in pv_set]
col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}
ctrl_cv_col_idx = {ln: col_to_idx[LOOPS[ln].cv]
                   for ln in CTRL_LOOPS if LOOPS[ln].cv in col_to_idx}

# ── 2. Load gru_scenario_weighted + add HAIEND head ───────────────────────────
log.info("Step 2: Loading gru_scenario_weighted checkpoint...")
ckpt_path = WARMSTART_DIR / "gru_plant.pt"
if not ckpt_path.exists():
    raise FileNotFoundError(f"Warm-start checkpoint not found: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
hidden = ckpt.get('hidden', 512)
layers = ckpt.get('layers', 2)
dropout = ckpt.get('dropout', 0.054)

plant_model = GRUPlant(
    n_plant_in=N_PLANT_IN, n_pv=N_PV,
    hidden=hidden, layers=layers,
    n_scenarios=N_SCENARIOS, dropout=0.0,
    n_haiend=N_HAIEND,
).to(device)

# Load existing weights (strict=False so haiend_head initialises fresh)
missing, unexpected = plant_model.load_state_dict(ckpt['model_state'], strict=False)
log.info("Loaded weights. Missing (new): %s", missing)
if unexpected:
    log.warning("Unexpected keys (removed from checkpoint): %s", unexpected)

# Freeze everything except haiend_head
for name, param in plant_model.named_parameters():
    if 'haiend_head' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable = sum(p.numel() for p in plant_model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in plant_model.parameters())
log.info("Trainable params: %s / %s (only haiend_head)",
         f"{trainable:,}", f"{total:,}")

# Load controller models (frozen, used only for CV prediction)
ctrl_models = {}
for ln in CTRL_LOOPS:
    n_in = ctrl_data[ln]['X_train'].shape[-1]
    h    = CTRL_HIDDEN_PER_LOOP[ln]
    if ln == 'CC':
        ctrl_models[ln] = CCSequenceModel(
            n_inputs=n_in, hidden=h, layers=2,
            dropout=0.0, output_len=TARGET_LEN).to(device)
    else:
        ctrl_models[ln] = GRUController(
            n_inputs=n_in, hidden=h, layers=2,
            dropout=0.0, output_len=TARGET_LEN).to(device)
    p = WARMSTART_DIR / f"gru_ctrl_{ln.lower()}.pt"
    if p.exists():
        c = torch.load(p, map_location=device, weights_only=True)
        ctrl_models[ln].load_state_dict(c['model_state'], strict=False)
    ctrl_models[ln].eval()
    for param in ctrl_models[ln].parameters():
        param.requires_grad = False

# ── 3. Optimiser (only HAIEND head) ───────────────────────────────────────────
opt = torch.optim.Adam(
    filter(lambda p: p.requires_grad, plant_model.parameters()),
    lr=LR, weight_decay=WD
)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=SCH_PAT, factor=SCH_FAC)
mse = nn.MSELoss()

# ── 4. Training loop ───────────────────────────────────────────────────────────
log.info("Step 3: Fine-tuning HAIEND head (%d epochs, PATIENCE=%d)", EPOCHS, PATIENCE)

X_tr        = plant_data['X_train']
X_cv_tr     = plant_data['X_cv_target_train']
pv_init_tr  = plant_data['pv_init_train']
pv_tgt_tr   = plant_data['pv_target_train']
haiend_tr   = plant_data['haiend_target_train']
sc_tr       = plant_data['scenario_train']

X_vl        = plant_data['X_val']
X_cv_vl     = plant_data['X_cv_target_val']
pv_init_vl  = plant_data['pv_init_val']
pv_tgt_vl   = plant_data['pv_target_val']
haiend_vl   = plant_data['haiend_target_val']
sc_vl       = plant_data['scenario_val']

N_train = len(X_tr)
N_val   = len(X_vl)

best_val   = float('inf')
best_state = None
wait       = 0
history    = {'train': [], 'val': []}

for epoch in range(1, EPOCHS + 1):
    plant_model.train()
    idx    = np.random.permutation(N_train)
    tr_loss = 0.0
    steps   = 0

    for i in range(0, N_train, BATCH):
        sl   = idx[i:i + BATCH]
        x_cv = torch.tensor(X_tr[sl]).float().to(device)
        xct  = torch.tensor(X_cv_tr[sl]).float().to(device).clone()
        pvi  = torch.tensor(pv_init_tr[sl]).float().to(device)
        pv_t = torch.tensor(pv_tgt_tr[sl]).float().to(device)
        hae  = torch.tensor(haiend_tr[sl]).float().to(device)
        sc   = torch.tensor(sc_tr[sl]).long().to(device)

        # inject CV predictions
        with torch.no_grad():
            for ln in CTRL_LOOPS:
                if ln not in ctrl_cv_col_idx: continue
                Xc      = torch.tensor(ctrl_data[ln]['X_train'][sl]).float().to(device)
                cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
                ci      = ctrl_cv_col_idx[ln]
                xct[:, :, ci:ci+1] = cv_pred

        opt.zero_grad()
        # use teacher forcing (alpha=0) — only HAIEND head trains
        pv_pred, haiend_pred = plant_model(x_cv, xct, pvi, sc, pv_teacher=pv_t, ss_ratio=0.0)

        loss = mse(haiend_pred, hae)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(plant_model.parameters(), GRAD_CLIP)
        opt.step()
        tr_loss += loss.item()
        steps   += 1

    tr_loss /= steps

    # validation
    plant_model.eval()
    vl_loss = 0.0
    vsteps  = 0
    with torch.no_grad():
        for i in range(0, N_val, BATCH):
            sl   = slice(i, i + BATCH)
            x_cv = torch.tensor(X_vl[sl]).float().to(device)
            xct  = torch.tensor(X_cv_vl[sl]).float().to(device).clone()
            pvi  = torch.tensor(pv_init_vl[sl]).float().to(device)
            pv_t = torch.tensor(pv_tgt_vl[sl]).float().to(device)
            hae  = torch.tensor(haiend_vl[sl]).float().to(device)
            sc   = torch.tensor(sc_vl[sl]).long().to(device)
            for ln in CTRL_LOOPS:
                if ln not in ctrl_cv_col_idx: continue
                Xc      = torch.tensor(ctrl_data[ln]['X_val'][sl]).float().to(device)
                cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
                ci      = ctrl_cv_col_idx[ln]
                xct[:, :, ci:ci+1] = cv_pred
            pv_pred, haiend_pred = plant_model(x_cv, xct, pvi, sc, pv_teacher=pv_t, ss_ratio=0.0)
            vl_loss += mse(haiend_pred, hae).item()
            vsteps  += 1

    vl_loss /= vsteps
    sch.step(vl_loss)
    history['train'].append(tr_loss)
    history['val'].append(vl_loss)

    if epoch % 5 == 0 or epoch == 1:
        log.info("Epoch %3d/%d | train=%.5f  val=%.5f", epoch, EPOCHS, tr_loss, vl_loss)

    if vl_loss < best_val:
        best_val   = vl_loss
        best_state = {k: v.cpu().clone() for k, v in plant_model.state_dict().items()}
        wait       = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            log.info("Early stop at epoch %d  (best val=%.5f)", epoch, best_val)
            break

# ── 5. Save ────────────────────────────────────────────────────────────────────
log.info("Step 4: Saving to %s", OUT_DIR)
plant_model.load_state_dict(best_state)

torch.save({
    'model_state': best_state,
    'hidden':      hidden,
    'layers':      layers,
    'n_haiend':    N_HAIEND,
    'val_loss':    best_val,
}, OUT_DIR / "gru_plant.pt")

for ln in CTRL_LOOPS:
    src = WARMSTART_DIR / f"gru_ctrl_{ln.lower()}.pt"
    if src.exists():
        shutil.copy(src, OUT_DIR / f"gru_ctrl_{ln.lower()}.pt")
    else:
        log.warning("Controller checkpoint not found: %s", src)
log.info("Controller checkpoints copied from gru_scenario_weighted")

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(history['train'], label='Train (HAIEND MSE)')
ax.plot(history['val'],   label='Val (HAIEND MSE)')
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
ax.set_title("HAIEND head fine-tune loss")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "haiend_loss_curves.png", dpi=150)
plt.close(fig)

log.info("Best val HAIEND loss: %.5f", best_val)
log.info("Done. Run monitor.py with --ckpt outputs/pipeline/gru_scenario_haiend/gru_plant.pt")
