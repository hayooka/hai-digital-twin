"""
tune_transformer.py — Evolutionary hyperparameter search for TransformerPlant.

Same design as tune_gru.py — see that file for full explanation.

TRANSFORMER-SPECIFIC NOTES:
  - d_model MUST be divisible by n_heads (enforced in search space).
  - Batch is capped at 32 due to O(T^2) attention memory on 10GB GPU.
    (T=180 steps: batch=32 uses ~4GB, batch=64 causes OOM on RTX 3080)
  - n_heads choices are paired with d_model choices to always be divisible.

Run:
    python 03_model/tune_transformer.py
"""

import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import optuna
from optuna.samplers import CmaEsSampler
from optuna.pruners import MedianPruner

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from transformer import TransformerPlant, TransformerController, TransformerCCClassifierRegressor
from config import LOOPS, PV_COLS

# ── Fixed settings ────────────────────────────────────────────────────────────
SEED           = 42
N_TRIALS       = 30
N_TRIAL_EPOCHS = 40
CTRL_HIDDEN    = 64
CTRL_LAYERS    = 2
CC_HIDDEN      = 32
CC_HEADS       = 4
CC_LAYERS      = 2
CTRL_LR        = 1e-3
WD             = 1e-5
GRAD_CLIP      = 1.0
SS_START       = 10
SCH_PAT        = 5
SCH_FAC        = 0.5
# Batch is capped for Transformer — attention is O(T^2), 10GB VRAM limit
MAX_BATCH      = 32

OUT_DIR = Path("outputs/transformer_tune")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load data ONCE ────────────────────────────────────────────────────────────
print("Loading data...")
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

N_PLANT_IN  = plant_data['n_plant_in']
N_PV        = plant_data['n_pv']
N_SCENARIOS = data['metadata']['n_scenarios']
N           = len(X_train)

pv_set      = set(PV_COLS)
non_pv_cols = [c for c in sensor_cols if c not in pv_set]
col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}
CTRL_LOOPS  = ['PC', 'LC', 'FC', 'TC']
print(f"Data loaded. Train: {X_train.shape}  Val: {X_val.shape}")


def objective(trial: optuna.Trial) -> float:
    # d_model / n_heads pairs that are always divisible
    # CMA-ES will choose from these valid combinations
    arch = trial.suggest_categorical("arch", [
        "64_4", "128_4", "128_8", "256_4", "256_8",
    ])
    d_model, n_heads = int(arch.split("_")[0]), int(arch.split("_")[1])

    n_layers = trial.suggest_categorical("n_layers", [2, 3, 4])
    emb_dim  = trial.suggest_categorical("emb_dim",  [16, 32, 64])
    dropout  = trial.suggest_float("dropout", 0.05, 0.3)
    lr       = trial.suggest_float("lr",      1e-4,  1e-2, log=True)
    batch    = trial.suggest_categorical("batch", [8, 16, 32])
    ss_end   = trial.suggest_int("ss_end",  30, 100)
    ss_max   = trial.suggest_float("ss_max", 0.3, 0.7)

    # Safety cap — never exceed MAX_BATCH to avoid OOM
    batch = min(batch, MAX_BATCH)

    def ss_ratio(epoch):
        if epoch < SS_START:  return 0.0
        if epoch >= ss_end:   return ss_max
        return ss_max * (epoch - SS_START) / (ss_end - SS_START)

    plant = TransformerPlant(
        n_plant_in  = N_PLANT_IN,
        n_pv        = N_PV,
        d_model     = d_model,
        n_heads     = n_heads,
        n_layers    = n_layers,
        n_scenarios = N_SCENARIOS,
        emb_dim     = emb_dim,
        dropout     = dropout,
    ).to(device)

    ctrl_models = {
        ln: TransformerController(
            n_inputs   = ctrl_data[ln]['X_train'].shape[-1],
            d_model    = CTRL_HIDDEN,
            n_heads    = 4,
            n_layers   = CTRL_LAYERS,
            dropout    = dropout,
            output_len = TARGET_LEN,
        ).to(device)
        for ln in CTRL_LOOPS
    }

    cc = TransformerCCClassifierRegressor(
        n_inputs=2, d_model=CC_HIDDEN, n_heads=CC_HEADS,
        n_layers=CC_LAYERS, dropout=dropout,
    ).to(device)

    plant_opt = torch.optim.Adam(plant.parameters(), lr=lr, weight_decay=WD)
    plant_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        plant_opt, patience=SCH_PAT, factor=SCH_FAC)
    ctrl_opts = {ln: torch.optim.Adam(m.parameters(), lr=CTRL_LR, weight_decay=WD)
                 for ln, m in ctrl_models.items()}
    cc_opt    = torch.optim.Adam(cc.parameters(), lr=CTRL_LR, weight_decay=WD)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    best_val = float("inf")

    for epoch in range(1, N_TRIAL_EPOCHS + 1):
        ss = ss_ratio(epoch)

        # Controllers
        for m in ctrl_models.values(): m.train()
        idx_c = np.random.permutation(len(ctrl_data['PC']['X_train']))
        for i in range(0, len(idx_c), batch):
            b = idx_c[i:i + batch]
            for ln in CTRL_LOOPS:
                Xc = torch.tensor(ctrl_data[ln]['X_train'][b]).float().to(device)
                yc = torch.tensor(ctrl_data[ln]['y_train'][b]).float().to(device)
                loss = mse(ctrl_models[ln](Xc, y_cv_teacher=yc, ss_ratio=ss), yc)
                ctrl_opts[ln].zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ctrl_models[ln].parameters(), GRAD_CLIP)
                ctrl_opts[ln].step()

        # CC
        cc.train()
        X_cc, y_cc = ctrl_data['CC']['X_train'], ctrl_data['CC']['y_train']
        for i in range(0, len(X_cc), batch):
            b  = np.random.permutation(len(X_cc))[i:i + batch]
            xi = torch.tensor(X_cc[b]).float().to(device)
            cv = torch.tensor(y_cc[b, 0, 0]).float().to(device)
            lbl = (cv > 0).float().unsqueeze(-1)
            lg, sp = cc(xi)
            lc = bce(lg, lbl)
            om = lbl.squeeze(-1).bool()
            lr_ = mse(sp[om], cv.unsqueeze(-1)[om]) if om.any() else torch.tensor(0., device=device)
            cc_opt.zero_grad(); (lc + lr_).backward(); cc_opt.step()

        # Plant
        plant.train()
        idx_p = np.random.permutation(N)
        for i in range(0, N, batch):
            b = idx_p[i:i + batch]
            x_cv   = torch.tensor(X_train[b]).float().to(device)
            x_cv_t = torch.tensor(X_cv_tgt_train[b]).float().to(device)
            pv_i   = torch.tensor(pv_init_train[b]).float().to(device)
            pv_tf  = torch.tensor(pv_teacher_tr[b]).float().to(device)
            pv_tgt = torch.tensor(pv_target_tr[b]).float().to(device)
            sc     = torch.tensor(scenario_train[b]).long().to(device)
            loss   = mse(plant(x_cv, x_cv_t, pv_i, sc, pv_teacher=pv_tf, ss_ratio=ss), pv_tgt)
            plant_opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(plant.parameters(), GRAD_CLIP)
            plant_opt.step()

        # Validation
        plant.eval()
        pval, nb = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(X_val), batch):
                x_cv   = torch.tensor(X_val[i:i+batch]).float().to(device)
                x_cv_t = torch.tensor(X_cv_tgt_val[i:i+batch]).float().to(device)
                pv_i   = torch.tensor(pv_init_val[i:i+batch]).float().to(device)
                pv_tf  = torch.tensor(pv_teacher_val[i:i+batch]).float().to(device)
                pv_tgt = torch.tensor(pv_target_val[i:i+batch]).float().to(device)
                sc     = torch.tensor(scenario_val[i:i+batch]).long().to(device)
                pval  += mse(plant(x_cv, x_cv_t, pv_i, sc, pv_teacher=pv_tf, ss_ratio=0.0), pv_tgt).item()
                nb    += 1

        pval /= max(1, nb)
        plant_sch.step(pval)
        best_val = min(best_val, pval)

        trial.report(pval, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val


sampler = CmaEsSampler(seed=SEED)
pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
study   = optuna.create_study(
    direction="minimize",
    sampler=sampler,
    pruner=pruner,
    study_name="transformer_plant_tune",
)

print(f"\nStarting CMA-ES search: {N_TRIALS} trials × {N_TRIAL_EPOCHS} epochs each")
print(f"Batch capped at {MAX_BATCH} to avoid GPU OOM.\n")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best = study.best_trial
print(f"\n{'='*60}")
print(f"Best trial: val_loss = {best.value:.6f}")
print(f"Best hyperparameters:")
for k, v in best.params.items():
    print(f"  {k}: {v}")

best_params = {"val_loss": best.value, "params": best.params}
with open(OUT_DIR / "best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
print(f"\nSaved: {OUT_DIR}/best_params.json")
print("Now plug these values into train_transformer.py and run a full training.")
