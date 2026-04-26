"""
finetune_lstm.py — Grid search fine-tuning for LSTMPlant.

WHAT THIS DOES:
    Loads the already-trained LSTMPlant checkpoint and continues training
    (fine-tuning) with a lower learning rate, while searching over 4 training
    hyperparameters that can be changed without rebuilding the architecture:

        ss_max    — how aggressively autoregressive the model becomes
        ss_end    — epoch at which scheduled sampling reaches ss_max
        dropout   — regularisation strength (rebuilds model with new dropout,
                    loads old weights — same architecture so weights are compatible)
        patience  — early stopping patience

    WHY FINE-TUNING INSTEAD OF RETRAINING:
        Starting from trained weights means each run converges in ~15-25 epochs
        instead of 90+. The full grid (16 combinations) takes ~1 hour on RTX 3080
        vs ~20 hours for 16 full training runs from scratch.

    WHY THESE 4 PARAMETERS:
        - ss_max / ss_end  : control autoregressive exposure — the biggest lever
                             found empirically (SS_END 40→80 gave 68% NRMSE drop)
        - dropout          : can regularise more or less without changing model size
        - patience         : controls how long to wait before stopping

    PREREQUISITE:
        lstm_plant.pt must exist in outputs/lstm_plant/.
        If it does not exist, retrain first:
            python 03_model/train_lstm.py

    SEARCH GRID (2 values each = 16 combinations):
        ss_max   : 0.5, 0.6
        ss_end   : 80, 100
        dropout  : 0.05, 0.1
        patience : 20, 30

    FINE-TUNE LR:
        1e-4 (10x lower than original 1e-3 — avoids destroying learned features)

Run:
    python 03_model/finetune_lstm.py
"""

import sys
import json
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from lstm import LSTMPlant, LSTMController, LSTMCCClassifierRegressor
from config import LOOPS, PV_COLS

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_DIR   = Path("outputs/lstm_plant")
PLANT_CKPT = CKPT_DIR / "lstm_plant.pt"
OUT_DIR    = Path("outputs/lstm_finetune")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Check checkpoint exists before doing anything else ───────────────────────
if not PLANT_CKPT.exists():
    print("=" * 60)
    print("ERROR: No checkpoint found at outputs/lstm_plant/lstm_plant.pt")
    print()
    print("The LSTM model must be trained first. Run:")
    print("    python 03_model/train_lstm.py")
    print()
    print("After training completes, run this script again.")
    print("=" * 60)
    sys.exit(1)

# ── Fixed settings (not searched) ────────────────────────────────────────────
SEED        = 42
FINETUNE_LR = 1e-4       # 10x lower than original — avoids destroying learned weights
EPOCHS      = 150        # upper bound; early stopping will trigger much sooner
CTRL_LR     = 1e-4       # also fine-tune controllers gently
WD          = 1e-5
GRAD_CLIP   = 1.0
SS_START    = 10         # keep same start point as original training
SCH_PAT     = 5
SCH_FAC     = 0.5
CTRL_HIDDEN = 64
CTRL_LAYERS = 2
CC_HIDDEN   = 32

# ── Grid search space ─────────────────────────────────────────────────────────
GRID = {
    "ss_max":   [0.5, 0.6],
    "ss_end":   [80, 100],
    "dropout":  [0.05, 0.1],
    "patience": [20, 30],
}

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load data ONCE ────────────────────────────────────────────────────────────
print("\nLoading data...")
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
ctrl_cv_col_idx = {ln: col_to_idx[LOOPS[ln].cv]
                   for ln in CTRL_LOOPS if LOOPS[ln].cv in col_to_idx}

print(f"Train: {X_train.shape}  Val: {X_val.shape}")

# ── Load original checkpoint metadata ────────────────────────────────────────
ckpt = torch.load(PLANT_CKPT, map_location="cpu")
HIDDEN      = ckpt["hidden"]
LAYERS      = ckpt["layers"]
print(f"\nLoaded checkpoint: hidden={HIDDEN}, layers={LAYERS}")
print(f"Original val_loss: {ckpt.get('val_loss', 'unknown'):.6f}" if 'val_loss' in ckpt else "")


# ── Helper: build and load a plant model with given dropout ───────────────────
def load_plant(dropout: float) -> LSTMPlant:
    """
    Rebuild LSTMPlant with the given dropout rate, then load saved weights.
    Hidden/layers come from the checkpoint so architecture is always compatible.
    Dropout changes the fc layer and LSTM inter-layer dropout — safe to vary.
    """
    model = LSTMPlant(
        n_plant_in=N_PLANT_IN, n_pv=N_PV,
        hidden=HIDDEN, layers=LAYERS,
        n_scenarios=N_SCENARIOS, dropout=dropout,
    )
    # Load weights — strict=False allows minor mismatches in dropout (safe here)
    model.load_state_dict(ckpt["model_state"], strict=False)
    return model.to(device)


# ── Helper: load controller checkpoints ──────────────────────────────────────
def load_controllers() -> dict:
    models = {}
    for ln in CTRL_LOOPS:
        cp_path = CKPT_DIR / f"lstm_ctrl_{ln.lower()}.pt"
        if cp_path.exists():
            cp = torch.load(cp_path, map_location="cpu")
            m  = LSTMController(
                n_inputs   = cp["n_inputs"],
                hidden     = cp["hidden"],
                layers     = cp["layers"],
                dropout    = 0.1,
                output_len = TARGET_LEN,
            )
            m.load_state_dict(cp["model_state"])
            models[ln] = m.to(device)
        else:
            # No checkpoint — build fresh (controllers converge fast)
            models[ln] = LSTMController(
                n_inputs   = ctrl_data[ln]['X_train'].shape[-1],
                hidden     = CTRL_HIDDEN,
                layers     = CTRL_LAYERS,
                dropout    = 0.1,
                output_len = TARGET_LEN,
            ).to(device)
    return models


def load_cc() -> LSTMCCClassifierRegressor:
    cp_path = CKPT_DIR / "cc_model.pt"
    cc = LSTMCCClassifierRegressor(n_inputs=2, hidden=CC_HIDDEN, dropout=0.1).to(device)
    if cp_path.exists():
        cp = torch.load(cp_path, map_location="cpu")
        cc.load_state_dict(cp["model_state"])
    return cc


# ── Helper: compute closed-loop val NRMSE ────────────────────────────────────
def compute_nrmse(plant: LSTMPlant, ctrl_models: dict, batch: int = 64) -> float:
    plant.eval()
    for m in ctrl_models.values(): m.eval()

    N_val    = len(X_val)
    pv_preds = np.zeros((N_val, TARGET_LEN, N_PV), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, N_val, batch):
            sl        = slice(i, i + batch)
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

            pv_seq   = plant.predict(x_cv_b, xct_b, pv_init_b, sc_b)
            B_actual = pv_seq.size(0)
            pv_preds[i:i + B_actual] = pv_seq.cpu().numpy()

    nrmse = []
    for k in range(N_PV):
        true_k = pv_target_val[:, :, k]
        pred_k = pv_preds[:, :, k]
        rmse   = np.sqrt(np.mean((pred_k - true_k) ** 2))
        rng    = max(float(true_k.max() - true_k.min()), 1e-6)
        nrmse.append(rmse / rng)
    return float(np.mean(nrmse))


# ── Main grid search loop ─────────────────────────────────────────────────────
keys   = list(GRID.keys())
values = list(GRID.values())
combos = list(itertools.product(*values))

print(f"\n{'='*60}")
print(f"Fine-tuning grid search: {len(combos)} combinations")
print(f"Parameters: {keys}")
print(f"Fine-tune LR: {FINETUNE_LR}  (original was 1e-3)")
print(f"{'='*60}\n")

results_table = []
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()

for run_idx, combo in enumerate(combos, 1):
    params = dict(zip(keys, combo))
    ss_max   = params["ss_max"]
    ss_end   = params["ss_end"]
    dropout  = params["dropout"]
    patience = params["patience"]

    print(f"--- Run {run_idx}/{len(combos)}: "
          f"ss_max={ss_max}  ss_end={ss_end}  "
          f"dropout={dropout}  patience={patience} ---")

    # Load fresh model from checkpoint with this dropout
    plant       = load_plant(dropout)
    ctrl_models = load_controllers()
    cc          = load_cc()

    plant_opt = torch.optim.Adam(plant.parameters(), lr=FINETUNE_LR, weight_decay=WD)
    plant_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        plant_opt, patience=SCH_PAT, factor=SCH_FAC)
    ctrl_opts = {ln: torch.optim.Adam(m.parameters(), lr=CTRL_LR, weight_decay=WD)
                 for ln, m in ctrl_models.items()}
    cc_opt    = torch.optim.Adam(cc.parameters(), lr=CTRL_LR, weight_decay=WD)

    def ss_ratio(epoch):
        if epoch < SS_START:  return 0.0
        if epoch >= ss_end:   return ss_max
        return ss_max * (epoch - SS_START) / (ss_end - SS_START)

    best_val     = float("inf")
    best_state   = None
    pat_counter  = 0
    BATCH        = 128

    for epoch in range(1, EPOCHS + 1):
        ss = ss_ratio(epoch)

        # Controllers
        for m in ctrl_models.values(): m.train()
        idx_c = np.random.permutation(len(ctrl_data['PC']['X_train']))
        for i in range(0, len(idx_c), BATCH):
            b = idx_c[i:i + BATCH]
            for ln in CTRL_LOOPS:
                Xc   = torch.tensor(ctrl_data[ln]['X_train'][b]).float().to(device)
                yc   = torch.tensor(ctrl_data[ln]['y_train'][b]).float().to(device)
                loss = mse(ctrl_models[ln](Xc, y_cv_teacher=yc, ss_ratio=ss), yc)
                ctrl_opts[ln].zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ctrl_models[ln].parameters(), GRAD_CLIP)
                ctrl_opts[ln].step()

        # CC
        cc.train()
        X_cc, y_cc = ctrl_data['CC']['X_train'], ctrl_data['CC']['y_train']
        for i in range(0, len(X_cc), BATCH):
            b   = np.random.permutation(len(X_cc))[i:i + BATCH]
            xi  = torch.tensor(X_cc[b]).float().to(device)
            cv  = torch.tensor(y_cc[b, 0, 0]).float().to(device)
            lbl = (cv > 0).float().unsqueeze(-1)
            lg, sp = cc(xi)
            lc  = bce(lg, lbl)
            om  = lbl.squeeze(-1).bool()
            lr_ = mse(sp[om], cv.unsqueeze(-1)[om]) if om.any() else torch.tensor(0., device=device)
            cc_opt.zero_grad(); (lc + lr_).backward(); cc_opt.step()

        # Plant
        plant.train()
        idx_p = np.random.permutation(N)
        for i in range(0, N, BATCH):
            b      = idx_p[i:i + BATCH]
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
            for i in range(0, len(X_val), BATCH):
                x_cv   = torch.tensor(X_val[i:i+BATCH]).float().to(device)
                x_cv_t = torch.tensor(X_cv_tgt_val[i:i+BATCH]).float().to(device)
                pv_i   = torch.tensor(pv_init_val[i:i+BATCH]).float().to(device)
                pv_tf  = torch.tensor(pv_teacher_val[i:i+BATCH]).float().to(device)
                pv_tgt = torch.tensor(pv_target_val[i:i+BATCH]).float().to(device)
                sc     = torch.tensor(scenario_val[i:i+BATCH]).long().to(device)
                pval  += mse(plant(x_cv, x_cv_t, pv_i, sc, pv_teacher=pv_tf, ss_ratio=0.0), pv_tgt).item()
                nb    += 1
        pval /= max(1, nb)
        plant_sch.step(pval)

        if pval < best_val:
            best_val    = pval
            best_state  = {k: v.clone() for k, v in plant.state_dict().items()}
            pat_counter = 0
        else:
            pat_counter += 1
            if pat_counter >= patience:
                print(f"  Early stopping at epoch {epoch}  best_val={best_val:.6f}")
                break

    # Compute closed-loop NRMSE with best weights
    plant.load_state_dict(best_state)
    nrmse = compute_nrmse(plant, ctrl_models)

    print(f"  Mean NRMSE: {nrmse:.5f}  (val_loss={best_val:.6f})\n")

    row = {**params, "val_loss": best_val, "mean_nrmse": nrmse}
    results_table.append(row)

    # Save checkpoint for this run
    run_dir = OUT_DIR / f"run_{run_idx:02d}"
    run_dir.mkdir(exist_ok=True)
    torch.save({
        "model_state": best_state,
        "n_plant_in":  N_PLANT_IN,
        "n_pv":        N_PV,
        "n_scenarios": N_SCENARIOS,
        "hidden":      HIDDEN,
        "layers":      LAYERS,
        **params,
    }, run_dir / "lstm_plant.pt")

# ── Print results table ───────────────────────────────────────────────────────
results_table.sort(key=lambda r: r["mean_nrmse"])

print(f"\n{'='*70}")
print("RESULTS TABLE (sorted by mean NRMSE):")
print(f"{'ss_max':>8} {'ss_end':>8} {'dropout':>9} {'patience':>9} {'NRMSE':>10}")
print("-" * 70)
for r in results_table:
    marker = " <-- BEST" if r is results_table[0] else ""
    print(f"{r['ss_max']:>8.1f} {r['ss_end']:>8d} {r['dropout']:>9.2f} "
          f"{r['patience']:>9d} {r['mean_nrmse']:>10.5f}{marker}")

best = results_table[0]
print(f"\nBest configuration:")
for k, v in best.items():
    print(f"  {k}: {v}")

with open(OUT_DIR / "grid_results.json", "w") as f:
    json.dump(results_table, f, indent=2)
print(f"\nSaved full results: {OUT_DIR}/grid_results.json")
print("Best model checkpoint: outputs/lstm_finetune/run_XX/lstm_plant.pt")
