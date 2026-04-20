"""
train_gru_scenario_weighted.py — Fine-tune GRU-Causal-Plus with per-scenario loss weights.

Warm-starts from outputs/gru_causal_plus_added/gru_plant.pt (already trained).
Only change: weighted_mse now uses per-scenario weights instead of binary attack/normal.

Weight logic (data imbalance × reconstruction difficulty):
  Normal  (sc=0): 1.0  — baseline
  AP_no   (sc=1): 3.0  — decent already, keep moderate pressure
  AP_with (sc=2): 6.0  — hardest (10% NRMSE) AND among fewest windows
  AE_no   (sc=3): 2.0  — very few windows but already predicts well

Fine-tune config (not full retrain):
  LR=0.0003  (gentle — already near optimum)
  PATIENCE=20 (shorter — already near optimum)
  Controllers loaded from causal_plus_added, NOT retrained

Run:
    python 03_model/train_gru_scenario_weighted.py
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))
sys.path.insert(0, str(ROOT / "04_evaluate"))

import joblib
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCSequenceModel
from config import LOOPS, PV_COLS, PROCESSED_DATA_DIR
from plot_results import plot_training_curves

# ── Config ────────────────────────────────────────────────────────────────────
SEED    = 42

# Per-scenario weights (replaces binary ATTACK_WEIGHT)
SCENARIO_WEIGHTS = {
    0: 1.0,   # Normal     — baseline
    1: 3.0,   # AP_no      — moderate, already decent
    2: 6.0,   # AP_with    — hardest + fewest windows
    3: 2.0,   # AE_no      — very few windows but predicts well already
}
TIT03_LOSS_WEIGHT = 2.0   # keep TIT03 upweighting from causal_plus

HIDDEN  = 512
LAYERS  = 2
DROPOUT = 0.05368612920084348

CTRL_HIDDEN_PER_LOOP = {'PC': 64, 'LC': 64, 'FC': 128, 'TC': 64, 'CC': 64}
CTRL_LAYERS = 2

EXTRA_CHANNELS = {
    'PC': ['P1_PCV02D',  'P1_FT01',   'P1_TIT01'],
    'LC': ['P1_FT03',   'P1_FCV03D', 'P1_PCV01D'],
    'FC': ['P1_PIT01',  'P1_LIT01',  'P1_TIT03'],
    'TC': ['P1_FT02',   'P1_PIT02',  'P1_TIT02'],
    'CC': ['P1_PP04D',  'P1_FCV03D', 'P1_PCV02D'],
}

# Fine-tune settings (not full retrain)
EPOCHS    = 100
BATCH     = 64
LR        = 0.0003   # gentle — already near optimum from causal_plus
CTRL_LR   = 0.001    # controllers already trained, just gentle update
WD        = 1e-5
GRAD_CLIP = 1.0
SCH_PAT   = 6
SCH_FAC   = 0.5
PATIENCE  = 20       # shorter — already near optimum

# Scheduled sampling — start from SS_MAX immediately (already trained phase)
SS_START  = 0
SS_END    = 20
SS_MAX    = 0.483

# Warm-start from best causal_plus run
WARMSTART_CKPT     = ROOT / "outputs/pipeline/gru_causal_plus_tuned/gru_plant.pt"
WARMSTART_CTRL_DIR = ROOT / "outputs/pipeline/gru_causal_plus_tuned"

OUT_DIR = ROOT / "outputs/pipeline/gru_scenario_weighted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading data...")
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
attack_test    = plant_data['attack_test']

N_PLANT_IN  = plant_data['n_plant_in']
N_PV        = plant_data['n_pv']
N_SCENARIOS = data['metadata']['n_scenarios']
N           = len(X_train)

print(f"  Train: {X_train.shape}   Val: {X_val.shape}")
for sc_id, w in SCENARIO_WEIGHTS.items():
    n = (scenario_train == sc_id).sum()
    print(f"  sc={sc_id}: {n:5d} train windows  weight={w}")

# ── Augment ctrl_data ─────────────────────────────────────────────────────────
def augment_ctrl_data(ctrl_data: dict, sensor_cols: list) -> None:
    plant_scaler = joblib.load(f"{PROCESSED_DATA_DIR}/scaler.pkl")
    npz_train = np.load(f"{PROCESSED_DATA_DIR}/train_data.npz")
    npz_val   = np.load(f"{PROCESSED_DATA_DIR}/val_data.npz")
    npz_test  = np.load(f"{PROCESSED_DATA_DIR}/test_data.npz")
    col_idx   = {c: i for i, c in enumerate(sensor_cols)}
    for ln, extra_cols in EXTRA_CHANNELS.items():
        for extra_col in extra_cols:
            if extra_col not in col_idx: continue
            ei      = col_idx[extra_col]
            mean_e  = plant_scaler.mean_[ei]
            scale_e = plant_scaler.scale_[ei]
            for split, npz in [('train', npz_train), ('val', npz_val), ('test', npz_test)]:
                raw = npz['X'][:, :, [ei]].astype(np.float32)
                sc  = (raw - mean_e) / scale_e
                ctrl_data[ln][f'X_{split}'] = np.concatenate(
                    [ctrl_data[ln][f'X_{split}'], sc], axis=-1)
        print(f"  {ln}: → n_inputs={ctrl_data[ln]['X_train'].shape[-1]}")

print("\nAugmenting controller data...")
augment_ctrl_data(ctrl_data, sensor_cols)

pv_set      = set(PV_COLS)
non_pv_cols = [c for c in sensor_cols if c not in pv_set]
col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}
CTRL_LOOPS  = ['PC', 'LC', 'FC', 'TC', 'CC']
ctrl_cv_col_idx = {ln: col_to_idx[LOOPS[ln].cv]
                   for ln in CTRL_LOOPS if LOOPS[ln].cv in col_to_idx}

# ── 2. Build & load models ────────────────────────────────────────────────────
print(f"\nStep 2: Loading models... (device: {device})")

plant_model = GRUPlant(
    n_plant_in=N_PLANT_IN, n_pv=N_PV,
    hidden=HIDDEN, layers=LAYERS,
    n_scenarios=N_SCENARIOS, dropout=DROPOUT,
).to(device)

ckpt = torch.load(WARMSTART_CKPT, map_location=device)
plant_model.load_state_dict(ckpt["model_state"])
print(f"  Loaded plant from {WARMSTART_CKPT.name}"
      f"  (epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss','?')})")

# Load controllers from causal_plus_added
ctrl_models = {}
for ln in CTRL_LOOPS:
    n_in = ctrl_data[ln]['X_train'].shape[-1]
    h    = CTRL_HIDDEN_PER_LOOP[ln]
    ctrl_path = WARMSTART_CTRL_DIR / f"gru_ctrl_{ln.lower()}.pt"
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
    if ctrl_path.exists():
        c = torch.load(ctrl_path, map_location=device)
        ctrl_models[ln].load_state_dict(c["model_state"])
        print(f"  Loaded controller {ln} from {ctrl_path.name}")
    else:
        print(f"  WARNING: {ctrl_path.name} not found — random init")

# ── 3. Loss & optimizers ──────────────────────────────────────────────────────
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()

# Build weight tensor once (maps scenario id → scalar weight)
sc_weight_tensor = torch.tensor(
    [SCENARIO_WEIGHTS[i] for i in range(N_SCENARIOS)],
    dtype=torch.float32, device=device
)

def weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                 scenario: torch.Tensor) -> torch.Tensor:
    """Per-scenario weighted MSE with TIT03 channel upweighting."""
    loss_main  = ((pred[:, :, :4] - target[:, :, :4]) ** 2).mean(dim=(1, 2))
    loss_tit03 = ((pred[:, :, 4:] - target[:, :, 4:]) ** 2).mean(dim=(1, 2))
    loss    = loss_main + TIT03_LOSS_WEIGHT * loss_tit03
    weights = sc_weight_tensor[scenario]   # (B,) — per-sample scenario weight
    return (loss * weights).mean()

def ss_ratio_for(epoch: int) -> float:
    if epoch < SS_START:  return 0.0
    if epoch >= SS_END:   return SS_MAX
    return SS_MAX * (epoch - SS_START) / (SS_END - SS_START)

plant_opt = torch.optim.Adam(plant_model.parameters(), lr=LR, weight_decay=WD)
plant_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
    plant_opt, patience=SCH_PAT, factor=SCH_FAC)
ctrl_opts = {ln: torch.optim.Adam(m.parameters(), lr=CTRL_LR, weight_decay=WD)
             for ln, m in ctrl_models.items()}

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
            ctrl_opts[ln].zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ctrl_models[ln].parameters(), GRAD_CLIP)
            ctrl_opts[ln].step()
            totals[ln] += loss.item()
    return {ln: totals[ln] / max(1, N_ctrl // BATCH) for ln in CTRL_LOOPS}

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
print(f"\nStep 3: Fine-tuning {EPOCHS} epochs  LR={LR}  PATIENCE={PATIENCE}")
print(f"  Scenario weights: {SCENARIO_WEIGHTS}")

best_val   = float("inf")
best_state = {k: v.clone() for k, v in plant_model.state_dict().items()}
patience_ctr = 0
train_losses, val_losses, ss_ratios = [], [], []

for epoch in range(1, EPOCHS + 1):
    ss = ss_ratio_for(epoch)
    train_controllers(ss)

    plant_model.train()
    idx, total = np.random.permutation(N), 0.0
    for i in range(0, N, BATCH):
        b      = idx[i:i + BATCH]
        x_cv   = torch.tensor(X_train[b]).float().to(device)
        x_cv_t = torch.tensor(X_cv_tgt_train[b]).float().to(device)
        pv_ini = torch.tensor(pv_init_train[b]).float().to(device)
        pv_tf  = torch.tensor(pv_teacher_tr[b]).float().to(device)
        pv_tgt = torch.tensor(pv_target_tr[b]).float().to(device)
        sc     = torch.tensor(scenario_train[b]).long().to(device)
        pred   = plant_model(x_cv, x_cv_t, pv_ini, sc, pv_teacher=pv_tf, ss_ratio=ss)
        loss   = weighted_mse(pred, pv_tgt, sc)
        plant_opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(plant_model.parameters(), GRAD_CLIP)
        plant_opt.step()
        total += loss.item()

    plant_model.eval()
    pval, n_b = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X_val), BATCH):
            x_cv   = torch.tensor(X_val[i:i+BATCH]).float().to(device)
            x_cv_t = torch.tensor(X_cv_tgt_val[i:i+BATCH]).float().to(device)
            pv_ini = torch.tensor(pv_init_val[i:i+BATCH]).float().to(device)
            pv_tf  = torch.tensor(pv_teacher_val[i:i+BATCH]).float().to(device)
            pv_tgt = torch.tensor(pv_target_val[i:i+BATCH]).float().to(device)
            sc     = torch.tensor(scenario_val[i:i+BATCH]).long().to(device)
            pval  += mse(plant_model(x_cv, x_cv_t, pv_ini, sc,
                                     pv_teacher=pv_tf, ss_ratio=0.0), pv_tgt).item()
            n_b   += 1

    pval /= max(1, n_b)
    plant_sch.step(pval)
    train_losses.append(total / max(1, N // BATCH))
    val_losses.append(pval)
    ss_ratios.append(ss)

    if pval < best_val:
        best_val   = pval
        best_state = {k: v.clone() for k, v in plant_model.state_dict().items()}
        patience_ctr = 0
        torch.save({
            "model_state": best_state,
            "n_plant_in":  N_PLANT_IN,
            "n_pv":        N_PV,
            "n_scenarios": N_SCENARIOS,
            "hidden":      HIDDEN,
            "layers":      LAYERS,
            "epoch":       epoch,
            "val_loss":    best_val,
        }, OUT_DIR / "gru_plant.pt")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | train={total/max(1,N//BATCH):.5f}"
              f"  val={pval:.5f}  ss={ss:.3f}")

plant_model.load_state_dict(best_state)
print(f"\n  Best val loss: {best_val:.5f}")

plot_training_curves(
    train_losses, val_losses, ss_ratios,
    model_name="GRU-Scenario-Weighted",
    save_path=OUT_DIR / "gru_loss_curves.png",
)

# ── 5. Test evaluation ────────────────────────────────────────────────────────
print(f"\nStep 4: Test-set evaluation ({len(X_test)} windows)...")
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
        pv_seq   = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
        pv_preds_te[i:i + pv_seq.size(0)] = pv_seq.cpu().numpy()

SCENARIO_SHORT = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}

print("\n  Per-scenario NRMSE (test):")
scenario_nrmse = {}
for sc_id, sc_name in SCENARIO_SHORT.items():
    mask = (scenario_test == sc_id)
    if mask.sum() == 0: continue
    sc_nrmse = []
    for k in range(N_PV):
        true_k = pv_target_test[mask, :, k]
        pred_k = pv_preds_te[mask, :, k]
        rmse   = np.sqrt(np.mean((pred_k - true_k) ** 2))
        rng    = max(float(true_k.max() - true_k.min()), 1e-6)
        sc_nrmse.append(rmse / rng)
    scenario_nrmse[sc_name] = sc_nrmse
    print(f"    {sc_name:<10}: {' '.join(f'{v:.4f}' for v in sc_nrmse)}  mean={np.mean(sc_nrmse):.4f}")

normal_mask = (attack_test == 0)
nrmse_test  = []
for k in range(N_PV):
    true_k = pv_target_test[normal_mask, :, k]
    pred_k = pv_preds_te[normal_mask, :, k]
    rmse   = np.sqrt(np.mean((pred_k - true_k) ** 2))
    rng    = max(float(true_k.max() - true_k.min()), 1e-6)
    nrmse_test.append(rmse / rng)

print(f"\n  Test NRMSE (normal): mean={np.mean(nrmse_test):.4f}")
for name, v in zip(PV_COLS, nrmse_test):
    print(f"    {name:<30s}: {v:.4f}")

anomaly_scores = np.mean((pv_preds_te - pv_target_test) ** 2, axis=(1, 2))
attack_metrics = {}
if attack_test.sum() > 0:
    auroc = roc_auc_score(attack_test, anomaly_scores)
    thresholds = np.percentile(anomaly_scores, np.linspace(50, 99, 100))
    best_f1, best_thresh = 0.0, thresholds[0]
    for t in thresholds:
        f1 = f1_score(attack_test, anomaly_scores > t, zero_division=0)
        if f1 > best_f1: best_f1, best_thresh = f1, t
    prec_arr, _, _ = precision_recall_curve(attack_test, anomaly_scores)
    attack_metrics = {
        "auroc":            float(auroc),
        "best_f1":          float(best_f1),
        "best_threshold":   float(best_thresh),
        "avg_precision":    float(np.mean(prec_arr)),
        "n_attack_windows": int(attack_test.sum()),
        "n_normal_windows": int((attack_test == 0).sum()),
    }
    print(f"\n  Attack detection: AUROC={auroc:.4f}  F1={best_f1:.4f}")

results = {
    "model": "GRU-Scenario-Weighted",
    "causal_layers": {ln: cols for ln, cols in EXTRA_CHANNELS.items()},
    "hyperparams": {
        "LR": LR, "CTRL_LR": CTRL_LR, "BATCH": BATCH, "PATIENCE": PATIENCE,
        "SCENARIO_WEIGHTS": SCENARIO_WEIGHTS,
        "TIT03_LOSS_WEIGHT": TIT03_LOSS_WEIGHT,
        "warmstart": str(WARMSTART_CKPT),
    },
    "best_val_loss":  float(best_val),
    "test_nrmse_per_pv": {name: float(v) for name, v in zip(PV_COLS, nrmse_test)},
    "test_mean_nrmse":   float(np.mean(nrmse_test)),
    "test_nrmse_per_scenario": {
        sc: {name: float(v) for name, v in zip(PV_COLS, vals)}
        for sc, vals in scenario_nrmse.items()
    },
    "attack_detection": attack_metrics,
    "train_losses": [float(x) for x in train_losses],
    "val_losses":   [float(x) for x in val_losses],
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved: results.json")

# ── 6. Save checkpoints ───────────────────────────────────────────────────────
print(f"\nStep 5: Saving to {OUT_DIR}/")
for ln, m in ctrl_models.items():
    torch.save({
        "model_state":   m.state_dict(),
        "arch":          "CCSequenceModel" if ln == 'CC' else "GRUController",
        "n_inputs":      ctrl_data[ln]['X_train'].shape[-1],
        "hidden":        CTRL_HIDDEN_PER_LOOP[ln],
        "layers":        CTRL_LAYERS,
        "causal_layers": EXTRA_CHANNELS[ln],
    }, OUT_DIR / f"gru_ctrl_{ln.lower()}.pt")
    print(f"  Saved: gru_ctrl_{ln.lower()}.pt")

print("\nDone.")
