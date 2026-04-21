"""
train_gru_normal_only.py — Digital twin trained exclusively on normal operation.

Goal:
  Learn pure plant physics under normal conditions (no attack knowledge).
  At inference, feed ANY data (including attacks) and the model outputs
  "what the plant should look like under normal physics given these inputs."
  The residual (actual - predicted) reveals the attack footprint.

Why this is useful:
  - No scenario label needed at inference — fully unsupervised anomaly detection
  - Works on unseen attack types (model never saw attacks)
  - Residual patterns differ per attack type → attack characterisation
  - Complements gru_scenario_weighted (which simulates attack behaviour)

Training:
  - Only normal windows (scenario=0): 10,148 samples
  - Warm-start from gru_causal_plus_tuned (already mostly normal-physics weights)
  - Fine-tune with normal-only loss → removes any attack contamination
  - No scenario embedding needed (n_scenarios=1, always sc=0)

Run:
    python 03_model/train_gru_normal_only.py
"""

import sys, json, random
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
from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCSequenceModel
from config import LOOPS, PV_COLS, PROCESSED_DATA_DIR
from plot_results import plot_training_curves

# ── Config ─────────────────────────────────────────────────────────────────────
SEED    = 42
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

EPOCHS    = 60
BATCH     = 256   # larger batch → fewer steps per epoch → faster
LR        = 0.0003
WD        = 1e-5
GRAD_CLIP = 1.0
SCH_PAT   = 6
SCH_FAC   = 0.5
PATIENCE  = 15
SS_START  = 0
SS_END    = 20
SS_MAX    = 0.483

WARMSTART_CKPT     = ROOT / "outputs/pipeline/gru_causal_plus_tuned/gru_plant.pt"
WARMSTART_CTRL_DIR = ROOT / "outputs/pipeline/gru_causal_plus_tuned"
OUT_DIR            = ROOT / "outputs/pipeline/gru_normal_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading data (normal windows only)...")
data        = load_and_prepare_data()
plant_data  = data['plant']
ctrl_data   = data['ctrl']
sensor_cols = data['metadata']['sensor_cols']
TARGET_LEN  = data['metadata']['target_len']

# Filter to normal windows only (scenario == 0)
scenario_train = plant_data['scenario_train']
scenario_val   = plant_data['scenario_val']
normal_tr  = scenario_train == 0
normal_val = scenario_val   == 0

X_train        = plant_data['X_train'][normal_tr]
X_cv_tgt_train = plant_data['X_cv_target_train'][normal_tr]
pv_init_train  = plant_data['pv_init_train'][normal_tr]
pv_teacher_tr  = plant_data['pv_teacher_train'][normal_tr]
pv_target_tr   = plant_data['pv_target_train'][normal_tr]
sc_train       = np.zeros(normal_tr.sum(), dtype=np.int64)  # always 0

X_val          = plant_data['X_val'][normal_val]
X_cv_tgt_val   = plant_data['X_cv_target_val'][normal_val]
pv_init_val    = plant_data['pv_init_val'][normal_val]
pv_teacher_val = plant_data['pv_teacher_val'][normal_val]
pv_target_val  = plant_data['pv_target_val'][normal_val]
sc_val         = np.zeros(normal_val.sum(), dtype=np.int64)

# Keep full test set (all scenarios) for residual evaluation
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

print(f"  Normal train windows : {N}  (of {len(scenario_train)} total)")
print(f"  Normal val   windows : {len(X_val)}  (of {len(scenario_val)} total)")
print(f"  Test windows (all)   : {len(X_test)}")

# ── Augment ctrl_data ──────────────────────────────────────────────────────────
def augment_ctrl_data(ctrl_data, sensor_cols):
    plant_scaler = joblib.load(f"{PROCESSED_DATA_DIR}/scaler.pkl")
    npz_train = np.load(f"{PROCESSED_DATA_DIR}/train_data.npz")
    npz_val   = np.load(f"{PROCESSED_DATA_DIR}/val_data.npz")
    npz_test  = np.load(f"{PROCESSED_DATA_DIR}/test_data.npz")
    col_idx   = {c: i for i, c in enumerate(sensor_cols)}
    for ln, extra_cols in EXTRA_CHANNELS.items():
        for extra_col in extra_cols:
            if extra_col not in col_idx: continue
            ei = col_idx[extra_col]
            mean_e, scale_e = plant_scaler.mean_[ei], plant_scaler.scale_[ei]
            for split, npz in [('train', npz_train), ('val', npz_val), ('test', npz_test)]:
                raw = npz['X'][:, :, [ei]].astype(np.float32)
                ctrl_data[ln][f'X_{split}'] = np.concatenate(
                    [ctrl_data[ln][f'X_{split}'], (raw - mean_e) / scale_e], axis=-1)
        print(f"  {ln}: → n_inputs={ctrl_data[ln]['X_train'].shape[-1]}")

print("\nAugmenting controller data...")
augment_ctrl_data(ctrl_data, sensor_cols)

pv_set      = set(PV_COLS)
non_pv_cols = [c for c in sensor_cols if c not in pv_set]
col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}
CTRL_LOOPS  = ['PC', 'LC', 'FC', 'TC', 'CC']
ctrl_cv_col_idx = {ln: col_to_idx[LOOPS[ln].cv]
                   for ln in CTRL_LOOPS if LOOPS[ln].cv in col_to_idx}

# ── 2. Build & warm-start model ────────────────────────────────────────────────
print(f"\nStep 2: Building model  (device: {device})")

plant_model = GRUPlant(
    n_plant_in=N_PLANT_IN, n_pv=N_PV,
    hidden=HIDDEN, layers=LAYERS,
    n_scenarios=N_SCENARIOS, dropout=DROPOUT,
).to(device)

src = torch.load(WARMSTART_CKPT, map_location=device)
plant_model.load_state_dict(src["model_state"])
print(f"  Warm-started from {WARMSTART_CKPT.name}")

ctrl_models = {}
for ln in CTRL_LOOPS:
    n_in = ctrl_data[ln]['X_train'].shape[-1]
    h    = CTRL_HIDDEN_PER_LOOP[ln]
    if ln == 'CC':
        ctrl_models[ln] = CCSequenceModel(
            n_inputs=n_in, hidden=h, layers=CTRL_LAYERS,
            dropout=DROPOUT, output_len=TARGET_LEN).to(device)
    else:
        ctrl_models[ln] = GRUController(
            n_inputs=n_in, hidden=h, layers=CTRL_LAYERS,
            dropout=DROPOUT, output_len=TARGET_LEN).to(device)
    ctrl_path = WARMSTART_CTRL_DIR / f"gru_ctrl_{ln.lower()}.pt"
    if ctrl_path.exists():
        c = torch.load(ctrl_path, map_location=device)
        ctrl_models[ln].load_state_dict(c["model_state"])

# ── 3. Optimiser ───────────────────────────────────────────────────────────────
plant_opt = torch.optim.Adam(plant_model.parameters(), lr=LR, weight_decay=WD)
plant_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(plant_opt, patience=SCH_PAT, factor=SCH_FAC)
ctrl_opts = {ln: torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
             for ln, m in ctrl_models.items()}
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()

def ss_ratio_for(epoch):
    if epoch < SS_START: return 0.0
    if epoch >= SS_END:  return SS_MAX
    return SS_MAX * (epoch - SS_START) / (SS_END - SS_START)

# ── 4. Training loop — normal windows only ─────────────────────────────────────
print(f"\nStep 3: Training on normal data only ({EPOCHS} epochs, PATIENCE={PATIENCE})")

best_val, best_state, patience_ctr = float("inf"), plant_model.state_dict(), 0
train_losses, val_losses, ss_ratios = [], [], []

for epoch in range(1, EPOCHS + 1):
    ss = ss_ratio_for(epoch)

    # Controllers (trained on full ctrl_data — 98.7% normal windows)
    for m in ctrl_models.values(): m.train()
    N_ctrl = len(ctrl_data['PC']['X_train'])
    idx_c  = np.random.permutation(N_ctrl)
    for i in range(0, N_ctrl, BATCH):
        b = idx_c[i:i + BATCH]
        for ln in CTRL_LOOPS:
            Xc = torch.tensor(ctrl_data[ln]['X_train'][b]).float().to(device)
            yc = torch.tensor(ctrl_data[ln]['y_train'][b]).float().to(device)
            if ln == 'CC':
                logits, cv_preds = ctrl_models[ln](Xc, y_teacher=yc, ss_ratio=ss)
                loss = bce(logits, (yc > 0).float()) + mse(cv_preds, yc)
            else:
                loss = mse(ctrl_models[ln](Xc, y_cv_teacher=yc, ss_ratio=ss), yc)
            ctrl_opts[ln].zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ctrl_models[ln].parameters(), GRAD_CLIP)
            ctrl_opts[ln].step()

    # Plant — normal only
    plant_model.train()
    idx   = np.random.permutation(N)
    total = 0.0
    for i in range(0, N, BATCH):
        b      = idx[i:i + BATCH]
        x_cv   = torch.tensor(X_train[b]).float().to(device)
        x_cv_t = torch.tensor(X_cv_tgt_train[b]).float().to(device)
        pv_ini = torch.tensor(pv_init_train[b]).float().to(device)
        pv_tf  = torch.tensor(pv_teacher_tr[b]).float().to(device)
        pv_tgt = torch.tensor(pv_target_tr[b]).float().to(device)
        sc     = torch.tensor(sc_train[b]).long().to(device)

        pred = plant_model(x_cv, x_cv_t, pv_ini, sc, pv_teacher=pv_tf, ss_ratio=ss)
        loss = mse(pred, pv_tgt)
        plant_opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(plant_model.parameters(), GRAD_CLIP)
        plant_opt.step()
        total += loss.item()

    # Validation — normal only
    plant_model.eval()
    pval, n_b = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X_val), BATCH):
            x_cv   = torch.tensor(X_val[i:i+BATCH]).float().to(device)
            x_cv_t = torch.tensor(X_cv_tgt_val[i:i+BATCH]).float().to(device)
            pv_ini = torch.tensor(pv_init_val[i:i+BATCH]).float().to(device)
            pv_tf  = torch.tensor(pv_teacher_val[i:i+BATCH]).float().to(device)
            pv_tgt = torch.tensor(pv_target_val[i:i+BATCH]).float().to(device)
            sc     = torch.tensor(sc_val[i:i+BATCH]).long().to(device)
            pval  += mse(plant_model(
                x_cv, x_cv_t, pv_ini, sc, pv_teacher=pv_tf, ss_ratio=0.0
            ), pv_tgt).item()
            n_b += 1

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
            "n_plant_in":  N_PLANT_IN, "n_pv": N_PV,
            "n_scenarios": N_SCENARIOS, "hidden": HIDDEN,
            "layers": LAYERS, "epoch": epoch, "val_loss": best_val,
            "normal_only": True,
        }, OUT_DIR / "gru_plant.pt")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | train={train_losses[-1]:.5f}  val={pval:.5f}  ss={ss:.3f}")

plant_model.load_state_dict(best_state)
print(f"\n  Best val loss (normal): {best_val:.5f}")

plot_training_curves(train_losses, val_losses, ss_ratios,
                     model_name="GRU-Normal-Only",
                     save_path=OUT_DIR / "gru_loss_curves.png")

# ── 5. Residual evaluation on ALL test scenarios ───────────────────────────────
print(f"\nStep 4: Residual evaluation on full test set (all scenarios)...")
for m in ctrl_models.values(): m.eval()
plant_model.eval()

N_test      = len(X_test)
pv_preds_te = np.zeros((N_test, TARGET_LEN, N_PV), dtype=np.float32)
sc_zero     = torch.zeros(BATCH, dtype=torch.long).to(device)  # always predict as normal

with torch.no_grad():
    for i in range(0, N_test, BATCH):
        sl        = slice(i, i + BATCH)
        x_cv_b    = torch.tensor(X_test[sl]).float().to(device)
        xct_b     = torch.tensor(X_cv_tgt_test[sl]).float().to(device).clone()
        pv_init_b = torch.tensor(pv_init_test[sl]).float().to(device)
        B_actual  = x_cv_b.size(0)
        sc_b      = torch.zeros(B_actual, dtype=torch.long).to(device)  # force normal

        for ln in CTRL_LOOPS:
            if ln not in ctrl_cv_col_idx: continue
            Xc      = torch.tensor(ctrl_data[ln]['X_test'][sl]).float().to(device)
            cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
            xct_b[:, :, ctrl_cv_col_idx[ln]:ctrl_cv_col_idx[ln] + 1] = cv_pred

        pv_seq   = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
        pv_preds_te[i:i + B_actual] = pv_seq.cpu().numpy()

# ── Residuals per scenario ─────────────────────────────────────────────────────
SCENARIO_NAMES = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}
residuals = pv_target_test - pv_preds_te  # actual - predicted_normal

print("\n  Mean absolute residual per scenario per PV:")
print(f"  {'Scenario':<12}", "  ".join(f"{p:<10}" for p in PV_COLS))
scenario_residuals = {}
for sc_id, sc_name in SCENARIO_NAMES.items():
    mask = (scenario_test == sc_id)
    if mask.sum() == 0: continue
    res = np.abs(residuals[mask]).mean(axis=(0, 1))  # mean over windows & time → per PV
    scenario_residuals[sc_name] = {name: float(v) for name, v in zip(PV_COLS, res)}
    print(f"  {sc_name:<12}", "  ".join(f"{v:.4f}    " for v in res))

# Normal NRMSE
normal_mask = (attack_test == 0)
nrmse_normal = []
for k in range(N_PV):
    true_k = pv_target_test[normal_mask, :, k]
    pred_k = pv_preds_te[normal_mask, :, k]
    rmse   = np.sqrt(np.mean((pred_k - true_k) ** 2))
    rng    = max(float(true_k.max() - true_k.min()), 1e-6)
    nrmse_normal.append(rmse / rng)
print(f"\n  Normal NRMSE: mean={np.mean(nrmse_normal):.4f}")
for name, v in zip(PV_COLS, nrmse_normal):
    print(f"    {name:<30}: {v:.4f}")

results = {
    "model": "GRU-Normal-Only",
    "description": "Trained on normal data only. Residual = actual - predicted_normal reveals attack footprint.",
    "normal_val_loss":      float(best_val),
    "normal_test_nrmse":    {name: float(v) for name, v in zip(PV_COLS, nrmse_normal)},
    "normal_test_mean_nrmse": float(np.mean(nrmse_normal)),
    "mean_abs_residual_per_scenario": scenario_residuals,
    "train_losses": [float(x) for x in train_losses],
    "val_losses":   [float(x) for x in val_losses],
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

# ── 6. Save checkpoints ────────────────────────────────────────────────────────
print(f"\nStep 5: Saving checkpoints to {OUT_DIR}/")
for ln, m in ctrl_models.items():
    torch.save({
        "model_state": m.state_dict(),
        "arch":        "CCSequenceModel" if ln == 'CC' else "GRUController",
        "n_inputs":    ctrl_data[ln]['X_train'].shape[-1],
        "hidden":      CTRL_HIDDEN_PER_LOOP[ln],
        "layers":      CTRL_LAYERS,
    }, OUT_DIR / f"gru_ctrl_{ln.lower()}.pt")

print("\nDone. Run 04_evaluate/plot_residuals.py to visualise attack footprints.")
