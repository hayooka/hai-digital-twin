"""
Digital Twin — Full Pipeline (Guided Generation + Generalization Gap)

Steps:
    1. Load trained Transformer + ISO Forest errors
    2. Load Diffusion-generated attack windows (240 steps each)
    3. Guided Generation:
         - Split each window: X = first 60, Y = last 180
         - Transformer → predict future → RMSE (physics check)
         - ISO Forest  → anomaly score  (detectability check)
         - Rejection Sampling: keep physics_ok + detectable
    4. Generalization Gap:
         - RMSE_known = Transformer RMSE on test2 real attacks
         - RMSE_novel = Transformer RMSE on filtered novel attacks
         - Gap = |RMSE_novel - RMSE_known|
    5. Final results table
"""
from __future__ import annotations

import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.transformer_model import TransformerSeq2Seq
from utils.prep import twin

# ── Config ────────────────────────────────────────────────────────────────────

ENC_LEN   = 60
DEC_LEN   = 180
N_FEAT    = 277
D_MODEL   = 256
N_HEADS   = 8
N_LAYERS  = 4
FFN_DIM   = 1024
DROPOUT   = 0.1
BATCH     = 64

Path("outputs").mkdir(exist_ok=True)


# ── 1. Load Data ──────────────────────────────────────────────────────────────

print("=" * 60)
print("  DIGITAL TWIN — GUIDED GENERATION PIPELINE")
print("=" * 60)

print("\n[1] Loading data...")
data   = twin(input_len=ENC_LEN, target_len=DEC_LEN, stride=60)
X_test = data["X_test"]         # (K, 60,  277)  test2
Y_test = data["Y_test"]         # (K, 180, 277)  test2
y_test = data["y_test_labels"]  # (K,)  0/1


# ── 2. Load Transformer ───────────────────────────────────────────────────────

print("\n[2] Loading Transformer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    Device: {device}")

checkpoint = torch.load("outputs/transformer_twin.pt",
                        map_location=device, weights_only=False)

model = TransformerSeq2Seq(
    n_features=N_FEAT, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT,
).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

train_errors = checkpoint["train_errors"]   # (N, 277) normal errors
train_errors = np.nan_to_num(train_errors, nan=0.0, posinf=0.0, neginf=0.0)
cap = np.percentile(train_errors, 99.9)
train_errors = np.clip(train_errors, 0, cap)
print(f"    train_errors {train_errors.shape}")


# ── 3. Fit ISO Forest on normal errors ────────────────────────────────────────

print("\n[3] Fitting ISO Forest + PCA on normal errors...")
pca = PCA(n_components=20, random_state=42)
train_reduced = pca.fit_transform(np.log1p(train_errors))
print(f"    PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

attack_rate = float(y_test.sum()) / len(y_test)
iso = IsolationForest(n_estimators=200, contamination=attack_rate,
                      random_state=42, n_jobs=-1)
iso.fit(train_reduced)
print("    ISO Forest trained.")


# ── Helper: RMSE for a batch of (X, Y) ───────────────────────────────────────

def compute_rmse(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Returns per-window scalar RMSE (N,)."""
    rmses = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            src    = torch.tensor(X[i:i+BATCH]).float().to(device)
            tgt    = torch.tensor(Y[i:i+BATCH]).float().to(device)
            dec_in = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
            pred   = model(src, dec_in)
            mse    = ((pred - tgt) ** 2).mean(dim=(1, 2)).cpu().numpy()
            rmses.append(np.sqrt(mse))
    return np.concatenate(rmses)


def compute_iso_scores(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Returns per-window ISO Forest anomaly scores (N,) — higher = more anomalous."""
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            src  = torch.tensor(X[i:i+BATCH]).float().to(device)
            tgt  = torch.tensor(Y[i:i+BATCH]).float().to(device)
            pred = model.predict(src, dec_len=tgt.size(1))
            err  = ((pred - tgt) ** 2).mean(dim=1).cpu().numpy()   # (B, 277)
            errs.append(err)
    errs = np.concatenate(errs)
    errs = np.nan_to_num(np.clip(errs, 0, cap), nan=0.0, posinf=0.0)
    reduced = pca.transform(np.log1p(errs))
    return -iso.score_samples(reduced)   # higher = more anomalous


# ── 4. Known attack RMSE (test2) ──────────────────────────────────────────────

print("\n[4] Computing RMSE on test2 (known attacks)...")
attack_mask  = y_test == 1
normal_mask  = y_test == 0

rmse_known_attacks = compute_rmse(X_test[attack_mask], Y_test[attack_mask])
rmse_known_normal  = compute_rmse(X_test[normal_mask], Y_test[normal_mask])

print(f"    RMSE normal windows  = {rmse_known_normal.mean():.5f}")
print(f"    RMSE attack windows  = {rmse_known_attacks.mean():.5f}")
print(f"    Attack/Normal ratio  = {rmse_known_attacks.mean()/rmse_known_normal.mean():.2f}x")


# ── 5. Load Diffusion generated windows ───────────────────────────────────────

print("\n[5] Loading Diffusion-generated attack windows...")
diff_ckpt = torch.load("outputs/diffusion_attacks.pt",
                       map_location="cpu", weights_only=False)
gen_windows = diff_ckpt["generated_windows"].numpy()   # (150, 240, 277)

X_novel = gen_windows[:, :ENC_LEN,  :]   # first 60  → encoder input
Y_novel = gen_windows[:, ENC_LEN:,  :]   # last 180  → decoder target
print(f"    Generated windows: {gen_windows.shape[0]}")
print(f"    X_novel {X_novel.shape}  Y_novel {Y_novel.shape}")


# ── 6. Guided Generation — Rejection Sampling ─────────────────────────────────

print("\n[6] Guided Generation — Rejection Sampling...")

rmse_novel  = compute_rmse(X_novel, Y_novel)
iso_novel   = compute_iso_scores(X_novel, Y_novel)

# Thresholds:
#   physics_ok:  RMSE > normal baseline (attack should have higher error)
#   detectable:  ISO score > 95th percentile of normal ISO scores
iso_normal_scores  = compute_iso_scores(X_test[normal_mask], Y_test[normal_mask])
physics_threshold  = float(rmse_known_normal.mean() * 5)   # 5x normal RMSE
detect_threshold   = float(np.percentile(iso_normal_scores, 99))

physics_ok   = rmse_novel  > physics_threshold
detectable   = iso_novel   > detect_threshold
keep_mask    = physics_ok & detectable

n_generated  = len(X_novel)
n_kept       = keep_mask.sum()
accept_rate  = n_kept / n_generated

print(f"    Physics threshold  = {physics_threshold:.5f}")
print(f"    Detect threshold   = {detect_threshold:.5f}")
print(f"    Generated          = {n_generated}")
print(f"    Kept (both checks) = {n_kept}")
print(f"    Acceptance rate    = {accept_rate:.2%}")


# ── 7. Generalization Gap ─────────────────────────────────────────────────────

print("\n[7] Computing Generalization Gap...")

rmse_novel_all      = rmse_novel.mean()
rmse_novel_filtered = rmse_novel[keep_mask].mean() if n_kept > 0 else float("nan")
rmse_known_mean     = rmse_known_attacks.mean()

gap_all      = abs(rmse_novel_all      - rmse_known_mean)
gap_filtered = abs(rmse_novel_filtered - rmse_known_mean) if n_kept > 0 else float("nan")


# ── 8. Final Results Table ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  DIGITAL TWIN — FINAL RESULTS")
print("=" * 60)
print("\n  SIMULATOR (Transformer Seq2Seq)")
print(f"    RMSE normal          = {rmse_known_normal.mean():.5f}")
print(f"    RMSE known attacks   = {rmse_known_mean:.5f}")
print(f"    Attack/Normal ratio  = {rmse_known_mean/rmse_known_normal.mean():.2f}x")
print("\n  GENERALIZATION GAP")
print(f"    RMSE novel (all)     = {rmse_novel_all:.5f}")
print(f"    RMSE novel (filtered)= {rmse_novel_filtered:.5f}")
print(f"    Gap (all)            = {gap_all:.5f}")
print(f"    Gap (filtered)       = {gap_filtered:.5f}")
print(f"    Acceptance rate      = {accept_rate:.2%}")
print("\n  GUIDED GENERATION")
print(f"    Generated windows    = {n_generated}")
print(f"    Kept after filtering = {n_kept}")
print("=" * 60)

results = {
    "rmse_normal":          float(rmse_known_normal.mean()),
    "rmse_known_attacks":   float(rmse_known_mean),
    "attack_normal_ratio":  float(rmse_known_mean / rmse_known_normal.mean()),
    "rmse_novel_all":       float(rmse_novel_all),
    "rmse_novel_filtered":  float(rmse_novel_filtered),
    "gap_all":              float(gap_all),
    "gap_filtered":         float(gap_filtered),
    "n_generated":          int(n_generated),
    "n_kept":               int(n_kept),
    "acceptance_rate":      float(accept_rate),
}

with open("outputs/pipeline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved: outputs/pipeline_results.json")
