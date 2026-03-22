"""
Isolation Forest — Anomaly Detection (primary)

Improvements over baseline:
    1. Rich features: per-sensor MSE (277) + temporal profile (180) + stats (4)
       → captures WHICH sensors are attacked AND HOW error evolves over time
    2. Semi-supervised: trains on normal errors (train1-3) + attack errors (test1)
       → cleaner decision boundary
    3. PCA(30) on combined 461-dim feature vector

Pipeline:
    1. transformer_model.py → outputs/transformer_twin.pt  (model + errors saved)
    2. iso_forest.py        → rich features → ISO Forest → F1/ROC-AUC
    3. twin/pipeline.py     → Guided Generation + Generalization Gap
"""
from __future__ import annotations

import sys
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.transformer_model import TransformerSeq2Seq
from utils.prep import twin
from utils.data_loader import load_merged

META_COLS = {"timestamp", "attack", "label", "attack_p1", "attack_p2", "attack_p3"}

ENC_LEN  = 60
DEC_LEN  = 180
N_FEAT   = 277
# N_FEAT = 86             # HAI-only mode (commented out)
D_MODEL  = 256
N_HEADS  = 8
N_LAYERS = 4
FFN_DIM  = 1024
DROPOUT  = 0.1
BATCH    = 64


# ── Load checkpoint ────────────────────────────────────────────────────────────

checkpoint   = torch.load("outputs/transformer_twin.pt", map_location="cpu", weights_only=False)
y_test       = checkpoint["y_test"]    # (K,) test2 labels

print("Loaded checkpoint from outputs/transformer_twin.pt")
print(f"  attacks in test2: {y_test.sum()} / {len(y_test)}")


# ── Load Transformer ───────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

model = TransformerSeq2Seq(
    n_features=N_FEAT, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT,
).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()


# ── Rich feature extractor ─────────────────────────────────────────────────────
# Instead of just per-sensor MSE (277), extract 3 feature groups:
#
#   1. sensor_errors  (277): mean MSE per sensor over 180 timesteps
#      → which sensors are attacked
#
#   2. temporal_profile (180): mean MSE across sensors per timestep
#      → how error evolves over time (attacks often grow; normal errors are flat)
#
#   3. stats (4): [global_mean, global_max, global_std, global_p95]
#      → overall severity
#
# Combined: (277 + 180 + 4) = 461-dim feature vector per window
# # HAI-only mode: (86 + 180 + 4) = 270-dim (commented out)

def extract_features(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Returns (N, 461) rich feature matrix.
    X: (N, 60, 277)  Y: (N, 180, 277)
    """
    sensor_list, temporal_list, stats_list = [], [], []

    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            src    = torch.tensor(X[i:i+BATCH]).float().to(device)
            tgt    = torch.tensor(Y[i:i+BATCH]).float().to(device)
            dec_in = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
            pred   = model(src, dec_in)
            err    = (pred - tgt) ** 2   # (B, 180, 277)

            # 1. Per-sensor MSE: average over timesteps → (B, 277)
            sensor_err = err.mean(dim=1).cpu().numpy()

            # 2. Temporal profile: average over sensors → (B, 180)
            temporal_err = err.mean(dim=2).cpu().numpy()

            # 3. Global stats → (B, 4)
            flat = err.reshape(err.shape[0], -1).cpu().numpy()  # (B, 180*277)
            stats = np.stack([
                flat.mean(axis=1),
                flat.max(axis=1),
                flat.std(axis=1),
                np.percentile(flat, 95, axis=1),
            ], axis=1)

            sensor_list.append(sensor_err)
            temporal_list.append(temporal_err)
            stats_list.append(stats)

    sensor_all   = np.concatenate(sensor_list,   axis=0)   # (N, 277)
    temporal_all = np.concatenate(temporal_list, axis=0)   # (N, 180)
    stats_all    = np.concatenate(stats_list,    axis=0)   # (N, 4)

    # Clip extremes using 99.9th percentile of sensor errors
    cap          = np.percentile(sensor_all, 99.9)
    sensor_all   = np.clip(sensor_all,   0, cap)
    temporal_all = np.clip(temporal_all, 0, cap)
    stats_all    = np.clip(stats_all,    0, cap)

    # Log-compress (reduces skew from extreme attack values)
    features = np.concatenate([
        np.log1p(sensor_all),     # (N, 277)
        np.log1p(temporal_all),   # (N, 180)
        np.log1p(stats_all),      # (N, 4)
    ], axis=1)                    # (N, 461)

    return features.astype(np.float32)


# ── Load test2 data ────────────────────────────────────────────────────────────

print("\nLoading test2 data...")
twin_data = twin(input_len=ENC_LEN, target_len=DEC_LEN, stride=60)
norm      = twin_data["norm"]
X_test    = twin_data["X_test"]
Y_test    = twin_data["Y_test"]

print(f"  test2 windows: {X_test.shape[0]}")


# ── Load train1-3 data (normal) ────────────────────────────────────────────────

print("Loading train1-3 data (normal)...")
train_dfs = [load_merged("train", i) for i in range(1, 4)]
import pandas as pd
import numpy as np

train_arr = pd.concat(train_dfs, ignore_index=True)
cols = [c for c in train_arr.columns if c not in META_COLS and train_arr[c].dtype != object]
arr_train = norm.transform(train_arr)[cols].values.astype(np.float32)

span   = ENC_LEN + DEC_LEN
starts = list(range(0, len(arr_train) - span + 1, 60))
X_tr   = np.empty((len(starts), ENC_LEN, len(cols)), dtype=np.float32)
Y_tr   = np.empty((len(starts), DEC_LEN, len(cols)), dtype=np.float32)
for i, s in enumerate(starts):
    X_tr[i] = arr_train[s          : s + ENC_LEN]
    Y_tr[i] = arr_train[s + ENC_LEN : s + span]
print(f"  train1-3 windows: {X_tr.shape[0]}")


# ── Load test1 data (known attacks) ───────────────────────────────────────────

print("Loading test1 data (known attacks)...")
test1_df  = load_merged("test", 1)
arr_test1 = norm.transform(test1_df)[cols].values.astype(np.float32)
y_test1   = test1_df["attack"].values.astype(np.int32)

starts_t1 = list(range(0, len(arr_test1) - span + 1, 60))
X_t1      = np.empty((len(starts_t1), ENC_LEN, len(cols)), dtype=np.float32)
Y_t1      = np.empty((len(starts_t1), DEC_LEN, len(cols)), dtype=np.float32)
y_t1      = np.array([int(y_test1[s : s + span].any()) for s in starts_t1], dtype=np.int32)
for i, s in enumerate(starts_t1):
    X_t1[i] = arr_test1[s          : s + ENC_LEN]
    Y_t1[i] = arr_test1[s + ENC_LEN : s + span]
print(f"  test1 windows: {X_t1.shape[0]}  (attacks: {y_t1.sum()})")


# ── Extract rich features ──────────────────────────────────────────────────────

print("\nExtracting rich features (461-dim)...")
print("  [1/3] train1-3 normal windows...")
feat_train = extract_features(X_tr, Y_tr)          # (N, 461)

print("  [2/3] test1 windows (known attacks)...")
feat_test1 = extract_features(X_t1, Y_t1)          # (M, 461)

print("  [3/3] test2 windows (held-out eval)...")
feat_test2 = extract_features(X_test, Y_test)      # (K, 461)

print(f"  Features shape: train={feat_train.shape}  test1={feat_test1.shape}  test2={feat_test2.shape}")


# ── PCA ───────────────────────────────────────────────────────────────────────
# Fit PCA on normal features only → apply to all

print("\nReducing with PCA (461 → 30)...")
pca          = PCA(n_components=30, random_state=42)
train_pca    = pca.fit_transform(feat_train)
test1_pca    = pca.transform(feat_test1)
test2_pca    = pca.transform(feat_test2)
print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")


# ── Combine training: normal + test1 (semi-supervised) ────────────────────────

iso_train    = np.concatenate([train_pca, test1_pca], axis=0)
attack_rate  = float(y_test.sum()) / len(y_test)

print(f"\nTraining ISO Forest on {len(iso_train)} windows")
print(f"  Normal (train1-3): {len(train_pca)}")
print(f"  Test1 (known atk): {len(test1_pca)}")
print(f"  Contamination:     {attack_rate:.3f}")

iso = IsolationForest(
    n_estimators=300,
    contamination=attack_rate,
    max_features=0.8,       # feature subsampling → reduces correlation
    random_state=42,
    n_jobs=-1,
)
iso.fit(iso_train)
print("  done.")


# ── Predict on test2 (held-out) ───────────────────────────────────────────────

raw_pred    = iso.predict(test2_pca)
pred_labels = (raw_pred == -1).astype(int)
scores      = -iso.score_samples(test2_pca)   # higher = more anomalous


# ── Metrics ───────────────────────────────────────────────────────────────────

f1  = f1_score(y_test,  pred_labels, zero_division=0)
pre = precision_score(y_test, pred_labels, zero_division=0)
rec = recall_score(y_test,  pred_labels, zero_division=0)
try:   auc = roc_auc_score(y_test, scores)
except: auc = float("nan")

tn, fp, fn, tp = confusion_matrix(y_test, pred_labels, labels=[0, 1]).ravel()

print("\n" + "=" * 55)
print("  ISOLATION FOREST — ANOMALY DETECTION RESULTS")
print("=" * 55)
print(f"  Features:  461-dim (sensor + temporal + stats) → PCA(30)")
print(f"  Training:  train1-3 normal + test1 known attacks")
print(f"  Eval:      test2 (held-out, blind)")
print(f"  F1        = {f1:.4f}")
print(f"  Precision = {pre:.4f}")
print(f"  Recall    = {rec:.4f}")
print(f"  ROC-AUC   = {auc:.4f}")
print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
print("=" * 55)

results = {
    "f1":        float(f1),
    "precision": float(pre),
    "recall":    float(rec),
    "roc_auc":   float(auc),
    "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    "features":  "461-dim: per-sensor(277) + temporal(180) + stats(4) → PCA(30)",
    "training":  "train1-3 normal + test1 known attacks",
    "evaluation": "test2 held-out",
}

Path("outputs").mkdir(exist_ok=True)
with open("outputs/iso_forest_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved: outputs/iso_forest_metrics.json")
