"""
Isolation Forest — Layer 2: Anomaly Detector (primary)

Runs AFTER transformer_model.py (or lstm_model.py) has saved its checkpoint.
Loads the trained twin, extracts rich reconstruction-error features, trains
Isolation Forest, evaluates on held-out test2 with eTaPR.

Feature vector per window  (all dynamic — sized by actual F at runtime):
    1. per-sensor MSE   (F dims)  — which sensors deviate
    2. temporal profile (180 dims) — how error evolves over the target window
    3. global stats     (4 dims)  — severity: mean, max, std, p95
    Total = F + 184 dims  →  PCA(30)

Pipeline:
    1. transformer_model.py → outputs/transformer_twin.pt
    2. iso_forest.py        → eTaPR detection results
"""
from __future__ import annotations

import sys
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.transformer_model import TransformerSeq2Seq, N_FEAT, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, DROPOUT, BATCH
from utils.prep import twin, generate
from utils.data_loader import load_merged, identify_common_constants
from utils.eval import evaluate_detector

ENC_LEN = 300   # must match twin() input_len
DEC_LEN = 180   # must match twin() target_len


# ── Load checkpoint ────────────────────────────────────────────────────────────

checkpoint = torch.load("outputs/transformer_twin.pt", map_location="cpu",
                        weights_only=False)
y_test     = checkpoint["y_test"]    # (K,) test2 window labels

print("Loaded checkpoint from outputs/transformer_twin.pt")
print(f"  N_FEAT in checkpoint: {N_FEAT}")
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

def extract_features(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Returns (N, F+184) rich feature matrix from reconstruction errors.

    F   = per-sensor MSE  (which sensors deviate)
    180 = temporal profile (how error evolves across the 180-step target)
    4   = global stats    (mean, max, std, p95)
    """
    sensor_list, temporal_list, stats_list = [], [], []

    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            src    = torch.tensor(X[i:i+BATCH]).float().to(device)
            tgt    = torch.tensor(Y[i:i+BATCH]).float().to(device)
            dec_in = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
            pred   = model(src, dec_in)
            err    = (pred - tgt) ** 2          # (B, DEC_LEN, F)

            sensor_list.append(err.mean(dim=1).cpu().numpy())      # (B, F)
            temporal_list.append(err.mean(dim=2).cpu().numpy())    # (B, DEC_LEN)
            flat = err.reshape(err.shape[0], -1).cpu().numpy()     # (B, DEC_LEN*F)
            stats_list.append(np.stack([
                flat.mean(axis=1),
                flat.max(axis=1),
                flat.std(axis=1),
                np.percentile(flat, 95, axis=1),
            ], axis=1))                                             # (B, 4)

    sensor_all   = np.concatenate(sensor_list,   axis=0)
    temporal_all = np.concatenate(temporal_list, axis=0)
    stats_all    = np.concatenate(stats_list,    axis=0)

    cap          = np.percentile(sensor_all, 99.9)
    sensor_all   = np.clip(sensor_all,   0, cap)
    temporal_all = np.clip(temporal_all, 0, cap)
    stats_all    = np.clip(stats_all,    0, cap)

    return np.concatenate([
        np.log1p(sensor_all),
        np.log1p(temporal_all),
        np.log1p(stats_all),
    ], axis=1).astype(np.float32)


# ── Load data via twin() — consistent pipeline with constant deletion ───────────

print("\nLoading data via twin() pipeline (consistent constant deletion)...")
twin_data  = twin(input_len=ENC_LEN, target_len=DEC_LEN, stride=60)
norm       = twin_data["norm"]
X_val      = twin_data["X_val"]     # last 20% of train1-4 — normal, unseen by twin
Y_val      = twin_data["Y_val"]
X_test     = twin_data["X_test"]    # test2 held-out
Y_test     = twin_data["Y_test"]

print(f"  val windows  (normal): {X_val.shape[0]}")
print(f"  test windows (test2):  {X_test.shape[0]}")


# ── Load test1 (known attacks) with same constant deletion ─────────────────────

print("Loading test1 (known attacks)...")
const_hai, const_hiend = identify_common_constants()
test1_df  = load_merged("test", 1, drop_constants=True, keep_hai_duplicates=True,
                         const_cols_hai=const_hai, const_cols_hiend=const_hiend)

import pandas as pd
META_COLS  = {"timestamp", "attack", "label", "attack_p1", "attack_p2", "attack_p3"}
cols       = [c for c in test1_df.columns if c not in META_COLS and test1_df[c].dtype != object]
arr_test1  = norm.transform(test1_df)[cols].values.astype(np.float32)
y_test1_raw = test1_df["attack"].values.astype(np.int32)

span      = ENC_LEN + DEC_LEN
starts_t1 = list(range(0, len(arr_test1) - span + 1, 60))
X_t1 = np.empty((len(starts_t1), ENC_LEN, len(cols)), dtype=np.float32)
Y_t1 = np.empty((len(starts_t1), DEC_LEN, len(cols)), dtype=np.float32)
y_t1 = np.array([int(y_test1_raw[s:s+span].any()) for s in starts_t1], dtype=np.int32)
for i, s in enumerate(starts_t1):
    X_t1[i] = arr_test1[s          : s + ENC_LEN]
    Y_t1[i] = arr_test1[s + ENC_LEN : s + span]
print(f"  test1 windows: {X_t1.shape[0]}  (attacks: {y_t1.sum()})")


# ── Extract features ───────────────────────────────────────────────────────────

print(f"\nExtracting features ({N_FEAT}+184-dim)...")
feat_val   = extract_features(X_val,  Y_val)    # normal — for ISO Forest training
feat_test1 = extract_features(X_t1,   Y_t1)    # known attacks — semi-supervised
feat_test2 = extract_features(X_test, Y_test)  # held-out evaluation

print(f"  val={feat_val.shape}  test1={feat_test1.shape}  test2={feat_test2.shape}")


# ── PCA ───────────────────────────────────────────────────────────────────────

print("\nPCA (→ 30 components)...")
pca        = PCA(n_components=30, random_state=42)
val_pca    = pca.fit_transform(feat_val)      # fit on normal only
test1_pca  = pca.transform(feat_test1)
test2_pca  = pca.transform(feat_test2)
print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")


# ── ISO Forest — semi-supervised (normal val + test1 known attacks) ────────────

iso_train   = np.concatenate([val_pca, test1_pca], axis=0)
attack_rate = float(y_test.sum()) / max(1, len(y_test))

print(f"\nTraining ISO Forest on {len(iso_train)} windows")
print(f"  Normal (val 20%):  {len(val_pca)}")
print(f"  Test1 (known atk): {len(test1_pca)}")
print(f"  Contamination:     {attack_rate:.3f}")

iso = IsolationForest(
    n_estimators=300,
    contamination=attack_rate,
    max_features=0.8,
    random_state=42,
    n_jobs=-1,
)
iso.fit(iso_train)
print("  done.")


# ── Score test2 ───────────────────────────────────────────────────────────────

scores = -iso.score_samples(test2_pca)   # higher = more anomalous


# ── Evaluate with eTaPR ───────────────────────────────────────────────────────

results = evaluate_detector(
    y_true=y_test,
    scores=scores,
    label="Isolation Forest (Layer 2)",
    theta_p=0.5,
    theta_r=0.1,
    save_path="outputs/iso_forest_metrics.json",
)

print("\nSaved: outputs/iso_forest_metrics.json")
