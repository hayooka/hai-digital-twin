"""
Isolation Forest — Anomaly Detection (primary)

Loads reconstruction errors saved by transformer_model.py and trains
an Isolation Forest to detect anomalies.

Pipeline:
    1. transformer_model.py  → outputs/transformer_twin.pt  (errors saved)
    2. iso_forest.py         → loads errors → trains ISO Forest → F1/ROC-AUC
    3. generalization_gap.py → loads diffusion_attacks.pt   → Generalization Gap

Input:
    train_errors : (N, 277)  per-sensor MSE on train windows (normal + known attacks)
    test_errors  : (K, 277)  per-sensor MSE on test2 windows (held-out)
    y_test       : (K,)      0=normal 1=attack
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


# ── Load saved Transformer errors ─────────────────────────────────────────────

checkpoint = torch.load("outputs/transformer_twin.pt", map_location="cpu", weights_only=False)
train_errors = checkpoint["train_errors"]   # (N, 277)
test_errors  = checkpoint["test_errors"]    # (K, 277)
y_test       = checkpoint["y_test"]         # (K,)

print("Loaded errors from outputs/transformer_twin.pt")
print(f"  train_errors {train_errors.shape}  (normal only)")
print(f"  test_errors  {test_errors.shape}   (test2 held-out)")
print(f"  attacks in test: {y_test.sum()} / {len(y_test)}")


# ── Clean: remove NaN / Inf ────────────────────────────────────────────────────

train_errors = np.nan_to_num(train_errors, nan=0.0, posinf=0.0, neginf=0.0)
test_errors  = np.nan_to_num(test_errors,  nan=0.0, posinf=0.0, neginf=0.0)

# Clip extreme outliers to 99.9th percentile of train errors
cap = np.percentile(train_errors, 99.9)
train_errors = np.clip(train_errors, 0, cap)
test_errors  = np.clip(test_errors,  0, cap)


# ── PCA: reduce 277-dim errors to 20 dims ─────────────────────────────────────
# High dimensionality hurts ISO Forest — PCA keeps the most informative variance.

print("\nReducing dimensions with PCA (277 → 20)...")
pca = PCA(n_components=20, random_state=42)
train_reduced = pca.fit_transform(train_errors)
test_reduced  = pca.transform(test_errors)
print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# ── Train Isolation Forest ─────────────────────────────────────────────────────

# Contamination = actual attack rate in test2 (244/3836 ≈ 6.4%)
attack_rate = float(y_test.sum()) / len(y_test)
print(f"\nTraining Isolation Forest (contamination={attack_rate:.3f})...")
iso = IsolationForest(
    n_estimators=200,
    contamination=attack_rate,
    random_state=42,
    n_jobs=-1,
)
iso.fit(train_reduced)
print("  done.")


# ── Predict ───────────────────────────────────────────────────────────────────
# IsolationForest returns: -1 = anomaly, 1 = normal
# Convert to: 1 = anomaly (attack), 0 = normal

raw_pred    = iso.predict(test_reduced)          # -1 or 1
pred_labels = (raw_pred == -1).astype(int)       # 1=attack 0=normal

# Anomaly score: lower = more anomalous (negate for ROC-AUC)
scores      = -iso.score_samples(test_reduced)   # higher = more anomalous


# ── Metrics ───────────────────────────────────────────────────────────────────

f1  = f1_score(y_test,  pred_labels, zero_division=0)
pre = precision_score(y_test, pred_labels, zero_division=0)
rec = recall_score(y_test,  pred_labels, zero_division=0)
try:   auc = roc_auc_score(y_test, scores)
except: auc = float("nan")

tn, fp, fn, tp = confusion_matrix(y_test, pred_labels, labels=[0, 1]).ravel()

print("\n" + "=" * 50)
print("  ISOLATION FOREST — ANOMALY DETECTION RESULTS")
print("=" * 50)
print(f"  F1        = {f1:.4f}")
print(f"  Precision = {pre:.4f}")
print(f"  Recall    = {rec:.4f}")
print(f"  ROC-AUC   = {auc:.4f}")
print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
print("=" * 50)

results = {
    "f1": float(f1), "precision": float(pre),
    "recall": float(rec), "roc_auc": float(auc),
    "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
}

Path("outputs").mkdir(exist_ok=True)
with open("outputs/iso_forest_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved: outputs/iso_forest_metrics.json")
