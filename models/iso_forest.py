"""
Isolation Forest — Layer 2: Anomaly Detector (primary)

Functions used by models/train.py after the Transformer has been trained.

Feature vector per window  (all dynamic — sized by actual F at runtime):
    1. per-sensor MSE   (F dims)  — which sensors deviate
    2. temporal profile (180 dims) — how error evolves over the target window
    3. global stats     (4 dims)  — severity: mean, max, std, p95
    Total = F + 184 dims  →  PCA(30)
"""
from __future__ import annotations

import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_features(model, device, X: np.ndarray, Y: np.ndarray,
                     batch_size: int = 64) -> np.ndarray:
    """
    Returns (N, F+184) rich feature matrix from reconstruction errors.

    F   = per-sensor MSE  (which sensors deviate)
    180 = temporal profile (how error evolves across the 180-step target)
    4   = global stats    (mean, max, std, p95)
    """
    sensor_list, temporal_list, stats_list = [], [], []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            src    = torch.tensor(X[i:i+batch_size]).float().to(device)
            tgt    = torch.tensor(Y[i:i+batch_size]).float().to(device)
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


def train_iso_forest(feat_val: np.ndarray, feat_test1: np.ndarray,
                     attack_rate: float, n_components: int = 30):
    """
    Fit PCA then ISO Forest (semi-supervised: normal val + known test1 attacks).

    Returns (iso, pca, val_pca, test1_pca)
    """
    pca       = PCA(n_components=n_components, random_state=42)
    val_pca   = pca.fit_transform(feat_val)      # fit on normal only
    test1_pca = pca.transform(feat_test1)

    iso_train = np.concatenate([val_pca, test1_pca], axis=0)

    print(f"  Normal (val 20%):  {len(val_pca)}")
    print(f"  Test1 (known atk): {len(test1_pca)}")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    iso = IsolationForest(
        n_estimators=300,
        contamination=attack_rate,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(iso_train)
    print("  done.")

    return iso, pca, val_pca, test1_pca
