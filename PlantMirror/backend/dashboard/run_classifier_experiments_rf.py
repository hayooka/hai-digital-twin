"""
run_classifier_experiments_rf.py - same 4 experiments as the XGBoost version,
but using the teammate's classifier strategy:
    - 30 engineered features per 180-s window (5 PVs x 6 stats)
    - RandomForestClassifier(n_estimators=500, random_state=42)
    - Binary task (attack vs normal, binarised from scenario labels)

Saves to cache/classifier_experiments_rf.json alongside the XGBoost results.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REAL_TEST1 = Path(r"C:\Users\PC GAMING\Desktop\new_ai\processed\test1.csv")
REAL_TEST2 = Path(r"C:\Users\PC GAMING\Desktop\new_ai\processed\test2.csv")
SYNTH_CSV  = Path(r"C:\Users\PC GAMING\Desktop\AI\HAI\synthetic_attacks.csv")
OUT_JSON   = HERE / "cache" / "classifier_experiments_rf.json"

PV_COLS = ["P1_PIT01", "P1_LIT01", "P1_FT03Z", "P1_TIT01", "P1_TIT03"]
WIN_SEC = 180
STRIDE  = 60
THR     = 0.5     # binary RF uses its own calibrated threshold

# Teammate's RF hparams (from inspecting trts_rf_classifier.pkl)
RF_PARAMS = {"n_estimators": 500, "random_state": 42, "n_jobs": -1}


def extract_features(traj: np.ndarray) -> np.ndarray:
    """(N, T, K) -> (N, K*6). Source: 05_detect/sec3_classification.py (farah)."""
    feats = []
    for k in range(traj.shape[-1]):
        r = traj[:, :, k]
        feats += [r.mean(1), r.std(1), r.min(1), r.max(1),
                  np.abs(r).mean(1), np.diff(r, axis=1).mean(1)]
    return np.stack(feats, axis=1).astype(np.float32)


def slide_windows_real(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Slide 180-s windows over a real CSV. Label = any(attack) in window."""
    pv = df[PV_COLS].to_numpy(dtype=np.float32)
    lab = (df["label"] > 0).astype(np.int8).to_numpy() if "label" in df.columns else np.zeros(len(df), dtype=np.int8)
    T = pv.shape[0]
    if T < WIN_SEC:
        return np.zeros((0, WIN_SEC, 5), dtype=np.float32), np.zeros((0,), dtype=np.int8)
    starts = np.arange(0, T - WIN_SEC + 1, STRIDE, dtype=np.int64)
    X = np.stack([pv[s:s + WIN_SEC] for s in starts], axis=0)
    y = np.array([int(lab[s:s + WIN_SEC].any()) for s in starts], dtype=np.int8)
    return X, y


def load_synthetic_windows() -> tuple[np.ndarray, np.ndarray]:
    """Each sample_id in synthetic_attacks.csv is already a 180-row window."""
    df = pd.read_csv(SYNTH_CSV, low_memory=False)
    if "sample_id" not in df.columns:
        # Fall back: pack every 180 rows together.
        n = (len(df) // WIN_SEC) * WIN_SEC
        df = df.iloc[:n]
        groups = [df.iloc[i:i + WIN_SEC] for i in range(0, n, WIN_SEC)]
    else:
        groups = [g for _, g in df.groupby("sample_id") if len(g) >= WIN_SEC]
        groups = [g.iloc[:WIN_SEC] for g in groups]
    if not groups:
        raise RuntimeError("no windows in synthetic_attacks.csv")
    X = np.stack([g[PV_COLS].to_numpy(dtype=np.float32) for g in groups], axis=0)
    y = np.array(
        [int((g["label"] > 0).any()) if "label" in g.columns else 0 for g in groups],
        dtype=np.int8,
    )
    return X, y


def fit_eval(X_tr_feat, y_tr, X_te_feat, y_te) -> dict:
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_feat)
    X_te_s = sc.transform(X_te_feat)
    clf = RandomForestClassifier(**RF_PARAMS)
    t0 = time.time()
    clf.fit(X_tr_s, y_tr)
    t_fit = time.time() - t0
    proba = clf.predict_proba(X_te_s)
    # Binary task: attack probability = 1 - P(class==0). Guards multiclass too.
    classes = clf.classes_
    if len(classes) == 2:
        attack_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
        p = proba[:, attack_idx]
    else:
        normal_col = int(np.where(classes == 0)[0][0]) if 0 in classes else 0
        p = 1.0 - proba[:, normal_col]
    pred = (p >= THR).astype(int)
    cm = confusion_matrix(y_te, pred).tolist()
    return {
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "n_train_attack": int(y_tr.sum()), "n_test_attack": int(y_te.sum()),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_te, p)) if len(set(y_te)) > 1 else None,
        "avg_precision": float(average_precision_score(y_te, p)) if len(set(y_te)) > 1 else None,
        "threshold": THR,
        "confusion_matrix": cm,
        "fit_time_s": round(t_fit, 2),
    }


def main() -> None:
    OUT_JSON.parent.mkdir(exist_ok=True)
    print("Loading data + windowing...")
    t0 = time.time()
    real_X, real_y = slide_windows_real(pd.concat(
        [pd.read_csv(REAL_TEST1, low_memory=False),
         pd.read_csv(REAL_TEST2, low_memory=False)],
        ignore_index=True,
    ))
    synth_X, synth_y = load_synthetic_windows()
    print(f"  real windows     : {real_X.shape} ({int(real_y.sum())} attacks)")
    print(f"  synthetic windows: {synth_X.shape} ({int(synth_y.sum())} attacks) [{time.time()-t0:.1f}s]")

    # Extract 30 features per window
    print("Extracting features (30-dim per window)...")
    real_feat  = extract_features(real_X)
    synth_feat = extract_features(synth_X)
    print(f"  real_feat  : {real_feat.shape}")
    print(f"  synth_feat : {synth_feat.shape}")

    # 70/30 split of real (temporal, no shuffle — same as XGBoost script)
    n = len(real_feat)
    split = int(n * 0.7)
    real_tr_f, real_te_f = real_feat[:split], real_feat[split:]
    real_tr_y, real_te_y = real_y[:split], real_y[split:]
    print(f"  real_tr: {real_tr_f.shape} ({int(real_tr_y.sum())} attacks), "
          f"real_te: {real_te_f.shape} ({int(real_te_y.sum())} attacks)")
    print()

    # A: Real -> Real
    print("[A] Real -> Real...")
    A = fit_eval(real_tr_f, real_tr_y, real_te_f, real_te_y)
    print(f"    F1={A['f1']:.4f} P={A['precision']:.4f} R={A['recall']:.4f} "
          f"AUROC={A['auroc'] or 0:.4f} ({A['fit_time_s']}s)")

    # B: Real -> Synthetic
    print("[B] Real -> Synthetic...")
    B = fit_eval(real_tr_f, real_tr_y, synth_feat, synth_y)
    print(f"    F1={B['f1']:.4f} P={B['precision']:.4f} R={B['recall']:.4f} "
          f"AUROC={B['auroc'] or 0:.4f} ({B['fit_time_s']}s)")

    # C: Synthetic -> Real
    print("[C] Synthetic -> Real...")
    C = fit_eval(synth_feat, synth_y, real_te_f, real_te_y)
    print(f"    F1={C['f1']:.4f} P={C['precision']:.4f} R={C['recall']:.4f} "
          f"AUROC={C['auroc'] or 0:.4f} ({C['fit_time_s']}s)")

    # D: Mixed -> Real
    print("[D] Mixed -> Real...")
    mixed_f = np.concatenate([real_tr_f, synth_feat], axis=0)
    mixed_y = np.concatenate([real_tr_y, synth_y], axis=0)
    D = fit_eval(mixed_f, mixed_y, real_te_f, real_te_y)
    print(f"    F1={D['f1']:.4f} P={D['precision']:.4f} R={D['recall']:.4f} "
          f"AUROC={D['auroc'] or 0:.4f} ({D['fit_time_s']}s)")

    out = {
        "classifier": "RandomForest (teammate's 30-feat strategy)",
        "hparams": RF_PARAMS,
        "n_features": 30,
        "window_sec": WIN_SEC,
        "stride_sec": STRIDE,
        "threshold": THR,
        "A": A, "B": B, "C": C, "D": D,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
