"""
run_classifier_experiments.py - run Experiments A, B, C from Student Guide §7.

A: Real -> Real        train on real, test on real (baseline)
B: Real -> Synthetic   train on real, test on synthetic (are synth attacks faithful?)
C: Synthetic -> Real   train on synthetic, test on real (can synth replace real?)
D: already shipped     via best_hai_classifier.pkl — read at dashboard runtime

Saves results to cache/classifier_experiments.json.
Uses the same 133 features and same XGBoost hyperparameters as the Guardian
pickle so the four experiments are apples-to-apples comparable.

Run once from the dashboard directory:
    python run_classifier_experiments.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent

REAL_TEST1 = Path(r"C:\Users\PC GAMING\Desktop\new_ai\processed\test1.csv")
REAL_TEST2 = Path(r"C:\Users\PC GAMING\Desktop\new_ai\processed\test2.csv")
SYNTH_CSV  = Path(r"C:\Users\PC GAMING\Desktop\AI\HAI\synthetic_attacks.csv")
GUARDIAN   = Path(r"C:\Users\PC GAMING\Desktop\AI\HAI\best_hai_classifier.pkl")
OUT_JSON   = HERE / "cache" / "classifier_experiments.json"
THR        = 0.35  # same as Guardian's peak-F1 threshold


# ── Hyperparameters: pull from the Guardian so all 4 experiments match ──

def _load_guardian_hparams() -> dict:
    """Return XGBoost hyperparameters extracted from the shipped Guardian pickle."""
    pipe = joblib.load(GUARDIAN)
    model = pipe["model"]
    params = model.get_params()
    out = {}
    for k in ["n_estimators", "max_depth", "learning_rate", "random_state",
              "eval_metric", "n_jobs"]:
        if k in params and params[k] is not None:
            out[k] = params[k]
    out.setdefault("n_jobs", -1)
    out.setdefault("eval_metric", "logloss")
    out.setdefault("random_state", 42)
    return out, list(pipe["features"])


def _load_data(features: list[str]):
    """Load real + synthetic sets with the Guardian's 133-col feature schema."""
    # Real: test1 + test2 concatenated
    real = pd.concat(
        [pd.read_csv(REAL_TEST1, low_memory=False),
         pd.read_csv(REAL_TEST2, low_memory=False)],
        ignore_index=True,
    )
    real_X = real[features].to_numpy(dtype=np.float32)
    real_y = (real["label"] > 0).astype(np.int8).to_numpy()

    # Synthetic: 36 k rows
    synth = pd.read_csv(SYNTH_CSV, low_memory=False)
    missing = [f for f in features if f not in synth.columns]
    if missing:
        raise RuntimeError(
            f"synthetic_attacks.csv missing {len(missing)} features: {missing[:3]}"
        )
    synth_X = synth[features].to_numpy(dtype=np.float32)
    synth_y = synth["label"].astype(np.int8).to_numpy()

    return real_X, real_y, synth_X, synth_y


def _fit_and_eval(X_tr, y_tr, X_te, y_te, hparams: dict) -> dict:
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    clf = XGBClassifier(**hparams)
    t0 = time.time()
    clf.fit(X_tr_s, y_tr)
    t_fit = time.time() - t0
    proba = clf.predict_proba(X_te_s)[:, 1]
    pred = (proba >= THR).astype(int)
    cm = confusion_matrix(y_te, pred).tolist()
    return {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_train_attack": int(y_tr.sum()),
        "n_test_attack": int(y_te.sum()),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_te, proba)) if len(set(y_te)) > 1 else None,
        "avg_precision": float(average_precision_score(y_te, proba)) if len(set(y_te)) > 1 else None,
        "threshold": THR,
        "confusion_matrix": cm,
        "fit_time_s": round(t_fit, 2),
    }


def main() -> None:
    OUT_JSON.parent.mkdir(exist_ok=True)
    hparams, features = _load_guardian_hparams()
    print(f"Loaded Guardian hparams: {hparams}")
    print(f"Features: {len(features)}")
    print()
    print("Loading data...")
    real_X, real_y, synth_X, synth_y = _load_data(features)
    print(f"  real:      {real_X.shape}, {int(real_y.sum())} attacks")
    print(f"  synthetic: {synth_X.shape}, {int(synth_y.sum())} attacks")
    print()

    # Split real 70/30 for A, C, D (D already shipped; we just re-derive the
    # A and C training + test sets consistently).
    real_X_tr, real_X_te, real_y_tr, real_y_te = train_test_split(
        real_X, real_y, test_size=0.3, shuffle=False,
    )
    print(f"  real train: {real_X_tr.shape}, real test: {real_X_te.shape}")
    print()

    # ── Experiment A: Real -> Real ──────────────────────────────────────
    print("[A] Real -> Real...")
    A = _fit_and_eval(real_X_tr, real_y_tr, real_X_te, real_y_te, hparams)
    print(f"    F1={A['f1']:.4f} P={A['precision']:.4f} R={A['recall']:.4f} "
          f"AUROC={A['auroc']:.4f} ({A['fit_time_s']}s)")

    # ── Experiment B: Real -> Synthetic ─────────────────────────────────
    print("[B] Real -> Synthetic...")
    B = _fit_and_eval(real_X_tr, real_y_tr, synth_X, synth_y, hparams)
    print(f"    F1={B['f1']:.4f} P={B['precision']:.4f} R={B['recall']:.4f} "
          f"AUROC={B['auroc']:.4f} ({B['fit_time_s']}s)")

    # ── Experiment C: Synthetic -> Real ─────────────────────────────────
    print("[C] Synthetic -> Real...")
    C = _fit_and_eval(synth_X, synth_y, real_X_te, real_y_te, hparams)
    print(f"    F1={C['f1']:.4f} P={C['precision']:.4f} R={C['recall']:.4f} "
          f"AUROC={C['auroc']:.4f} ({C['fit_time_s']}s)")

    out = {
        "threshold": THR,
        "hparams": hparams,
        "n_features": len(features),
        "A": A,
        "B": B,
        "C": C,
        # D is rendered at runtime from the Guardian + live threshold slider
        "_D_note": "Experiment D (Mixed -> Real) is the live Guardian in the dashboard.",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
