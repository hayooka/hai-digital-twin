"""
eval.py — Per-layer evaluation utilities for the HAI Digital Twin pipeline.

Layer 1 (Digital Twin):   evaluate_twin()      → RMSE on val and test
Layer 2 (Detector):       evaluate_detector()  → eTaPR + standard metrics
Layer 3 (Generator):      evaluate_generator() → TSTR placeholder
Layer 4 (Attack Twin):    evaluate_twin()      → reuses Layer 1 evaluator

eTaPR reference:
    Hwang et al. "Do You Know What You Are Doing? Understanding Time-series
    Anomaly Detection" (2022) — designed and validated on the HAI dataset.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# eTaPR core
# ─────────────────────────────────────────────────────────────────────────────

def _get_segments(y: np.ndarray) -> list[tuple[int, int]]:
    """Convert binary array to list of (start, end) inclusive index pairs."""
    segments: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, v in enumerate(y):
        if v == 1 and not in_seg:
            start = i
            in_seg = True
        elif v == 0 and in_seg:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, len(y) - 1))
    return segments


def etapr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    theta_p: float = 0.5,
    theta_r: float = 0.1,
) -> dict:
    """
    eTaPR: enhanced Time-aware Precision and Recall.

    Evaluates anomaly detection at the SEGMENT level, not timestamp level.
    This prevents gaming via the Point-Adjust trick and properly handles
    detection delay — catching an attack 30s late still counts.

    Args
    ----
    y_true   : (N,) binary ground truth  (0 = normal, 1 = attack)
    y_pred   : (N,) binary predictions
    theta_p  : minimum overlap fraction for a predicted segment to be a TP
               (default 0.5 — predicted segment must overlap ≥50% with a real one)
    theta_r  : minimum coverage fraction for a real segment to be "detected"
               (default 0.1 — detecting ≥10% of an attack window counts)

    Returns
    -------
    dict with eTaP, eTaR, eTaF1, n_pred_segs, n_real_segs
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    real_segs = _get_segments(y_true)
    pred_segs = _get_segments(y_pred)

    if not real_segs:
        # No actual attacks — any prediction is a FP
        return {
            "eTaP": 0.0 if pred_segs else 1.0,
            "eTaR": 1.0,
            "eTaF1": 0.0,
            "n_pred_segs": len(pred_segs),
            "n_real_segs": 0,
        }

    if not pred_segs:
        return {
            "eTaP": 0.0,
            "eTaR": 0.0,
            "eTaF1": 0.0,
            "n_pred_segs": 0,
            "n_real_segs": len(real_segs),
        }

    # ── Precision: for each predicted segment, does it overlap ≥ theta_p with any real seg?
    tp_p = 0
    for ps, pe in pred_segs:
        pred_len = pe - ps + 1
        best_overlap = max(
            max(0, min(pe, re) - max(ps, rs) + 1)
            for rs, re in real_segs
        )
        if best_overlap / pred_len >= theta_p:
            tp_p += 1
    eTaP = tp_p / len(pred_segs)

    # ── Recall: for each real segment, is it covered by ≥ theta_r of predicted detections?
    detected = 0.0
    for rs, re in real_segs:
        real_len = re - rs + 1
        covered = sum(
            max(0, min(pe, re) - max(ps, rs) + 1)
            for ps, pe in pred_segs
        )
        if covered / real_len >= theta_r:
            detected += 1.0
    eTaR = detected / len(real_segs)

    eTaF1 = (
        2 * eTaP * eTaR / (eTaP + eTaR)
        if (eTaP + eTaR) > 0
        else 0.0
    )

    return {
        "eTaP":       eTaP,
        "eTaR":       eTaR,
        "eTaF1":      eTaF1,
        "n_pred_segs": len(pred_segs),
        "n_real_segs": len(real_segs),
    }


def best_threshold_etapr(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_thresholds: int = 100,
    theta_p: float = 0.5,
    theta_r: float = 0.1,
) -> tuple[float, dict]:
    """
    Sweep anomaly score thresholds to find the one maximising eTaF1.

    Args
    ----
    y_true       : (N,) binary ground truth
    scores       : (N,) continuous anomaly scores (higher = more anomalous)
    n_thresholds : number of candidate thresholds to try
    theta_p/r    : passed to etapr()

    Returns
    -------
    (best_threshold, best_metrics_dict)
    """
    thresholds = np.percentile(scores, np.linspace(0, 100, n_thresholds))
    best_thr, best_metrics = thresholds[0], {"eTaF1": -1.0}

    for thr in thresholds:
        y_pred   = (scores >= thr).astype(np.int32)
        metrics  = etapr(y_true, y_pred, theta_p=theta_p, theta_r=theta_r)
        if metrics["eTaF1"] > best_metrics["eTaF1"]:
            best_thr     = thr
            best_metrics = metrics

    best_metrics["threshold"] = float(best_thr)
    return float(best_thr), best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 / Layer 4 — Digital Twin (Seq2Seq simulator)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_twin(
    predict_fn,
    X_val:  np.ndarray,
    Y_val:  np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    y_test: np.ndarray,
    label:  str = "Digital Twin",
    save_path: str | None = None,
) -> dict:
    """
    Evaluate a digital twin (Layer 1 or Layer 4) as a SIMULATOR.

    Metrics are RMSE-based — this model is NOT a detector.
    The separation gap (attack_rmse / normal_rmse) is the signal
    that Layer 2 (Isolation Forest) uses downstream.

    Args
    ----
    predict_fn : callable(X, Y) → predicted Y  (numpy arrays, same shape as Y)
                 For teacher-forced models: predict_fn(X, Y) uses Y as dec input.
                 For autoregressive: predict_fn(X, Y=None).
    X_val, Y_val   : (M, input_len, F) val windows (normal only)
    X_test, Y_test : (K, input_len, F) test2 windows (normal + attack)
    y_test         : (K,) binary labels for test2
    label          : display label for the printed table
    save_path      : optional JSON path to save results

    Returns
    -------
    dict with rmse_val, rmse_test_normal, rmse_test_attack, separation_ratio
    """
    def _rmse(X, Y, mask=None):
        Y_pred = predict_fn(X, Y)
        mse    = ((Y_pred - Y) ** 2).mean(axis=(1, 2))   # (N,)
        if mask is not None:
            mse = mse[mask]
        return float(np.sqrt(np.mean(mse))) if len(mse) > 0 else float("nan")

    normal_mask = (y_test == 0)
    attack_mask = (y_test == 1)

    rmse_val    = _rmse(X_val,  Y_val)
    rmse_normal = _rmse(X_test, Y_test, mask=normal_mask)
    rmse_attack = _rmse(X_test, Y_test, mask=attack_mask)
    ratio       = rmse_attack / rmse_normal if rmse_normal > 0 else float("nan")

    results = {
        "rmse_val":          rmse_val,
        "rmse_test_normal":  rmse_normal,
        "rmse_test_attack":  rmse_attack,
        "separation_ratio":  ratio,
    }

    print(f"\n{'=' * 55}")
    print(f"  {label.upper()} — SIMULATOR EVALUATION")
    print(f"{'=' * 55}")
    print(f"  RMSE val    (normal)         = {rmse_val:.5f}")
    print(f"  RMSE test   (normal windows) = {rmse_normal:.5f}")
    print(f"  RMSE test   (attack windows) = {rmse_attack:.5f}")
    print(f"  Separation ratio (atk/norm)  = {ratio:.2f}x")
    print(f"  {'↑ good' if ratio > 1.5 else '↓ weak — check model or data'}")
    print(f"{'=' * 55}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {save_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Anomaly Detector (Isolation Forest / LSTM-AE)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_detector(
    y_true:     np.ndarray,
    scores:     np.ndarray,
    label:      str = "Detector",
    theta_p:    float = 0.5,
    theta_r:    float = 0.1,
    save_path:  str | None = None,
) -> dict:
    """
    Evaluate an anomaly detector (Layer 2) using eTaPR + standard metrics.

    Sweeps anomaly score thresholds to find the best eTaF1, then also reports
    standard P/R/F1/AUC at that same threshold for comparison.

    Args
    ----
    y_true    : (N,) binary ground truth (window level: 1 = attack window)
    scores    : (N,) continuous anomaly score (higher = more anomalous)
    label     : display label
    theta_p   : eTaPR precision overlap threshold
    theta_r   : eTaPR recall coverage threshold
    save_path : optional JSON path

    Returns
    -------
    dict with eTaP, eTaR, eTaF1, threshold, standard_f1, precision, recall, roc_auc
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix,
    )

    # Best eTaF1 threshold
    best_thr, etapr_metrics = best_threshold_etapr(
        y_true, scores, theta_p=theta_p, theta_r=theta_r
    )
    y_pred = (scores >= best_thr).astype(np.int32)

    # Standard metrics at the same threshold
    std_f1  = f1_score(y_true,  y_pred, zero_division=0)
    std_pre = precision_score(y_true, y_pred, zero_division=0)
    std_rec = recall_score(y_true,  y_pred, zero_division=0)
    try:
        auc = float(roc_auc_score(y_true, scores))
    except Exception:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    results = {
        **etapr_metrics,
        "standard_f1":  float(std_f1),
        "precision":    float(std_pre),
        "recall":       float(std_rec),
        "roc_auc":      auc,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "theta_p": theta_p,
        "theta_r": theta_r,
    }

    print(f"\n{'=' * 55}")
    print(f"  {label.upper()} — ANOMALY DETECTION (eTaPR)")
    print(f"{'=' * 55}")
    print(f"  Threshold (best eTaF1) = {best_thr:.4f}")
    print(f"  ── eTaPR (segment-level) ──")
    print(f"  eTaP      = {etapr_metrics['eTaP']:.4f}  (θ_p={theta_p})")
    print(f"  eTaR      = {etapr_metrics['eTaR']:.4f}  (θ_r={theta_r})")
    print(f"  eTaF1     = {etapr_metrics['eTaF1']:.4f}")
    print(f"  Segments  : pred={etapr_metrics['n_pred_segs']}  real={etapr_metrics['n_real_segs']}")
    print(f"  ── Standard (timestamp-level, same threshold) ──")
    print(f"  F1        = {std_f1:.4f}")
    print(f"  Precision = {std_pre:.4f}")
    print(f"  Recall    = {std_rec:.4f}")
    print(f"  ROC-AUC   = {auc:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"{'=' * 55}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {save_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Attack Generator (Diffusion / VAE)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_generator(
    real_windows:      np.ndarray,
    generated_windows: np.ndarray,
    detector_scores_fn,
    y_real:            np.ndarray,
    label:             str = "Generator",
    save_path:         str | None = None,
) -> dict:
    """
    Evaluate the attack generator (Layer 3).

    TSTR (Train on Synthetic, Test on Real):
        1. Train a lightweight XGBoost classifier on generated attack windows
           (positive class) + normal windows (negative class).
        2. Test on real test2 windows.
        3. Report F1 — measures whether generated attacks have the right
           statistical signature to help a classifier generalise.

    Detectability:
        Run the trained Layer 2 detector on generated windows.
        Fraction flagged = detectability rate.
        Good generators produce attacks that ARE detectable.

    Args
    ----
    real_windows      : (N, window_len, F) real attack windows (from test1)
    generated_windows : (M, window_len, F) generated attack windows
    detector_scores_fn: callable(windows) → (M,) anomaly scores from Layer 2
    y_real            : (N,) labels for real test windows (for TSTR eval)
    label             : display label
    save_path         : optional JSON path

    Returns
    -------
    dict with tstr_f1, detectability_rate
    """
    # Detectability
    gen_scores    = detector_scores_fn(generated_windows)
    # Use median normal score as threshold proxy — rough but no val labels needed
    detect_rate   = float((gen_scores > np.median(gen_scores)).mean())

    results = {
        "detectability_rate": detect_rate,
        "n_real":             int(len(real_windows)),
        "n_generated":        int(len(generated_windows)),
    }

    print(f"\n{'=' * 55}")
    print(f"  {label.upper()} — GENERATION QUALITY")
    print(f"{'=' * 55}")
    print(f"  Generated windows    : {len(generated_windows)}")
    print(f"  Detectability rate   : {detect_rate:.3f}  (fraction flagged by Layer 2)")
    print(f"  (TSTR evaluation requires XGBoost — run separately)")
    print(f"{'=' * 55}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {save_path}")

    return results
