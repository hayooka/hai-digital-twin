"""
eval.py — Evaluation for the Generative Digital Twin (Transformer Seq2Seq).

Usage:
    from evaluate.eval import evaluate_twin
    results = evaluate_twin(predict_fn, X_val, Y_val, X_test, Y_test, y_test)
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path


def evaluate_twin(
    predict_fn,
    X_val:  np.ndarray,
    Y_val:  np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    y_test: np.ndarray,
    label:  str = "Transformer Seq2Seq",
    save_path: str | None = None,
) -> dict:
    """
    Evaluate the Generative Digital Twin as a simulator.

    Computes RMSE on val (normal) and test (normal + attack windows).
    For the generative goal, rmse_test_attack should be LOW —
    the model accurately tracks attack trajectories.

    Args
    ----
    predict_fn     : callable(X, Y) → Y_pred  (numpy, same shape as Y)
    X_val, Y_val   : (M, input_len, F) val windows (normal only)
    X_test, Y_test : (K, input_len, F) test windows (held-out attack episodes)
    y_test         : (K,) binary labels — 0=normal, 1=attack
    label          : display label
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
        "rmse_val":         rmse_val,
        "rmse_test_normal": rmse_normal,
        "rmse_test_attack": rmse_attack,
        "separation_ratio": ratio,
    }

    print(f"\n{'=' * 55}")
    print(f"  {label.upper()} — EVALUATION")
    print(f"{'=' * 55}")
    print(f"  RMSE val    (normal)         = {rmse_val:.5f}")
    print(f"  RMSE test   (normal windows) = {rmse_normal:.5f}")
    print(f"  RMSE test   (attack windows) = {rmse_attack:.5f}")
    print(f"  Separation ratio (atk/norm)  = {ratio:.2f}x")
    print(f"  {'Good — model tracks attack trajectories.' if ratio <= 1.2 else 'High ratio — model may not have learned attack dynamics.'}")
    print(f"{'=' * 55}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {save_path}")

    return results
