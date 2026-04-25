"""Train 3 one-class anomaly detectors on normal-only features, evaluate on
the labeled test set, and save all metrics + artifacts.

Methods:
  A) Isolation Forest  (sklearn)
  B) One-Class SVM     (sklearn) — RBF kernel, nu=0.05
  C) MLP Autoencoder   (pytorch) — reconstruction MSE as score

Per plan: threshold is picked on a 10% calibration slice of the test set
(AUROC-maximizing F1); metrics are reported on the remaining 90%.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "classifier"
FEATS_PATH = OUT_DIR / "features.npz"
RNG = np.random.RandomState(0)


# ── Autoencoder ─────────────────────────────────────────────────────────────

class MLPAutoencoder(nn.Module):
    def __init__(self, d_in: int = 50, d_hidden: int = 32, d_bottle: int = 8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_bottle), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(d_bottle, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_in),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


def train_ae(X_train: np.ndarray, epochs: int = 50, lr: float = 1e-3) -> MLPAutoencoder:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPAutoencoder(d_in=X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    x = torch.from_numpy(X_train).float().to(device)
    bs = 256
    n = x.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, bs):
            batch = x[perm[i : i + bs]]
            opt.zero_grad()
            rec = model(batch)
            loss = loss_fn(rec, batch)
            loss.backward()
            opt.step()
            total += float(loss.item()) * batch.shape[0]
        if (ep + 1) % 10 == 0:
            print(f"    AE epoch {ep + 1}/{epochs}  loss={total / n:.5f}")
    return model


def ae_score(model: MLPAutoencoder, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X).float().to(device)
        rec = model(x)
        err = ((rec - x) ** 2).mean(dim=1).cpu().numpy()
    return err.astype(np.float64)


# ── Eval helpers ────────────────────────────────────────────────────────────

def _best_f1_threshold(y: np.ndarray, s: np.ndarray) -> Tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    # thr has length len(prec) - 1
    if thr.size == 0:
        return 0.5, 0.0
    k = int(np.argmax(f1[:-1]))
    return float(thr[k]), float(f1[k])


def _eval(name: str, y: np.ndarray, scores: np.ndarray, cal_frac: float = 0.10) -> dict:
    # Split test into calibration / reported eval
    n = y.shape[0]
    idx = np.arange(n)
    RNG.shuffle(idx)
    n_cal = max(10, int(n * cal_frac))
    cal_idx, eval_idx = idx[:n_cal], idx[n_cal:]
    y_cal, s_cal = y[cal_idx], scores[cal_idx]
    y_ev, s_ev = y[eval_idx], scores[eval_idx]

    thr, f1_cal = _best_f1_threshold(y_cal, s_cal)
    pred = (s_ev >= thr).astype(np.int8)
    try:
        auroc = float(roc_auc_score(y_ev, s_ev))
    except ValueError:
        auroc = float("nan")
    f1 = float(f1_score(y_ev, pred, zero_division=0))
    prec = float(precision_score(y_ev, pred, zero_division=0))
    rec = float(recall_score(y_ev, pred, zero_division=0))
    cm = confusion_matrix(y_ev, pred, labels=[0, 1]).tolist()
    print(
        f"  [{name}] AUROC={auroc:.4f}  F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}"
        f"  thr={thr:.5f}  (cal n={n_cal}, eval n={len(eval_idx)})"
    )
    return {
        "model": name,
        "auroc": auroc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "threshold": thr,
        "confusion_matrix": cm,
        "n_cal": int(n_cal),
        "n_eval": int(len(eval_idx)),
    }


def _plot_curves(results: dict, scores_by_model: dict, y_eval_by_model: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, s in scores_by_model.items():
        y = y_eval_by_model[name]
        try:
            fpr, tpr, _ = roc_curve(y, s)
            axes[0].plot(fpr, tpr, label=f"{name} (AUROC={auc(fpr, tpr):.3f})")
            prec, rec, _ = precision_recall_curve(y, s)
            axes[1].plot(rec, prec, label=f"{name} (AUPRC={auc(rec, prec):.3f})")
        except Exception:
            pass
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_title("ROC")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend(loc="lower right")
    axes[1].set_title("Precision / Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    out = OUT_DIR / "roc_pr_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  plot → {out}")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    if not FEATS_PATH.exists():
        print(f"ERROR: {FEATS_PATH} missing — run build_features.py first.", file=sys.stderr)
        return 1
    z = np.load(FEATS_PATH, allow_pickle=True)
    X_train = z["X_train"]
    X_test = z["X_test"]
    y_test = z["y_test"].astype(np.int8)
    print(f"Loaded: X_train {X_train.shape} | X_test {X_test.shape} | positives {int((y_test == 1).sum())}")

    results = []
    scores_for_plot = {}
    y_for_plot = {}

    # A) Isolation Forest
    print("\n[A] Isolation Forest")
    t0 = time.time()
    iforest = IsolationForest(n_estimators=200, contamination="auto", random_state=0, n_jobs=-1)
    iforest.fit(X_train)
    s_iforest = -iforest.score_samples(X_test)
    print(f"    trained in {time.time() - t0:.1f}s")
    results.append(_eval("iforest", y_test, s_iforest))
    joblib.dump(iforest, OUT_DIR / "iforest.joblib")
    scores_for_plot["iforest"] = s_iforest
    y_for_plot["iforest"] = y_test

    # B) One-Class SVM (subsample train for tractability)
    print("\n[B] One-Class SVM")
    t0 = time.time()
    n_sub = min(5000, X_train.shape[0])
    sub = RNG.choice(X_train.shape[0], size=n_sub, replace=False)
    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(X_train[sub])
    s_ocsvm = -ocsvm.decision_function(X_test)
    print(f"    trained on {n_sub} samples in {time.time() - t0:.1f}s")
    results.append(_eval("ocsvm", y_test, s_ocsvm))
    joblib.dump(ocsvm, OUT_DIR / "ocsvm.joblib")
    scores_for_plot["ocsvm"] = s_ocsvm
    y_for_plot["ocsvm"] = y_test

    # C) Autoencoder
    print("\n[C] Autoencoder")
    t0 = time.time()
    ae = train_ae(X_train)
    s_ae = ae_score(ae, X_test)
    print(f"    trained in {time.time() - t0:.1f}s")
    results.append(_eval("ae", y_test, s_ae))
    torch.save(ae.state_dict(), OUT_DIR / "ae.pt")
    scores_for_plot["ae"] = s_ae
    y_for_plot["ae"] = y_test

    # Plot
    _plot_curves(results, scores_for_plot, y_for_plot)

    # Save JSON
    bench = {"results": results}
    (OUT_DIR / "bench.json").write_text(json.dumps(bench, indent=2))
    print(f"\nSaved → {OUT_DIR / 'bench.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
