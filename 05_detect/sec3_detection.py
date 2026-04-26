"""
sec3_detection.py — Section 3: Attack Detection

Uses PV residuals from gru_scenario_weighted for anomaly scoring.

Produces:
  figures/s3/s3_residual_dist.png          — PV residual distributions per scenario (violin)
  figures/s3/s3_roc_curve.png              — ROC curve (binary: normal vs attack)
  figures/s3/s3_pr_curve.png               — Precision-Recall curve
  figures/s3/s3_confusion_matrix.png       — Confusion matrix at best F1 threshold
  figures/s3/s3_detection_per_attack.png   — Detection summary table (replaces bar chart)
  figures/s3/s3_haiend_heatmap.png         — Mean absolute HAIEnd residual per channel per scenario
  figures/s3/s3_residual_timeline.png      — Residual score over time per scenario with threshold
  figures/s3/s3_threshold_sensitivity.png  — Recall vs FPR as threshold varies

Usage:
    python report_plots/code/sec3_detection.py
"""

import sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              average_precision_score, confusion_matrix,
                              f1_score)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from gru import GRUPlant
from config import HAIEND_COLS

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH    = 128
CKPT_DIR = ROOT / "outputs" / "pipeline" / "gru_scenario_haiend"
OUT_DIR  = ROOT / "report_plots" / "figures" / "s3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS  = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}
SC_COLORS  = {0: "#2196F3", 1: "#FF5722", 2: "#E91E63", 3: "#9C27B0"}
HAIEND_SHORT = [c.replace("-OUT", "").replace("DM-", "") for c in HAIEND_COLS]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model():
    ckpt = torch.load(CKPT_DIR / "gru_plant.pt", map_location=DEVICE)
    ms   = ckpt["model_state"]
    emb  = ms["scenario_emb.weight"].shape[1]
    m = GRUPlant(
        n_plant_in  = ckpt.get("n_plant_in", ms["encoder.weight_ih_l0"].shape[1] - emb),
        n_pv        = ckpt.get("n_pv",       ms["fc.3.weight"].shape[0]),
        hidden      = ckpt["hidden"],
        layers      = ckpt["layers"],
        n_scenarios = ms["scenario_emb.weight"].shape[0],
        n_haiend    = ms["haiend_head.3.weight"].shape[0],
    ).to(DEVICE)
    m.load_state_dict(ms, strict=False)
    m.eval()
    return m


def run_inference(model, plant_data):
    X, Xcv   = plant_data["X_test"], plant_data["X_cv_target_test"]
    pv_init  = plant_data["pv_init_test"]
    sc       = plant_data["scenario_test"]
    N, TL    = len(X), plant_data["pv_target_test"].shape[1]
    n_pv     = plant_data["n_pv"]
    n_haiend = plant_data["n_haiend"]

    pv_pred     = np.zeros((N, TL, n_pv),     dtype=np.float32)
    haiend_pred = np.zeros((N, TL, n_haiend), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl = slice(i, i + BATCH)
            pv_out, h_out = model.predict(
                torch.tensor(X[sl]).float().to(DEVICE),
                torch.tensor(Xcv[sl]).float().to(DEVICE),
                torch.tensor(pv_init[sl]).float().to(DEVICE),
                torch.tensor(sc[sl]).long().to(DEVICE),
            )
            pv_pred[i:i + pv_out.size(0)] = pv_out.cpu().numpy()
            if h_out is not None:
                haiend_pred[i:i + h_out.size(0)] = h_out.cpu().numpy()
    return pv_pred, haiend_pred


def pv_score(pred, target):
    """Mean absolute PV residual per window: (N, TL, K) → (N,)"""
    return np.abs(pred - target).mean(axis=(1, 2))


# ── Plot 1: Residual distributions (violin) ───────────────────────────────────

def plot_residual_dist(scores, sc_arr):
    sc_ids = [0, 1, 2, 3]
    data   = [scores[sc_arr == s] for s in sc_ids]
    labels = [SCENARIOS[s] for s in sc_ids]
    colors = [SC_COLORS[s] for s in sc_ids]

    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(data, positions=range(len(sc_ids)),
                          showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)
    for key in ("cbars", "cmaxes", "cmins"):
        parts[key].set_color("gray")

    ax.set_xticks(range(len(sc_ids)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean Absolute PV Residual", fontsize=12)
    ax.set_title("PV Residual Distribution per Scenario", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for i, d in enumerate(data):
        med = np.median(d)
        ax.text(i, med * 1.05, f"{med:.4f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    fig.tight_layout()
    path = OUT_DIR / "s3_residual_dist.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 2: ROC curve ────────────────────────────────────────────────────────

def plot_roc(y_true_bin, scores):
    fpr, tpr, _ = roc_curve(y_true_bin, scores)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2.5, label=f"PV Residual (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Attack Detection", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "s3_roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return roc_auc


# ── Plot 3: Precision-Recall curve ───────────────────────────────────────────

def plot_pr(y_true_bin, scores):
    prec, rec, thresholds = precision_recall_curve(y_true_bin, scores)
    ap  = average_precision_score(y_true_bin, scores)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best = np.argmax(f1s)
    best_thresh = thresholds[best]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec, prec, color="#FF5722", lw=2.5, label=f"PR (AP = {ap:.3f})")
    ax.scatter(rec[best], prec[best], color="black", s=80, zorder=5,
               label=f"Best F1={f1s[best]:.3f} @ thr={best_thresh:.4f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Attack Detection", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "s3_pr_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return best_thresh


# ── Plot 4: Confusion matrix ─────────────────────────────────────────────────

def plot_confusion(y_true_bin, scores, threshold):
    y_pred = (scores >= threshold).astype(int)
    cm     = confusion_matrix(y_true_bin, y_pred)
    labels = ["Normal", "Attack"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    thresh_cm = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > thresh_cm else "black")

    f1 = f1_score(y_true_bin, y_pred)
    ax.set_title(f"Confusion Matrix — F1={f1:.3f} (thr={threshold:.4f})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "s3_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return f1


# ── Plot 5: Detection summary table (replaces bar chart) ─────────────────────
# A table is more informative than a bar chart here: it shows n (sample size),
# recall, and avg alert time together, and makes the tiny AE_no sample visible.

def plot_detection_per_attack(sc_arr, scores, threshold):
    attack_info = {
        1: ("AP_no",   "#FF5722"),
        2: ("AP_with", "#E91E63"),
        3: ("AE_no",   "#9C27B0"),
    }

    rows = []
    for sc_id, (name, _) in attack_info.items():
        mask      = sc_arr == sc_id
        n         = int(mask.sum())
        detected  = int((scores[mask] >= threshold).sum())
        recall    = detected / n if n > 0 else 0.0
        rows.append([name, str(n), str(detected), f"{recall:.1%}"])

    # Normal FPR row
    normal_mask = sc_arr == 0
    n_norm      = int(normal_mask.sum())
    fp          = int((scores[normal_mask] >= threshold).sum())
    fpr         = fp / n_norm if n_norm > 0 else 0.0
    rows.append(["Normal (FPR)", str(n_norm), str(fp), f"{fpr:.1%}"])

    col_labels = ["Scenario", "Windows (n)", "Detected", "Recall / FPR"]

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.axis("off")

    row_colors = [
        ["#FFF3E0", "#FFF3E0", "#FFF3E0", "#FFF3E0"],
        ["#FCE4EC", "#FCE4EC", "#FCE4EC", "#FCE4EC"],
        ["#F3E5F5", "#F3E5F5", "#F3E5F5", "#F3E5F5"],
        ["#E3F2FD", "#E3F2FD", "#E3F2FD", "#E3F2FD"],
    ]

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.0)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#B0BEC5")

    ax.set_title(f"Attack Detection Summary  (threshold = {threshold:.4f})",
                 fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    path = OUT_DIR / "s3_detection_per_attack.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 6: HAIEnd residual heatmap per scenario ──────────────────────────────

def plot_haiend_heatmap(haiend_pred, haiend_tgt, sc_arr):
    abs_res = np.abs(haiend_pred - haiend_tgt)
    per_win = abs_res.mean(axis=1)
    sc_ids  = [0, 1, 2, 3]
    mat     = np.array([per_win[sc_arr == s].mean(axis=0) for s in sc_ids])

    fig, ax = plt.subplots(figsize=(max(10, len(HAIEND_COLS) * 0.45), 4))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Mean |Residual|")
    ax.set_yticks(range(len(sc_ids)))
    ax.set_yticklabels([SCENARIOS[s] for s in sc_ids], fontsize=10)
    ax.set_xticks(range(len(HAIEND_COLS)))
    ax.set_xticklabels(HAIEND_SHORT, rotation=60, ha="right", fontsize=7)
    ax.set_title("Mean Absolute HAIEnd Residual per Channel per Scenario",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "s3_haiend_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 7: Residual timeline per scenario ────────────────────────────────────
# Shows the raw anomaly score over sequential windows for each scenario,
# with the detection threshold drawn. Makes the alert moment visible.

def plot_residual_timeline(scores, sc_arr, threshold):
    sc_ids = [0, 1, 2, 3]
    fig, axes = plt.subplots(len(sc_ids), 1, figsize=(11, 8), sharex=False)

    for ax, sc_id in zip(axes, sc_ids):
        mask = sc_arr == sc_id
        s    = scores[mask]
        x    = np.arange(len(s))
        color = SC_COLORS[sc_id]
        label = SCENARIOS[sc_id]

        ax.plot(x, s, color=color, lw=1.2, alpha=0.85)
        ax.axhline(threshold, color="black", lw=1.2, linestyle="--", label=f"Threshold ({threshold:.4f})")

        # shade detected windows
        detected = s >= threshold
        ax.fill_between(x, 0, s, where=detected, color="red", alpha=0.25, label="Detected")

        n_det = detected.sum()
        recall = n_det / len(s) if len(s) > 0 else 0
        ax.set_ylabel("Score", fontsize=9)
        ax.set_title(f"{label}  —  n={len(s)}, detected={n_det} ({recall:.1%})",
                     fontsize=10, fontweight="bold", loc="left")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Window Index", fontsize=11)
    fig.suptitle("Anomaly Score Timeline per Scenario", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "s3_residual_timeline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 8: Threshold sensitivity (Recall vs FPR) ────────────────────────────
# Shows how recall for each attack type and FPR on normal data change as the
# detection threshold varies — makes the operating point choice explicit.

def plot_threshold_sensitivity(scores, sc_arr, best_thresh):
    thresholds = np.linspace(scores.min(), scores.max(), 300)

    def recall_at(sc_id, thr):
        mask = sc_arr == sc_id
        n = mask.sum()
        return (scores[mask] >= thr).sum() / n if n > 0 else 0.0

    fpr_curve = np.array([recall_at(0, t) for t in thresholds])
    ap_no     = np.array([recall_at(1, t) for t in thresholds])
    ap_with   = np.array([recall_at(2, t) for t in thresholds])
    ae_no     = np.array([recall_at(3, t) for t in thresholds])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, ap_no,   color="#FF5722", lw=2,   label="Recall — AP_no")
    ax.plot(thresholds, ap_with, color="#E91E63", lw=2,   label="Recall — AP_with")
    ax.plot(thresholds, ae_no,   color="#9C27B0", lw=2,   label="Recall — AE_no")
    ax.plot(thresholds, fpr_curve, color="#2196F3", lw=2, linestyle="--", label="FPR — Normal")
    ax.axvline(best_thresh, color="black", lw=1.5, linestyle=":", label=f"Best-F1 threshold ({best_thresh:.4f})")

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title("Threshold Sensitivity — Recall per Attack Type vs FPR",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "s3_threshold_sensitivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    data       = load_and_prepare_data()
    plant_data = data["plant"]
    sc_arr     = plant_data["scenario_test"]
    haiend_tgt = plant_data["haiend_target_test"]
    attack_arr = plant_data["attack_test"]

    print("Loading model...")
    model = load_model()

    print("Running inference...")
    pv_pred, haiend_pred = run_inference(model, plant_data)

    pv_true = plant_data["pv_target_test"]
    scores  = pv_score(pv_pred, pv_true)

    if attack_arr.ndim > 1:
        y_true_bin = (attack_arr.max(axis=1) > 0).astype(int)
    else:
        y_true_bin = (attack_arr > 0).astype(int)

    print(f"\n  Test windows : {len(scores)}")
    print(f"  Attack windows: {y_true_bin.sum()} / {len(y_true_bin)}")
    print(f"  Normal score median: {np.median(scores[y_true_bin==0]):.5f}")
    print(f"  Attack score median: {np.median(scores[y_true_bin==1]):.5f}")

    print("\n[Plot 1] Residual distributions...")
    plot_residual_dist(scores, sc_arr)

    print("[Plot 2] ROC curve...")
    roc_auc = plot_roc(y_true_bin, scores)

    print("[Plot 3] PR curve...")
    best_thresh = plot_pr(y_true_bin, scores)

    print("[Plot 4] Confusion matrix...")
    f1 = plot_confusion(y_true_bin, scores, best_thresh)

    print("[Plot 5] Detection summary table...")
    plot_detection_per_attack(sc_arr, scores, best_thresh)

    print("[Plot 6] HAIEnd residual heatmap...")
    plot_haiend_heatmap(haiend_pred, haiend_tgt, sc_arr)

    print("[Plot 7] Residual timeline...")
    plot_residual_timeline(scores, sc_arr, best_thresh)

    print("[Plot 8] Threshold sensitivity...")
    plot_threshold_sensitivity(scores, sc_arr, best_thresh)

    stats = {
        "roc_auc":        float(roc_auc),
        "best_f1":        float(f1),
        "best_threshold": float(best_thresh),
        "n_test":         int(len(scores)),
        "n_attack":       int(y_true_bin.sum()),
        "normal_median":  float(np.median(scores[y_true_bin==0])),
        "attack_median":  float(np.median(scores[y_true_bin==1])),
    }
    with open(OUT_DIR / "s3_detection_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved: s3_detection_stats.json")
    print(f"\nAll Section 3 plots saved to: {OUT_DIR}/")
