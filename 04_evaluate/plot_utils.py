"""
plot_utils.py — Paper plots for GRU-Scenario-Weighted Digital Twin
Includes: loss curves, NRMSE bars, per-loop performance, scenario NRMSE,
error growth, heatmap, ROC/PR curves, residual analysis, confusion matrix,
detection rates, and scenario overlays.
"""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))
sys.path.insert(0, str(ROOT / "04_evaluate"))

from config import LOOPS, PV_COLS
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CTRL_LOOPS = ["PC", "LC", "FC", "TC", "CC"]
SCENARIO_SHORT = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}
SCENARIO_NAMES = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}
PV_SHORT = [c.replace("P1_", "") for c in PV_COLS]

# Map PV to control loop
PV_TO_LOOP = {
    "P1_PIT01": "PC",  # Pressure
    "P1_LIT01": "LC",  # Level
    "P1_FT03Z": "FC",  # Flow
    "P1_TIT01": "TC",  # Temperature
    "P1_TIT03": "CC",  # Cooling
}

TARGET_LEN = 180
NRMSE_THRESHOLD = 0.10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helper Functions ────────────────────────────────────────────────────────────────

def _nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NRMSE normalised by signal range."""
    if y_true.size == 0 or y_pred.size == 0:
        return float('nan')
    if y_true.shape != y_pred.shape:
        return float('nan')
    try:
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        sig_range = float(y_true.max() - y_true.min())
        if sig_range <= 1e-10:
            return 0.0 if np.allclose(y_true, y_pred, rtol=1e-5, atol=1e-5) else float('inf')
        return rmse / sig_range
    except Exception:
        return float('nan')


def compute_nrmse(pv_true: np.ndarray, pv_preds: np.ndarray) -> Dict[str, float]:
    """Compute NRMSE per PV channel."""
    nrmse = {}
    for k, pv_name in enumerate(PV_COLS):
        true_k = pv_true[:, :, k].flatten()
        pred_k = pv_preds[:, :, k].flatten()
        nrmse[pv_name] = _nrmse(true_k, pred_k)
    return nrmse


def compute_nrmse_per_scenario(pv_true, pv_preds, scenario_labels) -> Dict[int, Dict[str, float]]:
    """Compute NRMSE per PV per scenario."""
    result = {}
    for sc_id in [0, 1, 2, 3]:
        mask = scenario_labels == sc_id
        if mask.sum() > 0:
            result[sc_id] = compute_nrmse(pv_true[mask], pv_preds[mask])
    result[-1] = compute_nrmse(pv_true, pv_preds)  # overall
    return result


def compute_error_growth(
    pv_true: np.ndarray,
    pv_preds: np.ndarray,
    horizons: List[int] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compute NRMSE at multiple prediction horizons.
    
    Args:
        pv_true: Ground truth (N, T, 5)
        pv_preds: Predictions (N, T, 5)
        horizons: List of horizon steps (default: 60,180,300,600,900,1800,3600,5400)
    
    Returns:
        Dict: horizon -> {pv_name: nrmse, 'overall': mean_nrmse}
    """
    if horizons is None:
        horizons = [60, 180, 300, 600, 900, 1800, 3600, 5400]
    
    max_steps = pv_true.shape[1]
    horizon_results = {}
    
    for h in horizons:
        if h > max_steps:
            continue
        
        steps = min(h, max_steps)
        true_h = pv_true[:, :steps, :]
        pred_h = pv_preds[:, :steps, :]
        
        horizon_results[h] = {}
        pv_nrmse = []
        
        for k, pv_name in enumerate(PV_COLS):
            true_k = true_h[:, :, k].flatten()
            pred_k = pred_h[:, :, k].flatten()
            nrmse_val = _nrmse(true_k, pred_k)
            horizon_results[h][pv_name] = nrmse_val
            pv_nrmse.append(nrmse_val)
        
        horizon_results[h]['overall'] = np.mean(pv_nrmse)
    
    return horizon_results


# ── PLOT 1: Training Loss Curves ───────────────────────────────────────────────────

def plot_loss_curves(results_path: Path, model_name: str, out_dir: Path) -> None:
    """Plot training and validation loss curves (Figure 1)."""
    if not results_path.exists():
        logger.warning(f"Results file not found: {results_path}")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    train_losses = results.get("train_losses", [])
    val_losses = results.get("val_losses", [])
    
    if not train_losses:
        logger.warning("No training losses found")
        return
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(epochs, train_losses, color="steelblue", label="Train", linewidth=2)
    if val_losses:
        ax1.plot(epochs[:len(val_losses)], val_losses, color="darkorange", label="Val", linewidth=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss (log scale)")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    # Scheduled sampling
    ss_start, ss_end, ss_max = 10, 100, 0.5
    ss_ratios = [
        0.0 if e < ss_start else ss_max if e >= ss_end else ss_max * (e - ss_start) / (ss_end - ss_start)
        for e in epochs
    ]
    ax2.plot(epochs, ss_ratios, color="seagreen", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("SS Ratio")
    ax2.set_title("Scheduled Sampling Ratio")
    ax2.set_ylim(-0.05, ss_max * 1.2)
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    fig.suptitle(f"{model_name} — Training Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    
    path = out_dir / "loss_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path.name}")
    plt.close(fig)


# ── PLOT 2: NRMSE per PV Bar Chart ────────────────────────────────────────────────

def plot_nrmse_per_pv(nrmse: Dict[str, float], model_name: str, out_dir: Path) -> None:
    """Bar chart of NRMSE per PV (Figure 2)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    pv_names = list(nrmse.keys())
    values = list(nrmse.values())
    colors = ['steelblue', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    
    bars = ax.bar(pv_names, values, color=colors[:len(pv_names)], alpha=0.8)
    
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    
    ax.set_ylabel("NRMSE (closed-loop)", fontsize=12)
    ax.set_xlabel("Process Variable", fontsize=12)
    ax.set_title(f"{model_name} — NRMSE per PV", fontsize=14)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    
    path = out_dir / "nrmse_per_pv.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path.name}")
    plt.close(fig)


# ── PLOT 3: Per-Control Loop Performance ──────────────────────────────────────────

def plot_per_loop_performance(nrmse: Dict[str, float], model_name: str, out_dir: Path) -> None:
    """Group NRMSE by control loop (Figure 3)."""
    loop_nrmse = {"PC": [], "LC": [], "FC": [], "TC": [], "CC": []}
    
    for pv, val in nrmse.items():
        loop = PV_TO_LOOP.get(pv, "Other")
        if loop in loop_nrmse:
            loop_nrmse[loop].append(val)
    
    loop_avg = {loop: np.mean(vals) for loop, vals in loop_nrmse.items() if vals}
    loops = list(loop_avg.keys())
    values = list(loop_avg.values())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['steelblue', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    bars = ax.bar(loops, values, color=colors[:len(loops)], alpha=0.8)
    
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11)
    
    ax.set_ylabel("NRMSE", fontsize=12)
    ax.set_xlabel("Control Loop", fontsize=12)
    ax.set_title(f"{model_name} — NRMSE per Control Loop", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    
    fig.tight_layout()
    path = out_dir / "per_loop_performance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path.name}")
    plt.close(fig)


# ── PLOT 4: NRMSE per Scenario (Grouped Bar) ──────────────────────────────────────

def plot_nrmse_per_scenario(
    nrmse_by_sc: Dict[int, Dict[str, float]],
    model_name: str,
    out_dir: Path,
) -> None:
    """Grouped bar chart: NRMSE per PV per scenario (Figure 4)."""
    scenarios = [0, 1, 2, 3]
    pv_names = list(PV_SHORT[:5])
    n_pv = len(pv_names)
    
    x = np.arange(n_pv)
    width = 0.2
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, sc_id in enumerate(scenarios):
        if sc_id not in nrmse_by_sc:
            continue
        nrmse_vals = [nrmse_by_sc[sc_id].get(PV_COLS[k], 0) for k in range(n_pv)]
        offset = (i - len(scenarios)/2 + 0.5) * width
        bars = ax.bar(x + offset, nrmse_vals, width, label=SCENARIO_SHORT[sc_id],
                      color=colors[i], alpha=0.8)
        
        for bar, v in zip(bars, nrmse_vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(pv_names, fontsize=10)
    ax.set_ylabel("NRMSE", fontsize=12)
    ax.set_title(f"{model_name} — NRMSE per PV per Scenario", fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    
    fig.tight_layout()
    path = out_dir / "nrmse_per_scenario.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path.name}")
    plt.close(fig)


# ── PLOT 5: Scenario Overlay (Predictions vs Ground Truth) ────────────────────────

def plot_scenario_overlay(
    pv_true: np.ndarray,
    pv_preds: np.ndarray,
    scenario_labels: np.ndarray,
    pv_name: str = "P1_PIT01",
    model_name: str = "",
    out_dir: Optional[Path] = None
) -> None:
    """Overlay predictions vs ground truth across all scenarios (Figure 5)."""
    pv_idx = PV_COLS.index(pv_name) if pv_name in PV_COLS else 0
    pv_short = PV_SHORT[pv_idx]
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    for idx, sc_id in enumerate([0, 1, 2, 3]):
        mask = scenario_labels == sc_id
        if mask.sum() == 0:
            continue
        
        # Take first window for each scenario
        first_idx = np.where(mask)[0][0]
        true_vals = pv_true[first_idx, :, pv_idx]
        pred_vals = pv_preds[first_idx, :, pv_idx]
        time = np.arange(len(true_vals))
        
        ax = axes[idx]
        ax.plot(time, true_vals, color="steelblue", linewidth=1.5, label="Ground Truth")
        ax.plot(time, pred_vals, color="crimson", linewidth=1.2, linestyle="--", label="Prediction")
        ax.fill_between(time, true_vals, pred_vals, color="red", alpha=0.15)
        
        ax.set_ylabel("Value (normalised)", fontsize=10)
        ax.set_title(f"{SCENARIO_SHORT[sc_id]}", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        if sc_id > 0:
            ax.axvspan(0, len(true_vals)-1, alpha=0.08, color="red")
    
    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    fig.suptitle(f"{model_name} — Generated vs Real PV: {pv_short}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    if out_dir:
        path = out_dir / "scenario_overlay.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path.name}")
    plt.close(fig)


# ── PLOT 6: Error Growth Curve (NRMSE vs Horizon) ─────────────────────────────────

def plot_error_growth_curve(
    pv_true: np.ndarray,
    pv_preds: np.ndarray,
    out_dir: Path,
    model_name: str = "",
    horizons: List[int] = None
) -> None:
    """
    NRMSE vs prediction horizon (Figure 6).
    Automatically computes error growth from predictions.
    """
    if horizons is None:
        horizons = [60, 180, 300, 600, 900, 1800, 3600, 5400]
    
    # Compute error growth
    horizon_results = compute_error_growth(pv_true, pv_preds, horizons)
    
    if not horizon_results:
        logger.warning("No horizon results to plot")
        return
    
    horizon_vals = sorted(horizon_results.keys())
    overall = [horizon_results[h]['overall'] for h in horizon_vals]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(horizon_vals, overall, 'b-o', linewidth=2, markersize=8, label='Overall Average')
    
    # Add individual PVs
    for i, pv_name in enumerate(PV_COLS[:5]):
        pv_short = PV_SHORT[i]
        values = [horizon_results[h].get(pv_name, 0) for h in horizon_vals]
        ax.plot(horizon_vals, values, '--', linewidth=1.5, alpha=0.7, label=pv_short)
    
    # Threshold line
    ax.axhline(y=NRMSE_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
               label=f'{NRMSE_THRESHOLD*100:.0f}% Threshold')
    
    ax.set_xlabel('Prediction Horizon (seconds)', fontsize=12)
    ax.set_ylabel('NRMSE', fontsize=12)
    ax.set_title(f'{model_name} — Error Accumulation Over Time', fontsize=14)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add vertical line at TARGET_LEN (180s)
    ax.axvline(x=TARGET_LEN, color='gray', linestyle=':', linewidth=1.5,
               label=f'Detection Window ({TARGET_LEN}s)')
    
    fig.tight_layout()
    path = out_dir / "error_growth_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path.name}")
    plt.close(fig)


# ── PLOT 7: Error Heatmap (PV × Horizon) ─────────────────────────────────────────

def plot_error_heatmap(
    pv_true: np.ndarray,
    pv_preds: np.ndarray,
    out_dir: Path,
    model_name: str = "",
    horizons: List[int] = None
) -> None:
    """
    Heatmap showing NRMSE for each PV at different horizons (Figure 7).
    Automatically computes error growth from predictions.
    """
    if horizons is None:
        horizons = [60, 180, 300, 600, 900, 1800, 3600, 5400]
    
    # Compute error growth
    horizon_results = compute_error_growth(pv_true, pv_preds, horizons)
    
    if not horizon_results:
        logger.warning("No horizon results to plot")
        return
    
    horizon_vals = sorted(horizon_results.keys())
    pv_names = PV_SHORT[:5]
    
    error_matrix = np.zeros((len(pv_names), len(horizon_vals)))
    for i, pv_name in enumerate(PV_COLS[:5]):
        for j, h in enumerate(horizon_vals):
            error_matrix[i, j] = horizon_results[h].get(pv_name, 0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(error_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.25)
    
    # Set ticks
    ax.set_xticks(range(len(horizon_vals)))
    ax.set_yticks(range(len(pv_names)))
    
    # Format x-axis labels (convert seconds to minutes for readability)
    x_labels = []
    for h in horizon_vals:
        if h >= 3600:
            x_labels.append(f'{h//3600}h')
        elif h >= 60:
            x_labels.append(f'{h//60}min')
        else:
            x_labels.append(f'{h}s')
    
    ax.set_xticklabels(x_labels, fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels(pv_names, fontsize=10)
    
    # Add text annotations
    for i in range(len(pv_names)):
        for j in range(len(horizon_vals)):
            val = error_matrix[i, j]
            color = "white" if val > 0.12 else "black"
            ax.text(j, i, f'{val:.3f}', ha="center", va="center", color=color, fontsize=9)
    
    ax.set_xlabel('Prediction Horizon', fontsize=12)
    ax.set_ylabel('Process Variable', fontsize=12)
    ax.set_title(f'{model_name} — NRMSE Heatmap (PV vs Horizon)', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, label='NRMSE')
    cbar.ax.tick_params(labelsize=10)
    
    # Add vertical line for TARGET_LEN (180s) position
    try:
        target_idx = horizon_vals.index(TARGET_LEN)
        ax.axvline(x=target_idx - 0.5, color='cyan', linestyle='-', linewidth=1.5, alpha=0.7)
        ax.text(target_idx - 0.5, -0.5, '← Detection Window →', 
                ha='center', va='top', fontsize=8, color='cyan', rotation=0)
    except ValueError:
        pass
    
    fig.tight_layout()
    path = out_dir / "error_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path.name}")
    plt.close(fig)


# ── PLOT 8: ROC Curve ───────────────────────────────────────────────────────────

def plot_roc_curve(
    attack_labels: np.ndarray,
    anomaly_scores: np.ndarray,
    model_name: str,
    out_dir: Path,
) -> None:
    """ROC curve with AUC (Figure 8)."""
    fpr, tpr, _ = roc_curve(attack_labels, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    ax.plot(fpr, tpr, linewidth=2, color='#2ecc71', label=f'GRU (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    # Mark best threshold
    best_idx = np.argmax(tpr - fpr)
    ax.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, zorder=5,
               label=f'Best Threshold (FPR={fpr[best_idx]:.3f}, TPR={tpr[best_idx]:.3f})')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{model_name} — ROC Curve', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal')
    
    fig.tight_layout()
    path = out_dir / "roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path.name}")
    plt.close(fig)


# ── PLOT 9: Precision-Recall Curve ──────────────────────────────────────────────

def plot_pr_curve(
    attack_labels: np.ndarray,
    anomaly_scores: np.ndarray,
    best_f1: float,
    best_threshold: float,
    model_name: str,
    out_dir: Path,
) -> None:
    """Precision-Recall curve (Figure 9)."""
    precision, recall, _ = precision_recall_curve(attack_labels, anomaly_scores)
    avg_precision = average_precision_score(attack_labels, anomaly_scores)
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    ax.plot(recall, precision, linewidth=2, color='#3498db',
            label=f'GRU (AP = {avg_precision:.4f})')
    ax.scatter(recall[best_idx], precision[best_idx], color='red', s=100, zorder=5,
               label=f'Best F1 = {best_f1:.4f}')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'{model_name} — Precision-Recall Curve', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fig.tight_layout()
    path = out_dir / "pr_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path.name}")
    plt.close(fig)


# ── PLOT 10: Residual Box Plot per Scenario ──────────────────────────────────────

def plot_residual_boxplot(
    anomaly_scores: np.ndarray,
    scenario_labels: np.ndarray,
    threshold: float,
    model_name: str,
    out_dir: Path,
) -> None:
    """Box plot of residuals per scenario (Figure 10)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = []
    labels = []
    for sc_id in [0, 1, 2, 3]:
        mask = scenario_labels == sc_id
        if mask.sum() > 0:
            data.append(anomaly_scores[mask])
            labels.append(SCENARIO_SHORT[sc_id])
    
    positions = np.arange(1, len(data) + 1)
    bp = ax.boxplot(data, positions=positions, patch_artist=True, showfliers=False)
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db']
    for patch, color in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
               label=f'Alert Threshold = {threshold:.4f}')
    
    # Add scatter overlay
    for i, (sc_data, label) in enumerate(zip(data, labels)):
        sample = np.random.choice(sc_data, min(500, len(sc_data)), replace=False)
        x_jitter = np.random.normal(i + 1, 0.04, size=len(sample))
        ax.scatter(x_jitter, sample, alpha=0.3, s=5, color=colors[i])
    
    ax.set_ylabel('Anomaly Score (MSE)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title(f'{model_name} — Residual Distribution by Scenario', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    fig.tight_layout()
    path = out_dir / "residual_boxplot.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {path.name}")
    plt.close(fig)


# ── PLOT 11: Residual Timeline with Attack Regions ──────────────────────────────

def plot_residual_timeline(
    anomaly_scores: np.ndarray,
    attack_labels: np.ndarray,
    threshold: float,
    model_name: str,
    out_dir: Path,
) -> None:
    """Timeline of residuals with attack regions (Figure 11)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    x = np.arange(len(anomaly_scores))
    
    # Top: Residuals
    ax1.plot(x, anomaly_scores, linewidth=1, color='steelblue', alpha=0.7)
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.4f})')
    
    # Shade attack regions
    attack_regions = []
    in_attack = False
    start = 0
    for i, label in enumerate(attack_labels):
        if label == 1 and not in_attack:
            start = i
            in_attack = True
        elif label == 0 and in_attack:
            attack_regions.append((start, i))
            in_attack = False
    if in_attack:
        attack_regions.append((start, len(attack_labels)))
    
    for start, end in attack_regions:
        ax1.axvspan(start, end, alpha=0.3, color='red',
                    label='Attack' if start == attack_regions[0][0] else '')
    
    ax1.set_ylabel('Anomaly Score (MSE)', fontsize=11)
    ax1.set_title(f'{model_name} — Residual Timeline', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_yscale('log')
    
    # Bottom: Alerts
    alerts = anomaly_scores > threshold
    ax2.fill_between(x, 0, alerts.astype(int), color='red', alpha=0.5, label='Alert')
    ax2.plot(x, attack_labels, linewidth=1, color='green', alpha=0.7, label='Actual Attack')
    ax2.set_xlabel('Window Index', fontsize=11)
    ax2.set_ylabel('Alert / Attack', fontsize=11)
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    path = out_dir / "residual_timeline.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {path.name}")
    plt.close(fig)


# ── PLOT 12: Confusion Matrix ───────────────────────────────────────────────────

def plot_confusion_matrix_attack(
    attack_labels: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
    model_name: str,
    out_dir: Path,
) -> None:
    """Confusion matrix at threshold (Figure 12)."""
    y_pred = (anomaly_scores > threshold).astype(int)
    cm = confusion_matrix(attack_labels, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    total = cm.sum()
    # Update text with percentages
    for i, text_obj in enumerate(ax.texts):
        if i < 4:
            row = i // 2
            col = i % 2
            val = cm[row, col]
            pct = val / total * 100
            text_obj.set_text(f'{val}\n({pct:.1f}%)')
    
    ax.set_title(f'{model_name} — Confusion Matrix (Threshold = {threshold:.4f})', fontsize=12)
    
    fig.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {path.name}")
    plt.close(fig)


# ── PLOT 13: Detection Rate per Attack Type ─────────────────────────────────────

def plot_detection_rate_per_attack(
    attack_labels: np.ndarray,
    anomaly_scores: np.ndarray,
    scenario_labels: np.ndarray,
    threshold: float,
    model_name: str,
    out_dir: Path,
) -> None:
    """Bar chart of detection rate by attack type (Figure 13)."""
    y_pred = (anomaly_scores > threshold).astype(int)
    
    detection_rates = {}
    for sc_id in [1, 2, 3]:  # AP_no, AP_with, AE_no
        mask = scenario_labels == sc_id
        if mask.sum() > 0:
            tp = np.sum((y_pred[mask] == 1) & (attack_labels[mask] == 1))
            fn = np.sum((y_pred[mask] == 0) & (attack_labels[mask] == 1))
            detection_rates[SCENARIO_SHORT[sc_id]] = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    
    fig, ax = plt.subplots(figsize=(8, 5))
    attacks = list(detection_rates.keys())
    rates = list(detection_rates.values())
    colors = ['#f39c12', '#e74c3c', '#3498db']
    
    bars = ax.bar(attacks, rates, color=colors[:len(attacks)], alpha=0.8)
    
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)
    
    ax.set_ylabel('Detection Rate (%)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title(f'{model_name} — Detection Rate by Attack Type', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    fig.tight_layout()
    path = out_dir / "detection_rate_per_attack.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {path.name}")
    plt.close(fig)


# ── NEW: Combined function to generate all plots ────────────────────────────────

def generate_all_paper_plots(
    pv_true: np.ndarray,
    pv_preds: np.ndarray,
    scenario_labels: np.ndarray,
    attack_labels: np.ndarray,
    anomaly_scores: np.ndarray,
    nrmse_overall: Dict[str, float],
    nrmse_by_scenario: Dict[int, Dict[str, float]],
    best_threshold: float,
    best_f1: float,
    auroc: float,
    results_path: Path,
    out_dir: Path,
    model_name: str = "GRU-Scenario-Weighted"
) -> None:
    """
    Generate all 13 paper plots from computed data.
    
    Args:
        pv_true: Ground truth PV values (N, T, 5)
        pv_preds: Predicted PV values (N, T, 5)
        scenario_labels: Scenario labels per window (N,)
        attack_labels: Binary attack labels (N,)
        anomaly_scores: MSE residuals per window (N,)
        nrmse_overall: Dict of NRMSE per PV
        nrmse_by_scenario: Dict of NRMSE per scenario
        best_threshold: Optimal detection threshold
        best_f1: Best F1 score
        auroc: AUROC score
        results_path: Path to results.json (for loss curves)
        out_dir: Output directory for plots
        model_name: Name for plot titles
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Loss curves
    print("\n[Plot 1] Loss curves...")
    plot_loss_curves(results_path, model_name, out_dir)
    
    # Plot 2: NRMSE per PV
    print("\n[Plot 2] NRMSE per PV...")
    plot_nrmse_per_pv(nrmse_overall, model_name, out_dir)
    
    # Plot 3: Per-loop performance
    print("\n[Plot 3] Per-loop performance...")
    plot_per_loop_performance(nrmse_overall, model_name, out_dir)
    
    # Plot 4: NRMSE per scenario
    print("\n[Plot 4] NRMSE per scenario...")
    plot_nrmse_per_scenario(nrmse_by_scenario, model_name, out_dir)
    
    # Plot 5: Scenario overlay
    print("\n[Plot 5] Scenario overlay...")
    plot_scenario_overlay(pv_true, pv_preds, scenario_labels, 
                          pv_name="P1_PIT01", model_name=model_name, out_dir=out_dir)
    
    # Plot 6: Error growth curve
    print("\n[Plot 6] Error growth curve...")
    plot_error_growth_curve(pv_true, pv_preds, out_dir, model_name)
    
    # Plot 7: Error heatmap
    print("\n[Plot 7] Error heatmap...")
    plot_error_heatmap(pv_true, pv_preds, out_dir, model_name)
    
    # Plot 8: ROC curve
    print("\n[Plot 8] ROC curve...")
    plot_roc_curve(attack_labels, anomaly_scores, model_name, out_dir)
    
    # Plot 9: Precision-Recall curve
    print("\n[Plot 9] Precision-Recall curve...")
    plot_pr_curve(attack_labels, anomaly_scores, best_f1, best_threshold, model_name, out_dir)
    
    # Plot 10: Residual boxplot
    print("\n[Plot 10] Residual boxplot...")
    plot_residual_boxplot(anomaly_scores, scenario_labels, best_threshold, model_name, out_dir)
    
    # Plot 11: Residual timeline
    print("\n[Plot 11] Residual timeline...")
    plot_residual_timeline(anomaly_scores, attack_labels, best_threshold, model_name, out_dir)
    
    # Plot 12: Confusion matrix
    print("\n[Plot 12] Confusion matrix...")
    plot_confusion_matrix_attack(attack_labels, anomaly_scores, best_threshold, model_name, out_dir)
    
    # Plot 13: Detection rate per attack
    print("\n[Plot 13] Detection rate per attack...")
    plot_detection_rate_per_attack(attack_labels, anomaly_scores, scenario_labels, 
                                    best_threshold, model_name, out_dir)
    
    print(f"\n✅ All 13 plots saved to: {out_dir}/")