"""
plot_utils.py — Shared inference and plotting utilities for all model plot scripts.
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

from config import LOOPS, PV_COLS

CTRL_LOOPS = ["PC", "LC", "FC", "TC"]

SCENARIO_NAMES = {
    0: "Normal",
    1: "AP_no  (Actuator Pollution, no combination)",
    2: "AP_with (Actuator Pollution, with combination)",
    3: "AE_no  (Actuator Emulation, no combination)",
}
SCENARIO_SHORT = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}

PV_SHORT = [c.replace("P1_", "") for c in PV_COLS]

# Plotting constants
HORIZONS = [300, 600, 900, 1800]
HORIZON_LABELS = {300: "5 min", 600: "10 min", 900: "15 min", 1800: "30 min"}
NRMSE_THRESHOLD = 0.10   # pass/fail gate (same as config.yaml causal_threshold)
TARGET_LEN = 180  # one window = 180 steps = 3 min
N_WINDOWS = 8     # windows to chain per horizon

# Controller loop name for each PV
_PV_TO_LOOP = {pv: ln for ln, loop in LOOPS.items() for pv in [loop.pv]}


# ── Helper Functions ────────────────────────────────────────────────────────────────

def _validate_inputs(pv_true: np.ndarray, pv_preds: np.ndarray, name: str = "data") -> None:
    """Validate input arrays for shape and content."""
    if pv_true.shape != pv_preds.shape:
        raise ValueError(f"{name}: Shape mismatch - true {pv_true.shape} vs pred {pv_preds.shape}")
    if np.any(np.isnan(pv_true)) or np.any(np.isnan(pv_preds)):
        logger.warning(f"{name}: Contains NaN values")
    if np.any(np.isinf(pv_true)) or np.any(np.isinf(pv_preds)):
        logger.warning(f"{name}: Contains infinite values")


def _nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NRMSE normalised by signal range.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        NRMSE value (0 to inf), NaN for invalid inputs
    """
    if y_true.size == 0 or y_pred.size == 0:
        logger.warning("Empty arrays passed to _nrmse")
        return float('nan')
    
    if y_true.shape != y_pred.shape:
        logger.warning(f"Shape mismatch in _nrmse: {y_true.shape} vs {y_pred.shape}")
        return float('nan')
    
    try:
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        sig_range = float(y_true.max() - y_true.min())
        
        if sig_range <= 1e-10:
            # Constant signal case - check if predictions match
            if np.allclose(y_true, y_pred, rtol=1e-5, atol=1e-5):
                return 0.0
            else:
                return float('inf')
        
        return rmse / sig_range
    except Exception as e:
        logger.error(f"Error computing NRMSE: {e}")
        return float('nan')


def _pick_best_start_window(pv_true: np.ndarray, indices: np.ndarray) -> int:
    """
    Pick the window with the highest variance across all PVs to start chaining.
    
    Args:
        pv_true: Ground truth PV values (N, target_len, n_pv)
        indices: Array of window indices to choose from
    
    Returns:
        Selected starting window index
    """
    if len(indices) == 0:
        raise ValueError("Empty window_indices array")
    
    # Calculate variance across all PVs for each window
    variances = pv_true[indices].var(axis=(1, 2))
    return int(indices[variances.argmax()])


def _get_consecutive_chain(
    start_idx: int,
    scenario_labels: np.ndarray,
    target_scenario: int,
    max_windows: int
) -> List[int]:
    """
    Build a chain of consecutive windows all belonging to the same scenario.
    
    Args:
        start_idx: Starting index
        scenario_labels: Array of scenario labels for each window
        target_scenario: The scenario we want to chain
        max_windows: Maximum number of windows to include
    
    Returns:
        List of consecutive indices belonging to the target scenario
    """
    chain = []
    for idx in range(start_idx, len(scenario_labels)):
        if scenario_labels[idx] == target_scenario:
            chain.append(idx)
        if len(chain) >= max_windows:
            break
    return chain


# ── Inference ────────────────────────────────────────────────────────────────

def run_closed_loop(
    plant_model: Any,
    ctrl_models: Dict[str, Any],
    data: Dict,
    split: str,
    device: torch.device,
    batch_size: int = 128,
    verbose: bool = True
) -> np.ndarray:
    """
    Run closed-loop rollout on a data split.
    
    Args:
        plant_model: Trained plant model
        ctrl_models: Dictionary of controller models
        data: Data dictionary from pipeline
        split: Data split name ('val' or 'test')
        device: Torch device
        batch_size: Batch size for inference
        verbose: Print progress information
    
    Returns:
        pv_preds: (N, target_len, n_pv) array of predictions
    """
    from pipeline import load_and_prepare_data
    
    # Extract data
    plant_data = data["plant"]
    ctrl_data = data["ctrl"]
    target_len = data["metadata"]["target_len"]
    n_pv = plant_data["n_pv"]
    
    # Get split-specific data
    X = plant_data.get(f"X_{split}")
    X_cv_tgt = plant_data.get(f"X_cv_target_{split}")
    pv_init = plant_data.get(f"pv_init_{split}")
    scenario = plant_data.get(f"scenario_{split}")
    
    if X is None:
        raise ValueError(f"No data found for split '{split}'")
    
    # Map control variable indices
    non_pv_cols = [c for c in data["metadata"]["sensor_cols"] if c not in set(PV_COLS)]
    col_to_idx = {c: i for i, c in enumerate(non_pv_cols)}
    ctrl_cv_col_idx = {
        ln: col_to_idx[LOOPS[ln].cv]
        for ln in CTRL_LOOPS 
        if LOOPS[ln].cv in col_to_idx
    }
    
    N = len(X)
    pv_preds = np.zeros((N, target_len, n_pv), dtype=np.float32)
    
    # Set models to eval mode
    for m in ctrl_models.values():
        m.eval()
    plant_model.eval()
    
    if verbose:
        logger.info(f"Running closed-loop inference on {split} split ({N} samples)")
    
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc=f"Processing {split}", disable=not verbose):
            end_idx = min(i + batch_size, N)
            sl = slice(i, end_idx)
            
            # Move data to device
            x_cv_b = torch.tensor(X[sl]).float().to(device)
            xct_b = torch.tensor(X_cv_tgt[sl]).float().to(device).clone()
            pv_init_b = torch.tensor(pv_init[sl]).float().to(device)
            sc_b = torch.tensor(scenario[sl]).long().to(device)
            
            current_batch_size = x_cv_b.size(0)
            
            # Apply controller predictions
            for ln in CTRL_LOOPS:
                if ln not in ctrl_cv_col_idx:
                    continue
                
                # Check if controller data exists for this split
                if f"X_{split}" not in ctrl_data.get(ln, {}):
                    if verbose and i == 0:
                        logger.warning(f"No controller data for {ln} in split {split}, using ground-truth CVs")
                    continue
                
                # Get controller predictions
                Xc = torch.tensor(ctrl_data[ln][f"X_{split}"][sl]).float().to(device)
                
                try:
                    cv_pred = ctrl_models[ln].predict(Xc, target_len=target_len)
                except Exception as e:
                    logger.error(f"Error predicting with controller {ln}: {e}")
                    continue
                
                # Validate batch size
                if cv_pred.size(0) != current_batch_size:
                    logger.warning(f"Batch size mismatch for {ln}: expected {current_batch_size}, got {cv_pred.size(0)}")
                    continue
                
                # Ensure correct dimensions
                if cv_pred.dim() == 2:
                    cv_pred = cv_pred.unsqueeze(-1)
                
                # Update control variable
                cv_i = ctrl_cv_col_idx[ln]
                if cv_i < xct_b.shape[2]:
                    xct_b[:, :, cv_i:cv_i + 1] = cv_pred
                else:
                    logger.warning(f"CV index {cv_i} out of range for xct_b shape {xct_b.shape}")
            
            # Get plant predictions
            try:
                pv_seq = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
                B_actual = pv_seq.size(0)
                pv_preds[i:i + B_actual] = pv_seq.cpu().numpy()
            except Exception as e:
                logger.error(f"Error in plant model prediction for batch {i}: {e}")
                continue
    
    if verbose:
        logger.info(f"Completed inference for {split} split")
    
    return pv_preds


def compute_nrmse(pv_true: np.ndarray, pv_preds: np.ndarray) -> List[float]:
    """
    Compute NRMSE per PV channel.
    
    Args:
        pv_true: Ground truth PV values (N, target_len, n_pv)
        pv_preds: Predicted PV values (N, target_len, n_pv)
    
    Returns:
        List of NRMSE values per PV channel
    """
    _validate_inputs(pv_true, pv_preds, "compute_nrmse")
    
    nrmse = []
    for k in range(pv_true.shape[-1]):
        true_k = pv_true[:, :, k]
        pred_k = pv_preds[:, :, k]
        nrmse.append(_nrmse(true_k, pred_k))
    
    return nrmse


# ── Plot: NRMSE bar ───────────────────────────────────────────────────────────

def plot_nrmse_bar(nrmse: List[float], model_name: str, out_dir: Path) -> None:
    """Create bar plot of NRMSE values per PV channel."""
    if not nrmse:
        logger.warning("No NRMSE values to plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(PV_SHORT[:len(nrmse)], nrmse, color="steelblue", alpha=0.85)
    
    for bar, v in zip(bars, nrmse):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    
    ax.set_ylabel("NRMSE (closed-loop)")
    ax.set_title(f"{model_name} — Closed-Loop NRMSE per PV")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    
    valid_nrmse = [v for v in nrmse if not np.isnan(v)]
    if valid_nrmse:
        ax.set_ylim(0, max(valid_nrmse) * 1.1)
    
    fig.tight_layout()
    path = out_dir / "nrmse_per_pv.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path.name}")
    plt.close(fig)


# ── Plot: training loss curves ─────────────────────────────────────────────────

def plot_loss_curves(results_path: Path, model_name: str, out_dir: Path) -> None:
    """Plot training and validation loss curves with scheduled sampling ratio."""
    if not results_path.exists():
        logger.warning(f"Results file not found: {results_path}")
        return
    
    try:
        with open(results_path) as f:
            results = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading results file: {e}")
        return
    
    if "train_losses" not in results:
        logger.info("No training losses found in results file")
        return
    
    train_losses = results.get("train_losses", [])
    val_losses = results.get("val_losses", [])
    
    if not train_losses:
        logger.warning("Empty training losses")
        return
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    ax1.plot(epochs, train_losses, color="steelblue", label="Train", linewidth=2)
    if val_losses:
        ax1.plot(epochs[:len(val_losses)], val_losses, color="darkorange", label="Val", linewidth=2)
    
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss (log scale)")
    ax1.set_title("Plant Training Loss")
    ax1.legend()
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    
    ss_start, ss_end, ss_max = 10, 100, 0.5
    ss_ratios = [
        0.0 if e < ss_start
        else ss_max if e >= ss_end
        else ss_max * (e - ss_start) / (ss_end - ss_start)
        for e in epochs
    ]
    
    ax2.plot(epochs, ss_ratios, color="seagreen", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("SS Ratio")
    ax2.set_title("Scheduled-Sampling Ratio")
    ax2.set_ylim(-0.05, ss_max * 1.2)
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    fig.suptitle(f"{model_name} — Training Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    
    path = out_dir / "loss_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path.name}")
    plt.close(fig)


# ── Plot: scenario predictions ────────────────────────────────────────────────

def plot_scenario_predictions(
    pv_true: np.ndarray,
    pv_preds: np.ndarray,
    scenario_labels: np.ndarray,
    model_name: str,
    out_dir: Path,
) -> None:
    """For each scenario, plot the most representative window with predictions."""
    _validate_inputs(pv_true, pv_preds, "plot_scenario_predictions")
    
    if len(scenario_labels) != pv_true.shape[0]:
        raise ValueError(f"Scenario labels length {len(scenario_labels)} doesn't match data shape {pv_true.shape[0]}")
    
    target_len = pv_true.shape[1]
    t = np.arange(target_len)
    
    for sc_id, sc_short in SCENARIO_SHORT.items():
        indices = np.where(scenario_labels == sc_id)[0]
        if len(indices) == 0:
            logger.info(f"No windows for scenario {sc_id} ({sc_short}), skipping.")
            continue
        
        # Pick best window based on overall variance across all PVs
        try:
            variances = pv_true[indices].var(axis=(1, 2))
            idx = int(indices[variances.argmax()])
        except Exception as e:
            logger.error(f"Error picking sample for scenario {sc_id}: {e}")
            continue
        
        true = pv_true[idx]
        pred = pv_preds[idx]
        
        is_attack = sc_id > 0
        n_windows_label = "attack windows" if is_attack else "normal windows"
        
        n_pv = true.shape[1]
        fig, axes = plt.subplots(n_pv, 1, figsize=(14, max(3 * n_pv, 4)), sharex=True)
        if n_pv == 1:
            axes = [axes]
        
        fig.suptitle(
            f"{model_name} — Scenario {sc_id}: {SCENARIO_NAMES[sc_id]}\n"
            f"(window {idx} of {len(indices)} {n_windows_label})",
            fontsize=12, fontweight="bold", y=1.01,
        )
        
        for k, (ax, pv_name) in enumerate(zip(axes, PV_SHORT[:n_pv])):
            ax.plot(t, true[:, k], color="steelblue", linewidth=1.5,
                    label="Ground truth")
            ax.plot(t, pred[:, k], color="crimson", linewidth=1.2,
                    linestyle="--", label="Prediction (closed-loop)")
            ax.fill_between(t, true[:, k], pred[:, k],
                            color="red", alpha=0.15, label="Error")
            ax.set_title(pv_name, fontsize=10)
            ax.set_ylabel("Value (normalised)", fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.35)
            
            if is_attack:
                ax.axvspan(0, target_len - 1, alpha=0.04, color="red")
        
        axes[-1].set_xlabel("Target time step", fontsize=10)
        fig.tight_layout()
        
        path = out_dir / f"scenario_{sc_id}_{sc_short}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path.name}")
        plt.close(fig)


# ── Plot: multi-horizon with chained windows for all scenarios ────────────────

def plot_multi_horizon(
    pv_true: np.ndarray,
    pv_preds: np.ndarray,
    scenario_labels: np.ndarray,
    model_name: str,
    out_dir: Path,
    data: Optional[Dict] = None,
    plant_model: Any = None,
    ctrl_models: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    For each scenario and each PV channel, create one plot with all horizons stacked vertically.
    
    Output structure:
    - One image per scenario per PV channel
    - Each image shows all horizons (300s, 600s, 900s, 1800s) stacked vertically
    - Easy to compare how prediction degrades across horizons for a single PV
    """
    _validate_inputs(pv_true, pv_preds, "plot_multi_horizon")
    
    n_pv = pv_true.shape[2]
    
    logger.info("\n" + "=" * 60)
    logger.info(f"MULTI-HORIZON PLOTS — CHAINED WINDOWS ({model_name})")
    logger.info("=" * 60)
    
    # For each scenario
    for sc_id in [0, 1, 2, 3]:
        sc_short = SCENARIO_SHORT.get(sc_id, f"sc{sc_id}")
        sc_name = SCENARIO_NAMES.get(sc_id, f"Scenario {sc_id}")
        
        # Find windows for this scenario
        indices = np.where(scenario_labels == sc_id)[0]
        if len(indices) == 0:
            logger.info(f"No windows for scenario {sc_id} ({sc_short}), skipping.")
            continue
        
        # Pick best starting window based on variance
        try:
            best_start = _pick_best_start_window(pv_true, indices)
        except Exception as e:
            logger.error(f"Error picking start window for scenario {sc_id}: {e}")
            continue
        
        # Build consecutive chain
        chain = _get_consecutive_chain(best_start, scenario_labels, sc_id, N_WINDOWS)
        
        # If not enough consecutive windows, use whatever is available
        if len(chain) < 2:
            chain = list(indices[:N_WINDOWS])
        
        logger.info(f"Scenario {sc_id} ({sc_short}): {len(chain)} windows starting at idx {chain[0]}")
        
        # Prepare data for each horizon
        horizon_data = []
        for horizon_s in HORIZONS:
            n_steps = max(1, int(np.ceil(horizon_s / TARGET_LEN)))
            n_steps = min(n_steps, len(chain))
            actual_horizon = n_steps * TARGET_LEN
            horizon_min = actual_horizon / 60
            
            # Extract chained data
            true_parts = []
            pred_parts = []
            for idx in chain[:n_steps]:
                true_parts.append(pv_true[idx])
                pred_parts.append(pv_preds[idx])
            
            true_arr = np.concatenate(true_parts, axis=0)
            pred_arr = np.concatenate(pred_parts, axis=0)
            
            horizon_data.append({
                'horizon_s': horizon_s,
                'actual_horizon': actual_horizon,
                'horizon_min': horizon_min,
                'n_steps': n_steps,
                'true_arr': true_arr,
                'pred_arr': pred_arr,
            })
        
        # For each PV channel, create a separate plot with all horizons stacked
        for pv_idx in range(n_pv):
            pv_name = PV_SHORT[pv_idx] if pv_idx < len(PV_SHORT) else f"PV{pv_idx}"
            pv_full = PV_COLS[pv_idx] if pv_idx < len(PV_COLS) else f"P1_{pv_name}"
            controller = _PV_TO_LOOP.get(pv_full, "")
            
            # Create figure: one subplot per horizon (stacked vertically)
            n_horizons = len(horizon_data)
            fig_height = 4.0 * n_horizons  # 4 inches per horizon
            fig_width = 14
            
            fig, axes = plt.subplots(n_horizons, 1, figsize=(fig_width, fig_height), sharex=True)
            
            if n_horizons == 1:
                axes = [axes]
            
            # Title includes scenario and PV info
            fig.suptitle(
                f"{model_name} — Scenario {sc_id}: {sc_name} | PV: {pv_name} (Controller: {controller})\n"
                f"Prediction performance across all horizons",
                fontsize=14, fontweight="bold", y=0.98
            )
            
            # For each horizon (subplot)
            for h_idx, hd in enumerate(horizon_data):
                ax = axes[h_idx]
                
                actual_horizon = hd['actual_horizon']
                horizon_min = hd['horizon_min']
                n_steps = hd['n_steps']
                true_arr = hd['true_arr']
                pred_arr = hd['pred_arr']
                
                # Extract data for this PV only
                true_k = true_arr[:, pv_idx]
                pred_k = pred_arr[:, pv_idx]
                
                t = np.arange(len(true_k))
                
                # Plot
                ax.plot(t, true_k, color="steelblue", linewidth=2.0, 
                        label="Ground truth", zorder=3)
                ax.plot(t, pred_k, color="crimson", linewidth=1.8,
                        linestyle="--", label="Prediction", zorder=3)
                
                # Error fill
                ax.fill_between(t, true_k, pred_k,
                               color="red", alpha=0.15, label="Error", zorder=2)
                
                # Window boundaries
                for w in range(1, n_steps):
                    ax.axvline(x=w * TARGET_LEN, color="gray",
                              linewidth=0.8, linestyle=":", alpha=0.7)
                
                # NRMSE calculation
                nrmse_val = _nrmse(true_k, pred_k)
                status = "✓ PASS" if nrmse_val < NRMSE_THRESHOLD else "✗ FAIL"
                status_color = "green" if nrmse_val < NRMSE_THRESHOLD else "red"
                
                # Subplot title with horizon info
                ax.set_title(
                    f"Horizon: {hd['horizon_s']}s ({horizon_min:.0f}min) | "
                    f"{n_steps} windows | NRMSE = {nrmse_val:.4f} [{status}]",
                    fontsize=11, color=status_color
                )
                
                ax.set_ylabel("Value", fontsize=10)
                ax.tick_params(axis='both', labelsize=9)
                ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
                ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
                
                # Attack shading
                if sc_id > 0:
                    ax.axvspan(0, len(true_k) - 1, alpha=0.04, color="red")
            
            # X-axis label on bottom subplot only
            axes[-1].set_xlabel("Time steps (1 step = 1 second)", fontsize=11)
            
            plt.tight_layout()
            
            # Save: one file per scenario per PV
            path = out_dir / f"scenario_{sc_id}_{sc_short}_PV{pv_idx:02d}_{pv_name}_all_horizons.png"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            logger.info(f"Saved: {path.name}")
            plt.close(fig)
              
# ── Main plotting function ────────────────────────────────────────────────────

def plot_all(
    plant_model: Any,
    ctrl_models: Dict[str, Any],
    data: Dict,
    model_name: str,
    out_dir: Path,
    results_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 128,
    verbose: bool = True
) -> None:
    """
    Run all plots for a trained model.
    
    Args:
        plant_model: Trained plant model
        ctrl_models: Dictionary of controller models
        data: Data dictionary from pipeline
        model_name: Name of the model for plot titles
        out_dir: Output directory for saving plots
        results_path: Optional path to results.json for loss curves
        device: Torch device (auto-detected if not provided)
        batch_size: Batch size for inference
        verbose: Print progress information
    """
    # Set up device
    if device is None:
        device = next(plant_model.parameters()).device
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Plotting: {model_name}  →  {out_dir}/")
    logger.info(f"{'='*60}")
    
    # Loss curves (only if results.json has history)
    if results_path:
        plot_loss_curves(results_path, model_name, out_dir)
    
    # Validation split closed-loop
    logger.info("Running closed-loop on val split...")
    try:
        pv_preds_val = run_closed_loop(plant_model, ctrl_models, data,
                                       "val", device, batch_size, verbose)
        pv_true_val = data["plant"]["pv_target_val"]
        
        _validate_inputs(pv_true_val, pv_preds_val, "validation data")
        nrmse_val = compute_nrmse(pv_true_val, pv_preds_val)
        
        logger.info("Val NRMSE per PV:")
        for name, v in zip(PV_COLS, nrmse_val):
            if not np.isnan(v):
                logger.info(f"  {name:<30s}: {v:.4f}")
        
        valid_nrmse = [v for v in nrmse_val if not np.isnan(v)]
        if valid_nrmse:
            logger.info(f"Mean val NRMSE: {np.mean(valid_nrmse):.4f}")
        
        plot_nrmse_bar(nrmse_val, model_name, out_dir)
    except Exception as e:
        logger.error(f"Error processing validation split: {e}")
    
    # Test split closed-loop + scenario plots
    logger.info("Running closed-loop on test split...")
    try:
        pv_preds_test = run_closed_loop(plant_model, ctrl_models, data,
                                        "test", device, batch_size, verbose)
        pv_true_test = data["plant"]["pv_target_test"]
        scenario_test = data["plant"]["scenario_test"]
        
        _validate_inputs(pv_true_test, pv_preds_test, "test data")
        
        logger.info("Generating scenario prediction plots...")
        plot_scenario_predictions(
            pv_true_test, pv_preds_test, scenario_test,
            model_name, out_dir,
        )
        
        logger.info("Generating multi-horizon plots (all scenarios with chained windows)...")
        plot_multi_horizon(
            pv_true_test, pv_preds_test, scenario_test,
            model_name, out_dir,
            data=data,
            plant_model=plant_model,
            ctrl_models=ctrl_models,
            device=device,
        )
    except Exception as e:
        logger.error(f"Error processing test split: {e}")
    
    logger.info(f"Done. All plots saved to {out_dir}/")