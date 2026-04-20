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
sys.path.insert(0, str(ROOT / "04_evaluate"))

from config import LOOPS, PV_COLS

CTRL_LOOPS = ["PC", "LC", "FC", "TC", "CC"]

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


def run_autoregressive_chain(
    plant_model: Any,
    ctrl_models: Dict[str, Any],
    data: Dict,
    split: str,
    start_idx: int,
    n_windows: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Open-loop autoregressive rollout across n_windows consecutive windows.

    The plant encoder runs **once** on window ``start_idx``.  The decoder then
    rolls out for n_windows * target_len steps without ever resetting to
    ground-truth PVs — the last predicted PV and the GRU hidden state are
    carried across every window boundary.

    Controller CVs are predicted by ``ctrl_models`` for each window.

    Returns
    -------
    true_chain : (n_windows * target_len, n_pv)   ground truth (concatenated)
    pred_chain : (n_windows * target_len, n_pv)   open-loop predictions
    """
    plant_data    = data["plant"]
    ctrl_data_map = data["ctrl"]
    target_len    = data["metadata"]["target_len"]
    n_pv          = plant_data["n_pv"]

    X           = plant_data[f"X_{split}"]
    X_cv_tgt    = plant_data[f"X_cv_target_{split}"]
    pv_init_arr = plant_data[f"pv_init_{split}"]
    scenario_arr = plant_data[f"scenario_{split}"]
    pv_true_arr = plant_data[f"pv_target_{split}"]

    non_pv_cols = [c for c in data["metadata"]["sensor_cols"] if c not in set(PV_COLS)]
    col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}
    ctrl_cv_col_idx = {
        ln: col_to_idx[LOOPS[ln].cv]
        for ln in CTRL_LOOPS
        if LOOPS[ln].cv in col_to_idx
    }

    plant_model.eval()
    for m in ctrl_models.values():
        m.eval()

    true_parts: List[np.ndarray] = []
    pred_parts: List[np.ndarray] = []

    with torch.no_grad():
        # ── Encode once from the first window ──────────────────────────────
        x_cv_0  = torch.tensor(X[start_idx:start_idx + 1]).float().to(device)
        sc_0    = torch.tensor(scenario_arr[start_idx:start_idx + 1]).long().to(device)
        pv_prev = torch.tensor(pv_init_arr[start_idx:start_idx + 1]).float().to(device)

        emb = plant_model.scenario_emb(sc_0).unsqueeze(1).expand(-1, x_cv_0.size(1), -1)
        _, h = plant_model.encoder(torch.cat([x_cv_0, emb], dim=-1))

        # ── Decode autoregressively across n_windows (no re-encoding) ──────
        for w in range(n_windows):
            idx = start_idx + w
            if idx >= len(X):
                break

            true_parts.append(pv_true_arr[idx])   # (target_len, n_pv)

            # Controller CV predictions for this window
            xct_w = torch.tensor(X_cv_tgt[idx:idx + 1]).float().to(device).clone()
            sc_w  = torch.tensor(scenario_arr[idx:idx + 1]).long().to(device)

            for ln in CTRL_LOOPS:
                if ln not in ctrl_cv_col_idx:
                    continue
                if f"X_{split}" not in ctrl_data_map.get(ln, {}):
                    continue
                Xc = torch.tensor(
                    ctrl_data_map[ln][f"X_{split}"][idx:idx + 1]
                ).float().to(device)
                cv_pred = ctrl_models[ln].predict(Xc, target_len=target_len)
                if cv_pred.dim() == 2:
                    cv_pred = cv_pred.unsqueeze(-1)
                cv_i = ctrl_cv_col_idx[ln]
                if cv_i < xct_w.shape[2]:
                    xct_w[:, :, cv_i:cv_i + 1] = cv_pred

            # Step-by-step decoder (hidden state h and pv_prev carried over)
            win_preds: List[torch.Tensor] = []
            for t in range(target_len):
                dec_in = torch.cat([xct_w[:, t, :], pv_prev], dim=-1).unsqueeze(1)
                out, h = plant_model.decoder(dec_in, h)
                h_out  = out.squeeze(1)

                if getattr(plant_model, "scenario_heads", False):
                    pv_pred = torch.zeros(1, n_pv, device=device)
                    for sc_id in range(plant_model.n_scenarios):
                        mask = (sc_w == sc_id)
                        if mask.any():
                            pv_pred[mask] = plant_model.fc_heads[sc_id](h_out[mask])
                else:
                    pv_pred = plant_model.fc(h_out)

                win_preds.append(pv_pred)
                pv_prev = pv_pred                  # carry predicted PV

            pred_parts.append(
                torch.stack(win_preds, dim=1).squeeze(0).cpu().numpy()  # (target_len, n_pv)
            )

    n_actual   = min(len(true_parts), len(pred_parts))
    true_chain = np.concatenate(true_parts[:n_actual], axis=0)
    pred_chain = np.concatenate(pred_parts[:n_actual], axis=0)
    return true_chain, pred_chain


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


def compute_nrmse_per_scenario(
    pv_true: np.ndarray,
    pv_preds: np.ndarray,
    scenario_labels: np.ndarray,
) -> Dict[int, List[float]]:
    """
    Compute NRMSE per PV channel, broken down by scenario.

    Args:
        pv_true:         Ground truth PV values  (N, target_len, n_pv)
        pv_preds:        Predicted PV values      (N, target_len, n_pv)
        scenario_labels: Scenario label per window (N,)

    Returns:
        Dict mapping scenario_id → [nrmse_per_pv].
        Key -1 holds the overall NRMSE (all windows combined).
    """
    result: Dict[int, List[float]] = {}
    for sc_id in sorted(set(scenario_labels.tolist())):
        mask = scenario_labels == sc_id
        if mask.sum() == 0:
            continue
        result[int(sc_id)] = compute_nrmse(pv_true[mask], pv_preds[mask])
    result[-1] = compute_nrmse(pv_true, pv_preds)   # overall
    return result


def _log_nrmse_table(nrmse_by_scenario: Dict[int, List[float]], split: str) -> None:
    """Print a formatted per-scenario NRMSE table to the log."""
    scenario_ids = [k for k in sorted(nrmse_by_scenario.keys()) if k >= 0]
    n_pv = len(next(iter(nrmse_by_scenario.values())))
    col_w = 10

    header = f"{'Scenario':<18}" + "".join(f"{PV_SHORT[k]:<{col_w}}" for k in range(n_pv)) + "  Mean"
    sep    = "-" * (18 + col_w * n_pv + 8)
    logger.info(f"\n{split.upper()} NRMSE per scenario:\n{header}\n{sep}")

    for sc_id in scenario_ids:
        nrmse = nrmse_by_scenario[sc_id]
        valid = [v for v in nrmse if not np.isnan(v)]
        mean_v = np.mean(valid) if valid else float("nan")
        row = f"{SCENARIO_SHORT.get(sc_id, f'Sc{sc_id}'):<18}"
        row += "".join(f"{v:<{col_w}.4f}" for v in nrmse)
        row += f"  {mean_v:.4f}"
        logger.info(row)

    overall = nrmse_by_scenario.get(-1)
    if overall:
        valid = [v for v in overall if not np.isnan(v)]
        mean_v = np.mean(valid) if valid else float("nan")
        logger.info(sep)
        row = f"{'Overall':<18}"
        row += "".join(f"{v:<{col_w}.4f}" for v in overall)
        row += f"  {mean_v:.4f}"
        logger.info(row)


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


# ── Plot: NRMSE per scenario grouped bar ──────────────────────────────────────

def plot_nrmse_per_scenario(
    nrmse_by_scenario: Dict[int, List[float]],
    model_name: str,
    out_dir: Path,
    split: str = "test",
) -> None:
    """
    Grouped bar chart: one group per PV channel, one bar per scenario.
    Saved as nrmse_per_scenario_{split}.png
    """
    scenario_ids = [k for k in sorted(nrmse_by_scenario.keys()) if k >= 0]
    if not scenario_ids:
        logger.warning("No scenario data to plot.")
        return

    n_pv   = len(nrmse_by_scenario[scenario_ids[0]])
    x      = np.arange(n_pv)
    n_sc   = len(scenario_ids)
    width  = 0.8 / max(n_sc, 1)
    colors = ["steelblue", "tomato", "darkorange", "seagreen"]

    fig, ax = plt.subplots(figsize=(max(10, n_pv * 2.5), 5))

    for i, sc_id in enumerate(scenario_ids):
        nrmse  = nrmse_by_scenario[sc_id]
        offset = (i - n_sc / 2 + 0.5) * width
        bars   = ax.bar(
            x + offset, nrmse, width=width * 0.9,
            label=SCENARIO_SHORT.get(sc_id, f"Sc{sc_id}"),
            color=colors[i % len(colors)], alpha=0.85,
        )
        for bar, v in zip(bars, nrmse):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(PV_SHORT[:n_pv], fontsize=10)
    ax.set_ylabel("NRMSE (closed-loop)", fontsize=11)
    ax.set_title(
        f"{model_name} — NRMSE per PV per Scenario ({split})",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    all_vals = [v for sc_id in scenario_ids for v in nrmse_by_scenario[sc_id] if not np.isnan(v)]
    if all_vals:
        ax.set_ylim(0, min(max(all_vals) * 1.2, 0.5))

    fig.tight_layout()
    path = out_dir / f"nrmse_per_scenario_{split}.png"
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


# ── Plot: normal operation only (no attack) ───────────────────────────────────

def plot_normal_only(
    pv_true: np.ndarray,
    pv_preds: np.ndarray,
    scenario_labels: np.ndarray,
    model_name: str,
    out_dir: Path,
    plant_model: Any = None,
    ctrl_models: Optional[Dict[str, Any]] = None,
    data: Optional[Dict] = None,
    device: Optional[torch.device] = None,
    split: str = "val",
    n_windows: int = 3,
) -> None:
    """
    Best / Median / Worst prediction windows for normal operation (scenario=0) only.
    One row per PV, three columns (Best / Median / Worst) based on per-window NRMSE.

    When plant_model / ctrl_models / data are provided, each column shows an
    n_windows-long open-loop autoregressive chain (model never resets to ground
    truth at window boundaries).  Otherwise falls back to concatenating
    independent predictions.

    Saved as normal_no_attack.png
    """
    indices = np.where(scenario_labels == 0)[0]
    if len(indices) == 0:
        logger.warning("No normal (scenario=0) windows found — skipping normal plot.")
        return

    n_pv       = pv_true.shape[2]
    target_len = pv_true.shape[1]

    # Per-window NRMSE averaged across all PVs (use 1-window predictions for ranking)
    per_win = np.zeros(len(indices))
    for k in range(n_pv):
        true_k = pv_true[indices, :, k]
        pred_k = pv_preds[indices, :, k]
        rmse_k = np.sqrt(np.mean((pred_k - true_k) ** 2, axis=1))
        rng_k  = np.maximum(true_k.max(axis=1) - true_k.min(axis=1), 1e-6)
        per_win += rmse_k / rng_k
    per_win /= n_pv

    best_i   = int(np.argmin(per_win))
    median_i = int(np.argmin(np.abs(per_win - np.median(per_win))))
    worst_i  = int(np.argmax(per_win))
    samples  = [("Best",   int(indices[best_i])),
                ("Median", int(indices[median_i])),
                ("Worst",  int(indices[worst_i]))]

    use_chain = (plant_model is not None and ctrl_models is not None and data is not None)
    chain_len = n_windows * target_len

    fig, axes = plt.subplots(n_pv, 3, figsize=(7 * n_windows, 3.2 * n_pv), sharex=True)
    if n_pv == 1:
        axes = [axes]

    for col, (label, start_idx) in enumerate(samples):
        # Build the chain for this representative window
        if use_chain:
            try:
                true_chain, pred_chain = run_autoregressive_chain(
                    plant_model, ctrl_models, data, split, start_idx, n_windows, device
                )
            except Exception as e:
                logger.warning(f"Autoregressive chain failed for {label} (idx={start_idx}): {e}. "
                               "Falling back to concatenated windows.")
                use_chain_here = False
            else:
                use_chain_here = True
        else:
            use_chain_here = False

        if not use_chain_here:
            end_idx    = min(start_idx + n_windows, pv_true.shape[0])
            true_chain = np.concatenate([pv_true[i] for i in range(start_idx, end_idx)], axis=0)
            pred_chain = np.concatenate([pv_preds[i] for i in range(start_idx, end_idx)], axis=0)

        actual_steps = len(true_chain)
        t = np.arange(actual_steps)

        for row, pv_name in enumerate(PV_SHORT[:n_pv]):
            ax        = axes[row][col]
            true_pv   = true_chain[:, row]
            pred_pv   = pred_chain[:, row]
            nrmse_win = _nrmse(true_pv, pred_pv)

            ax.plot(t, true_pv, color="steelblue", linewidth=1.5, label="True")
            ax.plot(t, pred_pv, color="tomato", linewidth=1.5,
                    linestyle="--", label="Predicted")
            ax.fill_between(t, true_pv, pred_pv, color="red", alpha=0.12, label="Error")

            # Window boundary markers
            actual_windows = actual_steps // target_len
            for w in range(1, actual_windows):
                ax.axvline(x=w * target_len, color="gray",
                           linewidth=0.8, linestyle=":", alpha=0.7)

            ax.set_title(f"{pv_name} — {label}  (NRMSE={nrmse_win:.4f})", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.35)
            if col == 0:
                ax.set_ylabel("Value (normalised)", fontsize=9)
            if row == 0:
                ax.legend(loc="upper right", fontsize=8)

    for col in range(3):
        axes[-1][col].set_xlabel(
            f"Time step  (1 step = 1 s,  {n_windows} windows × {target_len} steps"
            f" = {n_windows * target_len} s)",
            fontsize=9,
        )

    mean_nrmse = float(np.mean(per_win))
    chain_label = "open-loop autoregressive" if use_chain else "concatenated"
    fig.suptitle(
        f"{model_name} — Normal Operation (No Attack)\n"
        f"Best / Median / Worst  |  {n_windows}-window {chain_label} chain  |  "
        f"Mean NRMSE={mean_nrmse:.4f}  ({len(indices)} normal windows)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    path = out_dir / "normal_no_attack.png"
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
    plant_model: Any = None,
    ctrl_models: Optional[Dict[str, Any]] = None,
    data: Optional[Dict] = None,
    device: Optional[torch.device] = None,
    split: str = "test",
    n_windows: int = 3,
) -> None:
    """
    For each scenario, plot the most representative window with predictions.

    When plant_model / ctrl_models / data are provided, the plot shows an
    n_windows-long open-loop autoregressive chain starting from the chosen
    window — the model encodes once and decodes without ever resetting to
    ground-truth PVs across window boundaries.
    """
    _validate_inputs(pv_true, pv_preds, "plot_scenario_predictions")

    if len(scenario_labels) != pv_true.shape[0]:
        raise ValueError(f"Scenario labels length {len(scenario_labels)} doesn't match "
                         f"data shape {pv_true.shape[0]}")

    target_len = pv_true.shape[1]
    use_chain  = (plant_model is not None and ctrl_models is not None and data is not None)

    for sc_id, sc_short in SCENARIO_SHORT.items():
        indices = np.where(scenario_labels == sc_id)[0]
        if len(indices) == 0:
            logger.info(f"No windows for scenario {sc_id} ({sc_short}), skipping.")
            continue

        # Pick window by prediction error:
        #   - Normal (sc_id=0): median NRMSE — most representative
        #   - Attack scenarios: highest NRMSE — shows attack impact
        try:
            n_pv = pv_true.shape[2]
            per_win_err = np.zeros(len(indices))
            for k in range(n_pv):
                true_k = pv_true[indices, :, k]
                pred_k = pv_preds[indices, :, k]
                rmse_k = np.sqrt(np.mean((pred_k - true_k) ** 2, axis=1))
                rng_k  = np.maximum(true_k.max(axis=1) - true_k.min(axis=1), 1e-6)
                per_win_err += rmse_k / rng_k
            per_win_err /= n_pv

            if sc_id == 0:
                start_idx = int(indices[np.argmin(np.abs(per_win_err - np.median(per_win_err)))])
            else:
                start_idx = int(indices[per_win_err.argmax()])
        except Exception as e:
            logger.error(f"Error picking sample for scenario {sc_id}: {e}")
            continue

        # Build the prediction chain (open-loop autoregressive or concatenated fallback)
        if use_chain:
            try:
                true_chain, pred_chain = run_autoregressive_chain(
                    plant_model, ctrl_models, data, split, start_idx, n_windows, device
                )
                chain_label = "open-loop autoregressive"
            except Exception as e:
                logger.warning(f"Autoregressive chain failed for scenario {sc_id}: {e}. "
                               "Falling back to concatenated windows.")
                use_chain_here = False
            else:
                use_chain_here = True
        else:
            use_chain_here = False

        if not use_chain_here:
            end_idx    = min(start_idx + n_windows, pv_true.shape[0])
            true_chain = np.concatenate([pv_true[i] for i in range(start_idx, end_idx)], axis=0)
            pred_chain = np.concatenate([pv_preds[i] for i in range(start_idx, end_idx)], axis=0)
            chain_label = "concatenated"

        actual_steps   = len(true_chain)
        actual_windows = actual_steps // target_len
        t              = np.arange(actual_steps)
        is_attack      = sc_id > 0
        n_pv           = true_chain.shape[1]

        fig_width  = max(14, 5 * actual_windows)
        fig, axes  = plt.subplots(n_pv, 1, figsize=(fig_width, max(3 * n_pv, 4)),
                                  sharex=True)
        if n_pv == 1:
            axes = [axes]

        fig.suptitle(
            f"{model_name} — Scenario {sc_id}: {SCENARIO_NAMES[sc_id]}\n"
            f"{actual_windows}-window {chain_label} chain  "
            f"(starting window {start_idx},  {len(indices)} total scenario windows)",
            fontsize=12, fontweight="bold", y=1.01,
        )

        for k, (ax, pv_name) in enumerate(zip(axes, PV_SHORT[:n_pv])):
            ax.plot(t, true_chain[:, k], color="steelblue", linewidth=1.5,
                    label="Ground truth")
            ax.plot(t, pred_chain[:, k], color="crimson", linewidth=1.2,
                    linestyle="--", label="Prediction (open-loop)")
            ax.fill_between(t, true_chain[:, k], pred_chain[:, k],
                            color="red", alpha=0.15, label="Error")

            # Window boundary markers
            for w in range(1, actual_windows):
                ax.axvline(x=w * target_len, color="gray",
                           linewidth=0.9, linestyle=":", alpha=0.7,
                           label="Window boundary" if (w == 1 and k == 0) else None)

            ax.set_title(pv_name, fontsize=10)
            ax.set_ylabel("Value (normalised)", fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.35)

            if is_attack:
                ax.axvspan(0, actual_steps - 1, alpha=0.04, color="red")

        axes[-1].set_xlabel(
            f"Time step  (1 step = 1 s,  {actual_windows} windows × {target_len} steps"
            f" = {actual_steps} s)",
            fontsize=10,
        )
        fig.tight_layout()

        path = out_dir / f"scenario_{sc_id}_{sc_short}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path.name}")
        plt.close(fig)


# ── Plot: multi-horizon with autoregressive chain for all scenarios ───────────

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
    split: str = "test",
) -> None:
    """
    For each scenario × PV, one image with all 4 horizons stacked vertically.

    **Key design**: the model encodes once from ``best_start`` and decodes for the
    full 1.5 h without ever resetting to ground truth.  Every shorter horizon
    subplot is a **truncation of the same rollout** — so the 10-min row is
    literally steps 0-600 of the 1.5-h row.  Error accumulation is directly
    visible: if the model drifts, you see it grow down the rows from the exact
    same starting condition.

    Falls back to concatenating independent predictions when models are absent.
    """
    _validate_inputs(pv_true, pv_preds, "plot_multi_horizon")

    n_pv = pv_true.shape[2]

    custom_horizons   = [600, 1800, 3600, 5400]           # 10 min, 30 min, 1h, 1.5h
    HORIZON_LABEL_MAP = {600: "10 min", 1800: "30 min",
                         3600: "1 hour", 5400: "1.5 hours"}
    max_horizon_s     = max(custom_horizons)
    max_windows       = int(np.ceil(max_horizon_s / TARGET_LEN))   # 30

    use_chain = (plant_model is not None and ctrl_models is not None and data is not None)

    logger.info("\n" + "=" * 60)
    logger.info(
        f"MULTI-HORIZON PLOTS — "
        f"{'OPEN-LOOP AUTOREGRESSIVE' if use_chain else 'CONCATENATED'} CHAIN ({model_name})"
    )
    logger.info("=" * 60)

    for sc_id in [0, 1, 2, 3]:
        sc_short = SCENARIO_SHORT.get(sc_id, f"sc{sc_id}")
        sc_name  = SCENARIO_NAMES.get(sc_id, f"Scenario {sc_id}")

        indices = np.where(scenario_labels == sc_id)[0]
        if len(indices) == 0:
            logger.info(f"No windows for scenario {sc_id} ({sc_short}), skipping.")
            continue

        try:
            best_start = _pick_best_start_window(pv_true, indices)
        except Exception as e:
            logger.error(f"Error picking start window for scenario {sc_id}: {e}")
            continue

        # ── One autoregressive rollout for the full 1.5 h ───────────────────
        use_chain_here = False
        if use_chain:
            try:
                full_true, full_pred = run_autoregressive_chain(
                    plant_model, ctrl_models, data, split,
                    best_start, max_windows, device,
                )
                chain_label    = "open-loop autoregressive"
                use_chain_here = True
                logger.info(
                    f"Scenario {sc_id} ({sc_short}): autoregressive chain "
                    f"{len(full_true)} steps from window {best_start}"
                )
            except Exception as e:
                logger.warning(
                    f"Autoregressive chain failed for scenario {sc_id}: {e}. Falling back."
                )

        if not use_chain_here:
            chain = _get_consecutive_chain(best_start, scenario_labels, sc_id, max_windows)
            if len(chain) < max_windows:
                logger.warning(
                    f"Scenario {sc_id}: only {len(chain)} consecutive windows "
                    f"(needed {max_windows}). Plots truncated."
                )
            full_true  = np.concatenate([pv_true[i]  for i in chain], axis=0)
            full_pred  = np.concatenate([pv_preds[i] for i in chain], axis=0)
            chain_label = "concatenated (independent windows)"

        # ── Slice the SAME chain at each horizon ─────────────────────────────
        # All horizons share the identical starting point and model state.
        # Shorter horizons are strict prefixes of longer ones → error growth
        # is directly comparable across subplot rows.
        horizon_data = []
        for horizon_s in custom_horizons:
            end_step = min(horizon_s, len(full_true))
            n_steps  = max(1, int(np.ceil(end_step / TARGET_LEN)))
            end_step = min(n_steps * TARGET_LEN, len(full_true))

            horizon_data.append({
                'horizon_s': horizon_s,
                'label':     HORIZON_LABEL_MAP.get(horizon_s, f"{horizon_s} s"),
                'n_steps':   n_steps,
                'true_arr':  full_true[:end_step],
                'pred_arr':  full_pred[:end_step],
            })

        # ── One image per PV ─────────────────────────────────────────────────
        for pv_idx in range(n_pv):
            pv_name    = PV_SHORT[pv_idx] if pv_idx < len(PV_SHORT) else f"PV{pv_idx}"
            pv_full    = PV_COLS[pv_idx]  if pv_idx < len(PV_COLS)  else f"P1_{pv_name}"
            controller = _PV_TO_LOOP.get(pv_full, "")

            n_horizons = len(horizon_data)
            fig, axes  = plt.subplots(n_horizons, 1,
                                      figsize=(16, 4.5 * n_horizons),
                                      sharex=False)
            if n_horizons == 1:
                axes = [axes]

            fig.suptitle(
                f"{model_name} — Scenario {sc_id}: {sc_name} | "
                f"PV: {pv_name}  (Controller: {controller})\n"
                f"{chain_label} from window {best_start}  —  "
                f"all horizons are truncations of the same rollout  "
                f"(each row extends the one above it)",
                fontsize=12, fontweight="bold", y=1.00,
            )

            for h_idx, hd in enumerate(horizon_data):
                ax      = axes[h_idx]
                true_k  = hd['true_arr'][:, pv_idx]
                pred_k  = hd['pred_arr'][:, pv_idx]
                n_steps = hd['n_steps']
                t       = np.arange(len(true_k))

                ax.plot(t, true_k, color="steelblue", linewidth=2.0,
                        label="Ground truth", zorder=3)
                ax.plot(t, pred_k, color="crimson",   linewidth=1.8,
                        linestyle="--", label="Prediction (open-loop)", zorder=3)
                ax.fill_between(t, true_k, pred_k,
                                color="red", alpha=0.15, label="Error", zorder=2)

                # Window boundary markers
                for w in range(1, n_steps):
                    ax.axvline(x=w * TARGET_LEN, color="gray",
                               linewidth=0.8, linestyle=":", alpha=0.6,
                               label="Window boundary" if w == 1 else None)

                nrmse_val    = _nrmse(true_k, pred_k)
                status       = "PASS" if nrmse_val < NRMSE_THRESHOLD else "FAIL"
                status_color = "green" if nrmse_val < NRMSE_THRESHOLD else "red"

                ax.set_title(
                    f"Horizon: {hd['label']}  ({len(true_k)} s,  {n_steps} windows)"
                    f"  |  NRMSE = {nrmse_val:.4f}  [{status}]",
                    fontsize=11, color=status_color,
                )
                ax.set_ylabel("Value (norm.)", fontsize=10)
                ax.set_xlabel("Time step  (1 step = 1 s)", fontsize=9)
                ax.tick_params(axis="both", labelsize=9)
                ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
                ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

                if sc_id > 0:
                    ax.axvspan(0, len(true_k) - 1, alpha=0.04, color="red")

            plt.tight_layout()

            path = out_dir / (
                f"scenario_{sc_id}_{sc_short}_PV{pv_idx:02d}_{pv_name}_all_horizons.png"
            )
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

        # Per-scenario NRMSE (val)
        scenario_val = data["plant"]["scenario_val"]
        nrmse_by_sc_val = compute_nrmse_per_scenario(pv_true_val, pv_preds_val, scenario_val)
        _log_nrmse_table(nrmse_by_sc_val, "val")
        plot_nrmse_per_scenario(nrmse_by_sc_val, model_name, out_dir, split="val")

        # Normal-only plot (val split, scenario=0 windows) — 3-window open-loop chain
        plot_normal_only(
            pv_true_val, pv_preds_val, scenario_val, model_name, out_dir,
            plant_model=plant_model, ctrl_models=ctrl_models,
            data=data, device=device, split="val",
        )

    except Exception as e:
        logger.error(f"Error processing validation split: {e}")
    
    # Test split closed-loop + scenario plots
    logger.info("Running closed-loop on test split...")
    try:
        pv_preds_test = run_closed_loop(plant_model, ctrl_models, data,
                                        "test", device, batch_size, verbose)
        pv_true_test  = data["plant"]["pv_target_test"]
        scenario_test = data["plant"]["scenario_test"]

        _validate_inputs(pv_true_test, pv_preds_test, "test data")

        nrmse_test = compute_nrmse(pv_true_test, pv_preds_test)
        logger.info("Test NRMSE per PV:")
        for name, v in zip(PV_COLS, nrmse_test):
            if not np.isnan(v):
                logger.info(f"  {name:<30s}: {v:.4f}")
        valid_test = [v for v in nrmse_test if not np.isnan(v)]
        if valid_test:
            logger.info(f"Mean test NRMSE: {np.mean(valid_test):.4f}")

        # Per-scenario NRMSE (test)
        nrmse_by_sc_test = compute_nrmse_per_scenario(pv_true_test, pv_preds_test, scenario_test)
        _log_nrmse_table(nrmse_by_sc_test, "test")
        plot_nrmse_per_scenario(nrmse_by_sc_test, model_name, out_dir, split="test")

        logger.info("Generating scenario prediction plots...")
        plot_scenario_predictions(
            pv_true_test, pv_preds_test, scenario_test,
            model_name, out_dir,
            plant_model=plant_model, ctrl_models=ctrl_models,
            data=data, device=device, split="test",
        )
        
        logger.info("Generating multi-horizon plots (all scenarios with autoregressive chain)...")
        plot_multi_horizon(
            pv_true_test, pv_preds_test, scenario_test,
            model_name, out_dir,
            data=data,
            plant_model=plant_model,
            ctrl_models=ctrl_models,
            device=device,
            split="test",
        )
    except Exception as e:
        logger.error(f"Error processing test split: {e}")
    
    logger.info(f"Done. All plots saved to {out_dir}/")