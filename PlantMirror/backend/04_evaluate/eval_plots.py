"""
eval_plots.py — Shared diagnostic plots for HAI Digital-Twin evaluation.

Generated plots
───────────────
  0. autoregressive_composite.png  — 4 sensors × 4 columns (Train4 + one segment per attack type)
                                     Left half = ground truth only | Right half = ground truth + prediction
  a. reconstruction_error_boxplot.png — Per-sensor squared-error boxplot, grouped by scenario
  b. error_over_time.png              — Per-timestep MSE for one attack episode
  c. rmse_per_scenario.png            — Bar chart of RMSE per scenario with std error bars
  d. residual_acf.png                 — ACF of P1_PIT01 residuals on normal test windows
  e. scatter_true_vs_pred.png         — True vs predicted scatter for P1_PIT01 (normal)
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Sensors & display constants ───────────────────────────────────────────────
# 4 sensors chosen to represent different physical variables
KEY_SENSORS = ["P1_PIT01", "P1_LIT01", "P1_FT03Z", "P1_TIT01", "P1_TIT03"]

SCENARIO_NAMES = {
    0: "Normal",
    1: "AP no-comb",
    2: "AP with-comb",
    3: "AE no-comb",
}
SCENARIO_COLORS = {
    0: "#2ca02c",   # green
    1: "#ff7f0e",   # orange
    2: "#d62728",   # red
    3: "#9467bd",   # purple
}

CONTEXT_LEN  = 300
PRED_LEN     = 180
STRIDE       = 60
XTICK_STRIDE = 300   # x-axis tick spacing = 5 minutes (1 step = 1 s)


# ── Index helpers ─────────────────────────────────────────────────────────────
def _sidx(sensor_cols: list[str], name: str) -> int | None:
    """Return index of first column whose name contains `name`, or None."""
    for i, c in enumerate(sensor_cols):
        if name in c:
            return i
    return None


def _key_idx(sensor_cols: list[str]) -> list[int]:
    """Return indices for KEY_SENSORS (skipping any not found)."""
    return [i for i in [_sidx(sensor_cols, n) for n in KEY_SENSORS] if i is not None]


def _set_xticks(ax, T: int, stride: int = XTICK_STRIDE) -> None:
    """Set evenly-spaced x-axis ticks (every `stride` seconds) labelled in minutes."""
    ticks = np.arange(0, T + 1, stride)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 60}" for t in ticks], fontsize=8)


def _load_raw_attack_block(
    sensor_cols: list[str],
    scaler,
    scenario_id: int,
) -> np.ndarray | None:
    """
    Scan test1.csv then test2.csv for the LONGEST attack segment matching
    `scenario_id`.  Returns a (T, F) normalised block, or None if not found.
    """
    for fname in ("test1.csv", "test2.csv"):
        path = ROOT / "data/processed" / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        if "label" not in df.columns:
            continue

        # Find contiguous attack blocks
        mask = (df["label"] > 0).values
        i = 0
        best_blk: np.ndarray | None = None
        best_len = 0
        while i < len(mask):
            if mask[i]:
                j = i
                while j < len(mask) and mask[j]:
                    j += 1
                # Check scenario for this block
                attack_rows = df.iloc[i:j]
                a_type = attack_rows["attack_type"].iloc[0] if "attack_type" in attack_rows.columns else None
                combo  = attack_rows.get("combination", pd.Series(["no"])).iloc[0] if "combination" in attack_rows.columns else "no"
                # Inline scenario label logic (mirrors scaled_split.get_scenario_label)
                if pd.isna(a_type) or a_type == "normal":
                    sc = 0
                elif a_type == "AP":
                    sc = 1 if str(combo).strip().lower() == "no" else 2
                elif a_type == "AE":
                    sc = 3 if str(combo).strip().lower() == "no" else 4
                else:
                    sc = 0
                if sc == scenario_id:
                    # Extract with before/after context
                    lo  = max(0, i - 300)
                    hi  = min(len(df), j + 180)
                    seg = df.iloc[lo:hi][sensor_cols].values.astype(np.float32)
                    if len(seg) > best_len:
                        best_len = len(seg)
                        best_blk = scaler.transform(seg).astype(np.float32)
                i = j
            else:
                i += 1

        if best_blk is not None:
            return best_blk

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Core: rolling autoregressive prediction on a (T, F) normalised block
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _autoregress(
    model,
    block_norm: np.ndarray,       # (T, F) already normalised
    device: torch.device,
    scenario_id: int = 0,
    context_start: int | None = None,  # timestep where prediction begins (default: CONTEXT_LEN)
) -> np.ndarray:
    """
    Run the model in fully-autoregressive mode over `block_norm`.

    context_start : first timestep to predict (must be >= CONTEXT_LEN=300).
                    The CONTEXT_LEN steps before it are used as the first context window.
                    Defaults to CONTEXT_LEN (300).

    Returns
    -------
    pred_out : (T, F) — NaN before context_start, then rolling predictions.
    """
    T, F      = block_norm.shape
    ctx_start = context_start if context_start is not None else CONTEXT_LEN
    # Ensure context_start is at least CONTEXT_LEN
    ctx_start = max(ctx_start, CONTEXT_LEN)

    pred_out = np.full((T, F), np.nan, dtype=np.float32)
    combined = block_norm[:ctx_start].copy()   # real context
    t_pos    = ctx_start

    model.eval()
    while t_pos < T:
        ctx  = combined[-CONTEXT_LEN:]
        src  = torch.tensor(ctx[None]).float().to(device)
        sc   = torch.tensor([scenario_id]).long().to(device)
        pred = model.predict(src, dec_len=PRED_LEN, scenario=sc
                             ).cpu().numpy()[0]                    # (180, F)
        end     = min(t_pos + PRED_LEN, T)
        n_write = end - t_pos
        pred_out[t_pos:end] = pred[:n_write]
        # Fully autoregressive: feed own predictions as next context
        combined = np.vstack([combined, pred[:n_write]])
        t_pos    = end

    return pred_out


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: episode-run detection and block reconstruction from windows
# ─────────────────────────────────────────────────────────────────────────────
def _find_episode_runs(
    scenario_labels: np.ndarray,
    min_run: int = 5,
) -> list[tuple[int, int, int]]:
    """
    Find contiguous runs of non-zero scenario_labels in the test array.

    Returns
    -------
    List of (start_idx, end_idx, scenario_id) — one entry per attack episode.
    """
    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < len(scenario_labels):
        sc = int(scenario_labels[i])
        if sc != 0:
            j = i + 1
            while j < len(scenario_labels) and int(scenario_labels[j]) == sc:
                j += 1
            if (j - i) >= min_run:
                runs.append((i, j, sc))
            i = j
        else:
            i += 1
    return runs


def _reconstruct_block(
    X_windows: np.ndarray,   # (N, 300, F)
    Y_windows: np.ndarray,   # (N, 180, F)
    run_start: int,
    run_end:   int,
) -> np.ndarray:
    """
    Stitch a contiguous (T, F) block from overlapping sliding windows.

    Each consecutive window advances by STRIDE=60 steps.
    The final Y window is appended to extend the block into the future.

    Output length ≈ 300 + (run_end - run_start - 1)*60 + 180.
    """
    parts = [X_windows[run_start]]               # full first encoder window
    for i in range(run_start + 1, run_end):
        parts.append(X_windows[i][-STRIDE:])     # 60 unique new steps each
    parts.append(Y_windows[run_end - 1])          # 180 future steps of last window
    return np.concatenate(parts, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Batch predict helper
# ─────────────────────────────────────────────────────────────────────────────
def _batch_predict(
    predict_fn,
    X: np.ndarray,
    Y: np.ndarray,
    scenario_ids: np.ndarray | None = None,
    batch: int = 64,
) -> np.ndarray:
    preds = []
    N  = len(X)
    sc = scenario_ids if scenario_ids is not None else np.zeros(N, dtype=np.int32)
    for i in range(0, N, batch):
        preds.append(predict_fn(X[i:i+batch], Y[i:i+batch], sc[i:i+batch]))
    return np.concatenate(preds, axis=0)


# ═════════════════════════════════════════════════════════════════════════════
# Plot 0: Autoregressive composite  (Train4 block + all attack episodes)
# ═════════════════════════════════════════════════════════════════════════════
def plot_autoregressive_composite(
    model,
    sensor_cols: list[str],
    device: torch.device,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    test_scenario_labels: np.ndarray,
    output_dir: Path,
    model_name: str = "Model",
) -> None:
    """
    Produces 4 separate images (4 sensor rows each):

      pred_train4.png         — Train4 wide block (10 000 steps)
      pred_attack_sc1.png     — AP no-combination
      pred_attack_sc2.png     — AP with-combination
      pred_attack_sc3.png     — AE no-combination

    Layout of each image
    --------------------
    4 rows = 4 sensors (P1_PIT01, P1_LIT01, P1_FT03Z, P1_TIT01).
    X-axis split at the midpoint:
      Left half  — ground truth only (green solid)
      Right half — ground truth (green) + autoregressive prediction (red dashed)
    The model uses the last 300 steps of the left half as its initial context.
    """

    def _save_segment_figure(
        blk: np.ndarray,        # (T, F) normalised
        pred_out: np.ndarray,   # (T, F) predictions, NaN before midpoint
        mid: int,               # timestep where prediction starts
        sc_id: int,
        title: str,
        filename: str,
    ) -> None:
        key_idx   = _key_idx(sensor_cols)
        n_sensors = len(key_idx)
        T         = len(blk)
        t_axis    = np.arange(T)
        col_color = SCENARIO_COLORS[sc_id]

        fig, axes = plt.subplots(n_sensors, 1,
                                 figsize=(20, 4 * n_sensors),
                                 sharex=True)
        if n_sensors == 1:
            axes = [axes]

        for row, (ax, sidx) in enumerate(zip(axes, key_idx)):
            real_s = blk[:, sidx]
            pred_s = pred_out[:, sidx]
            valid  = ~np.isnan(pred_s)

            ax.plot(t_axis, real_s, color="green", lw=1.0, alpha=0.9,
                    label="Ground truth" if row == 0 else "_nolegend_",
                    zorder=3)
            if valid.any():
                ax.plot(t_axis[valid], pred_s[valid], color="red", lw=1.1,
                        linestyle="--", alpha=0.9,
                        label="Prediction" if row == 0 else "_nolegend_",
                        zorder=4)

            # Midpoint divider + right-half shading
            ax.axvline(mid, color="black", lw=1.2, linestyle="--", alpha=0.6)
            ax.axvspan(mid, T, alpha=0.07, color=col_color, zorder=1)

            ax.set_ylabel(sensor_cols[sidx][:22], fontsize=9)
            ax.grid(True, alpha=0.2)

            if row == 0:
                # Half labels
                ax.text(mid * 0.5, 1.03, "◀  original data",
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8, color="dimgray")
                ax.text(mid + (T - mid) * 0.5, 1.03, "predicted  ▶",
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8, color="red")
                ax.legend(fontsize=9, loc="upper right")

        _set_xticks(axes[-1], T)
        axes[-1].set_xlabel("Time (min)", fontsize=10)
        fig.suptitle(
            f"{model_name} — {title}\n"
            "Green = Ground Truth   |   Red dashed = Autoregressive Prediction",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        out = output_dir / "plots" / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close("all")
        print(f"  Saved: {out}")

    # ── Train4: full file — first half = ground truth only, second half = predicted ──
    scaler = joblib.load(ROOT / "outputs/scaled_split/scaler.pkl")
    df4    = pd.read_csv(ROOT / "data/processed/train4.csv")
    raw4   = df4[sensor_cols].values.astype(np.float32)
    block4 = scaler.transform(raw4).astype(np.float32)
    mid4   = max(len(block4) // 2, CONTEXT_LEN)
    pred4  = _autoregress(model, block4, device, scenario_id=0, context_start=mid4)
    _save_segment_figure(block4, pred4, mid4, 0,
                         "Normal (train4)", "pred_train4.png")

    # ── Attack segments: one per scenario type ────────────────────────────────
    episode_runs = _find_episode_runs(test_scenario_labels)
    # Pick the longest run for each scenario type
    best: dict[int, tuple[int, int]] = {}
    for rs, re, sc_id in episode_runs:
        if sc_id not in best or (re - rs) > (best[sc_id][1] - best[sc_id][0]):
            best[sc_id] = (rs, re)

    for sc_id in [1, 2, 3]:
        blk: np.ndarray | None = None
        if sc_id in best:
            rs, re = best[sc_id]
            blk = _reconstruct_block(X_test, Y_test, rs, re)
        else:
            print(f"  INFO: no test windows for scenario {sc_id} — "
                  f"falling back to raw CSV data.")
            blk = _load_raw_attack_block(sensor_cols, scaler, sc_id)
            if blk is None:
                print(f"  WARNING: scenario {sc_id} not found in raw data either — skipping.")
                continue
        mid  = max(len(blk) // 2, CONTEXT_LEN)
        pred = _autoregress(model, blk, device,
                            scenario_id=sc_id, context_start=mid)
        _save_segment_figure(
            blk, pred, mid, sc_id,
            SCENARIO_NAMES[sc_id],
            f"pred_attack_sc{sc_id}.png",
        )


# ═════════════════════════════════════════════════════════════════════════════
# Plot (a): Reconstruction-error boxplot  —  per sensor × scenario
# ═════════════════════════════════════════════════════════════════════════════
def plot_reconstruction_boxplot(
    predict_fn,
    sensor_cols: list[str],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    test_scenario_labels: np.ndarray,
    output_dir: Path,
    model_name: str = "Model",
) -> None:
    """
    For each of the 6 key sensors: side-by-side boxplot of (y_true - y_pred)²
    grouped by scenario (Normal / AP-no-comb / AP-with-comb / AE-no-comb).
    Shows which sensors are most affected by which attack type.
    """
    key_idx = _key_idx(sensor_cols)
    n_k     = len(key_idx)

    Y_pred  = _batch_predict(predict_fn, X_test, Y_test,
                              test_scenario_labels.astype(np.int32))
    sq_err  = (Y_test - Y_pred) ** 2   # (N, T, F)

    present_scenarios = [s for s in [0, 1, 2, 3]
                         if (test_scenario_labels == s).any()]

    fig, axes = plt.subplots(1, n_k, figsize=(4.5 * n_k, 5), sharey=False)
    if n_k == 1:
        axes = [axes]

    for ax, sidx in zip(axes, key_idx):
        data_by_scenario = []
        tick_labels      = []
        box_colors       = []
        for sc_id in present_scenarios:
            mask   = (test_scenario_labels == sc_id)
            errors = sq_err[mask, :, sidx].flatten()
            data_by_scenario.append(errors)
            tick_labels.append(SCENARIO_NAMES[sc_id])
            box_colors.append(SCENARIO_COLORS[sc_id])

        bp = ax.boxplot(data_by_scenario, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.5))
        for patch, col in zip(bp["boxes"], box_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)

        ax.set_title(sensor_cols[sidx][:20], fontsize=8)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=25)
        ax.set_ylabel("Squared error (norm.)" if sidx == key_idx[0] else "")
        ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle(f"{model_name} — Reconstruction Error per Sensor × Scenario", fontsize=11)
    fig.tight_layout()

    out = output_dir / "plots" / "reconstruction_error_boxplot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot (b): Reconstruction error over time  —  one attack episode
# ═════════════════════════════════════════════════════════════════════════════
def plot_error_over_time(
    predict_fn,
    sensor_cols: list[str],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    test_scenario_labels: np.ndarray,
    output_dir: Path,
    model_name: str = "Model",
) -> None:
    """
    For one attack episode run, slide a prediction window and plot
    per-timestep MSE (averaged over all sensors).
    Shows when the error spikes after attack injection.
    """
    episode_runs = _find_episode_runs(test_scenario_labels)
    if not episode_runs:
        print("  No attack runs found — skipping error-over-time plot.")
        return

    # Prefer AP no-comb (sc=1), else first available
    run = next((r for r in episode_runs if r[2] == 1), episode_runs[0])
    rs, re, sc_id = run

    block = _reconstruct_block(X_test, Y_test, rs, re)   # (T, F)
    T, F  = block.shape

    errors_t = np.full(T, np.nan, dtype=np.float32)
    for t in range(CONTEXT_LEN, T, PRED_LEN):
        X_w = block[t - CONTEXT_LEN: t][None]         # (1, 300, F)
        end = min(t + PRED_LEN, T)
        Y_w = block[t: end]
        n   = len(Y_w)
        if n < PRED_LEN:
            Y_w = np.pad(Y_w, ((0, PRED_LEN - n), (0, 0)))
        Y_w    = Y_w[None]                             # (1, 180, F)
        sc_arr = np.array([sc_id], dtype=np.int32)
        Y_pred = predict_fn(X_w, Y_w, sc_arr)[0]      # (180, F)
        mse_t  = ((block[t: t + n] - Y_pred[:n]) ** 2).mean(axis=1)  # (n,)
        errors_t[t: t + n] = mse_t

    fig, ax = plt.subplots(figsize=(14, 4))
    t_axis = np.arange(T)
    valid  = ~np.isnan(errors_t)
    ax.plot(t_axis[valid], errors_t[valid], color="steelblue", lw=1.2, label="MSE (all sensors)")
    ax.fill_between(t_axis[valid], errors_t[valid], alpha=0.2, color="steelblue")
    ax.axvspan(CONTEXT_LEN, T, alpha=0.06, color=SCENARIO_COLORS[sc_id],
               label=f"Prediction region ({SCENARIO_NAMES[sc_id]})")
    ax.axvline(CONTEXT_LEN, color="gray", linestyle="--", lw=1.2, label="Context ends (t=300)")

    _set_xticks(ax, T)
    ax.set_xlabel("Time (min)", fontsize=10)
    ax.set_ylabel("Mean Squared Error (norm.)", fontsize=10)
    ax.set_title(
        f"{model_name} — Error over Time  ({SCENARIO_NAMES[sc_id]} episode)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out = output_dir / "plots" / "error_over_time.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot (c): RMSE per scenario  —  bar chart with std error bars
# ═════════════════════════════════════════════════════════════════════════════
def plot_rmse_per_scenario(
    predict_fn,
    sensor_cols: list[str],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    test_scenario_labels: np.ndarray,
    output_dir: Path,
    model_name: str = "Model",
) -> None:
    """
    Bar chart: mean RMSE per scenario, with ± 1 std over individual windows.
    Quantifies the detection contrast between normal and attack regimes.
    """
    results: dict[int, tuple[float, float]] = {}
    for sc_id in [0, 1, 2, 3]:
        mask = (test_scenario_labels == sc_id)
        if mask.sum() == 0:
            continue
        sc_arr   = np.full(mask.sum(), sc_id, dtype=np.int32)
        Y_pred   = _batch_predict(predict_fn, X_test[mask], Y_test[mask], sc_arr)
        rmse_per = np.sqrt(((Y_test[mask] - Y_pred) ** 2).mean(axis=(1, 2)))  # (N,)
        results[sc_id] = (float(rmse_per.mean()), float(rmse_per.std()))

    sc_ids  = sorted(results)
    means   = [results[s][0] for s in sc_ids]
    stds    = [results[s][1] for s in sc_ids]
    colors  = [SCENARIO_COLORS[s] for s in sc_ids]
    labels  = [SCENARIO_NAMES[s]  for s in sc_ids]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(sc_ids)), means, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", width=0.6,
                  error_kw=dict(elinewidth=1.5, capthick=1.5))

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + max(stds) * 0.02,
                f"{m:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(sc_ids)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("RMSE (normalised)", fontsize=10)
    ax.set_title(f"{model_name} — RMSE per Scenario", fontsize=11)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()

    out = output_dir / "plots" / "rmse_per_scenario.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot (d): Residual ACF  —  P1_PIT01, normal test windows
# ═════════════════════════════════════════════════════════════════════════════
def plot_residual_acf(
    predict_fn,
    sensor_cols: list[str],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    test_scenario_labels: np.ndarray,
    output_dir: Path,
    model_name: str = "Model",
    n_lags: int = 40,
) -> None:
    """
    ACF of (y_true - y_pred) for P1_PIT01 on normal test windows.
    White-noise residuals indicate the model has captured all temporal structure.
    """
    sidx = _sidx(sensor_cols, "P1_PIT01")
    if sidx is None:
        print("  P1_PIT01 not found — skipping ACF plot.")
        return

    mask = (test_scenario_labels == 0)
    if mask.sum() == 0:
        print("  No normal test windows — skipping ACF.")
        return

    sc_arr = np.zeros(mask.sum(), dtype=np.int32)
    Y_pred = _batch_predict(predict_fn, X_test[mask], Y_test[mask], sc_arr)
    resid  = (Y_test[mask] - Y_pred)[:, :, sidx].flatten()   # (N*T,)
    resid -= resid.mean()

    n  = len(resid)
    c0 = float(np.dot(resid, resid)) / n
    lags = np.arange(n_lags + 1)
    acf  = np.array(
        [1.0] + [float(np.dot(resid[: n - k], resid[k:])) / (n * c0)
                 for k in range(1, n_lags + 1)]
    )
    conf = 1.96 / np.sqrt(n)   # 95% confidence band

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.vlines(lags, 0, acf, colors="steelblue", lw=1.5)
    ax.axhline(0,     color="black", lw=0.8)
    ax.axhline( conf, color="red",   linestyle="--", lw=1, label="95% CI (white-noise)")
    ax.axhline(-conf, color="red",   linestyle="--", lw=1)
    ax.set_xlim(-0.5, n_lags + 0.5)
    ax.set_xlabel("Lag", fontsize=10)
    ax.set_ylabel("Autocorrelation", fontsize=10)
    ax.set_title(
        f"{model_name} — Residual ACF  (P1_PIT01, normal test set)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out = output_dir / "plots" / "residual_acf.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot (e): Scatter true vs predicted  —  P1_PIT01, normal windows
# ═════════════════════════════════════════════════════════════════════════════
def plot_scatter_true_vs_pred(
    predict_fn,
    sensor_cols: list[str],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    test_scenario_labels: np.ndarray,
    output_dir: Path,
    model_name: str = "Model",
) -> None:
    """
    Scatter: true vs predicted for P1_PIT01 across all normal test windows.
    Includes y = x reference line, R², and RMSE annotation.
    Shows overall calibration and any systematic bias.
    """
    sidx = _sidx(sensor_cols, "P1_PIT01")
    if sidx is None:
        print("  P1_PIT01 not found — skipping scatter plot.")
        return

    mask = (test_scenario_labels == 0)
    if mask.sum() == 0:
        print("  No normal test windows — skipping scatter.")
        return

    sc_arr = np.zeros(mask.sum(), dtype=np.int32)
    Y_pred = _batch_predict(predict_fn, X_test[mask], Y_test[mask], sc_arr)

    true_v = Y_test[mask][:, :, sidx].flatten()
    pred_v = Y_pred[:, :, sidx].flatten()

    # Subsample for clarity / speed
    if len(true_v) > 50_000:
        rng    = np.random.default_rng(42)
        idx    = rng.choice(len(true_v), 50_000, replace=False)
        true_v = true_v[idx]
        pred_v = pred_v[idx]

    rmse   = float(np.sqrt(np.mean((true_v - pred_v) ** 2)))
    ss_res = float(np.sum((true_v - pred_v) ** 2))
    ss_tot = float(np.sum((true_v - true_v.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    lo = min(float(true_v.min()), float(pred_v.min()))
    hi = max(float(true_v.max()), float(pred_v.max()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_v, pred_v, s=1, alpha=0.3, color="steelblue", rasterized=True)
    ax.plot([lo, hi], [lo, hi], color="red", lw=1.5, label="y = x  (perfect)")
    ax.set_xlabel("True (normalised)", fontsize=10)
    ax.set_ylabel("Predicted (normalised)", fontsize=10)
    ax.set_title(
        f"{model_name} — True vs Predicted  (P1_PIT01, normal)", fontsize=11)
    ax.text(0.05, 0.93,
            f"R² = {r2:.4f}\nRMSE = {rmse:.5f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    out = output_dir / "plots" / "scatter_true_vs_pred.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════════════
def run_all_plots(
    model,
    predict_fn,
    sensor_cols: list[str],
    device: torch.device,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    test_scenario_labels: np.ndarray,
    output_dir: Path,
    model_name: str = "Model",
) -> None:
    """Run all 6 diagnostic plot functions in sequence."""
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTIC PLOTS — {model_name}")
    print(f"  Output: {output_dir / 'plots'}")
    print(f"{'='*60}")

    steps = [
        ("[0/6] Autoregressive composite (Train4 + attack episodes)...",
         lambda: plot_autoregressive_composite(
             model, sensor_cols, device, X_test, Y_test,
             test_scenario_labels, output_dir, model_name)),
        ("[a/6] Reconstruction error boxplot per sensor × scenario...",
         lambda: plot_reconstruction_boxplot(
             predict_fn, sensor_cols, X_test, Y_test,
             test_scenario_labels, output_dir, model_name)),
        ("[b/6] Error over time (attack episode)...",
         lambda: plot_error_over_time(
             predict_fn, sensor_cols, X_test, Y_test,
             test_scenario_labels, output_dir, model_name)),
        ("[c/6] RMSE per scenario bar chart...",
         lambda: plot_rmse_per_scenario(
             predict_fn, sensor_cols, X_test, Y_test,
             test_scenario_labels, output_dir, model_name)),
        ("[d/6] Residual ACF (P1_PIT01, normal)...",
         lambda: plot_residual_acf(
             predict_fn, sensor_cols, X_test, Y_test,
             test_scenario_labels, output_dir, model_name)),
        ("[e/6] Scatter true vs predicted (P1_PIT01, normal)...",
         lambda: plot_scatter_true_vs_pred(
             predict_fn, sensor_cols, X_test, Y_test,
             test_scenario_labels, output_dir, model_name)),
    ]

    for msg, fn in steps:
        print(f"\n{msg}")
        try:
            fn()
        except Exception as exc:
            print(f"  WARNING: skipped — {exc}")

    print(f"\n  All plots saved to: {output_dir / 'plots'}")
