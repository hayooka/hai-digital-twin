"""
sec1_1_shared.py — Section 1.1: Shared Results

Produces:
  figures/s1_1_loss_curves.png         — training & validation loss curves
  figures/s1_1_nrmse_per_pv.png        — overall NRMSE per PV (bar chart)
  figures/s1_1_nrmse_per_scenario.png  — NRMSE per PV per scenario (grouped bar)
  figures/s1_1_error_growth.png        — NRMSE vs prediction horizon (chained rollout)

Model used: gru_scenario_weighted (best PV prediction model)

Usage:
    python report_plots/code/sec1_1_shared.py
"""

import sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from gru import GRUPlant
from config import PV_COLS, LOOPS

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH    = 128
CKPT_DIR = ROOT / "outputs" / "pipeline" / "gru_scenario_weighted"
OUT_DIR  = ROOT / "report_plots" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PV_SHORT       = [p.replace("P1_", "") for p in PV_COLS]
SCENARIO_NAMES = {0: "Normal", 1: "AP\_no", 2: "AP\_with", 3: "AE\_no"}
SC_COLORS      = {0: "#2196F3", 1: "#FF5722", 2: "#E91E63", 3: "#9C27B0"}

CTRL_LOOPS = ["PC", "LC", "FC", "TC", "CC"]

# ── helpers ───────────────────────────────────────────────────────────────────

def _nrmse(t, p):
    rmse = np.sqrt(np.mean((t - p) ** 2))
    r    = float(t.max() - t.min())
    return 0.0 if r < 1e-10 else rmse / r


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
        n_haiend    = ckpt.get("n_haiend", 0),
    ).to(DEVICE)
    m.load_state_dict(ms)
    m.eval()
    return m


def run_inference(model, plant_data, split="test"):
    X, Xcv  = plant_data[f"X_{split}"], plant_data[f"X_cv_target_{split}"]
    pv_init = plant_data[f"pv_init_{split}"]
    sc      = plant_data[f"scenario_{split}"]
    N, TL   = X.shape[0], plant_data[f"pv_target_{split}"].shape[1]
    preds   = np.zeros((N, TL, plant_data["n_pv"]), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl = slice(i, i + BATCH)
            pv_out, _ = model.predict(
                torch.tensor(X[sl]).float().to(DEVICE),
                torch.tensor(Xcv[sl]).float().to(DEVICE),
                torch.tensor(pv_init[sl]).float().to(DEVICE),
                torch.tensor(sc[sl]).long().to(DEVICE),
            )
            preds[i:i + pv_out.size(0)] = pv_out.cpu().numpy()
    return preds


def chained_rollout(model, data, split="test", n_windows=30):
    """Autoregressive rollout across n_windows consecutive normal windows."""
    plant_data = data["plant"]
    sc_arr     = plant_data[f"scenario_{split}"]
    pv_target  = plant_data[f"pv_target_{split}"]
    X          = plant_data[f"X_{split}"]
    Xcv        = plant_data[f"X_cv_target_{split}"]
    pv_init    = plant_data[f"pv_init_{split}"]
    TL         = pv_target.shape[1]
    NP         = plant_data["n_pv"]

    normal_idx = np.where(sc_arr == 0)[0]
    variances  = [np.var(pv_target[i]) for i in normal_idx]
    start      = normal_idx[np.argsort(variances)[len(variances) // 2]]

    sensor_cols = data["metadata"]["sensor_cols"]
    non_pv      = [c for c in sensor_cols if c not in set(PV_COLS)]
    c2i         = {c: i for i, c in enumerate(non_pv)}
    ctrl_cv_idx = {ln: c2i[LOOPS[ln].cv] for ln in CTRL_LOOPS if LOOPS[ln].cv in c2i}

    true_parts, pred_parts = [], []
    with torch.no_grad():
        x0  = torch.tensor(X[start:start+1]).float().to(DEVICE)
        sc0 = torch.tensor(sc_arr[start:start+1]).long().to(DEVICE)
        emb = model.scenario_emb(sc0).unsqueeze(1).expand(-1, x0.size(1), -1)
        _, h = model.encoder(torch.cat([x0, emb], dim=-1))
        pv_prev = torch.tensor(pv_target[start:start+1, 0, :]).float().to(DEVICE)

        for w in range(n_windows):
            idx = start + w
            if idx >= len(X): break
            true_parts.append(pv_target[idx])
            xct_w = torch.tensor(Xcv[idx:idx+1]).float().to(DEVICE).clone()
            sc_w  = torch.tensor(sc_arr[idx:idx+1]).long().to(DEVICE)
            win_preds = []
            for t in range(TL):
                dec_in   = torch.cat([xct_w[:, t, :], pv_prev], dim=-1).unsqueeze(1)
                out, h   = model.decoder(dec_in, h)
                pv_pred  = model.fc(out.squeeze(1))
                win_preds.append(pv_pred)
                pv_prev  = pv_pred
            pred_parts.append(torch.stack(win_preds, dim=1).squeeze(0).cpu().numpy())

    return np.concatenate(true_parts), np.concatenate(pred_parts)


# ── Plot 1: Loss curves ───────────────────────────────────────────────────────

def plot_loss_curves():
    with open(CKPT_DIR / "results.json") as f:
        res = json.load(f)
    train_l = res["train_losses"]
    val_l   = res["val_losses"]
    epochs  = range(1, len(train_l) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_l, color="#2196F3", lw=2, label="Train")
    ax.plot(epochs[:len(val_l)], val_l, color="#FF5722", lw=2, label="Validation")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss (log scale)", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = OUT_DIR / "s1_1_loss_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ── Plot 2: Overall NRMSE per PV ─────────────────────────────────────────────

def plot_nrmse_per_pv(pv_true, preds):
    nrmse_vals = [_nrmse(pv_true[:,:,k].flatten(), preds[:,:,k].flatten())
                  for k in range(len(PV_COLS))]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
    bars   = ax.bar(PV_SHORT, nrmse_vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, nrmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("NRMSE", fontsize=12)
    ax.set_xlabel("Process Variable", fontsize=12)
    ax.set_title("Overall NRMSE per PV — Full Test Set", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = OUT_DIR / "s1_1_nrmse_per_pv.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ── Plot 3: NRMSE per PV per Scenario ────────────────────────────────────────

def plot_nrmse_per_scenario(pv_true, preds, sc_arr):
    sc_ids  = [0, 1, 2, 3]
    sc_lbls = [SCENARIO_NAMES[i] for i in sc_ids]
    n_sc    = len(sc_ids)
    x       = np.arange(len(PV_COLS))
    w       = 0.8 / n_sc
    sc_clrs = [SC_COLORS[i] for i in sc_ids]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (sc_id, lbl, col) in enumerate(zip(sc_ids, sc_lbls, sc_clrs)):
        mask   = sc_arr == sc_id
        vals   = [_nrmse(pv_true[mask,:,k].flatten(), preds[mask,:,k].flatten())
                  for k in range(len(PV_COLS))]
        offset = (i - n_sc/2 + 0.5) * w
        bars   = ax.bar(x + offset, vals, w, label=lbl, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(PV_SHORT, fontsize=11)
    ax.set_ylabel("NRMSE", fontsize=12)
    ax.set_title("NRMSE per PV per Scenario", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "s1_1_nrmse_per_scenario.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ── Plot 4: Error growth (chained rollout) ────────────────────────────────────

def plot_error_growth(model, data):
    print("  Running chained rollout (30 windows)...")
    true_chain, pred_chain = chained_rollout(model, data, n_windows=30)
    horizons = [60, 180, 300, 600, 900, 1800, 2700, 3600, 4500, 5400]
    horizons = [h for h in horizons if h <= len(true_chain)]

    overall, pv_curves = [], {pv: [] for pv in PV_COLS}
    for h in horizons:
        vals = [_nrmse(true_chain[:h, k], pred_chain[:h, k]) for k in range(len(PV_COLS))]
        overall.append(np.mean(vals))
        for k, pv in enumerate(PV_COLS):
            pv_curves[pv].append(vals[k])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(horizons, overall, "b-o", lw=2.5, markersize=6, label="Overall Average", zorder=5)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
    for (pv, vals), col in zip(pv_curves.items(), colors):
        ax.plot(horizons, vals, "--", lw=1.5, alpha=0.6, color=col,
                label=pv.replace("P1_", ""))
    ax.axvline(x=180, color="gray", linestyle=":", lw=1.5, label="Training horizon (180s)")
    ax.set_xlabel("Prediction Horizon (seconds)", fontsize=12)
    ax.set_ylabel("NRMSE", fontsize=12)
    ax.set_title("Prediction Error Growth — Autoregressive Chained Rollout", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.3)

    # x-axis labels in minutes/hours
    def fmt(s):
        if s >= 3600: return f"{s//3600}h"
        return f"{s//60}min"
    ax.set_xticks(horizons)
    ax.set_xticklabels([fmt(h) for h in horizons], fontsize=9)

    fig.tight_layout()
    path = OUT_DIR / "s1_1_error_growth.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    data       = load_and_prepare_data()
    plant_data = data["plant"]
    pv_true    = plant_data["pv_target_test"]
    sc_arr     = plant_data["scenario_test"]

    print("Loading model...")
    model = load_model()

    print("Running inference...")
    preds = run_inference(model, plant_data)

    print("\n[Plot 1] Loss curves...")
    plot_loss_curves()

    print("[Plot 2] NRMSE per PV...")
    plot_nrmse_per_pv(pv_true, preds)

    print("[Plot 3] NRMSE per scenario...")
    plot_nrmse_per_scenario(pv_true, preds, sc_arr)

    print("[Plot 4] Error growth...")
    plot_error_growth(model, data)

    print(f"\nAll Section 1.1 plots saved to: {OUT_DIR}/")
