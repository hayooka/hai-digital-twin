"""
compare_models.py — Compare NRMSE across all trained models.

Runs fresh inference on the test set for each available model and produces:
  plots/compare/nrmse_per_pv.png        — grouped bar: NRMSE per PV per model
  plots/compare/nrmse_per_scenario.png  — grouped bar: mean NRMSE per scenario per model
  plots/compare/nrmse_overall.png       — single bar: overall mean NRMSE per model

Usage:
    python 04_evaluate/compare_models.py
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from gru import GRUPlant
from config import PV_COLS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH  = 128

MODELS = {
    "Causal-Plus":        ROOT / "outputs" / "pipeline" / "gru_causal_plus_tuned",
    "Scenario-Weighted":  ROOT / "outputs" / "pipeline" / "gru_scenario_weighted",
    "Scenario-HaiEnd":    ROOT / "outputs" / "pipeline" / "gru_scenario_haiend",
}
MODEL_COLORS = {
    "Causal-Plus":       "#5b9bd5",
    "Scenario-Weighted": "#ed7d31",
    "Scenario-HaiEnd":   "#70ad47",
}

SCENARIO_NAMES = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}
PV_SHORT       = [p.replace("P1_", "") for p in PV_COLS]

OUT_DIR = ROOT / "plots" / "compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _nrmse(true, pred):
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    r    = true.max() - true.min()
    return 0.0 if r < 1e-10 else rmse / r


def load_model(ckpt_dir: Path) -> GRUPlant:
    ckpt = torch.load(ckpt_dir / "gru_plant.pt", map_location=DEVICE)
    ms   = ckpt["model_state"]
    emb  = ms["scenario_emb.weight"].shape[1]
    model = GRUPlant(
        n_plant_in  = ckpt.get("n_plant_in",  ms["encoder.weight_ih_l0"].shape[1] - emb),
        n_pv        = ckpt.get("n_pv",        ms["fc.3.weight"].shape[0]),
        hidden      = ckpt["hidden"],
        layers      = ckpt["layers"],
        n_scenarios = ms["scenario_emb.weight"].shape[0],
        n_haiend    = ckpt.get("n_haiend", 0),
    ).to(DEVICE)
    model.load_state_dict(ms)
    model.eval()
    return model


def run_inference(model, plant_data, split="test"):
    X       = plant_data[f"X_{split}"]
    X_cv    = plant_data[f"X_cv_target_{split}"]
    pv_init = plant_data[f"pv_init_{split}"]
    sc      = plant_data[f"scenario_{split}"]
    N, TL   = X.shape[0], plant_data[f"pv_target_{split}"].shape[1]
    NP      = plant_data["n_pv"]
    preds   = np.zeros((N, TL, NP), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl = slice(i, i + BATCH)
            pv_out, _ = model.predict(
                torch.tensor(X[sl]).float().to(DEVICE),
                torch.tensor(X_cv[sl]).float().to(DEVICE),
                torch.tensor(pv_init[sl]).float().to(DEVICE),
                torch.tensor(sc[sl]).long().to(DEVICE),
            )
            preds[i:i + pv_out.size(0)] = pv_out.cpu().numpy()
    return preds


def compute_nrmse(true, pred, scenario_arr):
    """Returns per_pv, per_scenario mean NRMSE, and per_pv NRMSE for attack scenarios."""
    per_pv = {pv: _nrmse(true[:, :, k].flatten(), pred[:, :, k].flatten())
              for k, pv in enumerate(PV_COLS)}
    per_sc = {}
    attack_per_pv = {}  # NRMSE per PV on attack windows only (sc 1,2,3)
    for sc_id, sc_name in SCENARIO_NAMES.items():
        mask = scenario_arr == sc_id
        if mask.sum() == 0:
            continue
        vals = [_nrmse(true[mask, :, k].flatten(), pred[mask, :, k].flatten())
                for k in range(len(PV_COLS))]
        per_sc[sc_name] = np.mean(vals)

    attack_mask = scenario_arr > 0
    if attack_mask.sum() > 0:
        attack_per_pv = {pv: _nrmse(true[attack_mask, :, k].flatten(),
                                     pred[attack_mask, :, k].flatten())
                         for k, pv in enumerate(PV_COLS)}
    return per_pv, per_sc, attack_per_pv


# ── run all models ─────────────────────────────────────────────────────────────

print("Loading data...")
data       = load_and_prepare_data()
plant_data = data["plant"]
sc_arr     = plant_data["scenario_test"]
pv_true    = plant_data["pv_target_test"]

results = {}   # model_name → {per_pv, per_sc, overall}

for name, ckpt_dir in MODELS.items():
    if not (ckpt_dir / "gru_plant.pt").exists():
        print(f"  Skipping {name} — checkpoint not found")
        continue
    print(f"  Running inference: {name}...")
    model   = load_model(ckpt_dir)
    preds   = run_inference(model, plant_data)
    per_pv, per_sc, attack_per_pv = compute_nrmse(pv_true, preds, sc_arr)
    results[name] = {
        "per_pv":        per_pv,
        "per_sc":        per_sc,
        "attack_per_pv": attack_per_pv,
        "overall":       np.mean(list(per_pv.values())),
    }
    print(f"    Overall NRMSE: {results[name]['overall']:.4f}")

model_names = list(results.keys())
colors      = [MODEL_COLORS[n] for n in model_names]
n_models    = len(model_names)
x           = np.arange(len(PV_COLS))
w           = 0.8 / n_models


# ── Plot 1: NRMSE per PV ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
for i, (name, col) in enumerate(zip(model_names, colors)):
    vals   = [results[name]["per_pv"][pv] for pv in PV_COLS]
    offset = (i - n_models / 2 + 0.5) * w
    bars   = ax.bar(x + offset, vals, w, label=name, color=col, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(PV_SHORT, fontsize=11)
ax.set_ylabel("NRMSE", fontsize=11)
ax.set_title("Model Comparison — NRMSE per PV (test set)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "nrmse_per_pv.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved nrmse_per_pv.png")


# ── Plot 2: Mean NRMSE per scenario ──────────────────────────────────────────

sc_names = list(SCENARIO_NAMES.values())
x2 = np.arange(len(sc_names))

fig, ax = plt.subplots(figsize=(10, 5))
for i, (name, col) in enumerate(zip(model_names, colors)):
    vals   = [results[name]["per_sc"].get(s, 0) for s in sc_names]
    offset = (i - n_models / 2 + 0.5) * w
    bars   = ax.bar(x2 + offset, vals, w, label=name, color=col, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x2)
ax.set_xticklabels(sc_names, fontsize=11)
ax.set_ylabel("Mean NRMSE across PVs", fontsize=11)
ax.set_title("Model Comparison — Mean NRMSE per Scenario", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "nrmse_per_scenario.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved nrmse_per_scenario.png")


# ── Plot 3: Overall mean NRMSE ────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5))
overall_vals = [results[n]["overall"] for n in model_names]
bars = ax.bar(model_names, overall_vals, color=colors, alpha=0.85, width=0.5)
for bar, v in zip(bars, overall_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{v:.4f}", ha="center", va="bottom", fontsize=10)

ax.set_ylabel("Mean NRMSE", fontsize=11)
ax.set_title("Model Comparison — Overall Mean NRMSE", fontsize=13, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "nrmse_overall.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved nrmse_overall.png")


# ── summary table ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print(f"{'Model':<22}", end="")
for pv in PV_SHORT:
    print(f"  {pv:>8}", end="")
print(f"  {'Overall':>8}")
print("-" * 65)
for name in model_names:
    print(f"{name:<22}", end="")
    for pv in PV_COLS:
        print(f"  {results[name]['per_pv'][pv]:>8.4f}", end="")
    print(f"  {results[name]['overall']:>8.4f}")
print("=" * 65)
print(f"\nAll plots saved to: {OUT_DIR}/")


# ── Plot 4: Attack NRMSE per PV ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
for i, (name, col) in enumerate(zip(model_names, colors)):
    vals   = [results[name]["attack_per_pv"].get(pv, 0) for pv in PV_COLS]
    offset = (i - n_models / 2 + 0.5) * w
    bars   = ax.bar(x + offset, vals, w, label=name, color=col, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(PV_SHORT, fontsize=11)
ax.set_ylabel("NRMSE (attack windows only)", fontsize=11)
ax.set_title("Model Comparison — Prediction Error on Attack Scenarios (Scenarios 1–3)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "nrmse_attack_per_pv.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved nrmse_attack_per_pv.png")
