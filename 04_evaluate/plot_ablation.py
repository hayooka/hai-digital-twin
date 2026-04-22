"""
plot_ablation.py — Ablation comparison: Stage 1 vs Stage 2.

Reads results.json from both checkpoints and produces:
  plots/ablation/ablation_nrmse.png    — NRMSE per PV, side-by-side
  plots/ablation/ablation_detection.png — AUROC / F1 / AvgPrec bar chart
  plots/ablation/ablation_scenario_nrmse.png — mean NRMSE per scenario

No model loading required — reads from saved results.json files only.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "plots" / "ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_PATH = ROOT / "outputs" / "pipeline" / "gru_causal_plus_tuned" / "results.json"
STAGE2_PATH = ROOT / "outputs" / "pipeline" / "gru_scenario_weighted"  / "results.json"

PV_COLS  = ["P1_PIT01", "P1_LIT01", "P1_FT03Z", "P1_TIT01", "P1_TIT03"]
PV_SHORT = [p.replace("P1_", "") for p in PV_COLS]
SCENARIO_NAMES = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}
COLORS = {"Stage 1\n(Causal+Tuned)": "#5b9bd5", "Stage 2\n(Scenario-Weighted)": "#ed7d31"}


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


s1 = load(STAGE1_PATH)
s2 = load(STAGE2_PATH)

labels = list(COLORS.keys())
colors = list(COLORS.values())

# ── Plot 1: NRMSE per PV ──────────────────────────────────────────────────────

s1_nrmse = [s1["test_nrmse_per_pv"].get(p, 0) for p in PV_COLS]
s2_nrmse = [s2["test_nrmse_per_pv"].get(p, 0) for p in PV_COLS]

x = np.arange(len(PV_COLS))
w = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w/2, s1_nrmse, w, label=labels[0], color=colors[0], alpha=0.85)
b2 = ax.bar(x + w/2, s2_nrmse, w, label=labels[1], color=colors[1], alpha=0.85)

for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.0003,
                f'{h:.4f}', ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x); ax.set_xticklabels(PV_SHORT, fontsize=11)
ax.set_ylabel("NRMSE (test set)", fontsize=11)
ax.set_title("Ablation — NRMSE per Process Variable\nStage 1 vs Stage 2",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_nrmse.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved ablation_nrmse.png")


# ── Plot 2: Detection metrics ──────────────────────────────────────────────────

metrics      = ["AUROC", "Best F1", "Avg Precision"]
s1_det = s1["attack_detection"]
s2_det = s2["attack_detection"]
s1_vals = [s1_det["auroc"], s1_det["best_f1"], s1_det["avg_precision"]]
s2_vals = [s2_det["auroc"], s2_det["best_f1"], s2_det["avg_precision"]]

x = np.arange(len(metrics))
w = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - w/2, s1_vals, w, label=labels[0], color=colors[0], alpha=0.85)
b2 = ax.bar(x + w/2, s2_vals, w, label=labels[1], color=colors[1], alpha=0.85)

for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f'{h:.4f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel("Score", fontsize=11); ax.set_ylim(0, 1.1)
ax.set_title("Ablation — Attack Detection Metrics\nStage 1 vs Stage 2",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_detection.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved ablation_detection.png")


# ── Plot 3: Per-scenario mean NRMSE ───────────────────────────────────────────

def mean_nrmse_per_scenario(results):
    out = {}
    sc_data = results.get("test_nrmse_per_scenario", {})
    for sc_name, pv_dict in sc_data.items():
        out[sc_name] = np.mean(list(pv_dict.values()))
    return out

s1_sc = mean_nrmse_per_scenario(s1)
s2_sc = mean_nrmse_per_scenario(s2)

sc_names = [n for n in ["Normal", "AP_no", "AP_with", "AE_no"] if n in s1_sc]
s1_sc_vals = [s1_sc.get(n, 0) for n in sc_names]
s2_sc_vals = [s2_sc.get(n, 0) for n in sc_names]

x = np.arange(len(sc_names))
w = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, s1_sc_vals, w, label=labels[0], color=colors[0], alpha=0.85)
b2 = ax.bar(x + w/2, s2_sc_vals, w, label=labels[1], color=colors[1], alpha=0.85)

for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                f'{h:.4f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x); ax.set_xticklabels(sc_names, fontsize=11)
ax.set_ylabel("Mean NRMSE across PVs", fontsize=11)
ax.set_title("Ablation — Mean NRMSE per Scenario\nStage 1 vs Stage 2",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_scenario_nrmse.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved ablation_scenario_nrmse.png")


# ── summary table ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print(f"{'Metric':<28} {'Stage 1':>10} {'Stage 2':>10}")
print("-"*55)
print(f"{'Mean NRMSE':<28} {s1['test_mean_nrmse']:>10.5f} {s2['test_mean_nrmse']:>10.5f}")
print(f"{'TIT03 NRMSE':<28} {s1['test_nrmse_per_pv'].get('P1_TIT03',0):>10.5f} {s2['test_nrmse_per_pv'].get('P1_TIT03',0):>10.5f}")
print(f"{'AUROC':<28} {s1_det['auroc']:>10.4f} {s2_det['auroc']:>10.4f}")
print(f"{'Best F1':<28} {s1_det['best_f1']:>10.4f} {s2_det['best_f1']:>10.4f}")
print(f"{'Avg Precision':<28} {s1_det['avg_precision']:>10.4f} {s2_det['avg_precision']:>10.4f}")
print(f"{'Best Val Loss':<28} {s1.get('best_val_loss',0):>10.6f} {s2.get('best_val_loss',0):>10.6f}")
print("="*55)
print(f"\nAll plots saved to {OUT_DIR}/")
