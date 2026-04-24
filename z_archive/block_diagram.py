"""
block_diagram.py — Block diagram of the full system pipeline.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

# ── Colors ────────────────────────────────────────────────────────────────────
BG      = "#F8F9FA"
BLUE    = "#1565C0"
LBLUE   = "#BBDEFB"
GREEN   = "#2E7D32"
LGREEN  = "#C8E6C9"
ORANGE  = "#E65100"
LORANGE = "#FFE0B2"
PURPLE  = "#6A1B9A"
LPURPLE = "#E1BEE7"
GRAY    = "#455A64"
LGRAY   = "#ECEFF1"
RED     = "#B71C1C"
LRED    = "#FFCDD2"
TEAL    = "#00695C"
LTEAL   = "#B2DFDB"
DARK    = "#212121"

fig, ax = plt.subplots(figsize=(18, 11), facecolor=BG)
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis('off')
ax.set_facecolor(BG)


def box(x, y, w, h, fc, ec, label, fontsize=9, bold=False,
        sublabel=None, sub_fs=7):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.12",
                          facecolor=fc, edgecolor=ec, linewidth=1.8)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ty = y + h/2 + (0.12 if sublabel else 0)
    ax.text(x + w/2, ty, label,
            ha='center', va='center', fontsize=fontsize,
            color=DARK, fontweight=weight)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                ha='center', va='center', fontsize=sub_fs,
                color=GRAY, style='italic')


def arrow(x1, y1, x2, y2, color=GRAY, label='', lw=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle='arc3,rad=0.0'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.05, my + 0.1, label,
                fontsize=6.5, color=color, ha='center', style='italic')


def bracket_label(x, y1, y2, label, color):
    ax.annotate('', xy=(x, y1), xytext=(x, y2),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.5))
    ax.text(x + 0.15, (y1+y2)/2, label,
            fontsize=7, color=color, va='center', fontweight='bold')


# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(9, 10.55, "Conditional Generative GRU Surrogate — System Block Diagram",
        ha='center', va='center', fontsize=14, fontweight='bold', color=DARK)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — DATA
# ══════════════════════════════════════════════════════════════════════════════
ax.text(0.3, 9.7, "DATA", fontsize=8, fontweight='bold', color=GRAY)

box(0.3, 8.9, 2.4, 0.7, LGRAY, GRAY,
    "HAI 23.05 Dataset", fontsize=9, bold=True,
    sublabel="4 scenarios · 86 sensors")

arrow(2.7, 9.25, 3.9, 9.25, GRAY, "sliding\nwindow")

box(3.9, 8.9, 2.8, 0.7, LGRAY, GRAY,
    "Preprocessing  (pipeline.py)", fontsize=9,
    sublabel="300s input · 180s target · 60s stride")

arrow(6.7, 9.25, 7.9, 9.25, GRAY, "scaled\nwindows")

box(7.9, 8.9, 2.5, 0.7, LGRAY, GRAY,
    "Scenario Labels", fontsize=9,
    sublabel="0=Normal  1=AP_no  2=AP_with  3=AE_no")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(0.3, 8.3, "TRAINING PIPELINE", fontsize=8, fontweight='bold', color=BLUE)

# Stage A
box(0.3, 7.2, 2.4, 0.9, LBLUE, BLUE,
    "Stage A", fontsize=9, bold=True,
    sublabel="train_gru_normal_only.py")
ax.text(1.5, 7.12, "Normal windows only", ha='center', fontsize=6.5, color=BLUE)

arrow(2.7, 7.65, 3.9, 7.65, BLUE)

# Stage B
box(3.9, 7.2, 2.8, 0.9, LBLUE, BLUE,
    "Stage B", fontsize=9, bold=True,
    sublabel="train_gru_scenario_weighted.py")
ax.text(5.3, 7.12, "All 4 scenarios · per-scenario loss weights",
        ha='center', fontsize=6.5, color=BLUE)

arrow(6.7, 7.65, 7.9, 7.65, BLUE, "warm\nstart")

# Stage C
box(7.9, 7.2, 2.8, 0.9, LBLUE, BLUE,
    "Stage C  ★  FINAL MODEL", fontsize=9, bold=True,
    sublabel="finetune_haiend.py")
ax.text(9.3, 7.12, "Freeze all · train HAIEND head (36 PLC signals)",
        ha='center', fontsize=6.5, color=BLUE)

# checkpoint arrow down
arrow(9.3, 7.2, 9.3, 6.4, BLUE, "gru_scenario_haiend/")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — MODEL INTERNALS
# ══════════════════════════════════════════════════════════════════════════════
ax.text(0.3, 6.7, "MODEL INTERNALS  (gru.py)", fontsize=8, fontweight='bold', color=PURPLE)

# Scenario embedding
box(0.3, 5.5, 2.0, 0.75, LPURPLE, PURPLE,
    "Scenario\nEmbedding", fontsize=8, bold=True,
    sublabel="label → 32-dim vec")

# Input
box(0.3, 4.5, 2.0, 0.75, LGRAY, GRAY,
    "Input Window", fontsize=8,
    sublabel="300s · sensor history")

# Encoder
arrow(2.3, 5.875, 3.3, 5.875, PURPLE)
arrow(2.3, 4.875, 3.3, 5.5,   PURPLE)
box(3.3, 5.35, 2.0, 0.9, LPURPLE, PURPLE,
    "GRU Encoder", fontsize=9, bold=True,
    sublabel="history + scenario → h")

# Controllers
box(0.3, 3.4, 2.0, 0.75, LGREEN, GREEN,
    "Controllers ×5", fontsize=8, bold=True,
    sublabel="PC·LC·FC·TC·CC")
arrow(2.3, 3.775, 3.3, 3.775, GREEN, "CV seq")

# Decoder
box(3.3, 3.4, 2.0, 0.9, LPURPLE, PURPLE,
    "GRU Decoder", fontsize=9, bold=True,
    sublabel="autoregressive · 180 steps")
arrow(5.3, 5.8, 5.3, 4.3, PURPLE, "hidden\nstate h")
arrow(5.3, 3.775, 5.3, 3.4, GREEN)

# Output heads
arrow(5.3, 3.85, 6.3, 4.5,  ORANGE)
arrow(5.3, 3.85, 6.3, 3.2,  TEAL)

box(6.3, 4.2, 2.2, 0.7, LORANGE, ORANGE,
    "PV Head", fontsize=9, bold=True,
    sublabel="→ 5 process variables")

box(6.3, 2.9, 2.2, 0.7, LTEAL, TEAL,
    "HAIEND Head", fontsize=9, bold=True,
    sublabel="→ 36 internal PLC signals")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — OUTPUTS / APPLICATIONS
# ══════════════════════════════════════════════════════════════════════════════
ax.text(9.0, 6.7, "OUTPUTS & EVALUATION", fontsize=8, fontweight='bold', color=ORANGE)

# Generation output
box(9.0, 5.5, 2.8, 0.75, LORANGE, ORANGE,
    "Synthetic Trajectory", fontsize=9, bold=True,
    sublabel="180s · conditioned on scenario")

arrow(8.5, 4.55, 9.0, 5.5, ORANGE, "PV output")

# Simulation output
box(9.0, 4.4, 2.8, 0.75, LGRAY, GRAY,
    "Simulation", fontsize=9,
    sublabel="predicted vs actual · residuals")

# Evaluation
box(12.2, 5.5, 2.8, 0.75, LORANGE, ORANGE,
    "evaluate_generation.py", fontsize=8, bold=True,
    sublabel="Exp A / B / C  ·  F1 = 0.95")
arrow(11.8, 5.875, 12.2, 5.875, ORANGE)

# Augmentation
box(12.2, 4.4, 2.8, 0.75, LORANGE, ORANGE,
    "augment_idea3_noise.py", fontsize=8, bold=True,
    sublabel="6 seeds × 50 noise → 300 AE windows")
arrow(11.8, 4.775, 12.2, 4.775, ORANGE)

# Anomaly detection
box(12.2, 3.2, 2.8, 0.75, LGREEN, GREEN,
    "monitor.py", fontsize=8, bold=True,
    sublabel="normal-only residuals · AUROC=0.92")
arrow(11.8, 4.4, 12.2, 3.575, GREEN)

# Dashboard
box(15.4, 5.0, 2.3, 1.3, LPURPLE, PURPLE,
    "Dashboard", fontsize=10, bold=True,
    sublabel="Page 1: Simulate\nPage 2: Generate")
arrow(15.0, 5.875, 15.4, 5.65, PURPLE)
arrow(15.0, 4.775, 15.4, 5.3,  PURPLE)

# Classifier result callout
box(12.2, 2.1, 2.8, 0.75, LRED, RED,
    "Classifier Transfer", fontsize=8, bold=True,
    sublabel="Train Synth → Test Real  ·  F1=0.95")
arrow(13.6, 4.4, 13.6, 2.85, RED)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (LBLUE,   BLUE,   "Training stages"),
    (LPURPLE, PURPLE, "Model architecture"),
    (LGREEN,  GREEN,  "Controllers / Detection"),
    (LORANGE, ORANGE, "Generation / Evaluation"),
    (LRED,    RED,    "Key result"),
]
lx, ly = 0.3, 1.5
ax.text(lx, ly + 0.55, "Legend:", fontsize=8, fontweight='bold', color=DARK)
for i, (fc, ec, label) in enumerate(legend_items):
    bx = lx + i * 3.3
    rect = FancyBboxPatch((bx, ly), 0.4, 0.35,
                          boxstyle="round,pad=0.05",
                          facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(bx + 0.55, ly + 0.175, label,
            fontsize=7.5, va='center', color=DARK)

# data flow label on top
ax.text(9.0, 10.2,
        "Input: real 300s sensor window  +  scenario label  →  Output: 180s synthetic trajectory",
        ha='center', fontsize=9, color=GRAY, style='italic')

plt.tight_layout(pad=0.5)
fig.savefig(OUT / "block_diagram.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"Saved: {OUT}/block_diagram.png")
