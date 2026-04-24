"""
dashboard_schematic.py — Static mockup of the dashboard UI layout.
Saves a single PNG showing how both pages look.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

BLUE   = "#2196F3"
ORANGE = "#FF5722"
PINK   = "#E91E63"
PURPLE = "#9C27B0"
DARK   = "#1E1E2E"
PANEL  = "#2A2A3E"
TEXT   = "#FFFFFF"
SUBTEXT= "#AAAACC"
GREEN  = "#4CAF50"

def rounded_box(ax, x, y, w, h, color, alpha=1.0, radius=0.02, label=None,
                fontsize=9, text_color=TEXT, bold=False):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0",
                         facecolor=color, edgecolor='none', alpha=alpha,
                         transform=ax.transAxes, clip_on=False)
    ax.add_patch(box)
    if label:
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, label,
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight=weight)

def fake_line(ax, x0, x1, y_base, color, noise=0.03, lw=1.5, alpha=1.0):
    t = np.linspace(x0, x1, 80)
    y = y_base + np.cumsum(np.random.normal(0, noise/10, 80))
    y = y_base + 0.6*(y - y.mean())/(y.std()+1e-6)*noise
    ax.plot(t, y, color=color, lw=lw, alpha=alpha, transform=ax.transAxes)

np.random.seed(7)

fig = plt.figure(figsize=(18, 11), facecolor=DARK)

# ── Title bar ─────────────────────────────────────────────────────────────────
title_ax = fig.add_axes([0, 0.93, 1, 0.07], facecolor=PANEL)
title_ax.set_xlim(0,1); title_ax.set_ylim(0,1); title_ax.axis('off')
title_ax.text(0.5, 0.55,
    "HAI Digital Twin — Conditional Generative GRU Surrogate",
    ha='center', va='center', fontsize=14, color=TEXT, fontweight='bold')
title_ax.text(0.5, 0.15,
    "Cyber-Physical Attack Simulation for Industrial Control Systems",
    ha='center', va='center', fontsize=9, color=SUBTEXT)

# ── Page tabs ─────────────────────────────────────────────────────────────────
tab_ax = fig.add_axes([0, 0.88, 1, 0.05], facecolor=DARK)
tab_ax.set_xlim(0,1); tab_ax.set_ylim(0,1); tab_ax.axis('off')
rounded_box(tab_ax, 0.01, 0.1, 0.2, 0.8, BLUE, label="Page 1 — Simulate", bold=True, fontsize=10)
rounded_box(tab_ax, 0.22, 0.1, 0.22, 0.8, PANEL, label="Page 2 — Generate Synthetic", fontsize=10)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SIMULATE
# ══════════════════════════════════════════════════════════════════════════════
p1 = fig.add_axes([0, 0, 0.48, 0.88], facecolor=DARK)
p1.set_xlim(0,1); p1.set_ylim(0,1); p1.axis('off')

p1.text(0.5, 0.97, "Page 1 — Simulate", ha='center', va='top',
        fontsize=12, color=TEXT, fontweight='bold')
p1.text(0.5, 0.93, "What would the plant look like under a given attack scenario?",
        ha='center', va='top', fontsize=8, color=SUBTEXT)

# ── Sidebar controls ──────────────────────────────────────────────────────────
rounded_box(p1, 0.02, 0.55, 0.22, 0.35, PANEL, alpha=1.0)
p1.text(0.13, 0.895, "Controls", ha='center', fontsize=9,
        color=TEXT, fontweight='bold', transform=p1.transAxes)

p1.text(0.04, 0.86, "Scenario", fontsize=7, color=SUBTEXT, transform=p1.transAxes)
rounded_box(p1, 0.04, 0.80, 0.18, 0.055, "#3A3A5E",
            label="AE_no  ▼", fontsize=8)

p1.text(0.04, 0.79, "Seed Window", fontsize=7, color=SUBTEXT, transform=p1.transAxes)
rounded_box(p1, 0.04, 0.73, 0.18, 0.055, "#3A3A5E",
            label="Window #142  ◀ ▶", fontsize=7)

p1.text(0.04, 0.72, "Model", fontsize=7, color=SUBTEXT, transform=p1.transAxes)
rounded_box(p1, 0.04, 0.66, 0.18, 0.055, "#3A3A5E",
            label="gru_scenario_haiend", fontsize=6)

rounded_box(p1, 0.04, 0.575, 0.18, 0.065, PURPLE,
            label="▶  Generate", fontsize=9, bold=True)

# ── PV charts 2×3 grid ───────────────────────────────────────────────────────
pv_names = ["P1_PIT01", "P1_LIT01", "P1_FT03Z", "P1_TIT01", "P1_TIT03"]
colors_pv = [BLUE, GREEN, ORANGE, PINK, PURPLE]

positions = [
    (0.27, 0.60), (0.53, 0.60), (0.77, 0.60),
    (0.27, 0.30), (0.53, 0.30),
]
w, h = 0.21, 0.27

for k, (px, py) in enumerate(positions):
    sub = fig.add_axes([px*0.48, py*0.88, w*0.48, h*0.88], facecolor=PANEL)
    sub.set_xlim(0,1); sub.set_ylim(-0.5, 0.5); sub.axis('off')

    # actual
    t = np.linspace(0, 1, 80)
    y_real  = 0.1*np.sin(2*np.pi*t) + np.cumsum(np.random.normal(0,0.008,80))
    y_real  -= y_real.mean()
    y_pred  = y_real + np.random.normal(0, 0.04, 80)

    sub.plot(t, y_real, color='white', lw=1.3, label='Actual')
    sub.plot(t, y_pred, color=colors_pv[k], lw=1.3, linestyle='--', label='Predicted')
    sub.fill_between(t, y_real, y_pred, alpha=0.2, color=colors_pv[k])
    sub.set_title(pv_names[k], fontsize=7, color=TEXT, pad=2)
    if k == 0:
        sub.legend(fontsize=5, loc='upper right',
                   labelcolor=TEXT, facecolor=PANEL, edgecolor='none')

# legend label bottom-left of page 1
p1.text(0.27, 0.26, "— Actual    -- Predicted    shading = residual",
        fontsize=7, color=SUBTEXT, transform=p1.transAxes)

# ── Divider ───────────────────────────────────────────────────────────────────
div = fig.add_axes([0.485, 0, 0.003, 0.88], facecolor="#444466")
div.axis('off')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — GENERATE SYNTHETIC
# ══════════════════════════════════════════════════════════════════════════════
p2 = fig.add_axes([0.49, 0, 0.51, 0.88], facecolor=DARK)
p2.set_xlim(0,1); p2.set_ylim(0,1); p2.axis('off')

p2.text(0.5, 0.97, "Page 2 — Generate Synthetic Data", ha='center', va='top',
        fontsize=12, color=TEXT, fontweight='bold')
p2.text(0.5, 0.93, "Generate realistic attack trajectories conditioned on scenario label",
        ha='center', va='top', fontsize=8, color=SUBTEXT)

# ── Controls row ──────────────────────────────────────────────────────────────
rounded_box(p2, 0.02, 0.80, 0.96, 0.10, PANEL)

p2.text(0.05, 0.875, "Scenario", fontsize=7, color=SUBTEXT, transform=p2.transAxes)
rounded_box(p2, 0.05, 0.815, 0.17, 0.052, "#3A3A5E", label="AP_no  ▼", fontsize=8)

p2.text(0.26, 0.875, "N synthetic windows", fontsize=7, color=SUBTEXT, transform=p2.transAxes)
rounded_box(p2, 0.26, 0.815, 0.25, 0.052, "#3A3A5E",
            label="━━━━●━━━━━━━━━  200", fontsize=7)

rounded_box(p2, 0.55, 0.815, 0.18, 0.052, ORANGE,
            label="▶  Generate", fontsize=9, bold=True)

# stat cards
rounded_box(p2, 0.75, 0.815, 0.10, 0.052, "#1A3A1A",
            label="F1 = 0.95", fontsize=8, text_color=GREEN, bold=True)
rounded_box(p2, 0.87, 0.815, 0.10, 0.052, "#3A1A1A",
            label="200 gen.", fontsize=8, text_color=ORANGE)

# ── Main chart: overlay of synthetic samples ──────────────────────────────────
p2.text(0.02, 0.775, "Generated Trajectories — P1_PIT01", fontsize=8,
        color=TEXT, fontweight='bold', transform=p2.transAxes)
p2.text(0.02, 0.755, "8 individual synthetic samples  |  synthetic mean  |  real mean",
        fontsize=7, color=SUBTEXT, transform=p2.transAxes)

main_chart = fig.add_axes([0.49 + 0.02*0.51, 0.38*0.88, 0.56*0.51, 0.34*0.88],
                           facecolor=PANEL)
main_chart.set_xlim(0, 180)
main_chart.set_ylim(-1.5, 1.5)
main_chart.set_facecolor(PANEL)
main_chart.tick_params(colors=SUBTEXT, labelsize=6)
for spine in main_chart.spines.values():
    spine.set_edgecolor("#444466")

t = np.linspace(0, 180, 180)
for _ in range(8):
    y = 0.3*np.sin(2*np.pi*t/60 + np.random.uniform(0,2)) \
        + np.cumsum(np.random.normal(0, 0.015, 180))
    y -= y.mean()
    main_chart.plot(t, y, color=ORANGE, lw=0.7, alpha=0.25)

y_synth_mean = 0.25*np.sin(2*np.pi*t/60)
y_real_mean  = 0.2*np.sin(2*np.pi*t/60 + 0.3) + 0.05
main_chart.plot(t, y_synth_mean, color=ORANGE, lw=2.0, label='Synthetic mean')
main_chart.plot(t, y_real_mean,  color='white', lw=1.8, linestyle='--', label='Real mean')
main_chart.legend(fontsize=6, facecolor=PANEL, edgecolor='none', labelcolor=TEXT)
main_chart.set_xlabel("t (s)", fontsize=7, color=SUBTEXT)
main_chart.set_ylabel("Scaled value", fontsize=7, color=SUBTEXT)

# ── Small per-PV grid (4 remaining PVs) ───────────────────────────────────────
p2.text(0.02, 0.365, "All 5 Process Variables", fontsize=8,
        color=TEXT, fontweight='bold', transform=p2.transAxes)

small_pvs  = ["P1_LIT01", "P1_FT03Z", "P1_TIT01", "P1_TIT03"]
small_cols = [GREEN, BLUE, PINK, PURPLE]
xs = [0.02, 0.27, 0.52, 0.76]

for k, (pv, col, xpos) in enumerate(zip(small_pvs, small_cols, xs)):
    sa = fig.add_axes([0.49 + xpos*0.51, 0.05*0.88, 0.215*0.51, 0.265*0.88],
                      facecolor=PANEL)
    sa.set_xlim(0, 180); sa.set_ylim(-1.2, 1.2)
    sa.set_facecolor(PANEL)
    sa.tick_params(colors=SUBTEXT, labelsize=5)
    for spine in sa.spines.values():
        spine.set_edgecolor("#444466")

    t2 = np.linspace(0, 180, 180)
    for _ in range(5):
        y = 0.3*np.sin(2*np.pi*t2/50 + np.random.uniform(0,3)) \
            + np.cumsum(np.random.normal(0, 0.012, 180))
        y -= y.mean()
        sa.plot(t2, y, color=col, lw=0.6, alpha=0.3)
    y_sm = 0.25*np.sin(2*np.pi*t2/50)
    y_rm = 0.2*np.sin(2*np.pi*t2/50 + 0.2)
    sa.plot(t2, y_sm, color=col, lw=1.5)
    sa.plot(t2, y_rm, color='white', lw=1.2, linestyle='--')
    sa.set_title(pv, fontsize=6, color=TEXT, pad=2)
    sa.set_xlabel("t (s)", fontsize=5, color=SUBTEXT)

fig.savefig(OUT / "dashboard_schematic.png", dpi=150, bbox_inches='tight',
            facecolor=DARK)
plt.close(fig)
print(f"Saved: {OUT}/dashboard_schematic.png")
