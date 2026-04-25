"""
components.py — tiny reusable UI pieces for the dashboard.
Keep each function under ~40 lines; bigger panels belong in tabs/.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ── palette (dark = default; constants kept for backwards compat) ───────

C_OK    = "#10b981"   # green
C_WARN  = "#f59e0b"   # amber
C_FAIL  = "#ef4444"   # red
C_BG    = "#0a1122"
C_PANEL = "#131b35"
C_TEXT  = "#e5e7eb"
C_MUTED = "#94a3b8"
C_ACCENT = "#60a5fa"


# ── CSS (loaded once from app.py) ────────────────────────────────────────

DARK_CSS = f"""
<style>
  .stApp {{ background: {C_BG}; color: {C_TEXT}; }}
  h1, h2, h3 {{ color: #c084fc; }}
  .pm-header  {{ background: {C_PANEL}; border-radius: 10px; padding: 12px 18px;
                 margin-bottom: 14px; border: 1px solid rgba(255,255,255,0.05); }}
  .pm-metric  {{ background: rgba(0,0,0,0.25); border-radius: 8px;
                 padding: 10px 14px; border: 1px solid rgba(255,255,255,0.05); }}
  .pm-metric-label {{ font-size: 0.72rem; color: {C_MUTED};
                      text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }}
  .pm-metric-value {{ font-size: 1.55rem; color: {C_TEXT}; font-weight: 800; }}
  .pm-pill    {{ display: inline-block; padding: 2px 10px; border-radius: 999px;
                 font-size: 0.72rem; font-weight: 700; letter-spacing: 1px; }}
  .pm-pill-ok   {{ background: rgba(16,185,129,0.12); color: {C_OK};  border: 1px solid {C_OK}; }}
  .pm-pill-warn {{ background: rgba(245,158,11,0.12); color: {C_WARN}; border: 1px solid {C_WARN}; }}
  .pm-pill-fail {{ background: rgba(239,68,68,0.12); color: {C_FAIL}; border: 1px solid {C_FAIL}; }}
  .pm-card    {{ background: {C_PANEL}; border-radius: 10px; padding: 14px 18px;
                 border: 1px solid rgba(255,255,255,0.05); margin-bottom: 12px; }}
  .pm-card-title {{ font-size: 0.85rem; color: {C_ACCENT}; text-transform: uppercase;
                    letter-spacing: 1px; font-weight: 700; margin-bottom: 8px; }}
  .pm-limit {{ background: rgba(245,158,11,0.05); border-left: 3px solid {C_WARN};
               padding: 10px 14px; border-radius: 0 4px 4px 0; margin-bottom: 8px; }}
  .pm-limit-id {{ color: {C_WARN}; font-weight: 700; font-size: 0.75rem; }}
  .pm-limit-title {{ color: {C_TEXT}; font-weight: 700; font-size: 0.9rem; margin: 2px 0 4px 0; }}
  .pm-limit-body {{ color: {C_MUTED}; font-size: 0.82rem; line-height: 1.4; }}
</style>
"""

# Light-mode overrides — applied AFTER DARK_CSS so they win via cascade.
# Inline-style colours from tabs (e.g. style='color:#e5e7eb') get overridden
# via attribute-substring selectors.
LIGHT_CSS = """
<style>
  /* Page chrome */
  .stApp { background: #f8fafc !important; color: #1e293b !important; }
  [data-testid="stSidebar"] { background: #eef2f7 !important; }
  [data-testid="stSidebar"] * { color: #1e293b !important; }
  h1, h2, h3 { color: #6d28d9 !important; }

  /* Cards / panels */
  .pm-header { background: #ffffff !important; border: 1px solid #e2e8f0 !important; }
  .pm-metric { background: #f1f5f9 !important; border: 1px solid #e2e8f0 !important; }
  .pm-metric-label { color: #64748b !important; }
  .pm-metric-value { color: #0f172a !important; }
  .pm-card    { background: #ffffff !important; border: 1px solid #e2e8f0 !important; }
  .pm-card-title { color: #2563eb !important; }
  .pm-limit { background: rgba(245,158,11,0.08) !important; }
  .pm-limit-title { color: #1e293b !important; }
  .pm-limit-body  { color: #475569 !important; }

  /* Tab labels */
  .stTabs [data-baseweb="tab"] { color: #475569 !important; }
  .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #2563eb !important; }

  /* Override common inline-style dark colours */
  .stApp [style*="color:#e5e7eb"], .stApp [style*="color: #e5e7eb"] { color: #1e293b !important; }
  .stApp [style*="color:#94a3b8"], .stApp [style*="color: #94a3b8"] { color: #64748b !important; }
  .stApp [style*="color:#cbd5e1"], .stApp [style*="color: #cbd5e1"] { color: #475569 !important; }
  .stApp [style*="background:rgba(255,255,255,0.03)"] { background: rgba(0,0,0,0.03) !important; }
  .stApp [style*="background:rgba(255,255,255,0.05)"] { background: rgba(0,0,0,0.05) !important; }
  .stApp [style*="background:rgba(0,0,0,0.25)"] { background: #f1f5f9 !important; }
  .stApp [style*="background:rgba(96,165,250,0.06)"] { background: rgba(37,99,235,0.08) !important; }
  .stApp [style*="background:rgba(96,165,250,0.08)"] { background: rgba(37,99,235,0.10) !important; }

  /* Code spans */
  .stApp code { background: #e2e8f0 !important; color: #0f172a !important; }

  /* Tables (dataframe) */
  .stApp [data-testid="stDataFrame"] { color: #0f172a !important; }
</style>
"""


def apply_css() -> None:
    """Inject base dark CSS, then light overrides if user picked light theme."""
    theme = st.session_state.get("ui_theme", "dark")
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    if theme == "light":
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)


def render_theme_toggle() -> None:
    """Sidebar widget to switch dark/light mode. Call from app.py inside
    `with st.sidebar:` (above the chatbot) so it appears at the very top."""
    if "ui_theme" not in st.session_state:
        st.session_state["ui_theme"] = "dark"
    choice = st.radio(
        "Theme",
        options=["🌙 Dark", "☀️ Light"],
        index=0 if st.session_state["ui_theme"] == "dark" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="ui_theme_radio",
    )
    new_theme = "dark" if choice.startswith("🌙") else "light"
    if new_theme != st.session_state["ui_theme"]:
        st.session_state["ui_theme"] = new_theme
        st.rerun()


# ── Metric card ──────────────────────────────────────────────────────────

def metric_card(label: str, value: str, hint: Optional[str] = None) -> str:
    hint_html = f"<div style='font-size:0.65rem;color:{C_MUTED};margin-top:4px'>{hint}</div>" if hint else ""
    return (
        f"<div class='pm-metric'>"
        f"<div class='pm-metric-label'>{label}</div>"
        f"<div class='pm-metric-value'>{value}</div>{hint_html}"
        f"</div>"
    )


# ── Pass/Fail pill ───────────────────────────────────────────────────────

def pass_fail_pill(status: str) -> str:
    cls = {"PASS": "pm-pill-ok", "WARN": "pm-pill-warn", "FAIL": "pm-pill-fail"}.get(status, "pm-pill-warn")
    return f"<span class='pm-pill {cls}'>{status}</span>"


# ── Horizontal bar chart (one bar per loop, pass/fail coloured) ──────────

def ks_fidelity_bars(ks_map: dict, thresholds: Tuple[float, float] = (0.10, 0.20)) -> go.Figure:
    loops = list(ks_map.keys())
    values = [ks_map[l]["ks"] for l in loops]
    colors = [
        C_OK if v < thresholds[0] else C_WARN if v < thresholds[1] else C_FAIL
        for v in values
    ]
    fig = go.Figure(go.Bar(
        x=values, y=loops, orientation="h",
        marker=dict(color=colors, line=dict(width=1, color="rgba(255,255,255,0.15)")),
        text=[f"{v:.3f}" for v in values], textposition="outside",
        textfont=dict(color=C_TEXT, size=12),
    ))
    fig.update_layout(
        template="plotly_dark",
        title=dict(text="KS distributional fidelity per loop (lower = better)",
                   font=dict(color=C_ACCENT, size=14)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, max(values) * 1.2 + 0.05], title=None, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title=None, autorange="reversed"),
        height=220, margin=dict(l=40, r=60, t=40, b=20),
    )
    # Vertical reference lines for the thresholds
    fig.add_vline(x=thresholds[0], line=dict(color=C_OK, dash="dot", width=1))
    fig.add_vline(x=thresholds[1], line=dict(color=C_FAIL, dash="dot", width=1))
    return fig


# ── Radar chart (per-loop: fidelity / speed / coupling / coverage / sensitivity) ──

def radar_chart(axes: Sequence[str], values: Sequence[float],
                compare_values: Optional[Sequence[float]] = None,
                label: str = "selected",
                compare_label: str = "mean of others") -> go.Figure:
    theta = list(axes) + [axes[0]]
    r1 = list(values) + [values[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r1, theta=theta, fill="toself", name=label,
        line=dict(color=C_ACCENT, width=2),
        fillcolor="rgba(96,165,250,0.25)",
    ))
    if compare_values is not None:
        r2 = list(compare_values) + [compare_values[0]]
        fig.add_trace(go.Scatterpolar(
            r=r2, theta=theta, fill="toself", name=compare_label,
            line=dict(color=C_MUTED, width=1, dash="dash"),
            fillcolor="rgba(148,163,184,0.1)",
        ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        ),
        height=320, margin=dict(l=30, r=30, t=30, b=30),
        showlegend=True, legend=dict(orientation="h", y=-0.05),
    )
    return fig


# ── Heatmap (5×5 loop coupling) ──────────────────────────────────────────

def coupling_heatmap(matrix: np.ndarray, loops: Sequence[str]) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix, x=list(loops), y=list(loops),
        colorscale="Blues", zmin=0, zmax=float(max(matrix.max(), 1e-6)),
        text=[[f"{v:.2f}" if v > 0 else "" for v in row] for row in matrix],
        texttemplate="%{text}", textfont=dict(color=C_TEXT, size=12),
        colorbar=dict(title="coupling"),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Cross-loop coupling (source → target)",
                   font=dict(color=C_ACCENT, size=13)),
        xaxis=dict(title="target loop"), yaxis=dict(title="source loop", autorange="reversed"),
        height=340, margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


# ── Mini SP/CV/PV chart for the attack gallery ───────────────────────────

def mini_signal_chart(series: Sequence[float], color: str = C_ACCENT, height: int = 60) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=list(series), mode="lines",
        line=dict(color=color, width=1.4),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


# ── Precision-recall bars (for Guardian card) ────────────────────────────

def classifier_metric_bars(f1: float, precision: float, recall: float,
                           auroc: Optional[float] = None) -> go.Figure:
    metrics = ["F1", "Precision", "Recall"]
    values = [f1, precision, recall]
    if auroc is not None:
        metrics.append("AUROC")
        values.append(auroc)
    fig = go.Figure(go.Bar(
        x=metrics, y=values,
        marker=dict(color=[C_ACCENT, C_OK, C_WARN, "#c084fc"][:len(metrics)],
                    line=dict(color="rgba(255,255,255,0.2)", width=1)),
        text=[f"{v:.3f}" for v in values], textposition="outside",
        textfont=dict(color=C_TEXT, size=12),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.05)"),
        height=220, margin=dict(l=30, r=10, t=10, b=30), showlegend=False,
    )
    return fig
