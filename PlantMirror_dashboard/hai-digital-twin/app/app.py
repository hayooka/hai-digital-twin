"""
app.py — ICS Digital Twin dashboard (SCADA operations view).

Two pages:
  LIVE DETECTION       — stream sensor data, detect anomalies, classify attack type,
                         compare predicted class vs ground truth (no leakage).
  SYNTHETIC GENERATION — pick a class label + setpoints, generate a synthetic attack,
                         analyse how close it is to the real training distribution.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_js_eval import streamlit_js_eval

from twin_core import (
    INPUT_LEN,
    PV_COLS,
    SCENARIO_MAPPING,
    TARGET_LEN,
    default_paths,
    load_bundle,
    load_replay,
)
from twin_runtime import ALERT_GAP_SEC, TwinRuntime, TwinSnapshot
import generative

st.set_page_config(
    page_title="ICS · DIGITAL TWIN",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── SCADA stylesheet (100/100 Premium UI) ───────────────────────────────────

SCADA_CSS = """
<style>
  /* Floating chat-bubble styling of the sidebar collapse control */
  @keyframes chatGlow {
    0%,100% { box-shadow: 0 0 0 0 rgba(168,85,247,0.6), 0 10px 28px rgba(168,85,247,0.6); }
    50%     { box-shadow: 0 0 0 14px rgba(168,85,247,0.0), 0 10px 28px rgba(168,85,247,0.6); }
  }
  #plantmirror-chat-fab {
    position: fixed;
    bottom: 28px;
    right: 28px;
    width: 78px;
    height: 78px;
    border-radius: 50%;
    background: linear-gradient(135deg, #c084fc 0%, #7c3aed 100%);
    border: 3px solid #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 999999;
    cursor: pointer;
    animation: chatGlow 2s infinite;
    font-size: 36px;
    user-select: none;
  }
  #plantmirror-chat-fab:hover { transform: scale(1.1); }

  section[data-testid="stSidebar"] {
    min-width: 380px !important;
    width: 380px !important;
  }
  @keyframes chatFlash {
    0%,100% { box-shadow: inset 0 0 0 0 rgba(168,85,247,0); }
    50%     { box-shadow: inset 0 0 0 6px rgba(168,85,247,0.8); }
  }
  section[data-testid="stSidebar"].chat-flash {
    animation: chatFlash 0.6s 3;
  }
  [data-testid="stSidebar"] { min-width: 380px !important; }

  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;700;800&display=swap');

  :root {
    --bg-0: #050914;
    --bg-1: #0a1122;
    --bg-2: #121c36;
    --border: #1e2d4a;
    --border-glow: rgba(79, 195, 247, 0.15);
    --text-0: #f8fafc;
    --text-1: #cbd5e1;
    --muted: #64748b;
    --accent: #00d2ff;
    --accent-glow: rgba(0, 210, 255, 0.4);
    --ok: #10b981;
    --warn: #f59e0b;
    --alert: #ef4444;
    --alert-glow: rgba(239, 68, 68, 0.4);
    --mono: 'JetBrains Mono', ui-monospace, 'Consolas', monospace;
    --sans: 'Inter', system-ui, sans-serif;
  }

  /* Core App Overrides */
  [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
  /* Hide Streamlit chrome (deploy button, hamburger, status bar) */
  #MainMenu, header [data-testid="stToolbar"], [data-testid="stDecoration"],
  [data-testid="stStatusWidget"], [data-testid="stHeader"] { display: none !important; visibility: hidden !important; }
  .stApp { background: var(--bg-0); font-family: var(--sans); }
  .block-container { max-width: 100% !important; padding: 1.5rem 2rem !important; }

  /* Header bar */
  .dt-header {
    display: flex; justify-content: space-between; align-items: center;
    background: linear-gradient(135deg, var(--bg-1) 0%, #0d1629 100%);
    border: 1px solid var(--border); 
    border-bottom: 2px solid var(--accent);
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5), 0 0 15px var(--border-glow);
    padding: 20px 30px; margin-bottom: 25px;
    border-radius: 12px;
  }
  .dt-title {
    font-family: var(--mono); font-weight: 800;
    font-size: 1.8rem; letter-spacing: 5px; color: var(--text-0);
    text-shadow: 0 0 15px var(--accent-glow);
    display: flex; align-items: center; gap: 15px;
  }
  .dt-title span { color: var(--accent); }
  .dt-right { display: flex; gap: 25px; align-items: center; }
  .dt-meta {
    font-family: var(--sans); font-size: 0.8rem; color: var(--text-1);
    text-align: right; letter-spacing: 1px; font-weight: 600;
  }
  .dt-meta .val { color: var(--accent); font-weight: 800; font-size: 1.2rem; }

  /* Status Pills */
  .pill {
    font-family: var(--mono); font-weight: 800; letter-spacing: 1.5px;
    font-size: 0.85rem; padding: 10px 22px; border-radius: 6px;
    text-transform: uppercase; box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .pill-ok    { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #059669; text-shadow: 0 0 8px rgba(52, 211, 153, 0.5); }
  .pill-idle  { background: rgba(100, 116, 139, 0.15); color: #94a3b8; border: 1px solid #475569; }
  .pill-alert {
    background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid #dc2626;
    text-shadow: 0 0 8px rgba(252, 165, 165, 0.6);
    animation: pulse-alert 2s infinite;
  }
  @keyframes pulse-alert {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
    70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
  }

  /* Premium Panels */
  .panel {
    background: rgba(10, 17, 34, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px; margin-bottom: 16px;
    box-shadow: inset 0 0 25px rgba(255,255,255,0.02), 0 8px 30px rgba(0,0,0,0.3);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
  }
  .panel:hover { border-color: rgba(0, 210, 255, 0.4); box-shadow: inset 0 0 25px rgba(0, 210, 255, 0.05), 0 8px 30px rgba(0,0,0,0.4); }

  .panel-title {
    font-family: var(--sans); font-size: 0.85rem; font-weight: 800;
    letter-spacing: 2px; color: var(--text-0);
    text-transform: uppercase; padding: 4px 0 12px 14px;
    border-left: 4px solid var(--accent); margin-bottom: 12px;
    display: flex; align-items: center; gap: 10px;
    text-shadow: 0 0 10px rgba(255,255,255,0.1);
  }

  /* Metric Grid */
  .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .metric-cell {
    background: linear-gradient(180deg, rgba(18, 28, 54, 0.8) 0%, rgba(10, 17, 34, 0.9) 100%);
    padding: 16px; border-radius: 6px;
    border: 1px solid var(--border);
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .metric-cell:hover { transform: translateY(-3px); border-color: var(--accent); }
  .metric-value {
    font-family: var(--mono); font-size: 1.6rem; font-weight: 800;
    color: var(--text-0); line-height: 1.1; text-shadow: 0 0 15px rgba(255,255,255,0.15);
  }
  .metric-label {
    font-family: var(--sans); font-size: 0.7rem; color: var(--muted);
    letter-spacing: 1.5px; text-transform: uppercase; margin-top: 8px; font-weight: 700;
  }

  /* Tabs Styling */
  div[data-baseweb="tab-list"] {
    gap: 12px !important; border-bottom: 1px solid var(--border); padding-bottom: 0;
  }
  button[data-baseweb="tab"] {
    font-family: var(--mono) !important; font-weight: 800 !important;
    letter-spacing: 2.5px !important; font-size: 0.9rem !important;
    text-transform: uppercase !important;
    padding: 14px 28px !important; background: transparent !important;
    border-radius: 6px 6px 0 0 !important;
    color: var(--text-1) !important;
    transition: all 0.3s ease !important;
  }
  button[data-baseweb="tab"]:hover {
    color: var(--text-0) !important; background: rgba(255,255,255,0.04) !important;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    background: linear-gradient(180deg, rgba(0, 210, 255, 0.15) 0%, transparent 100%) !important;
    border-bottom: 3px solid var(--accent) !important;
    text-shadow: 0 0 10px var(--accent-glow) !important;
  }

  /* Buttons */
  .stButton > button {
    font-family: var(--mono) !important; font-weight: 800 !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    font-size: 0.85rem !important; border-radius: 6px !important;
    background: var(--bg-2) !important; color: var(--text-0) !important;
    border: 1px solid var(--border) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    padding: 14px 24px !important;
  }
  .stButton > button:hover { 
    border-color: var(--accent) !important; color: var(--accent) !important; 
    box-shadow: 0 0 20px var(--border-glow) !important;
    transform: translateY(-2px);
  }

  /* Primary PLAY Button */
  .stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
    border: 1px solid #047857 !important; color: #ffffff !important;
    font-size: 1.05rem !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
  }
  .stButton > button[kind="primary"]:hover {
    transform: scale(1.03) translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.6) !important;
  }

  /* PAUSE Button is dynamically injected below */

  /* Dataframe & Select */
  div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
  div[data-baseweb="select"] > div { background-color: var(--bg-2) !important; border-color: var(--border) !important; color: var(--text-0) !important; }
  
  .caption-mono { font-family: var(--sans); font-size: 0.8rem; color: var(--muted); letter-spacing: 0.5px; line-height: 1.5; }
</style>
"""
st.markdown(SCADA_CSS, unsafe_allow_html=True)


# ── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Booting PyTorch GRU Core...")
def _load_bundle():
    paths = default_paths()
    return load_bundle(paths["ckpt_dir"], paths["split_dir"])



@st.cache_data(show_spinner="Loading Replay Data Matrix...")
def _load_replay(csv_path_str: str, _bundle_id: int):
    bundle = _load_bundle()
    return load_replay(Path(csv_path_str), bundle.scalers)


bundle = _load_bundle()
bundle.threshold = 0.02   # calibrated for live replay (Mode A vs Mode B mismatch)
paths = default_paths()

csv_choices = {p.stem: str(p) for p in paths["test_csvs"] if p.exists()}
if not csv_choices:
    st.error(f"FATAL: No test CSVs found under {paths['test_csvs'][0].parent}.")
    st.stop()
csv_name = next(iter(csv_choices))
src = _load_replay(csv_choices[csv_name], id(bundle))


# ── Runtime session state ───────────────────────────────────────────────────

def _get_or_make_runtime() -> TwinRuntime:
    key = f"rt::{csv_name}"
    if key not in st.session_state:
        st.session_state[key] = TwinRuntime(bundle, src)
    return st.session_state[key]

rt = _get_or_make_runtime()


# ── Responsive viewport sizing ──────────────────────────────────────────────

def _viewport_height() -> int:
    cached = st.session_state.get("viewport_h")
    if cached and cached > 400:
        return int(cached)
    vh = streamlit_js_eval(
        js_expressions="window.innerHeight",
        key="vh_probe",
        want_output=True,
    )
    if vh and vh > 400:
        st.session_state["viewport_h"] = int(vh)
        return int(vh)
    return 950

VH = _viewport_height()
LIVE_WORK_H = max(450, VH - 360)
H_RECON   = int(LIVE_WORK_H * 0.45)
H_OVERLAY = int(LIVE_WORK_H * 0.55)
H_GAUGE   = int(LIVE_WORK_H * 0.35)
H_ROOT    = int(LIVE_WORK_H * 0.30)
H_ALARM   = int(LIVE_WORK_H * 0.35)
H_TIMELINE = max(80, int(VH * 0.12))


# ── Plotly common theme (Premium) ───────────────────────────────────────────

PLOT_BG = "rgba(0,0,0,0)"
GRID = "rgba(255,255,255,0.05)"
TEXT = "#94a3b8"

def _apply_theme(fig: go.Figure, height: int = 260, title: str = "") -> go.Figure:
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Inter, sans-serif", color=TEXT, size=11),
        margin=dict(l=45, r=15, t=40 if title else 10, b=30),
        height=height, title=dict(text=title, x=0.005, xanchor="left",
                                   font=dict(color="#00d2ff", size=13, family="JetBrains Mono, monospace")),
        hovermode="x unified",
        uirevision="live",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, color=TEXT, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, color=TEXT, tickfont=dict(size=10))
    return fig


# ── Header ──────────────────────────────────────────────────────────────────

def _live_score(rt: TwinRuntime) -> Optional[float]:
    """The authentic 180-s batched window MSE — same quantity the detector
    thresholds at 0.326. Refreshes every 5 sim-ticks (see twin_runtime.step)."""
    if not rt.is_ready:
        return None
    return rt.anomaly_score


def _render_header(rt: TwinRuntime) -> None:
    playing = st.session_state.get("rt_playing", False)
    thr = float(bundle.threshold)
    score = _live_score(rt)
    attacking = score is not None and score > thr

    if score is None:
        health = 100.0
    else:
        ratio = min(score / thr, 1.0)
        health = 100.0 * (1.0 - ratio) ** 0.3
    if health >= 75:
        health_color = "#10b981"  # green
    elif health >= 40:
        health_color = "#f59e0b"  # amber
    else:
        health_color = "#ef4444"  # red

    if not playing:
        pill_html = '<span class="pill pill-idle">⏸ SYSTEM PAUSED</span>'
    elif attacking:
        pill_html = '<span class="pill pill-alert">▲ ATTACK DETECTED</span>'
    else:
        pill_html = '<span class="pill pill-ok">● NORMAL OPERATION</span>'

    st.markdown(
        f"""
        <div class="dt-header">
          <div>
            <div class="dt-title">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00d2ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                </svg>
                ICS <span>·</span> DIGITAL TWIN
            </div>
          </div>
          <div class="dt-right">
            <div class="dt-meta" style="margin-right: 15px; border-right: 1px solid rgba(255,255,255,0.1); padding-right: 25px;">
              HEALTH INDEX<br><span class="val" style="color: {health_color}; font-size: 1.4rem; text-shadow: 0 0 10px {health_color}55;">{health:.0f}%</span>
            </div>
            <div class="dt-meta" style="margin-right: 15px; border-right: 1px solid rgba(255,255,255,0.1); padding-right: 25px;">
              SYSTEM ALERTS<br><span class="val" style="color: {'#ef4444' if len(rt.alerts) > 0 else '#00d2ff'};">{len(rt.alerts)}</span>
            </div>
            {pill_html}
          </div>
        </div>
        <div style="font-family: var(--mono); font-size: 0.7rem; color: #64748b;
                    letter-spacing: 1px; margin-top: -18px; margin-bottom: 18px;
                    padding: 6px 12px; background: rgba(0, 210, 255, 0.04);
                    border: 1px solid rgba(0, 210, 255, 0.15); border-radius: 4px;">
            ℹ&nbsp; Live detector is pinned to scenario 0 = <b style="color:#94a3b8;">normal</b>
            baseline: the score measures deviation from what the twin expects under
            normal operation. Use the <b style="color:#94a3b8;">Generative</b> tab to
            swap embeddings.
        </div>
        """,
        unsafe_allow_html=True,
    )

if "active_snapshot" not in st.session_state:
    st.session_state["active_snapshot"] = None

# ── 2-page layout ────────────────────────────────────────────────────────────

tab_live, tab_gen = st.tabs(["LIVE DETECTION", "SYNTHETIC GENERATION"])


# ── Live-panel helpers ──────────────────────────────────────────────────────

def _panel_title(text: str) -> None:
    st.markdown(f'<div class="panel-title">{text}</div>', unsafe_allow_html=True)


def _attack_segments_in_buffer(rt: TwinRuntime) -> list[tuple[int, int]]:
    arrs = rt.rolling_arrays()
    if arrs["cursor_times"].size == 0:
        return []
    df = rt.src.df_raw
    if "label" not in df.columns:
        return []
    cursor_times = arrs["cursor_times"]
    sim_times = arrs["sim_times"]
    labels = df["label"].to_numpy()[cursor_times]
    segs: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i, lab in enumerate(labels):
        if lab > 0 and not in_seg:
            seg_start = int(sim_times[i]); in_seg = True
        elif lab == 0 and in_seg:
            segs.append((seg_start, int(sim_times[i]))); in_seg = False
    if in_seg:
        segs.append((seg_start, int(sim_times[-1]) + 1))
    return segs


def _panel_reconstruction_error(rt: TwinRuntime) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    _panel_title("Digital-Twin Reconstruction Error (Step MSE)")
    arrs = rt.rolling_arrays()
    if arrs["step_mse"].shape[0] == 0:
        st.markdown('<div class="caption-mono" style="padding: 40px; text-align: center;">Press ▶ PLAY to initialize data stream.</div></div>', unsafe_allow_html=True)
        return
    x = arrs["sim_times"]
    fig = go.Figure()
    
    for s, e in _attack_segments_in_buffer(rt):
        fig.add_vrect(
            x0=s, x1=e,
            fillcolor="rgba(239,68,68,0.22)", line_width=0, layer="below",
            annotation_text="ATTACK WINDOW", annotation_position="top left",
            annotation_font=dict(color="#fca5a5", size=10, family="JetBrains Mono"),
        )

    fig.add_trace(go.Scatter(
        x=x, y=arrs["step_mse"], mode="lines",
        line=dict(color="#00d2ff", width=2.5), name="Reconstruction Error",
        hovertemplate="t=%{x}s<br>MSE=%{y:.5f}<extra></extra>",
        fill="tozeroy", fillcolor="rgba(0, 210, 255, 0.08)"
    ))
    fig.add_hline(
        y=float(bundle.threshold),
        line=dict(color="#f59e0b", dash="dash", width=2),
        annotation_text=f"Detection Threshold ({bundle.threshold:.3f})",
        annotation_position="top right",
        annotation_font=dict(color="#f59e0b", size=11, family="JetBrains Mono"),
    )
    _apply_theme(fig, height=H_RECON)
    fig.update_xaxes(title="Simulation Clock (s)")
    fig.update_yaxes(title="MSE (Scaled)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


def _panel_sensor_overlay(rt: TwinRuntime) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    _panel_title("Key PV Readings — Twin vs Actual (Normalized 0–1)")
    arrs = rt.rolling_arrays()
    if arrs["actual_pv"].shape[0] == 0:
        st.markdown('</div>', unsafe_allow_html=True)
        return
    x = arrs["sim_times"]
    actual = arrs["actual_pv"]
    twin = arrs["twin_pv"]

    mn = np.minimum(actual.min(axis=0), twin.min(axis=0))
    mx = np.maximum(actual.max(axis=0), twin.max(axis=0))
    rng = np.where(mx - mn < 1e-6, 1.0, mx - mn)
    actual_n = (actual - mn) / rng
    twin_n = (twin - mn) / rng

    palette = ["#00d2ff", "#10b981", "#f59e0b", "#a855f7", "#ef4444"]
    fig = go.Figure()
    for s, e in _attack_segments_in_buffer(rt):
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(239,68,68,0.18)", line_width=0, layer="below")

    for i, pv in enumerate(PV_COLS):
        fig.add_trace(go.Scatter(
            x=x, y=actual_n[:, i], mode="lines",
            line=dict(color=palette[i], width=2),
            name=f"{pv} Actual", legendgroup=pv,
            hovertemplate=f"{pv} actual=%{{y:.3f}}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=twin_n[:, i], mode="lines",
            line=dict(color=palette[i], width=1.5, dash="dot"),
            name=f"{pv} Twin", legendgroup=pv, showlegend=False,
            hovertemplate=f"{pv} twin=%{{y:.3f}}<extra></extra>",
        ))
    _apply_theme(fig, height=H_OVERLAY)
    fig.update_xaxes(title="Simulation Clock (s)")
    fig.update_yaxes(title="Normalized Magnitude", range=[-0.05, 1.05])
    fig.update_layout(legend=dict(orientation="h", y=1.15, x=0, font=dict(size=11)))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


def _panel_gauge(rt: TwinRuntime) -> None:
    """Live Anomaly Score — circular gauge with tweened needle + snapshot button."""
    st.markdown('<div class="panel" style="display: flex; flex-direction: column; justify-content: center;">', unsafe_allow_html=True)
    _panel_title("Live Anomaly Score")
    score = _live_score(rt)
    thr = float(bundle.threshold)
    val = 0.0 if score is None else float(score)
    axis_max = max(thr * 1.6, val * 1.15, 1e-6)
    ratio = val / thr if thr > 0 else 0.0
    # Binary semantics: below threshold = normal (green), at/above = attack (red).
    bar_color = "#ef4444" if ratio >= 1.0 else "#10b981"

    gfig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number=dict(
            font=dict(family="JetBrains Mono", color=bar_color, size=34),
            valueformat=".5f",
        ),
        gauge=dict(
            axis=dict(range=[0, axis_max], tickcolor="#334155",
                      tickfont=dict(color="#64748b", size=9, family="JetBrains Mono"),
                      nticks=5),
            bar=dict(color=bar_color, thickness=0.28),
            bgcolor="rgba(255,255,255,0.03)",
            borderwidth=1,
            bordercolor="#1e2d4a",
            steps=[
                dict(range=[0, thr * 0.66], color="rgba(16,185,129,0.18)"),
                dict(range=[thr * 0.66, thr], color="rgba(245,158,11,0.22)"),
                dict(range=[thr, axis_max], color="rgba(239,68,68,0.22)"),
            ],
            threshold=dict(
                line=dict(color="#ffffff", width=3),
                thickness=0.85,
                value=thr,
            ),
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    gfig.update_layout(
        height=230, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        transition=dict(duration=600, easing="cubic-in-out"),
        uirevision="live-gauge",
    )
    st.plotly_chart(gfig, width="stretch", config={"displayModeBar": False})

    st.markdown(
        f'<div style="font-family: var(--sans); font-size: 0.7rem; color: #64748b; '
        f'letter-spacing: 1.5px; text-transform: uppercase; text-align:center; margin: -6px 0 10px 0;">'
        f'{ratio*100:.1f}% of threshold · thr = {thr:.3f}</div>',
        unsafe_allow_html=True,
    )

    # Snapshot = freeze the current twin state (h_plant, pv_twin, cursor) so the
    # Predictive / Generative / Assistive tabs analyse THIS moment even while
    # LIVE keeps running. Without it those tabs always read the latest live
    # cursor, which moves under your feet during investigation.
    has_snap = bool(st.session_state.get("active_snapshot"))
    snap_label = "📸 UPDATE SNAPSHOT" if has_snap else "📸 FREEZE THIS MOMENT FOR ANALYSIS"
    if st.button(snap_label, key="gauge_snap_btn", use_container_width=True,
                 help="Lock the twin's current state so Predictive / Generative / Assistive tabs can analyse this exact moment while LIVE keeps streaming."):
        if rt.is_ready:
            st.session_state["active_snapshot"] = rt.snapshot()
            st.toast("✅ Snapshot captured — sent to Predictive / Generative / Assistive.")
    if has_snap:
        st.markdown(
            '<div style="background: rgba(16,185,129,0.15); border: 1px solid #10b981; '
            'color: #10b981; padding: 6px; border-radius: 6px; text-align: center; '
            'font-family: var(--mono); font-size:0.7rem; font-weight:700; margin-top:6px;">'
            'SNAPSHOT ACTIVE</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


def _panel_root_cause(rt: TwinRuntime) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    _panel_title("Top PVs by Residual Contribution")
    arrs = rt.rolling_arrays()
    per_pv = arrs["per_pv_residual"]
    if per_pv.shape[0] == 0:
        st.markdown('<div class="caption-mono" style="text-align: center; padding: 20px;">Awaiting Stream Data...</div></div>', unsafe_allow_html=True)
        return
    window = per_pv[-min(TARGET_LEN, per_pv.shape[0]):]
    mean_per_pv = window.mean(axis=0)
    order = np.argsort(mean_per_pv)[::-1]
    
    palette = ["#00d2ff", "#10b981", "#f59e0b", "#a855f7", "#ef4444"]
    pv_color = {PV_COLS[i]: palette[i] for i in range(len(PV_COLS))}
    labels = [PV_COLS[i] for i in order]
    values = [float(mean_per_pv[i]) for i in order]
    colors = [pv_color[pv] for pv in labels]
    
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
        text=[f"{v:.5f}" for v in values], textposition="outside",
        textfont=dict(color="#f8fafc", size=11, family="JetBrains Mono"),
        hovertemplate="%{y}: %{x:.5f}<extra></extra>",
    ))
    _apply_theme(fig, height=H_ROOT)
    fig.update_yaxes(autorange="reversed", tickfont=dict(family="JetBrains Mono", size=12, color="#cbd5e1"))
    fig.update_xaxes(title=None, showticklabels=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


def _panel_alarm_log(rt: TwinRuntime) -> None:
    st.markdown('<div class="panel" style="display: flex; flex-direction: column;">', unsafe_allow_html=True)
    _panel_title(f"Alert Feed · Debounced (≥ {ALERT_GAP_SEC}s)")
    
    if st.session_state["active_snapshot"]:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            if st.button("📸 CAPTURE SYSTEM SNAPSHOT", help="Locks the current Twin state for deep dive analysis across tabs", use_container_width=True):
                if rt.is_ready:
                    st.session_state["active_snapshot"] = rt.snapshot()
                    st.toast("✅ Snapshot updated — deep-copy of twin state sent to Generative / Assistive tabs.")
        with c2:
            st.markdown('<div style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #10b981; padding: 10px; border-radius: 6px; text-align: center; font-family: var(--mono); font-weight: 700; height: 100%; display: flex; align-items: center; justify-content: center;">SNAPSHOT ACTIVE</div>', unsafe_allow_html=True)
    else:
        if st.button("📸 CAPTURE SYSTEM SNAPSHOT",
                     help="Deep-copy the twin state (h, pv_twin) so Generative / Assistive tabs can analyze it while the live feed keeps running.",
                     use_container_width=True):
            if rt.is_ready:
                st.session_state["active_snapshot"] = rt.snapshot()
                st.toast("✅ Snapshot captured! Proceed to Predictive or Generative tabs.")

    if not rt.alerts:
        st.markdown(
            '<div class="caption-mono" style="margin-top:20px;text-align:center;flex:1;">'
            'No alarms yet — press ▶ PLAY to start.</div></div>',
            unsafe_allow_html=True,
        )
        return

    # Latest alarm card — prominently shows predicted class vs ground truth
    latest = rt.alerts[-1]
    cls_color = {"Normal": "#10b981", "AP_no": "#f59e0b",
                 "AP_with": "#ef4444", "AE_no": "#a855f7"}.get(latest.predicted_class, "#94a3b8")
    st.markdown(
        f"""<div style="background:rgba({_hex_rgb(cls_color)},0.1);border:1px solid {cls_color};
        border-radius:6px;padding:12px 16px;margin-bottom:10px;">
        <div style="font-family:var(--mono);font-size:0.65rem;color:#64748b;letter-spacing:1px;">
            LATEST ALARM · t={latest.sim_clock}s · score {latest.score:.4f}
        </div>
        <div style="font-family:var(--mono);font-size:1.2rem;font-weight:700;color:{cls_color};margin:4px 0;">
            {latest.predicted_class}
        </div>
        <div style="font-family:var(--mono);font-size:0.7rem;color:#94a3b8;">
            culprit: {latest.top_pv}<br>
            ground truth: <span style="color:#cbd5e1;">{latest.ground_truth}</span>
        </div></div>""",
        unsafe_allow_html=True,
    )

    rows = []
    for a in rt.alerts[-8:][::-1]:
        rows.append({
            "T_SIM": a.sim_clock,
            "SCORE": round(a.score, 4),
            "PREDICTED": a.predicted_class,
            "GROUND_TRUTH": a.ground_truth,
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


def _hex_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _panel_session_timeline(rt: TwinRuntime) -> None:
    """Shows only the cursor position — attack bands are revealed post-hoc via alerts, not pre-drawn."""
    st.markdown('<div class="panel" style="padding: 10px 20px;">', unsafe_allow_html=True)
    df = rt.src.df_raw
    if "label" not in df.columns:
        st.markdown('<div class="caption-mono">No ground truth labels available.</div></div>', unsafe_allow_html=True)
        return
    labels = df["label"].to_numpy()
    total = len(labels)

    # Only show attack bands that the cursor has ALREADY PASSED — no future leakage
    passed_labels = labels[:rt.cursor] if rt.is_ready else np.zeros(0, dtype=int)

    # Compute contiguous attack segments only from already-passed data
    segments: list[dict] = []
    in_seg = False
    seg_start = 0
    for i, lab in enumerate(passed_labels):
        if lab > 0 and not in_seg:
            seg_start = i
            in_seg = True
        elif lab == 0 and in_seg:
            segments.append({"start": seg_start, "end": i})
            in_seg = False
    if in_seg:
        segments.append({"start": seg_start, "end": len(labels)})

    def _meta(col: str, sl: slice) -> str:
        if col in df.columns:
            vals = df[col].iloc[sl].dropna()
            if len(vals):
                return str(vals.mode().iloc[0])
        return "?"

    for idx, seg in enumerate(segments):
        sl = slice(seg["start"], seg["end"])
        seg["id"] = _meta("attack_id", sl)
        seg["scenario"] = _meta("attack_type", sl)
        combo = _meta("combination", sl) if "combination" in df.columns else ""
        if combo and combo != "?":
            seg["scenario"] = f"{seg['scenario']} · {combo}"
        seg["ctrl"] = _meta("target_controller", sl)
        seg["len"] = seg["end"] - seg["start"]

    fig = go.Figure()

    # Each attack as its own shape + invisible hover point in the middle so
    # mousing over the band surfaces full metadata.
    for seg in segments:
        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=seg["start"], x1=seg["end"], y0=0, y1=1,
            fillcolor="rgba(239,68,68,0.32)",
            line=dict(width=0), layer="below",
        )
        mid = (seg["start"] + seg["end"]) / 2
        fig.add_trace(go.Scatter(
            x=[mid], y=[0.5], mode="markers",
            marker=dict(size=14, color="rgba(239,68,68,0.01)",
                        line=dict(width=0)),
            showlegend=False,
            hovertemplate=(
                f"<b>Attack #{seg['id']}</b><br>"
                f"scenario: {seg['scenario']}<br>"
                f"target controller: {seg['ctrl']}<br>"
                f"starts t={seg['start']:,}<br>"
                f"length: {seg['len']}s<br>"
                f"ends t={seg['end']:,}"
                "<extra></extra>"
            ),
        ))

    if rt.is_ready:
        fig.add_vline(
            x=rt.cursor, line=dict(color="#00d2ff", width=2.5),
            annotation_text=f"CURSOR {rt.cursor:,}",
            annotation_position="top right",
            annotation_font=dict(color="#00d2ff", size=12, family="JetBrains Mono"),
        )
    _apply_theme(fig, height=H_TIMELINE)
    fig.update_layout(margin=dict(l=40, r=20, t=5, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hovermode="closest")
    fig.update_xaxes(title="Total Replay Ticks (Dataset Index)", showgrid=False,
                     tickfont=dict(size=11), range=[0, total])
    fig.update_yaxes(visible=False, range=[0, 1])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


# ── Transport bar ───────────────────────────────────────────────────────────

SPEED_CHOICES = ["1×", "10×", "50×", "200×"]
SPEED_MAP = {"1×": 1, "10×": 10, "50×": 50, "200×": 200}

def _transport_bar(rt: TwinRuntime) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    min_t = INPUT_LEN
    max_t = max(INPUT_LEN + 1, len(rt.src) - TARGET_LEN)

    pending = st.session_state.pop("_pending_jump_cursor", None)
    if pending is not None:
        st.session_state["rt_warm_cursor"] = int(pending)

    playing = st.session_state.get("rt_playing", False)
    if playing:
        # Amber pause — red is reserved for alerts / destructive actions in SCADA.
        st.markdown(
            """<style>
            .stButton > button[kind="primary"] {
                background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%) !important;
                border-color: #b45309 !important;
                box-shadow: 0 6px 20px rgba(245, 158, 11, 0.45) !important;
            }
            .stButton > button[kind="primary"]:hover {
                box-shadow: 0 8px 25px rgba(245, 158, 11, 0.65) !important;
            }
            </style>""", unsafe_allow_html=True
        )

    r1_play, r1_spd, r1_pos = st.columns([1.5, 4.0, 6.5])

    def _toggle_play():
        st.session_state["rt_playing"] = not st.session_state.get("rt_playing", False)

    with r1_play:
        label = "■ PAUSE" if playing else "▶ PLAY"
        st.button(label, key="rt_play_btn", type="primary",
                  use_container_width=True, on_click=_toggle_play)

    with r1_spd:
        spd = st.radio(
            "SPEED", options=SPEED_CHOICES, index=st.session_state.get("rt_speed_idx", 2),
            horizontal=True, key="rt_speed_sel", label_visibility="collapsed"
        )
        st.session_state["rt_speed_idx"] = SPEED_CHOICES.index(spd)

    cursor_target = st.session_state.get("rt_warm_cursor", min_t)
    with r1_pos:
        pos_cursor = rt.cursor if rt.is_ready else cursor_target
        total = max_t
        pct = 100.0 * (pos_cursor - min_t) / max(total - min_t, 1)
        st.markdown(
            f'<div style="font-family: var(--mono); font-size: 0.75rem; color:#94a3b8; '
            f'padding: 8px 14px; border: 1px solid rgba(255,255,255,0.08); '
            f'border-radius: 4px; text-align: center;">'
            f'CURSOR <b style="color:#00d2ff;">{pos_cursor:,}</b> / {total:,} '
            f'<span style="color:#64748b;">({pct:.1f}%)</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr style="border-color: rgba(255,255,255,0.05); margin: 15px 0;">', unsafe_allow_html=True)
    r2_rst, r2_step, r2_info = st.columns([1.5, 1.5, 9.0])

    with r2_rst:
        reset_clicked = st.button("⟲ RESET", key="rt_reset_btn", use_container_width=True)
    with r2_step:
        step_clicked = st.button("⏭ STEP", key="rt_step_btn", use_container_width=True)
    with r2_info:
        st.markdown(
            '<div style="font-family: var(--mono); font-size: 0.75rem; color: #64748b; '
            'letter-spacing: 1px; padding: 10px 14px; border: 1px dashed rgba(255,255,255,0.08); '
            'border-radius: 4px; text-align: center;">'
            'SCENARIO PINNED · 0 = NORMAL &nbsp;<span style="color:#475569;">'
            '(swap embeddings in Generative)</span></div>',
            unsafe_allow_html=True,
        )
    scenario = 0  # hard-coded for Live Monitoring

    reset_request = reset_clicked

    speed = SPEED_MAP[spd]

    if (not rt.is_ready) or reset_request:
        rt.warm_up(cursor=cursor_target, scenario=scenario)

    # Re-read the flag right before stepping. Guards against a race where
    # st_autorefresh fires a rerun between the PAUSE click and this block.
    really_playing = bool(st.session_state.get("rt_playing", False))
    if really_playing:
        st_autorefresh(interval=1000, key="rt_tick")
        rt.step(speed)
    elif step_clicked:
        rt.step(1)
    
    st.markdown('</div>', unsafe_allow_html=True)


# ── Transport bar + header run OUTSIDE tabs so autorefresh fires on every page
_transport_bar(rt)   # steps the runtime first
_render_header(rt)   # header reads the score AFTER this tick's step

# ── LIVE DETECTION TAB ──────────────────────────────────────────────────────

with tab_live:
    left, right = st.columns([3.3, 1.7])
    with left:
        _panel_reconstruction_error(rt)
        _panel_sensor_overlay(rt)
    with right:
        _panel_gauge(rt)
        _panel_alarm_log(rt)

    _panel_session_timeline(rt)

    # Toast notifications for newly-fired alerts. `_transport_bar` runs step()
    # which appends to rt.alerts, so this block must come AFTER it to see this
    # tick's new alerts. We track the last-seen count per session.
    _prev_alerts = st.session_state.get("_last_alert_count", 0)
    _curr_alerts = len(rt.alerts)
    if _curr_alerts > _prev_alerts:
        for a in rt.alerts[_prev_alerts:]:
            st.toast(
                f"⚠ ANOMALY · {a.predicted_class} · {a.top_pv} · t={a.sim_clock}s · score {a.score:.3f}",
                icon="🚨",
            )
        # Audible beep. Browsers block autoplay until the user interacts with
        # the page once (the PLAY button counts), so the first ever alert may
        # be silent in a fresh tab.
        st.markdown(
            """
            <script>
            (function(){
              try {
                const AC = window.AudioContext || window.webkitAudioContext;
                if (!AC) return;
                const ctx = new AC();
                const now = ctx.currentTime;
                for (let i = 0; i < 3; i++) {
                  const o = ctx.createOscillator();
                  const g = ctx.createGain();
                  o.type = "square";
                  o.frequency.value = 880;
                  g.gain.setValueAtTime(0.0001, now + i*0.22);
                  g.gain.exponentialRampToValueAtTime(0.25, now + i*0.22 + 0.01);
                  g.gain.exponentialRampToValueAtTime(0.0001, now + i*0.22 + 0.18);
                  o.connect(g); g.connect(ctx.destination);
                  o.start(now + i*0.22);
                  o.stop(now + i*0.22 + 0.2);
                }
              } catch(e) {}
            })();
            </script>
            """,
            unsafe_allow_html=True,
        )
    st.session_state["_last_alert_count"] = _curr_alerts


# ── SYNTHETIC GENERATION TAB ────────────────────────────────────────────────

with tab_gen:
    snap = st.session_state.get("active_snapshot")
    t_eval = snap.cursor if snap else (rt.cursor if rt.is_ready else INPUT_LEN + TARGET_LEN)
    generative.render(bundle, src, t_eval)


# ── CO-PILOT (toggleable panel via real Streamlit button) ──────────────────

import chatbot as _chatbot_module

if "chat_open" not in st.session_state:
    st.session_state["chat_open"] = True

# Toggle button — styled as a floating circle via CSS targeting its key
_bcol1, _bcol2, _bcol3 = st.columns([10, 1, 1])
with _bcol3:
    if st.button("💬", key="chat_fab_btn",
                 help="Open/close the Twin Co-Pilot",
                 use_container_width=True):
        st.session_state["chat_open"] = not st.session_state["chat_open"]
        st.rerun()

if st.session_state["chat_open"]:
    with st.container(border=True):
        _hdr_a, _hdr_b = st.columns([10, 1])
        with _hdr_a:
            st.markdown(
                '<div style="font-family:var(--mono);font-size:0.95rem;'
                'color:#a855f7;letter-spacing:2px;">💬 TWIN CO-PILOT</div>',
                unsafe_allow_html=True,
            )
        with _hdr_b:
            if st.button("✕", key="chat_close_btn", help="Close chat"):
                st.session_state["chat_open"] = False
                st.rerun()
        _chatbot_module.render(rt)

# Floating 💬 circle injected into the parent document (survives autorefresh,
# works regardless of Streamlit's internal test-ids).
import streamlit.components.v1 as _components
_components.html(
    """
    <script>
    (function(){
      const doc = window.parent.document;
      let fab = doc.getElementById('plantmirror-chat-fab');
      if(!fab){
        fab = doc.createElement('div');
        fab.id = 'plantmirror-chat-fab';
        fab.innerText = '💬';
        fab.title = 'Open Twin Co-Pilot';
        Object.assign(fab.style, {
          position:'fixed', bottom:'28px', right:'28px',
          width:'78px', height:'78px', borderRadius:'50%',
          background:'linear-gradient(135deg,#c084fc 0%,#7c3aed 100%)',
          border:'3px solid #ffffff', display:'flex',
          alignItems:'center', justifyContent:'center',
          zIndex:'999999', cursor:'pointer', fontSize:'36px',
          userSelect:'none',
          boxShadow:'0 10px 28px rgba(168,85,247,0.6)'
        });
        doc.body.appendChild(fab);
      }
      if(!fab.dataset.wired){
        fab.dataset.wired = '1';
        fab.onclick = function(e){
          e.preventDefault(); e.stopPropagation();
          const sb = doc.querySelector('section[data-testid="stSidebar"]');
          if(sb){
            sb.classList.remove('chat-flash');
            void sb.offsetWidth;
            sb.classList.add('chat-flash');
            sb.scrollIntoView({behavior:'smooth', block:'start'});
          }
        };
      }
    })();
    </script>
    """,
    height=0,
)


