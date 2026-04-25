"""
new_digital_twin.py — Standalone Live Monitoring Dashboard.

This is a dedicated, single-page application focused EXCLUSIVELY on live
monitoring, anomaly detection, and real-time streaming. 
Per the user's request, the Generative, Predictive, and Assistive layers 
have been removed, and the manual "Scenario Selector" has been stripped 
from the Live monitor (it will strictly assume Normal/0 physics to properly 
catch anomalies).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_js_eval import streamlit_js_eval

from twin_core import (
    INPUT_LEN,
    PV_COLS,
    TARGET_LEN,
    default_paths,
    load_bundle,
    load_replay,
)
from twin_runtime import ALERT_GAP_SEC, TwinRuntime

st.set_page_config(
    page_title="ICS · LIVE MONITOR",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── SCADA stylesheet (Premium UI) ───────────────────────────────────────────

SCADA_CSS = """
<style>
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
    --alert: #ef4444;
    --mono: 'JetBrains Mono', ui-monospace, 'Consolas', monospace;
    --sans: 'Inter', system-ui, sans-serif;
  }

  [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
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
  .dt-subtitle {
      font-family: var(--sans); font-size: 0.9rem; color: #10b981; 
      letter-spacing: 2px; text-transform: uppercase; margin-left: 15px;
      padding-left: 15px; border-left: 1px solid rgba(255,255,255,0.2);
  }
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
  }
  .pill-ok    { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #059669; }
  .pill-idle  { background: rgba(100, 116, 139, 0.15); color: #94a3b8; border: 1px solid #475569; }
  .pill-alert {
    background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid #dc2626;
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
  }

  .panel-title {
    font-family: var(--sans); font-size: 0.85rem; font-weight: 800;
    letter-spacing: 2px; color: var(--text-0);
    text-transform: uppercase; padding: 4px 0 12px 14px;
    border-left: 4px solid var(--accent); margin-bottom: 12px;
  }

  /* Buttons */
  .stButton > button {
    font-family: var(--mono) !important; font-weight: 800 !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    border-radius: 6px !important; background: var(--bg-2) !important; 
    color: var(--text-0) !important; border: 1px solid var(--border) !important;
  }
  .stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
    border: 1px solid #047857 !important; color: #ffffff !important;
  }

  /* Dataframe & Slider */
  div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
  .caption-mono { font-family: var(--sans); font-size: 0.8rem; color: var(--muted); letter-spacing: 0.5px; }
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
paths = default_paths()
csv_choices = {p.stem: str(p) for p in paths["test_csvs"] if p.exists()}
if not csv_choices:
    st.error(f"FATAL: No test CSVs found under {paths['test_csvs'][0].parent}.")
    st.stop()
csv_name = next(iter(csv_choices))
src = _load_replay(csv_choices[csv_name], id(bundle))


# ── Runtime session state ───────────────────────────────────────────────────

def _get_or_make_runtime() -> TwinRuntime:
    key = f"new_rt::{csv_name}"
    if key not in st.session_state:
        st.session_state[key] = TwinRuntime(bundle, src)
    return st.session_state[key]

rt = _get_or_make_runtime()


# ── Plotly Theme ────────────────────────────────────────────────────────────

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

def _render_header(rt: TwinRuntime) -> None:
    playing = st.session_state.get("rt_playing", False)
    score = rt.anomaly_score if rt.is_ready else None
    thr = float(bundle.threshold)
    attacking = score is not None and score > thr

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
                    <circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                DIGITAL TWIN <span>·</span> LIVE
                <div class="dt-subtitle">Monitoring App</div>
            </div>
          </div>
          <div class="dt-right">
            <div class="dt-meta" style="margin-right: 15px; border-right: 1px solid rgba(255,255,255,0.1); padding-right: 25px;">
              SESSION ALERTS<br><span class="val" style="color: {'#ef4444' if len(rt.alerts) > 0 else '#00d2ff'};">{len(rt.alerts)}</span>
            </div>
            {pill_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

_render_header(rt)


# ── Panels ──────────────────────────────────────────────────────────────────

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
    _panel_title("Reconstruction Error (Step MSE)")
    arrs = rt.rolling_arrays()
    if arrs["step_mse"].shape[0] == 0:
        st.markdown('<div class="caption-mono" style="padding: 40px; text-align: center;">Press ▶ PLAY to initialize data stream.</div></div>', unsafe_allow_html=True)
        return
    x = arrs["sim_times"]
    fig = go.Figure()
    
    for s, e in _attack_segments_in_buffer(rt):
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(239,68,68,0.15)", line_width=0, layer="below")
        
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
    _apply_theme(fig, height=300)
    fig.update_xaxes(title="Simulation Clock (s)")
    fig.update_yaxes(title="MSE (Scaled)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


def _panel_gauge(rt: TwinRuntime) -> None:
    st.markdown('<div class="panel" style="height: 100%; display: flex; flex-direction: column; justify-content: center;">', unsafe_allow_html=True)
    _panel_title("Live Anomaly Score")
    score = rt.anomaly_score
    thr = float(bundle.threshold)
    val = 0.0 if score is None else float(score)
    axis_max = max(thr * 1.8, val * 1.2, 0.05)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={"valueformat": ".5f", "font": {"color": "#f8fafc", "size": 32, "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, axis_max], "tickwidth": 1.5, "tickcolor": "#475569", "tickfont": {"size": 11}},
            "bar": {"color": "#00d2ff", "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0.2)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, thr * 0.66], "color": "rgba(16, 185, 129, 0.25)"},
                {"range": [thr * 0.66, thr], "color": "rgba(245, 158, 11, 0.25)"},
                {"range": [thr, axis_max], "color": "rgba(239, 68, 68, 0.5)"},
            ],
            "threshold": {"line": {"color": "#ffffff", "width": 4}, "thickness": 1, "value": thr},
        },
    ))
    fig.update_layout(paper_bgcolor=PLOT_BG, height=300, margin=dict(l=20, r=20, t=15, b=15), uirevision="live")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


def _panel_sensor_overlay(rt: TwinRuntime) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    _panel_title("Key PV Readings — Twin vs Actual")
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
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(239,68,68,0.1)", line_width=0, layer="below")
        
    for i, pv in enumerate(PV_COLS):
        fig.add_trace(go.Scatter(
            x=x, y=actual_n[:, i], mode="lines",
            line=dict(color=palette[i], width=2),
            name=f"{pv} Actual", legendgroup=pv,
        ))
        fig.add_trace(go.Scatter(
            x=x, y=twin_n[:, i], mode="lines",
            line=dict(color=palette[i], width=1.5, dash="dot"),
            name=f"{pv} Twin", legendgroup=pv, showlegend=False,
        ))
    _apply_theme(fig, height=350)
    fig.update_xaxes(title="Simulation Clock (s)")
    fig.update_yaxes(title="Normalized Magnitude", range=[-0.05, 1.05])
    fig.update_layout(legend=dict(orientation="h", y=1.15, x=0, font=dict(size=11)))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
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
    _apply_theme(fig, height=220)
    fig.update_yaxes(autorange="reversed", tickfont=dict(family="JetBrains Mono", size=12, color="#cbd5e1"))
    fig.update_xaxes(title=None, showticklabels=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


def _panel_alarm_log(rt: TwinRuntime) -> None:
    st.markdown('<div class="panel" style="height: 100%;">', unsafe_allow_html=True)
    _panel_title(f"Alert Feed (Debounced ≥ {ALERT_GAP_SEC}s)")
    
    rows = []
    if rt.alerts:
        for a in rt.alerts[-10:][::-1]:
            rows.append({
                "TIME (s)": a.sim_clock,
                "SCORE": round(a.score, 4),
                "CULPRIT_PV": a.top_pv,
                "GROUND_TRUTH": a.ground_truth,
            })
    else:
        # Empty dataframe placeholder
        rows = [{"TIME (s)": "-", "SCORE": "-", "CULPRIT_PV": "-", "GROUND_TRUTH": "-"}]
        
    st.dataframe(rows, width="stretch", hide_index=True, height=295)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Transport bar (REMOVED SCENARIO SELECTOR) ───────────────────────────────

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
        st.markdown(
            """<style>
            .stButton > button[kind="primary"] {
                background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%) !important;
                border-color: #b91c1c !important;
            }
            </style>""", unsafe_allow_html=True
        )

    r1_play, r1_spd, r1_scrub = st.columns([1.5, 3.0, 7.5])

    with r1_play:
        label = "■ PAUSE" if playing else "▶ PLAY"
        if st.button(label, key="rt_play_btn", type="primary", use_container_width=True):
            st.session_state["rt_playing"] = not playing
            playing = st.session_state["rt_playing"]

    with r1_spd:
        spd = st.radio(
            "SPEED", options=SPEED_CHOICES, index=st.session_state.get("rt_speed_idx", 2),
            horizontal=True, key="rt_speed_sel", label_visibility="collapsed"
        )
        st.session_state["rt_speed_idx"] = SPEED_CHOICES.index(spd)

    with r1_scrub:
        cursor_target = st.slider(
            "Warm-up cursor", min_value=min_t, max_value=max_t,
            value=st.session_state.get("rt_warm_cursor", min_t),
            step=60, key="rt_warm_cursor", label_visibility="collapsed"
        )

    st.markdown('<hr style="border-color: rgba(255,255,255,0.05); margin: 15px 0;">', unsafe_allow_html=True)
    r2_rst, r2_step, r2_jmp, _ = st.columns([1.5, 1.5, 4.0, 5.0])

    with r2_rst:
        reset_clicked = st.button("⟲ RESET", key="rt_reset_btn", use_container_width=True)
    with r2_step:
        step_clicked = st.button("⏭ STEP", key="rt_step_btn", use_container_width=True)
    with r2_jmp:
        jump_clicked = st.button("⚠ JUMP TO ATTACK", key="rt_jump_btn", use_container_width=True)

    reset_request = reset_clicked
    if jump_clicked and "label" in rt.src.df_raw.columns:
        attack_idx = rt.src.df_raw.index[rt.src.df_raw["label"] > 0]
        if len(attack_idx):
            target = max(min_t, min(int(attack_idx[0]) - 300, max_t))
            st.session_state["_pending_jump_cursor"] = target
            cursor_target = target
            reset_request = True

    speed = SPEED_MAP[spd]

    if (not rt.is_ready) or reset_request:
        # HARDCODE SCENARIO TO 0 (NORMAL) TO PREVENT CHEATING IN LIVE MONITORING
        rt.warm_up(cursor=cursor_target, scenario=0)

    if playing:
        st_autorefresh(interval=1000, key="rt_tick")
        rt.step(speed)
    elif step_clicked:
        rt.step(speed)
    
    st.markdown('</div>', unsafe_allow_html=True)


# ── App Layout Assembly ─────────────────────────────────────────────────────

left_col, right_col = st.columns([3, 1.5])

with left_col:
    _panel_reconstruction_error(rt)
    _panel_sensor_overlay(rt)

with right_col:
    _panel_gauge(rt)
    _panel_root_cause(rt)
    _panel_alarm_log(rt)

_transport_bar(rt)
