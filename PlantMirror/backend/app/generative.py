"""
generative.py — Streamlit view for the Generative twin.

Two modes share a common form:
  A. Scenario Explorer: run the closed loop under each of the 4 scenario labels
     and overlay the resulting PV trajectories. Highlights how the scenario
     embedding shapes plant response.
  B. Virtual Plant: sliders override the SP for each of 4 controller loops;
     the closed loop (5 controllers → plant) is rolled out 180s ahead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


_SP_BOUNDS_CACHE: Optional[Dict[str, Dict[str, float]]] = None


def _load_sp_bounds() -> Dict[str, Dict[str, float]]:
    """Load sp_bounds.json. Returns {} if the file is missing so sliders fall
    back to the heuristic range. p01/p99 = hard clamp, p05/p95 = soft warning."""
    global _SP_BOUNDS_CACHE
    if _SP_BOUNDS_CACHE is not None:
        return _SP_BOUNDS_CACHE
    here = Path(__file__).resolve().parent
    p = here.parent / "outputs" / "sp_bounds.json"
    if not p.exists():
        _SP_BOUNDS_CACHE = {}
        return _SP_BOUNDS_CACHE
    _SP_BOUNDS_CACHE = json.load(open(p))
    return _SP_BOUNDS_CACHE

from twin_core import (
    INPUT_LEN,
    LOOP_ORDER,
    LOOP_SPECS,
    PV_COLS,
    SCENARIO_MAPPING,
    TARGET_LEN,
    ReplaySource,
    TwinBundle,
    closed_loop_rollout,
)

def _apply_premium_theme(fig: go.Figure, title: str, height: int = 280) -> go.Figure:
    """Helper to maintain 100/100 aesthetic consistency across all charts."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="#00d2ff", family="JetBrains Mono", size=15)),
        height=height, margin=dict(l=45, r=20, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter", size=11),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)", title_font=dict(size=10, color="#64748b"))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)", title_font=dict(size=10, color="#64748b"))
    return fig


def render(bundle: TwinBundle, src: ReplaySource, t_end: int) -> None:
    # Use a custom styled radio button via CSS (already injected in app.py)
    mode = st.radio(
        "Generative Mode Selection",
        ["Scenario Explorer", "Virtual Plant (What-If Analysis)"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
    
    if mode == "Scenario Explorer":
        _scenario_explorer(bundle, src, t_end)
    else:
        _virtual_plant(bundle, src, t_end)


# ── A. Scenario Explorer ─────────────────────────────────────────────────────

def _scenario_explorer(bundle: TwinBundle, src: ReplaySource, t_end: int) -> None:
    st.markdown(
        """<div class="caption-mono" style="margin-bottom: 20px;">
        Replays the exact 300s history window and swaps the <b>scenario embedding fed to the plant GRU</b>
        across all 4 learned classes. Controllers are scenario-agnostic (they output the same CVs in every run),
        so this view isolates how the plant's scenario conditioning alone bends the 180s PV trajectory.
        </div>""", unsafe_allow_html=True
    )
    
    colors = {0: "#10b981", 1: "#f59e0b", 2: "#ef4444", 3: "#a855f7"}
    rollouts: Dict[int, np.ndarray] = {}
    
    with st.spinner("Generating scenario rollouts..."):
        for sc in range(4):
            out = closed_loop_rollout(bundle, src, t_end, scenario=sc)
            if out is None:
                st.error(f"Cannot generate rollout: Cursor t={t_end} is out of bounds.")
                return
            rollouts[sc] = out["pv_physical"]

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    fig = make_subplots(
        rows=len(PV_COLS), cols=1, shared_xaxes=True,
        subplot_titles=[f"<b>{pv}</b> Response" for pv in PV_COLS],
        vertical_spacing=0.04,
    )
    
    for i, pv in enumerate(PV_COLS):
        for sc, arr in rollouts.items():
            fig.add_trace(
                go.Scatter(
                    x=list(range(TARGET_LEN)),
                    y=arr[:, i], mode="lines",
                    name=SCENARIO_MAPPING[sc],
                    line=dict(color=colors[sc], width=2 if sc != 0 else 2.5, dash="solid" if sc == 0 else "dashdot"),
                    showlegend=(i == 0),
                    legendgroup=str(sc)
                ),
                row=i + 1, col=1,
            )
            
    fig = _apply_premium_theme(fig, "Closed-Loop PV Rollout Per Scenario (Physical Units)", height=180 * len(PV_COLS))
    fig.update_xaxes(title="Seconds Ahead (s)", row=len(PV_COLS), col=1)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


# ── B. Virtual Plant ─────────────────────────────────────────────────────────

def _virtual_plant(bundle: TwinBundle, src: ReplaySource, t_end: int) -> None:
    st.markdown(
        """<div class="caption-mono" style="margin-bottom: 20px;">
        Override controller setpoints dynamically. The twin will roll forward 180s simulating how the 
        5 PID-surrogate controllers and the MIMO plant respond to your injected conditions.
        </div>""", unsafe_allow_html=True
    )

    # Read current SPs at cursor
    spec_cols = {loop: LOOP_SPECS[loop]["base_cols"][0] for loop in LOOP_ORDER}
    
    try:
        current_sps = {
            loop: float(src.df_raw[spec_cols[loop]].iloc[t_end - 1])
            for loop in LOOP_ORDER
        }
    except Exception as e:
        st.error(f"Cannot read setpoints at t={t_end - 1}. Out of bounds.")
        return

    sp_overrides: Dict[str, float] = {}
    
    st.markdown('<div class="panel" style="padding: 20px;">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title" style="margin-bottom: 15px;">Target Condition Overrides</div>', unsafe_allow_html=True)
    
    col_scen, col_empty = st.columns([1, 2])
    with col_scen:
        scenario = st.selectbox(
            "Background Context",
            options=list(SCENARIO_MAPPING.keys()),
            format_func=lambda k: f"Scenario {k}: {SCENARIO_MAPPING[k]}",
            index=0,
        )

    st.markdown('<hr style="border-color: rgba(255,255,255,0.05); margin: 20px 0;">', unsafe_allow_html=True)
    
    # 5 Loops layout — sliders hard-clamped to training-distribution p01/p99
    # so the user cannot drive the twin into out-of-distribution hallucination.
    bounds = _load_sp_bounds()
    ncols = st.columns(len(LOOP_ORDER))
    soft_warnings: List[str] = []
    for col_widget, loop in zip(ncols, LOOP_ORDER):
        with col_widget:
            cur = current_sps[loop]
            b = bounds.get(loop)
            if b is not None:
                lo, hi = float(b["p01"]), float(b["p99"])
                soft_lo, soft_hi = float(b["p05"]), float(b["p95"])
                # Guard: value must lie inside the clamp or st.slider raises.
                cur_clamped = min(max(cur, lo), hi)
                step = max((hi - lo) / 200.0, 1e-4)
                caption = f"p01–p99 clamp&nbsp;&nbsp;[{lo:.3g}, {hi:.3g}]"
            else:
                span = max(abs(cur) * 0.25, 1e-3)
                lo, hi = float(cur - span * 4), float(cur + span * 4)
                soft_lo, soft_hi = lo, hi
                cur_clamped = cur
                step = float(max(span / 20.0, 1e-4))
                caption = "heuristic range (sp_bounds.json missing)"
            st.markdown(
                f"**{loop}** SP<br><span style='font-size: 0.7rem; color: #64748b;'>"
                f"{spec_cols[loop]}<br>{caption}</span>",
                unsafe_allow_html=True,
            )
            val = st.slider(
                f"{loop}_sp_slider",
                min_value=lo, max_value=hi,
                value=float(cur_clamped),
                step=float(step),
                label_visibility="collapsed",
            )
            sp_overrides[loop] = val
            if b is not None and (val < soft_lo or val > soft_hi):
                soft_warnings.append(
                    f"{loop}={val:.3g} is outside the p05–p95 soft band "
                    f"[{soft_lo:.3g}, {soft_hi:.3g}] — predictions may be less reliable."
                )

    if soft_warnings:
        st.markdown(
            "<div style='color:#f59e0b;font-family:var(--mono);font-size:0.75rem;margin-top:10px;'>"
            + "<br>".join("⚠ " + w for w in soft_warnings)
            + "</div>",
            unsafe_allow_html=True,
        )
            
    btn_col, _, _ = st.columns([1, 2, 2])
    with btn_col:
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        run_btn = st.button("🚀 SIMULATE ROLLOUT", type="primary", use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

    if run_btn:
        import time, hashlib
        t_start = time.time()
        with st.spinner("Simulating closed-loop dynamics..."):
            out_baseline = closed_loop_rollout(
                bundle, src, t_end, sp_overrides=None, scenario=scenario,
            )
            out_override = closed_loop_rollout(
                bundle, src, t_end, sp_overrides=sp_overrides, scenario=scenario,
            )
        compute_ms = (time.time() - t_start) * 1000.0

        if out_baseline is None or out_override is None:
            st.error("Rollout failed: window out of bounds.")
            return

        # Inference fingerprint — proves fresh computation (not cached)
        out_hash = hashlib.md5(out_override["pv_physical"].tobytes()).hexdigest()[:8]
        diff = out_override["pv_physical"] - out_baseline["pv_physical"]
        d_max = float(np.max(np.abs(diff)))
        d_mean = float(np.mean(np.abs(diff)))
        per_pv_delta = {PV_COLS[i]: float(np.max(np.abs(diff[:, i]))) for i in range(len(PV_COLS))}
        top_pv = max(per_pv_delta.items(), key=lambda x: x[1])

        st.markdown(
            f"""<div style="display:flex;gap:12px;margin:10px 0 18px 0;flex-wrap:wrap;">
              <div style="flex:1;min-width:160px;background:rgba(0,210,255,0.06);border:1px solid rgba(0,210,255,0.25);border-radius:6px;padding:10px 14px;">
                <div style="font-family:var(--mono);font-size:.7rem;color:#64748b;letter-spacing:1px;">INFERENCE HASH</div>
                <div style="font-family:var(--mono);font-size:1.1rem;color:#00d2ff;font-weight:700;">{out_hash}</div>
                <div style="font-family:var(--mono);font-size:.65rem;color:#475569;">computed in {compute_ms:.0f} ms</div>
              </div>
              <div style="flex:1;min-width:160px;background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.25);border-radius:6px;padding:10px 14px;">
                <div style="font-family:var(--mono);font-size:.7rem;color:#64748b;letter-spacing:1px;">MAX Δ (BASE vs WHAT-IF)</div>
                <div style="font-family:var(--mono);font-size:1.1rem;color:#f59e0b;font-weight:700;">{d_max:.4f}</div>
                <div style="font-family:var(--mono);font-size:.65rem;color:#475569;">mean Δ = {d_mean:.4f}</div>
              </div>
              <div style="flex:1;min-width:160px;background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.25);border-radius:6px;padding:10px 14px;">
                <div style="font-family:var(--mono);font-size:.7rem;color:#64748b;letter-spacing:1px;">MOST AFFECTED PV</div>
                <div style="font-family:var(--mono);font-size:1.1rem;color:#10b981;font-weight:700;">{top_pv[0]}</div>
                <div style="font-family:var(--mono);font-size:.65rem;color:#475569;">Δ = {top_pv[1]:.4f}</div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig = make_subplots(
            rows=len(PV_COLS), cols=1, shared_xaxes=True,
            subplot_titles=[f"<b>{pv}</b> Trajectory" for pv in PV_COLS],
            vertical_spacing=0.04,
        )
        
        for i, pv in enumerate(PV_COLS):
            fig.add_trace(
                go.Scatter(
                    x=list(range(TARGET_LEN)),
                    y=out_baseline["pv_physical"][:, i], mode="lines",
                    name="Baseline (Replay SPs)",
                    line=dict(color="rgba(255, 255, 255, 0.3)", width=2),
                    showlegend=(i == 0),
                    legendgroup="base"
                ),
                row=i + 1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(TARGET_LEN)),
                    y=out_override["pv_physical"][:, i], mode="lines",
                    name="What-If Injection",
                    line=dict(color="#00d2ff", width=2.5),
                    showlegend=(i == 0),
                    legendgroup="override"
                ),
                row=i + 1, col=1,
            )
            
        fig = _apply_premium_theme(fig, "Virtual Plant Response (Physical Units)", height=180 * len(PV_COLS))
        fig.update_xaxes(title="Seconds Ahead (s)", row=len(PV_COLS), col=1)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        _render_ctrl_cvs(out_override["ctrl_cv_scaled"])


def _render_ctrl_cvs(cv_preds: Dict[str, np.ndarray]) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    fig = go.Figure()
    colors = ["#00d2ff", "#10b981", "#f59e0b", "#a855f7", "#ef4444"]
    for i, (loop, cv) in enumerate(cv_preds.items()):
        fig.add_trace(go.Scatter(
            x=list(range(TARGET_LEN)),
            y=cv, mode="lines", name=f"{loop} CV",
            line=dict(width=2, color=colors[i % len(colors)])
        ))
        
    fig = _apply_premium_theme(fig, "Controller Actuation Signals (Scaled)", height=280)
    fig.update_xaxes(title="Seconds into horizon (s)")
    fig.update_yaxes(title="Control Value (Scaled)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)
