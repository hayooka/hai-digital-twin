"""
assistive.py — Streamlit view for the Assistive layer.

L1: rank the 5 PVs by their contribution to the window MSE (the "which PV
    drifted?" view).
L2: for the top-ranked PV, walk parents_full.json upstream to surface the
    sensors most likely to have caused the drift, with lag and level annotated.

The view is rendered *after* predictive.render has produced the per-PV MSE.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from causal_utils import CausalGraph, Edge
from twin_core import PV_COLS, ReplaySource


def _apply_premium_theme(fig: go.Figure, title: str, height: int = 280) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color="#00d2ff", family="JetBrains Mono", size=15)),
        height=height, margin=dict(l=45, r=20, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter", size=11),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)", title_font=dict(size=10, color="#64748b"))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)", title_font=dict(size=10, color="#64748b"))
    return fig


def render(
    graph: CausalGraph,
    predict_result: Optional[dict],
    src: ReplaySource,
) -> None:
    if predict_result is None:
        st.markdown(
            """<div class="panel" style="border-color: #f59e0b; background: rgba(245, 158, 11, 0.05);">
            <div class="panel-title" style="color: #f59e0b; border-color: #f59e0b;">⚠ NO ACTIVE PREDICTION</div>
            <div style="color: #cbd5e1; font-family: var(--sans);">
            The Assistive layer requires an active snapshot. Please trigger a <b>Snapshot</b> in the Live tab, 
            or ensure the Live Twin has successfully streamed data.
            </div>
            </div>""", unsafe_allow_html=True
        )
        return

    per_pv: np.ndarray = predict_result["per_pv"]

    st.markdown('<div class="panel" style="margin-bottom: 25px;">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">L1 — Root Cause PV Identification</div>', unsafe_allow_html=True)
    
    col_l1, col_l1_txt = st.columns([2, 1])
    
    with col_l1:
        order = np.argsort(per_pv)[::-1]
        ranked = [(PV_COLS[i], float(per_pv[i])) for i in order]
        total = float(per_pv.sum()) + 1e-12
        
        rank_fig = go.Figure(go.Bar(
            x=[f"{pv}<br><span style='font-size:10px; color:#64748b;'>{mse/total*100:.1f}%</span>" for pv, mse in ranked],
            y=[mse for _, mse in ranked],
            marker=dict(
                color=["#ef4444"] + ["#00d2ff"] * (len(ranked) - 1),
                line=dict(width=1, color="rgba(255,255,255,0.2)"),
                pattern_shape=["/" if i == 0 else "" for i in range(len(ranked))]
            ),
            text=[f"{mse:.4f}" for _, mse in ranked], textposition="outside",
            textfont=dict(color="#f8fafc", family="JetBrains Mono", size=11),
        ))
        rank_fig = _apply_premium_theme(rank_fig, "Residual MSE Distribution by PV", height=280)
        rank_fig.update_yaxes(title="MSE (Scaled)")
        st.plotly_chart(rank_fig, width="stretch", config={"displayModeBar": False})
        
    top_pv, top_mse = ranked[0]
    share = top_mse / total * 100
    
    with col_l1_txt:
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 8px; padding: 20px; height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <div style="color: #fca5a5; font-family: var(--sans); font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Primary Culprit</div>
            <div style="color: #ef4444; font-family: var(--mono); font-size: 2rem; font-weight: 800; margin: 10px 0; text-shadow: 0 0 10px rgba(239, 68, 68, 0.4);">{top_pv}</div>
            <div style="color: #cbd5e1; font-family: var(--sans); font-size: 0.9rem; line-height: 1.5;">
                This PV accounts for <b style="color: #f8fafc;">{share:.1f}%</b> of the total reconstruction error in the current window.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

    # ── L2: causal trace ──────────────────────────────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-title">L2 — Causal Graph Upstream Trace for `{top_pv}`</div>', unsafe_allow_html=True)
    
    col_opt1, col_opt2 = st.columns([1, 1])
    with col_opt1:
        level_cap = st.radio(
            "Causal Level Depth",
            options=[0, 1, 2],
            index=2,
            format_func=lambda v: {
                0: "L0 Only (Direct DCS Links)",
                1: "≤ L1 (+ 1-Hop Physical)",
                2: "All Levels (+ 2-Hop Physical)",
            }[v],
            horizontal=True,
            label_visibility="collapsed"
        )
    with col_opt2:
        max_depth = st.slider("Max Graph BFS Depth", min_value=1, max_value=4, value=2, label_visibility="collapsed")

    st.markdown('<hr style="border-color: rgba(255,255,255,0.05); margin: 20px 0;">', unsafe_allow_html=True)

    if top_pv not in graph.parents:
        fallback = None
        for pv, _ in ranked[1:]:
            if pv in graph.parents:
                fallback = pv
                break
        if fallback is None:
            st.markdown(
                f"""<div style="color: #cbd5e1; font-family: var(--sans); margin-bottom: 20px;">
                ⚠️ No PV in this window has parents mapped in <code>parents_full.json</code>.
                </div>""", unsafe_allow_html=True
            )
            _render_ground_truth(predict_result, src)
            st.markdown('</div>', unsafe_allow_html=True)
            return
        st.markdown(
            f"""<div style="color: #cbd5e1; font-family: var(--sans); margin-bottom: 15px;
            background: rgba(0,210,255,0.05); border-left: 3px solid #00d2ff; padding: 10px 14px;">
            ℹ Top-ranked <b>{top_pv}</b> has no parents mapped — tracing next-ranked
            <b style="color:#00d2ff;">{fallback}</b> instead.
            </div>""", unsafe_allow_html=True
        )
        top_pv = fallback

    suspects = graph.rank_suspects(top_pv, max_depth=max_depth, level_cap=level_cap)
    suspects = [(s, sc, p) for s, sc, p in suspects if s != top_pv]

    if not suspects:
        st.markdown('<div style="color: #94a3b8; font-family: var(--mono); padding: 20px; text-align: center; border: 1px dashed rgba(255,255,255,0.1); border-radius: 4px;">No causal suspects found at this depth/level constraint.</div>', unsafe_allow_html=True)
        _render_ground_truth(predict_result, src)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    top_k = min(10, len(suspects))
    
    # Styled Dataframe presentation
    st.markdown("<div style='font-family: var(--mono); color: #00d2ff; font-weight: 700; margin-bottom: 10px;'>MOST LIKELY CAUSAL ANTECEDENTS</div>", unsafe_allow_html=True)
    
    # Build as one flat string — leading whitespace triggers markdown code-block rendering.
    rows_html = ""
    for idx, (sensor, score, path) in enumerate(suspects[:top_k]):
        trail = " ← ".join(
            f"<span style='color:#00d2ff;'>{e.parent}</span> "
            f"<span style='font-size:0.7rem;color:#64748b;'>(-{e.lag}s, L{e.level})</span>"
            for e in path
        )
        row_bg = "background:rgba(255,255,255,0.02);" if idx % 2 == 0 else ""
        rows_html += (
            f"<tr style=\"{row_bg}border-bottom:1px solid rgba(255,255,255,0.05);\">"
            f"<td style=\"padding:12px 15px;font-weight:600;color:#f8fafc;font-family:var(--mono);\">{sensor}</td>"
            f"<td style=\"padding:12px 15px;color:#10b981;font-family:var(--mono);\">{score:.3f}</td>"
            f"<td style=\"padding:12px 15px;font-family:var(--mono);\">"
            f"<span style='color:#ef4444;font-weight:700;'>{top_pv}</span> ← {trail}</td>"
            f"</tr>"
        )
    table_html = (
        "<table style=\"width:100%;border-collapse:collapse;font-family:var(--sans);"
        "font-size:0.85rem;text-align:left;background:rgba(0,0,0,0.2);"
        "border-radius:6px;overflow:hidden;\">"
        "<thead><tr style=\"background:rgba(255,255,255,0.05);color:#cbd5e1;"
        "font-family:var(--mono);font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;\">"
        "<th style=\"padding:12px 15px;border-bottom:1px solid rgba(255,255,255,0.1);\">Sensor Node</th>"
        "<th style=\"padding:12px 15px;border-bottom:1px solid rgba(255,255,255,0.1);\">Causal Score</th>"
        "<th style=\"padding:12px 15px;border-bottom:1px solid rgba(255,255,255,0.1);\">Trace Path (Upstream ← Downstream)</th>"
        "</tr></thead><tbody>" + rows_html + "</tbody></table>"
    )
    st.markdown(table_html, unsafe_allow_html=True)
    
    st.markdown('<hr style="border-color: rgba(255,255,255,0.05); margin: 25px 0;">', unsafe_allow_html=True)
    _render_ground_truth(predict_result, src)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Ground-truth overlay ─────────────────────────────────────────────────────

def _render_ground_truth(predict_result: dict, src: ReplaySource) -> None:
    t0, t2 = predict_result["t0"], predict_result["t2"]
    slc = src.df_raw.iloc[t0:t2]

    if "label" not in slc.columns or slc["label"].sum() == 0:
        html = f"""
        <div style="background: rgba(16, 185, 129, 0.05); border-left: 4px solid #10b981; padding: 15px; border-radius: 0 4px 4px 0;">
            <div style="color: #10b981; font-family: var(--mono); font-weight: 700; font-size: 0.9rem; margin-bottom: 5px;">GROUND TRUTH: NORMAL</div>
            <div style="color: #cbd5e1; font-family: var(--sans); font-size: 0.8rem;">
                No attack labels present in this window. Detector assessed as: <b style="color: {'#ef4444' if predict_result['anomaly'] else '#10b981'}">{'ANOMALY (False Alarm)' if predict_result['anomaly'] else 'NORMAL (Correct)'}</b>.
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        return

    attacked = slc[slc["label"] > 0]
    ids = attacked["attack_id"].dropna().unique().tolist() if "attack_id" in attacked.columns else []
    tgt_ctrls = attacked["target_controller"].dropna().unique().tolist() if "target_controller" in attacked.columns else []
    tgt_pts = attacked["target_points"].dropna().unique().tolist() if "target_points" in attacked.columns else []

    caught = predict_result["anomaly"]
    
    html = f"""
    <div style="display: flex; gap: 20px;">
        <div style="flex: 1; background: rgba(239, 68, 68, 0.05); border-left: 4px solid #ef4444; padding: 15px; border-radius: 0 4px 4px 0;">
            <div style="color: #ef4444; font-family: var(--mono); font-weight: 700; font-size: 0.9rem; margin-bottom: 10px;">GROUND TRUTH: ATTACK DETECTED</div>
            <table style="width: 100%; font-family: var(--sans); font-size: 0.8rem; color: #cbd5e1;">
                <tr><td style="padding-bottom: 5px; color: #94a3b8;">Attack ID(s)</td><td style="font-family: var(--mono); color: #f8fafc;">{ids}</td></tr>
                <tr><td style="padding-bottom: 5px; color: #94a3b8;">Target Controller(s)</td><td style="font-family: var(--mono); color: #f8fafc;">{tgt_ctrls}</td></tr>
                <tr><td style="color: #94a3b8;">Target Point(s)</td><td style="font-family: var(--mono); color: #f8fafc;">{tgt_pts}</td></tr>
            </table>
        </div>
        <div style="width: 250px; background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div style="color: #64748b; font-family: var(--mono); font-size: 0.7rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 10px;">DETECTION STATUS</div>
            <div style="color: {'#10b981' if caught else '#ef4444'}; font-family: var(--mono); font-weight: 800; font-size: 1.5rem; text-shadow: 0 0 10px {'rgba(16, 185, 129, 0.4)' if caught else 'rgba(239, 68, 68, 0.4)'};">
                {'✓ CAUGHT' if caught else '✗ MISSED'}
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
