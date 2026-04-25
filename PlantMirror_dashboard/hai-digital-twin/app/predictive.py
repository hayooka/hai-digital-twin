"""
predictive.py — Streamlit view for the Predictive twin.

- Streams a ReplaySource, picks a window at the user's cursor.
- Runs one-shot plant prediction.
- Shows window-MSE vs threshold, per-PV MSE bars, per-timestep residuals,
  and overlay plots of predicted vs actual PV trajectories.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from twin_core import (
    INPUT_LEN,
    PV_COLS,
    TARGET_LEN,
    ReplaySource,
    TwinBundle,
    build_plant_window,
    per_pv_mse,
    per_step_residual,
    predict_plant,
    window_mse,
)


def render(bundle: TwinBundle, src: ReplaySource, t_end: int) -> Optional[dict]:
    """Render Predictive twin for the window ending at t_end (1-Hz index).

    Returns the prediction result dict (used by the Assistive layer), or None
    if the cursor is too close to the CSV edges to form a valid window.
    """
    win = build_plant_window(bundle, src, t_end)
    if win is None:
        st.error(
            f"Cursor t={t_end} is outside the valid replay range. "
            f"Need {INPUT_LEN} ≤ t ≤ {len(src) - TARGET_LEN}."
        )
        return None

    pv_pred = predict_plant(bundle, win)
    pv_true = win["pv_target"].squeeze(0).cpu().numpy()

    mse = window_mse(pv_pred, pv_true)
    per_pv = per_pv_mse(pv_pred, pv_true)
    per_step = per_step_residual(pv_pred, pv_true)

    anomaly = mse > bundle.threshold
    gt_label = _ground_truth_label(src, win)

    # ── Header metrics ────────────────────────────────────────────────────
    html = f"""
    <div class="metric-grid" style="grid-template-columns: repeat(4, 1fr); margin-bottom: 20px;">
      <div class="metric-cell">
        <div class="metric-value">{mse:.5f}</div>
        <div class="metric-label">Window MSE</div>
        <div class="metric-sub" style="color: {'#ef4444' if anomaly else '#10b981'}; font-size: 0.75rem; margin-top: 6px; font-weight: 600;">
            {mse - bundle.threshold:+.5f} vs threshold
        </div>
      </div>
      <div class="metric-cell">
        <div class="metric-value">{bundle.threshold:.5f}</div>
        <div class="metric-label">Detector Threshold</div>
      </div>
      <div class="metric-cell" style="border-bottom: 3px solid {'#ef4444' if anomaly else '#10b981'};">
        <div class="metric-value" style="color: {'#ef4444' if anomaly else '#10b981'}; text-shadow: 0 0 10px {'rgba(239, 68, 68, 0.5)' if anomaly else 'rgba(16, 185, 129, 0.5)'};">
            {'⚠ ANOMALY' if anomaly else '✓ NORMAL'}
        </div>
        <div class="metric-label">Digital Twin Assessment</div>
      </div>
      <div class="metric-cell" style="border-bottom: 3px solid {'#f59e0b' if gt_label != 'normal' else '#10b981'};">
        <div class="metric-value" style="font-size: 1.1rem; color: {'#f59e0b' if gt_label != 'normal' else '#10b981'};">{gt_label}</div>
        <div class="metric-label">Ground Truth Scenario</div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    # ── Premium Theme Helper ──────────────────────────────────────────────
    def _apply_premium_theme(fig: go.Figure, title: str, height: int = 280) -> go.Figure:
        fig.update_layout(
            title=dict(text=title, font=dict(color="#00d2ff", family="JetBrains Mono", size=14)),
            height=height, margin=dict(l=45, r=20, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter", size=11),
            hovermode="x unified",
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)", title_font=dict(size=10, color="#64748b"))
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)", title_font=dict(size=10, color="#64748b"))
        return fig

    # ── Charts Setup ──────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        # ── Per-PV MSE bars
        bars = go.Figure(go.Bar(
            x=PV_COLS, y=per_pv,
            marker=dict(
                color=["#ef4444" if v == per_pv.max() else "#00d2ff" for v in per_pv],
                line=dict(width=1, color="rgba(255,255,255,0.2)"),
                pattern_shape=["/" if v == per_pv.max() else "" for v in per_pv]
            ),
            text=[f"{v:.4f}" for v in per_pv], textposition="outside",
            textfont=dict(color="#f8fafc", family="JetBrains Mono", size=10),
        ))
        _apply_premium_theme(bars, "Per-PV contribution to window MSE (scaled space)")
        bars.update_yaxes(title="MSE Contribution")
        st.plotly_chart(bars, width="stretch", config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        # ── Per-step residual trace
        step_fig = go.Figure()
        step_fig.add_trace(go.Scatter(
            x=list(range(TARGET_LEN)), y=per_step,
            mode="lines", name="Per-Step MSE",
            line=dict(color="#ef4444" if anomaly else "#00d2ff", width=2.5),
            fill="tozeroy", fillcolor="rgba(239, 68, 68, 0.15)" if anomaly else "rgba(0, 210, 255, 0.15)"
        ))
        step_fig.add_hline(
            y=bundle.threshold, line=dict(color="#f59e0b", dash="dash", width=2),
            annotation_text=f"Threshold ({bundle.threshold:.3f})",
            annotation_position="top right", annotation_font=dict(color="#f59e0b", family="JetBrains Mono", size=11)
        )
        _apply_premium_theme(step_fig, "Per-timestep residual over 180s horizon")
        step_fig.update_xaxes(title="Seconds into horizon (s)")
        step_fig.update_yaxes(title="Mean Squared Error")
        st.plotly_chart(step_fig, width="stretch", config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)


    st.markdown('<div class="panel">', unsafe_allow_html=True)
    # ── Per-PV overlay (predicted vs actual, scaled)
    overlay = make_subplots(
        rows=len(PV_COLS), cols=1, shared_xaxes=True,
        subplot_titles=[
            f"<b>{pv}</b> | MSE={per_pv[i]:.4f}" for i, pv in enumerate(PV_COLS)
        ],
        vertical_spacing=0.04,
    )
    for i, pv in enumerate(PV_COLS):
        # Fill area between true and pred for visual emphasis of error
        overlay.add_trace(
            go.Scatter(
                x=list(range(TARGET_LEN)),
                y=pv_true[:, i], mode="lines",
                name=f"{pv} actual", line=dict(color="#10b981", width=2.5),
                showlegend=(i == 0),
                legendgroup="actual"
            ),
            row=i + 1, col=1,
        )
        overlay.add_trace(
            go.Scatter(
                x=list(range(TARGET_LEN)),
                y=pv_pred[:, i], mode="lines",
                name=f"{pv} predicted", line=dict(color="#ef4444", dash="dash", width=2),
                showlegend=(i == 0),
                legendgroup="predicted"
            ),
            row=i + 1, col=1,
        )
        # Highlight large errors in red fill
        overlay.add_trace(
            go.Scatter(
                x=list(range(TARGET_LEN)) + list(range(TARGET_LEN))[::-1],
                y=list(pv_true[:, i]) + list(pv_pred[:, i])[::-1],
                fill="toself",
                fillcolor="rgba(239, 68, 68, 0.1)" if per_pv[i] > (bundle.threshold / len(PV_COLS)) else "rgba(0, 210, 255, 0.05)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=i + 1, col=1,
        )

    overlay.update_layout(
        height=180 * len(PV_COLS),
        title=dict(text="Predicted vs Actual PV Trajectory Analysis (Scaled Units)", font=dict(color="#00d2ff", family="JetBrains Mono", size=16)),
        margin=dict(l=45, r=20, t=70, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    overlay.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)")
    overlay.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)")
    st.plotly_chart(overlay, width="stretch", config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    return {
        "pv_pred": pv_pred, "pv_true": pv_true,
        "per_pv": per_pv, "per_step": per_step,
        "mse": mse, "anomaly": anomaly,
        "t0": int(win["t0"].item()), "t1": int(win["t1"].item()),
        "t2": int(win["t2"].item()),
        "ground_truth": gt_label,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ground_truth_label(src: ReplaySource, win) -> str:
    """Label the window using the test CSV's attack columns."""
    t0, t2 = int(win["t0"].item()), int(win["t2"].item())
    slc = src.df_raw.iloc[t0:t2]
    if "label" not in slc.columns:
        return "unknown"
    if slc["label"].sum() == 0:
        return "normal"
    attacked = slc[slc["label"] > 0]
    if "attack_type" in attacked.columns:
        at = attacked["attack_type"].mode().iloc[0]
        ctrl = (
            attacked["target_controller"].mode().iloc[0]
            if "target_controller" in attacked.columns
            and not attacked["target_controller"].isna().all()
            else "?"
        )
        return f"ATTACK: {at} @ {ctrl}"
    return "ATTACK"
