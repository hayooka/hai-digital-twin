"""
Tab 2 — Rollout Tester.

Layout:
  1. Banner (target_len trained on + validated horizons + 1.5% threshold)
  2. Scenario selector (Normal / AP_no / AP_with / AE_no / Overall)
  3. NRMSE bar chart per PV (5 bars, coloured vs 1.5% threshold)
  4. Per-PV pass/fail cards
  5. Per-scenario summary table
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from components import (  # noqa: E402
    C_ACCENT, C_FAIL, C_MUTED, C_OK, C_TEXT, C_WARN, pass_fail_pill,
)
from data_loader import PV_COLS, get_eval_results, get_eval_1800s  # noqa: E402


THRESHOLD = 0.015   # Student Guide §9: "under 1.5% at 30 minutes"


def render() -> None:
    st.session_state["_current_tab"] = "Rollout Tester"
    st.markdown("## 📈 Rollout Tester")
    st.caption("How accurate is the plant forecast at different horizons?")

    eval_results_180  = get_eval_results()
    eval_results_1800 = get_eval_1800s()
    if not eval_results_180:
        st.error("`generator/weights/eval_results.json` missing. Cannot render.")
        return

    # ── 1. Horizon selector ────────────────────────────────────────────
    horizon_opts = ["3 min (180 s) — training horizon"]
    if eval_results_1800:
        horizon_opts.append("30 min (1800 s) — long-horizon plant (v2)")
    sel_horizon = st.radio(
        "Forecast horizon",
        horizon_opts,
        horizontal=True,
        key="rollout_horizon",
    )
    is_1800 = sel_horizon.startswith("30 min")
    eval_results = eval_results_1800 if is_1800 else eval_results_180

    # ── 1b. Banner ─────────────────────────────────────────────────────
    if is_1800:
        n_windows_note = (
            f"stride {eval_results.get('stride_s', '?')} s · open-loop "
            f"(real plant inputs fed to the plant; controllers not in the loop)"
        )
        banner_text = (
            f"<b>Long-horizon evaluation (30 min)</b> using the v2 plant "
            f"(<code>v2_weighted_init_best.pt</code>, trained at target_len=1800 on vast.ai). "
            f"{n_windows_note}. Threshold {THRESHOLD*100:.1f}% (Student Guide §9 target)."
        )
    else:
        banner_text = (
            "<b>Validated horizon:</b> 3 min (target_len=180 s, training horizon). "
            f"All numbers below are the weighted plant's reported NRMSE from "
            f"<code>eval_results.json</code>. "
            f"Threshold bar at {THRESHOLD*100:.1f}% (Student Guide §9)."
        )
    st.markdown(
        f"<div style='background:rgba(96,165,250,0.08);border-left:3px solid {C_ACCENT};"
        f"padding:10px 14px;border-radius:0 4px 4px 0;margin-bottom:16px'>"
        f"<div style='color:{C_TEXT};font-size:0.9rem'>{banner_text}</div></div>",
        unsafe_allow_html=True,
    )
    if not is_1800 and eval_results_1800 is None:
        st.caption(
            "ℹ The 30-min horizon row is hidden because `cache/eval_1800s.json` is missing. "
            "Run `python dashboard/run_long_horizon_eval.py` to populate it."
        )

    # ── 2. Scenario selector ───────────────────────────────────────────
    scenarios = ["Overall", "Normal", "AP_no", "AP_with", "AE_no"]
    sel_scen = st.radio("Scenario", scenarios, horizontal=True, key="rollout_scen")

    # Pull per-PV NRMSE for the selected scenario
    per_pv_per_scen = eval_results.get("nrmse_per_pv_per_scenario", {})
    overall_per_pv = eval_results.get("overall_nrmse_per_pv", {})
    if sel_scen == "Overall":
        nrmse_per_pv = overall_per_pv
        scen_label = "Overall (all scenarios mixed)"
    else:
        nrmse_per_pv = {pv: per_pv_per_scen.get(pv, {}).get(sel_scen, None) for pv in PV_COLS}
        scen_label = f"Scenario: {sel_scen}"

    # ── 3. NRMSE bar chart per PV ──────────────────────────────────────
    st.markdown("<div class='pm-card'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='pm-card-title'>PER-PV NRMSE · {scen_label}</div>",
        unsafe_allow_html=True,
    )
    valid = [(pv, nrmse_per_pv.get(pv)) for pv in PV_COLS if nrmse_per_pv.get(pv) is not None]
    if valid:
        names, values = zip(*valid)
        colors = [C_OK if v < THRESHOLD else C_WARN if v < 3 * THRESHOLD else C_FAIL for v in values]
        fig = go.Figure(go.Bar(
            x=list(names), y=list(values),
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
            text=[f"{v:.4f}" for v in values], textposition="outside",
            textfont=dict(color=C_TEXT, size=12),
        ))
        fig.add_hline(y=THRESHOLD, line=dict(color=C_FAIL, dash="dash", width=1),
                      annotation_text=f"1.5% threshold", annotation_position="right",
                      annotation_font_color=C_FAIL)
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="NRMSE (lower = better)", gridcolor="rgba(255,255,255,0.05)"),
            height=320, margin=dict(l=50, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        st.info("No NRMSE data for the selected scenario.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 4. Per-PV pass/fail cards ──────────────────────────────────────
    if valid:
        st.markdown(
            f"<div style='color:{C_MUTED};font-size:0.85rem;margin-bottom:8px'>"
            f"Pass/fail pills based on the {THRESHOLD*100:.1f}% threshold."
            f"</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(len(valid))
        for col, (pv, v) in zip(cols, valid):
            if v < THRESHOLD:
                status = "PASS"
                color = C_OK
            elif v < 3 * THRESHOLD:
                status = "WARN"
                color = C_WARN
            else:
                status = "FAIL"
                color = C_FAIL
            with col:
                st.markdown(
                    f"<div class='pm-card' style='text-align:center;margin-bottom:8px'>"
                    f"<div style='color:{C_MUTED};font-size:0.7rem;font-weight:700'>{pv}</div>"
                    f"<div style='font-size:1.3rem;color:{color};font-weight:800'>{v:.4f}</div>"
                    f"<div>{pass_fail_pill(status)}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── 5. Per-scenario summary table ──────────────────────────────────
    scen_nrmse = eval_results.get("nrmse_per_scenario", {})
    if scen_nrmse:
        st.markdown("<div class='pm-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='pm-card-title'>NRMSE BY SCENARIO</div>"
            f"<div style='color:{C_MUTED};font-size:0.8rem;margin-bottom:8px'>"
            "Mean NRMSE across all 5 PVs for each scenario. AP_with (coordinated) is the hardest."
            "</div>",
            unsafe_allow_html=True,
        )
        scen_order_full = ["Normal", "AP_no", "AP_with", "AE_no"]
        scen_pairs = [(s, scen_nrmse.get(s)) for s in scen_order_full]
        # Drop scenarios where the eval JSON has null (no windows of that scenario hit)
        scen_pairs = [(s, v) for s, v in scen_pairs if v is not None]
        missing = [s for s in scen_order_full if scen_nrmse.get(s) is None]
        if missing:
            st.caption(
                f"⚠ No windows tagged as {', '.join(missing)} were found at this horizon "
                f"— their bars are hidden. (Stride may not align with attack regions.)"
            )
        scen_order = [s for s, _ in scen_pairs]
        scen_vals  = [v for _, v in scen_pairs]
        colors = [C_OK if v < THRESHOLD else C_WARN if v < 3 * THRESHOLD else C_FAIL for v in scen_vals]
        fig_s = go.Figure(go.Bar(
            x=scen_order, y=scen_vals,
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
            text=[f"{v:.4f}" for v in scen_vals], textposition="outside",
            textfont=dict(color=C_TEXT, size=12),
        ))
        fig_s.add_hline(y=THRESHOLD, line=dict(color=C_FAIL, dash="dash", width=1))
        fig_s.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="NRMSE (mean across 5 PVs)", gridcolor="rgba(255,255,255,0.05)"),
            height=260, margin=dict(l=50, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_s, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── 6. Degradation curve (normal vs attack) ────────────────────────
    normal = eval_results.get("normal_nrmse_per_pv", {})
    attack = eval_results.get("attack_nrmse_per_pv", {})
    if normal and attack:
        st.markdown("<div class='pm-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='pm-card-title'>NORMAL vs ATTACK NRMSE (per PV)</div>"
            f"<div style='color:{C_MUTED};font-size:0.8rem;margin-bottom:8px'>"
            "Attack windows are harder to forecast. Ratio attack/normal tells you which "
            "PVs the twin handles worst under adversarial conditions."
            "</div>",
            unsafe_allow_html=True,
        )
        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(
            name="Normal", x=PV_COLS, y=[normal.get(pv, 0) for pv in PV_COLS],
            marker_color=C_OK, text=[f"{normal.get(pv, 0):.3f}" for pv in PV_COLS],
            textposition="outside", textfont=dict(color=C_TEXT, size=10),
        ))
        fig_d.add_trace(go.Bar(
            name="Attack", x=PV_COLS, y=[attack.get(pv, 0) for pv in PV_COLS],
            marker_color=C_FAIL, text=[f"{attack.get(pv, 0):.3f}" for pv in PV_COLS],
            textposition="outside", textfont=dict(color=C_TEXT, size=10),
        ))
        fig_d.add_hline(y=THRESHOLD, line=dict(color=C_WARN, dash="dash", width=1))
        fig_d.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="NRMSE", gridcolor="rgba(255,255,255,0.05)"),
            barmode="group", height=300, margin=dict(l=50, r=20, t=20, b=40),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_d, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── 7. 30-min horizon status ───────────────────────────────────────
    v2_path = HERE / "training" / "checkpoints" / "v2_weighted_init_best.pt"
    if v2_path.exists() and eval_results_1800 is None:
        st.info(
            "ℹ Long-horizon plant (v2, target_len=1800) is available locally at "
            f"`training/checkpoints/v2_weighted_init_best.pt`. "
            "Run `python dashboard/run_long_horizon_eval.py` to populate the 30-min horizon row."
        )
    elif eval_results_1800 is not None and is_1800:
        st.caption(
            f"Source: `cache/eval_1800s.json` · "
            f"horizon {eval_results_1800.get('horizon_s', 1800)} s · "
            f"stride {eval_results_1800.get('stride_s', 600)} s · "
            f"open-loop (plant only, controllers bypassed)."
        )
