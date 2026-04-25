"""
Tab 3 — Loop Explorer.

Layout:
  1. Loop reference table (5 rows x SP/PV/CV/range/delay)
  2. KS fidelity bars (green/amber/red; values from Student Guide §3.3)
  3. Loop selector (PC/LC/FC/TC/CC) — all below zooms to picked loop
  4. Spec panel for selected loop
  5. Radar chart (5 axes: fidelity/speed/coupling/coverage/sensitivity)
  6. Cross-loop coupling heatmap
  7. Insights card
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import plotly.graph_objects as go  # noqa: E402

from components import (  # noqa: E402
    C_ACCENT, C_FAIL, C_MUTED, C_OK, C_TEXT, C_WARN,
    ks_fidelity_bars, pass_fail_pill, radar_chart,
)
from data_loader import (  # noqa: E402
    LOOP_SPECS, get_causal_graph, get_coupling_matrix,
    get_eval_results, get_response_delays,
)
from student_guide_text import KS_PER_LOOP_REPORTED  # noqa: E402


LOOP_DESCRIPTIONS = {
    "PC": ("Pressure Control", "Regulates drum pressure via the main pressure-control valve. Fastest loop — CV→PV delay near 0 s."),
    "LC": ("Level Control",    "Regulates drum water level via the level-control valve. Slow — level responds to imbalance between inflow and outflow."),
    "FC": ("Flow Control",     "Regulates feedwater flow to the boiler. Mid-speed; tightly coupled to PC through valve positions."),
    "TC": ("Temperature Control", "Regulates steam-outlet temperature via the temp-regulating valve. Longest time constant (thermal inertia)."),
    "CC": ("Cooling Control",  "Regulates coolant temperature via the cooling pump (on/off gated). Hysteresis around switching threshold."),
}

LOOP_RANGES = {
    # (SP range, PV range) from Student Guide §4 + operating constraints
    "PC": ("0.1–0.3", "0–10 bar"),
    "LC": ("300–500 mm", "0–720 mm"),
    "FC": ("900–1100 L/h", "0–2500 L/h"),
    "TC": ("25–30 °C", "0–50 °C"),
    "CC": ("26–30 °C", "0–50 °C"),
}


def render() -> None:
    st.session_state["_current_tab"] = "Loop Explorer"
    st.markdown("## 🔄 Loop Explorer")
    st.caption("Per-loop dynamics, fidelity, and cross-loop coupling.")

    # ── 1. Loop reference table ────────────────────────────────────────
    delays = get_response_delays()
    rows = []
    for loop, spec in LOOP_SPECS.items():
        ks = KS_PER_LOOP_REPORTED.get(loop, {"ks": None, "status": "?"})
        rows.append({
            "Loop": loop,
            "Name": LOOP_DESCRIPTIONS[loop][0],
            "SP column": spec["base_cols"][0],
            "PV column": spec["base_cols"][1],
            "CV column": spec["cv_col"],
            "CV→PV lag (s)": delays.get(loop, "—"),
            "KS": f"{ks['ks']:.3f}" if ks['ks'] is not None else "—",
            "Status": ks['status'],
        })
    df = pd.DataFrame(rows)
    st.markdown("<div class='pm-card'><div class='pm-card-title'>Loop reference</div>", unsafe_allow_html=True)
    st.dataframe(df, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 2. KS fidelity bars ────────────────────────────────────────────
    st.markdown("<div class='pm-card'>", unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.plotly_chart(ks_fidelity_bars(KS_PER_LOOP_REPORTED),
                        width="stretch", config={"displayModeBar": False})
    with col_b:
        st.markdown(
            "<div style='padding: 10px'>"
            f"<div class='pm-card-title'>KS FIDELITY</div>"
            f"<div style='color:{C_MUTED};font-size:0.82rem;line-height:1.5'>"
            f"Kolmogorov-Smirnov statistic on tracking-error distributions. "
            f"Measures whether synthetic data from the twin matches real data.<br><br>"
            f"<span style='color:{C_OK}'>● &lt; 0.10 PASS</span><br>"
            f"<span style='color:{C_WARN}'>● 0.10–0.20 WARN</span><br>"
            f"<span style='color:{C_FAIL}'>● &gt; 0.20 FAIL</span><br><br>"
            f"Source: Student Guide §3.3 (reported).</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 3. Loop selector + zoomed-in panels ────────────────────────────
    st.markdown("### Zoom into one loop")
    loop_sel = st.radio(
        "Select a loop",
        options=list(LOOP_SPECS.keys()),
        horizontal=True,
        label_visibility="collapsed",
        key="loop_sel",
    )
    st.session_state["selected_loop"] = loop_sel

    # ── 4. Spec panel ──────────────────────────────────────────────────
    spec = LOOP_SPECS[loop_sel]
    name, desc = LOOP_DESCRIPTIONS[loop_sel]
    sp_range, pv_range = LOOP_RANGES[loop_sel]
    ks = KS_PER_LOOP_REPORTED[loop_sel]

    cols = st.columns([1, 1])
    with cols[0]:
        st.markdown(
            f"<div class='pm-card'>"
            f"<div class='pm-card-title'>{loop_sel} · {name}</div>"
            f"<div style='color:{C_TEXT};font-size:0.9rem;line-height:1.5'>{desc}</div>"
            f"<table style='margin-top:10px;font-size:0.85rem;color:{C_TEXT}'>"
            f"<tr><td style='color:{C_MUTED};padding:3px 10px 3px 0'>SP column</td><td><code>{spec['base_cols'][0]}</code></td></tr>"
            f"<tr><td style='color:{C_MUTED};padding:3px 10px 3px 0'>PV column</td><td><code>{spec['base_cols'][1]}</code></td></tr>"
            f"<tr><td style='color:{C_MUTED};padding:3px 10px 3px 0'>CV column</td><td><code>{spec['cv_col']}</code></td></tr>"
            f"<tr><td style='color:{C_MUTED};padding:3px 10px 3px 0'>SP range</td><td>{sp_range}</td></tr>"
            f"<tr><td style='color:{C_MUTED};padding:3px 10px 3px 0'>PV range</td><td>{pv_range}</td></tr>"
            f"<tr><td style='color:{C_MUTED};padding:3px 10px 3px 0'>CV→PV lag</td><td>{delays.get(loop_sel, '—')} s</td></tr>"
            f"<tr><td style='color:{C_MUTED};padding:3px 10px 3px 0'>Fidelity</td><td>{pass_fail_pill(ks['status'])} KS = {ks['ks']:.3f}</td></tr>"
            f"</table></div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        # Radar chart: 4 real axes (no placeholders), 0..1 higher=better.
        # Axis labels use <br> for line breaks (plotly polar supports HTML).
        radar_axes = [
            "Fidelity<br>(threshold-anchored)",
            "Response<br>speed",
            "Coupling<br>(outgoing)",
            "Forecast<br>accuracy",
        ]
        # 1. Fidelity — anchored to the §3.3 PASS/FAIL thresholds so the radar
        #    matches the bar chart above:
        #      KS = 0.00 → 1.00 (perfect)
        #      KS = 0.10 → 0.50 (the PASS line)
        #      KS = 0.20 → 0.00 (the FAIL line)
        #    Linear interpolation, capped to [0, 1].
        FAIL_KS = 0.20
        fidelity = max(0.0, min(1.0, 1.0 - ks["ks"] / FAIL_KS))
        # 2. Response speed: inverse of delay, normalised by the slowest loop
        max_delay = max(delays.values()) if delays else 1
        speed = 1.0 - (delays.get(loop_sel, 0) / max(max_delay, 1))
        # 3. Coupling: row sum of coupling matrix (this loop's outgoing influence)
        cm = get_coupling_matrix()
        loops = cm["loops"]
        mat = np.array(cm["matrix"])
        idx = loops.index(loop_sel)
        max_row_sum = float(mat.sum(axis=1).max()) or 1e-6
        coupling = float(mat[idx].sum()) / max_row_sum
        # 4. Forecast accuracy: 1 - (per-PV NRMSE / max NRMSE), real number from eval_results
        ev = get_eval_results()
        per_pv = ev.get("overall_nrmse_per_pv", {})
        loop_pv = LOOP_SPECS[loop_sel]["base_cols"][1]
        if per_pv and loop_pv in per_pv:
            max_nrmse = max(per_pv.values())
            forecast_acc = 1.0 - (per_pv[loop_pv] / max_nrmse) if max_nrmse > 0 else 0.0
        else:
            forecast_acc = 0.0

        values = [fidelity, speed, coupling, forecast_acc]
        # Compare vs mean of others (using the SAME 4 metrics)
        all_values = []
        for lp in LOOP_SPECS.keys():
            if lp == loop_sel:
                continue
            kks = KS_PER_LOOP_REPORTED[lp]
            f = max(0.0, min(1.0, 1.0 - kks["ks"] / FAIL_KS))
            sp_ = 1.0 - (delays.get(lp, 0) / max(max_delay, 1))
            cp = float(mat[loops.index(lp)].sum()) / max_row_sum
            other_pv = LOOP_SPECS[lp]["base_cols"][1]
            if per_pv and other_pv in per_pv:
                max_nrmse = max(per_pv.values())
                fa = 1.0 - (per_pv[other_pv] / max_nrmse) if max_nrmse > 0 else 0.0
            else:
                fa = 0.0
            all_values.append([f, sp_, cp, fa])
        compare = list(np.mean(all_values, axis=0)) if all_values else None

        st.plotly_chart(
            radar_chart(radar_axes, values, compare, label=loop_sel, compare_label="mean of others"),
            width="stretch", config={"displayModeBar": False},
            key=f"radar_{loop_sel}",
        )
        st.caption(
            "Axes: **Fidelity** = `1 - KS / 0.20`, threshold-anchored to §3.3 "
            "(KS=0 → 1.0 perfect, KS=0.10 → 0.5 PASS line, KS=0.20 → 0 FAIL line) — "
            "so a FAIL on the bar chart above shows up as a low fidelity here. "
            "**Speed** = inverse of CV->PV lag, normalised by slowest loop. "
            "**Coupling** = outgoing influence (row sum) on other loops. "
            "**Forecast accuracy** = `1 - NRMSE / max(NRMSE)` from `eval_results.json`. "
            "All 0..1, higher = better."
        )

    # ── 5. Per-scenario forecast accuracy for the selected loop ────────
    ev = get_eval_results()
    per_pv_per_scen = ev.get("nrmse_per_pv_per_scenario", {})
    loop_pv = LOOP_SPECS[loop_sel]["base_cols"][1]
    scen_data = per_pv_per_scen.get(loop_pv, {})
    if scen_data:
        st.markdown("<div class='pm-card'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='pm-card-title'>{loop_sel} ({loop_pv}) — FORECAST NRMSE BY SCENARIO</div>"
            f"<div style='color:{C_MUTED};font-size:0.8rem;margin-bottom:8px'>"
            "Lower bar = the twin forecasts this loop more accurately under that scenario. "
            "Threshold line at 1.5% (Student Guide §9). AP_with (combined) is usually the hardest."
            "</div>",
            unsafe_allow_html=True,
        )
        order = ["Normal", "AP_no", "AP_with", "AE_no"]
        vals = [scen_data.get(s, 0) for s in order]
        thr = 0.015
        colors = [C_OK if v < thr else C_WARN if v < 3 * thr else C_FAIL for v in vals]
        fig = go.Figure(go.Bar(
            x=order, y=vals,
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
            text=[f"{v:.4f}" for v in vals], textposition="outside",
            textfont=dict(color=C_TEXT, size=12),
        ))
        fig.add_hline(y=thr, line=dict(color=C_FAIL, dash="dash", width=1),
                      annotation_text="1.5% threshold", annotation_position="right",
                      annotation_font_color=C_FAIL)
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="NRMSE (lower = better)", gridcolor="rgba(255,255,255,0.05)"),
            height=280, margin=dict(l=50, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False},
                        key=f"scen_{loop_sel}")

        # 4 quick-summary cards under the bar chart
        cards = st.columns(4)
        labels = ["Normal", "AP_no\n(single-point)", "AP_with\n(coordinated)", "AE_no\n(sensor spoof)"]
        for c, sc, lbl, v in zip(cards, order, labels, vals):
            status = "PASS" if v < thr else "WARN" if v < 3 * thr else "FAIL"
            color  = C_OK if status == "PASS" else C_WARN if status == "WARN" else C_FAIL
            with c:
                st.markdown(
                    f"<div class='pm-card' style='text-align:center;margin-bottom:0'>"
                    f"<div style='color:{C_MUTED};font-size:0.7rem;font-weight:700'>{lbl.replace(chr(10),'<br>')}</div>"
                    f"<div style='font-size:1.2rem;color:{color};font-weight:800'>{v:.4f}</div>"
                    f"<div>{pass_fail_pill(status)}</div></div>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info(f"No per-scenario NRMSE breakdown for {loop_pv} in `eval_results.json`.")

    # ── 6. Insights ────────────────────────────────────────────────────
    insights = []
    if delays:
        slow_loop, slow_lag = max(delays.items(), key=lambda kv: kv[1])
        fast_loop, fast_lag = min(delays.items(), key=lambda kv: kv[1])
        insights.append(
            f"**{slow_loop}** has the longest CV→PV delay ({slow_lag} s) — "
            f"slowest physical time constant of the 5 loops. "
            f"**{fast_loop}** is the fastest at {fast_lag} s."
        )
    # Worst-case scenario per the eval JSON
    scen_means = ev.get("nrmse_per_scenario", {})
    if scen_means:
        worst = max(scen_means.items(), key=lambda kv: kv[1])
        insights.append(
            f"**{worst[0]}** is the hardest scenario for the twin overall "
            f"(mean NRMSE {worst[1]:.4f} across all 5 PVs). "
            "Combined attacks (AP_with) typically dominate this number."
        )
    # Per-loop tip
    if scen_data:
        worst_scen, worst_v = max(scen_data.items(), key=lambda kv: kv[1])
        insights.append(
            f"For **{loop_sel}** specifically, the twin struggles most under "
            f"**{worst_scen}** (NRMSE {worst_v:.4f}). Use the Attack Simulator "
            f"to inject {worst_scen}-shape attacks on this loop and see how the "
            "Guardian responds."
        )
    failing = [l for l, v in KS_PER_LOOP_REPORTED.items() if v["status"] == "FAIL"]
    if failing:
        insights.append(
            f"KS fidelity failures on **{', '.join(failing)}** — synthetic data "
            "from the twin is less statistically similar to real data for these loops. "
            "Report synthetic-data experiments on these loops with explicit caveats."
        )
    if insights:
        st.markdown(
            "<div class='pm-card'>"
            "<div class='pm-card-title'>KEY INSIGHTS</div>"
            "<ul style='margin:0;padding-left:20px;line-height:1.6'>" +
            "".join(f"<li style='color:#e5e7eb'>{ins}</li>" for ins in insights) +
            "</ul></div>",
            unsafe_allow_html=True,
        )
