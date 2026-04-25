"""
generative.py — Synthetic Generation & Quality Analysis page.
"""

from __future__ import annotations

import hashlib
import time
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from twin_core import (
    INPUT_LEN,
    PV_COLS,
    SCENARIO_MAPPING,
    TARGET_LEN,
    ReplaySource,
    TwinBundle,
    closed_loop_rollout,
)
from twin_runtime import (
    _attack_clf,
    _attack_scaler,
    _CLF_LABELS,
    classify_window,
    _extract_features,
)

# ── Attack seed catalogue (real attack windows from test2.csv) ───────────────

_SEED_CATALOGUE: List[Dict] = [
    {"scenario": 1, "target": "P1-LC", "cursor": 5591},
    {"scenario": 1, "target": "P1-LC", "cursor": 13387},
    {"scenario": 1, "target": "P1-LC", "cursor": 21122},
    {"scenario": 1, "target": "P1-PC", "cursor": 31358},
    {"scenario": 1, "target": "P1-CC", "cursor": 42047},
    {"scenario": 1, "target": "P1-CC", "cursor": 50026},
    {"scenario": 1, "target": "P1-TC", "cursor": 63574},
    {"scenario": 1, "target": "P1-TC", "cursor": 71150},
    {"scenario": 1, "target": "P1-PC", "cursor": 82067},
    {"scenario": 1, "target": "P1-FC", "cursor": 85449},
    {"scenario": 2, "target": "P1-LC / P2-SC", "cursor": 154259},
    {"scenario": 2, "target": "P1-LC / P2-TC", "cursor": 156390},
    {"scenario": 2, "target": "P1-PC / P1-FC", "cursor": 160687},
    {"scenario": 2, "target": "P1-FC / P1-LC", "cursor": 164827},
    {"scenario": 2, "target": "P1-TC / P1-PC", "cursor": 178141},
    {"scenario": 2, "target": "P1-FC / P3-LC", "cursor": 189588},
    {"scenario": 2, "target": "P1-TC / P2-SC", "cursor": 192018},
    {"scenario": 2, "target": "P1-LC / P2-TC", "cursor": 198486},
    {"scenario": 3, "target": "P1-LC",  "cursor": 136448},
    {"scenario": 3, "target": "P1-HC",  "cursor": 140672},
    {"scenario": 3, "target": "P1-PC",  "cursor": 145625},
    {"scenario": 3, "target": "P1-CC",  "cursor": 150229},
    {"scenario": 3, "target": "P1-TC",  "cursor": 207195},
    {"scenario": 3, "target": "P1-PC",  "cursor": 224674},
    {"scenario": 3, "target": "P1-LC",  "cursor": 226620},
]

# ── Attack type descriptions ─────────────────────────────────────────────────

_CLASS_DESC = {
    0: ("Normal operation", "No attack. GRU generates baseline PV behaviour."),
    1: ("AP — Setpoint manipulation, single controller",
        "Injects a false setpoint into one controller loop (e.g. P1-LC). "
        "The controller drives the process variable toward the wrong target, "
        "causing measurable deviation in the targeted PV."),
    2: ("AP — Setpoint manipulation, multiple controllers",
        "Same as AP_no but two controllers are attacked simultaneously "
        "(e.g. P1-LC and P2-SC). Combined deviation makes detection harder."),
    3: ("AE — Sensor evasion, single controller",
        "Manipulates sensor readings to hide the attack. The controller "
        "appears to operate normally but the physical process is being "
        "covertly driven off-target."),
}

# Known classifier F1 per class from experiment C (mixed training)
_CLASS_F1 = {0: 0.99, 1: 0.04, 2: 0.0, 3: 0.0}

_SCENARIO_OPTIONS = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}
_CLASS_COLORS     = {0: "#10b981", 1: "#f59e0b", 2: "#ef4444", 3: "#a855f7"}


def _hex_rgb(h: str) -> str:
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _theme(fig: go.Figure, title: str, height: int = 280) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color="#00d2ff", family="JetBrains Mono", size=14)),
        height=height, margin=dict(l=45, r=20, t=55, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter", size=11),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)")
    return fig


# ── Main entry point ─────────────────────────────────────────────────────────

def render(bundle: TwinBundle, src: ReplaySource, t_end: int) -> None:
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;'
        'color:#64748b;letter-spacing:2px;margin-bottom:18px;">'
        'SYNTHETIC GENERATION &nbsp;·&nbsp; QUALITY ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    # ── Controls ─────────────────────────────────────────────────────────────
    st.markdown('<div class="panel" style="padding:20px;">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title" style="margin-bottom:14px;">Attack Parameters</div>',
                unsafe_allow_html=True)

    col_cls, col_tgt = st.columns([1, 1])

    with col_cls:
        selected_class = st.selectbox(
            "Attack Class",
            options=[1, 2, 3, 0],
            format_func=lambda k: f"{_SCENARIO_OPTIONS[k]}",
            index=0,
        )

    color = _CLASS_COLORS[selected_class]

    # Attack description box
    cls_title, cls_desc = _CLASS_DESC[selected_class]
    st.markdown(
        f'<div style="background:rgba({_hex_rgb(color)},0.06);border:1px solid '
        f'rgba({_hex_rgb(color)},0.25);border-radius:6px;padding:10px 14px;margin:10px 0;">'
        f'<div style="font-family:var(--mono);font-size:0.7rem;color:{color};'
        f'font-weight:700;margin-bottom:4px;">{cls_title}</div>'
        f'<div style="font-family:var(--sans);font-size:0.78rem;color:#94a3b8;">{cls_desc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Target selector + seed resolution
    available = [s for s in _SEED_CATALOGUE if s["scenario"] == selected_class]

    if selected_class == 0:
        with col_tgt:
            st.markdown(
                '<div style="font-family:var(--mono);font-size:0.8rem;color:#64748b;'
                'padding-top:28px;">No target — Normal baseline</div>',
                unsafe_allow_html=True,
            )
        selected_target = None
        seed_cursors = [min(t_end, max(INPUT_LEN + 1, len(src) - TARGET_LEN - 1))]
    else:
        target_options = sorted(set(s["target"] for s in available))
        with col_tgt:
            selected_target = st.selectbox("Target Controller", options=target_options, index=0)

        # What does targeting this controller mean?
        _ctrl_desc = {
            "P1-LC": "Level Control loop — manipulates tank level setpoint",
            "P1-PC": "Pressure Control loop — manipulates pump pressure setpoint",
            "P1-FC": "Flow Control loop — manipulates flow valve setpoint",
            "P1-TC": "Temperature Control loop — manipulates temperature setpoint",
            "P1-CC": "Conductivity/Chemistry loop — manipulates dosing setpoint",
            "P1-HC": "Hydraulic Control loop",
        }
        single = selected_target if "/" not in selected_target else None
        if single and single in _ctrl_desc:
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:0.68rem;color:#64748b;'
                f'margin-top:4px;">→ {_ctrl_desc[single]}</div>',
                unsafe_allow_html=True,
            )
        elif "/" in selected_target:
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:0.68rem;color:#64748b;'
                f'margin-top:4px;">→ Combined attack: {selected_target}</div>',
                unsafe_allow_html=True,
            )

        matching = [s for s in available if s["target"] == selected_target]
        seed_cursors = [min(s["cursor"], len(src) - TARGET_LEN - 1) for s in matching]
        if not seed_cursors:
            seed_cursors = [min(t_end, len(src) - TARGET_LEN - 1)]

    # F1 warning for rare classes
    f1 = _CLASS_F1.get(selected_class, 0.0)
    if f1 < 0.1 and selected_class > 0:
        st.markdown(
            f'<div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);'
            f'border-radius:6px;padding:8px 14px;font-family:var(--mono);font-size:0.72rem;'
            f'color:#f59e0b;margin-top:6px;">'
            f'⚠ Note: {_SCENARIO_OPTIONS[selected_class]} is a rare class in training data '
            f'(classifier F1 = {f1:.2f}). Generation quality will be lower — the GRU learned '
            f'limited examples of this attack pattern. Multi-seed best-selection is applied.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="font-family:var(--mono);font-size:0.68rem;color:#64748b;margin:10px 0 4px 0;">'
        f'Seeds available: <span style="color:#94a3b8;">{len(seed_cursors)}</span>'
        f'&nbsp;·&nbsp; strategy: best of all seeds by P(target class)</div>',
        unsafe_allow_html=True,
    )

    btn_col, *_ = st.columns([1, 5])
    with btn_col:
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        run = st.button("⚡ GENERATE", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if not run:
        st.markdown(
            '<div class="caption-mono" style="margin-top:30px;text-align:center;">'
            'Select a class and target, then click GENERATE.</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Multi-seed rollout — keep best by P(target class) ────────────────────
    t0 = time.time()
    best_out, best_prob, best_cursor, n_tried = None, -1.0, seed_cursors[0], 0

    with st.spinner(f"Trying {len(seed_cursors)} seed(s), keeping best by classifier confidence…"):
        for sc in seed_cursors:
            candidate = closed_loop_rollout(bundle, src, sc, scenario=selected_class)
            if candidate is None:
                continue
            n_tried += 1
            win = candidate["pv_scaled"][-TARGET_LEN:]
            feats = _extract_features(win)
            feats_sc = _attack_scaler.transform(feats)
            proba_c = _attack_clf.predict_proba(feats_sc)[0] if _attack_clf is not None else None
            p_target = float(proba_c[selected_class]) if proba_c is not None else 0.0
            if p_target > best_prob:
                best_prob, best_out, best_cursor = p_target, candidate, sc

    elapsed_ms = (time.time() - t0) * 1000

    if best_out is None:
        st.error("All rollouts failed: seed windows out of bounds.")
        return

    pv_phys   = best_out["pv_physical"]
    pv_scaled = best_out["pv_scaled"]
    cv_preds  = best_out["ctrl_cv_scaled"]
    out_hash  = hashlib.md5(pv_phys.tobytes()).hexdigest()[:8]

    # ── Section A — Trajectory ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="font-family:var(--mono);font-size:0.7rem;color:#64748b;'
        'letter-spacing:2px;margin-bottom:10px;">SECTION A · GENERATED TRAJECTORY</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    fig = make_subplots(rows=len(PV_COLS), cols=1, shared_xaxes=True,
                        subplot_titles=[f"<b>{pv}</b>" for pv in PV_COLS],
                        vertical_spacing=0.04)
    for i, pv in enumerate(PV_COLS):
        fig.add_trace(
            go.Scatter(x=list(range(TARGET_LEN)), y=pv_phys[:, i],
                       mode="lines", name=pv,
                       line=dict(color=color, width=2), showlegend=False),
            row=i + 1, col=1,
        )
    label_str = _SCENARIO_OPTIONS[selected_class]
    if selected_target:
        label_str += f" → {selected_target}"
    fig = _theme(fig, f"Synthetic PV — {label_str} (Physical Units)",
                 height=150 * len(PV_COLS))
    fig.update_xaxes(title="Seconds", row=len(PV_COLS), col=1)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel" style="margin-top:10px;">', unsafe_allow_html=True)
    fig2 = go.Figure()
    cv_colors = ["#00d2ff", "#10b981", "#f59e0b", "#a855f7", "#ef4444"]
    for i, (loop, cv) in enumerate(cv_preds.items()):
        fig2.add_trace(go.Scatter(x=list(range(TARGET_LEN)), y=cv, mode="lines",
                                  name=f"{loop} CV",
                                  line=dict(width=2, color=cv_colors[i % len(cv_colors)])))
    fig2 = _theme(fig2, "Controller Actuation Signals (Scaled)", height=220)
    fig2.update_xaxes(title="Seconds")
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section B — Quality Analysis ─────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="font-family:var(--mono);font-size:0.7rem;color:#64748b;'
        'letter-spacing:2px;margin-bottom:10px;">SECTION B · QUALITY ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    predicted_cls = classify_window(pv_scaled)
    clf_match = predicted_cls == _SCENARIO_OPTIONS[selected_class]
    clf_color = "#10b981" if clf_match else "#ef4444"
    clf_icon  = "✓ MATCH" if clf_match else "✗ MISMATCH"

    proba: Optional[np.ndarray] = None
    if _attack_clf is not None and hasattr(_attack_clf, "predict_proba"):
        win = pv_scaled[-TARGET_LEN:] if len(pv_scaled) >= TARGET_LEN else pv_scaled
        feats = _extract_features(win)
        feats_sc = _attack_scaler.transform(feats)
        proba = _attack_clf.predict_proba(feats_sc)[0]

    prob_target = float(proba[selected_class]) if proba is not None else None
    prob_str = f"{prob_target*100:.1f}%" if prob_target is not None else "n/a"

    st.markdown(
        f'<div style="background:rgba(148,163,184,0.06);border:1px solid rgba(148,163,184,0.18);'
        f'border-radius:6px;padding:12px 16px;margin-bottom:16px;">'
        f'<div style="font-family:var(--mono);font-size:0.72rem;color:#cbd5e1;font-weight:700;'
        f'margin-bottom:6px;">Correct class generation check</div>'
        f'<div style="font-family:var(--sans);font-size:0.8rem;line-height:1.55;color:#94a3b8;">'
        f'This section evaluates the quality of the synthetic data produced by the digital twin&apos;s '
        f'conditional generation capability. The key question is whether the generated 180-second PV '
        f'trajectory is close enough to real recordings from <b>{_SCENARIO_OPTIONS[selected_class]}</b> '
        f'that the balanced classifier maps it back to the same class. The verdict below checks whether '
        f'the correct class was generated, while the target-class probability shows how strongly the '
        f'synthetic window matches the requested scenario.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;">
          <div style="flex:1;min-width:140px;background:rgba(0,210,255,0.06);
               border:1px solid rgba(0,210,255,0.2);border-radius:6px;padding:12px 16px;">
            <div style="font-family:var(--mono);font-size:.65rem;color:#64748b;letter-spacing:1px;">BEST SEED · TRIED</div>
            <div style="font-family:var(--mono);font-size:.9rem;color:#00d2ff;font-weight:700;">{best_cursor:,}</div>
            <div style="font-family:var(--mono);font-size:.65rem;color:#475569;">{n_tried} seeds · {elapsed_ms:.0f} ms · {out_hash}</div>
          </div>
          <div style="flex:1;min-width:140px;background:rgba({_hex_rgb(clf_color)},0.06);
               border:1px solid rgba({_hex_rgb(clf_color)},0.3);border-radius:6px;padding:12px 16px;">
            <div style="font-family:var(--mono);font-size:.65rem;color:#64748b;letter-spacing:1px;">CORRECT CLASS GENERATED?</div>
            <div style="font-family:var(--mono);font-size:.9rem;color:{clf_color};font-weight:700;">{clf_icon}</div>
            <div style="font-family:var(--mono);font-size:.65rem;color:#475569;">predicted: {predicted_cls} · target: {_SCENARIO_OPTIONS[selected_class]}</div>
          </div>
          <div style="flex:1;min-width:140px;background:rgba({_hex_rgb(color)},0.06);
               border:1px solid rgba({_hex_rgb(color)},0.3);border-radius:6px;padding:12px 16px;">
            <div style="font-family:var(--mono);font-size:.65rem;color:#64748b;letter-spacing:1px;">P(CORRECT CLASS)</div>
            <div style="font-family:var(--mono);font-size:.9rem;color:{color};font-weight:700;">{prob_str}</div>
            <div style="font-family:var(--mono);font-size:.65rem;color:#475569;">probability assigned to the requested scenario</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    if not clf_match:
        reason = (
            f"Classifier F1 for {_SCENARIO_OPTIONS[selected_class]} = {f1:.2f} on real test data — "
            "rare class with few training examples. The GRU embedding shifts PV behaviour "
            "but the classifier cannot reliably distinguish it from Normal."
            if f1 < 0.1 else
            f"Try a different seed window or check the trajectory above for expected PV deviations."
        )
        st.markdown(
            f'<div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.35);'
            f'border-radius:6px;padding:10px 16px;font-family:var(--mono);font-size:0.75rem;'
            f'color:#ef4444;margin-bottom:14px;">'
            f'⚠ Predicted: <b>{predicted_cls}</b> · Target: <b>{_SCENARIO_OPTIONS[selected_class]}</b><br>'
            f'<span style="color:#94a3b8;">{reason}</span></div>',
            unsafe_allow_html=True,
        )

    if proba is not None:
        st.markdown('<div class="panel" style="margin-top:4px;">', unsafe_allow_html=True)
        cls_labels = [_SCENARIO_OPTIONS[i] for i in range(4)]
        bar_colors = [
            _CLASS_COLORS[i] if i == selected_class else "rgba(148,163,184,0.35)"
            for i in range(4)
        ]
        fig3 = go.Figure(go.Bar(
            x=cls_labels, y=list(proba),
            marker_color=bar_colors,
            text=[f"{v*100:.1f}%" for v in proba],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=12, color="#f8fafc"),
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        ))
        fig3 = _theme(fig3,
                      "Does the Synthetic Window Match the Requested Real Class?",
                      height=280)
        fig3.update_yaxes(title="Probability", range=[0, 1.15])
        fig3.update_xaxes(tickfont=dict(size=13, family="JetBrains Mono", color="#cbd5e1"))
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
