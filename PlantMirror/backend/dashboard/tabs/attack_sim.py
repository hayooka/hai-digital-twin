"""
Tab 1 — Attack Simulator.

Two subtabs:
  A. Gallery  — grid of pre-rendered attacks from cache/attack_gallery.npz
  B. Custom   — live sliders; calls run_attack_sim directly
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
SAVED_DIR = HERE / "cache" / "saved_attacks"
SAVED_INDEX = SAVED_DIR / "index.json"

from components import C_ACCENT, C_FAIL, C_MUTED, C_OK, C_TEXT, mini_signal_chart  # noqa: E402
from data_loader import (  # noqa: E402
    AttackSpec, AttackType, InjectionPoint,
    LOOP_SPECS, PV_COLS, INPUT_LEN, TARGET_LEN,
    get_attack_gallery, get_bundle, get_replay, get_trts_rf, run_attack_sim,
)


# Per-loop slider ranges for the Custom subtab
MAG_DEFAULTS = {
    ("LC", "SP"): (-100.0, 100.0, 20.0),
    ("LC", "CV"): (-50.0, 50.0, 10.0),
    ("LC", "PV"): (-50.0, 50.0, 20.0),
    ("PC", "SP"): (-0.20, 0.20, 0.05),
    ("PC", "CV"): (-5.0, 5.0, 1.0),
    ("PC", "PV"): (-1.0, 1.0, 0.3),
    ("FC", "SP"): (-200.0, 200.0, 50.0),
    ("FC", "CV"): (-30.0, 30.0, 10.0),
    ("FC", "PV"): (-200.0, 200.0, 50.0),
    ("TC", "SP"): (-5.0, 5.0, 1.0),
    ("TC", "CV"): (-30.0, 30.0, 10.0),
    ("TC", "PV"): (-5.0, 5.0, 2.0),
    ("CC", "SP"): (-5.0, 5.0, 1.0),
    ("CC", "CV"): (-45.0, 45.0, 10.0),
    ("CC", "PV"): (-5.0, 5.0, 2.0),
}


def render() -> None:
    st.session_state["_current_tab"] = "Attack Simulator"
    st.markdown("## ⚔ Attack Simulator")
    st.caption("Generate counterfactual attack trajectories on the frozen twin.")

    sub_gallery, sub_custom = st.tabs(["🖼 Gallery (pre-rendered)", "🎛 Custom (live sliders)"])

    with sub_gallery:
        _render_gallery()

    with sub_custom:
        _render_custom()


# ============================================================================
# GALLERY
# ============================================================================

def _render_gallery() -> None:
    gal = get_attack_gallery()
    if gal is None:
        st.warning(
            "Gallery cache not found. Run `python precompute_gallery.py` from the "
            "dashboard directory to pre-render 30 attacks (takes ~25 seconds)."
        )
        return

    index = gal["index"]
    meta = gal["meta"]

    st.markdown(
        f"<div style='color:{C_MUTED};font-size:0.85rem'>"
        f"{meta.get('n', len(index))} attacks pre-rendered at cursor t_end={meta.get('cursor', '?')}, "
        f"start_offset={meta.get('start_offset', '?')}s, duration={meta.get('duration', '?')}s."
        f"</div>",
        unsafe_allow_html=True,
    )

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        loops = sorted(set(it["loop"] for it in index))
        sel_loop = st.multiselect("Target loop", loops, default=loops, key="gal_loop")
    with col_f2:
        ips = sorted(set(it["injection_point"] for it in index))
        sel_ip = st.multiselect("Injection point", ips, default=ips, key="gal_ip")
    with col_f3:
        types = sorted(set(it["attack_type"] for it in index))
        sel_type = st.multiselect("Attack type", types, default=types, key="gal_type")

    filtered_idx = [
        i for i, it in enumerate(index)
        if it["loop"] in sel_loop
        and it["injection_point"] in sel_ip
        and it["attack_type"] in sel_type
    ]
    if not filtered_idx:
        st.info("No attacks match the current filters.")
        return

    st.markdown(f"**{len(filtered_idx)} attacks shown**")

    # Grid of cards — 3 per row
    for row_start in range(0, len(filtered_idx), 3):
        cols = st.columns(3)
        for col, idx in zip(cols, filtered_idx[row_start:row_start + 3]):
            with col:
                _render_gallery_card(gal, idx)


def _render_gallery_card(gal: dict, idx: int) -> None:
    it = gal["index"][idx]
    card_key = f"gal_{idx}_{it['loop']}_{it['injection_point']}_{it['attack_type']}"
    b = gal["baseline_pv"][idx]   # (180, 5)
    a = gal["attacked_pv"][idx]   # (180, 5)
    loops = list(LOOP_SPECS.keys())
    pv_for_loop = LOOP_SPECS[it["loop"]]["base_cols"][1]
    pv_i = PV_COLS.index(pv_for_loop) if pv_for_loop in PV_COLS else 1
    b_pv = b[:, pv_i]
    a_pv = a[:, pv_i]
    peak_dpv = float(np.max(np.abs(a_pv - b_pv)))
    recovery = _recovery_time(a_pv, b_pv)

    with st.container():
        st.markdown(
            f"<div class='pm-card' style='margin-bottom:0'>"
            f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
            f"<div style='font-weight:700;color:{C_ACCENT};font-family:monospace;font-size:1rem'>"
            f"{it['loop']} · {it['injection_point']} · {it['attack_type']}</div>"
            f"<div style='color:{C_MUTED};font-size:0.7rem'>"
            f"mag={it['magnitude']:g}</div></div>",
            unsafe_allow_html=True,
        )
        # Mini chart: baseline vs attacked on the loop's PV
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=b_pv, mode="lines", name="baseline",
                                 line=dict(color=C_ACCENT, width=1.5), showlegend=False))
        fig.add_trace(go.Scatter(y=a_pv, mode="lines", name="attacked",
                                 line=dict(color=C_FAIL, width=1.5, dash="dash"), showlegend=False))
        # Attack window band
        lab = gal["attack_label"][idx]
        if lab.sum() > 0:
            a_start = int(np.argmax(lab))
            a_end = len(lab) - int(np.argmax(lab[::-1]))
            fig.add_vrect(x0=a_start, x1=a_end, fillcolor=C_FAIL,
                          opacity=0.08, line_width=0)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=110, margin=dict(l=5, r=5, t=5, b=5),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=card_key)
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;font-size:0.75rem;color:{C_MUTED};margin-top:-5px;margin-bottom:5px'>"
            f"<span>peak ΔPV = <b style='color:{C_TEXT}'>{peak_dpv:.3f}</b></span>"
            f"<span>recovery = <b style='color:{C_TEXT}'>{recovery}</b></span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def _recovery_time(attacked: np.ndarray, baseline: np.ndarray, tol: float = 0.05) -> str:
    diff = np.abs(attacked - baseline)
    scale = max(float(np.max(np.abs(baseline))), 1e-6)
    relative = diff / scale
    # Last second above tolerance
    above = np.where(relative > tol)[0]
    if above.size == 0:
        return "no drift"
    last = int(above[-1])
    if last >= len(attacked) - 1:
        return ">180 s"
    return f"{last} s"


# ============================================================================
# CUSTOM (live sliders)
# ============================================================================

def _render_custom() -> None:
    bundle = get_bundle()
    src = get_replay()

    max_t = len(src) - TARGET_LEN - 1
    min_t = INPUT_LEN

    cfg_row1 = st.columns([1, 1, 1, 2])
    with cfg_row1[0]:
        loop = st.selectbox("Target loop", list(LOOP_SPECS.keys()), index=1, key="cus_loop")
    with cfg_row1[1]:
        injection = st.selectbox("Injection point", [ip.value for ip in InjectionPoint],
                                 index=0, key="cus_ip")
    with cfg_row1[2]:
        atk_type = st.selectbox("Attack type", [at.value for at in AttackType], key="cus_at")
    with cfg_row1[3]:
        scenario = st.selectbox(
            "Scenario embedding",
            [0, 1, 2, 3],
            format_func=lambda s: {0: "0 · Normal", 1: "1 · AP_no", 2: "2 · AP_with", 3: "3 · AE_no"}[s],
            index=0,
            key="cus_scen",
        )

    cfg_row2 = st.columns([1, 1, 2])
    with cfg_row2[0]:
        start_offset = st.slider(
            "Start offset (s)", -299, 179, -30, key="cus_start",
            help=(
                "Seconds relative to the cursor. NEGATIVE = attack was "
                "already active before the cursor, so the controllers see the "
                "spoofed input in their 300-s history window and re-plan bad CVs "
                "-- this is what makes the 5-PV chart visibly drift. "
                "0 or POSITIVE = attack stays in the target window; for SP/PV "
                "attacks the plant's PV output barely changes because "
                "the controller's CV plan was already fixed at cursor time."
            ),
        )
    with cfg_row2[1]:
        duration = st.slider("Duration (s)", 1, 180, 90, key="cus_dur")
    with cfg_row2[2]:
        lo_m, hi_m, def_m = MAG_DEFAULTS.get((loop, injection), (-50.0, 50.0, 10.0))
        if atk_type == "replay":
            mag = st.slider("Replay lag (s)", 1, 180, 30, key="cus_mag")
        else:
            mag = st.slider("Magnitude (physical units)",
                            float(lo_m), float(hi_m), float(def_m),
                            step=float((hi_m - lo_m) / 100.0), key="cus_mag")

    cursor = st.slider("Replay cursor (t_end)", int(min_t), int(max_t),
                       int(min_t + 700), step=60, key="cus_cursor")

    if st.button("▶ Run attack", type="primary", use_container_width=True, key="cus_run"):
        spec = AttackSpec(
            target_loop=loop,
            injection_point=InjectionPoint(injection),
            attack_type=AttackType(atk_type),
            start_offset=int(start_offset),
            duration=int(duration),
            magnitude=float(mag),
        )
        with st.spinner("Running baseline + attacked rollouts…"):
            result = run_attack_sim(bundle, src, int(cursor), spec, scenario=int(scenario))
        if result is None:
            st.error("Cursor out of range.")
            return
        st.session_state["cus_result"] = result
        st.session_state["last_attack_spec"] = (
            f"{loop}/{injection}/{atk_type} start={start_offset:+d}s "
            f"dur={duration}s mag={mag:g} scenario={scenario}"
        )

    result = st.session_state.get("cus_result")
    if result is None:
        st.info("Configure an attack above and click **Run attack**.")
        return

    # ── 3-row target-loop chart + 5-row PV overlay ─────────────────────
    spec = result.spec
    sig = result.signals
    atk_start, atk_end = spec.target_window()
    hist_s, hist_e = spec.history_window()

    st.markdown(
        f"<div class='pm-card'>"
        f"<div class='pm-card-title'>{spec.target_loop} · {spec.injection_point.value} · {spec.attack_type.value}</div>"
        f"<div style='color:{C_MUTED};font-size:0.8rem'>"
        f"start_offset={spec.start_offset:+d}s · duration={spec.duration}s · "
        f"magnitude={spec.magnitude:g} · "
        f"history window: [{hist_s - INPUT_LEN}, {hist_e - INPUT_LEN}) · "
        f"target window: [{atk_start}, {atk_end})"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    fig1 = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=[
            f"{spec.target_loop} · SP (real vs seen)",
            f"{spec.target_loop} · CV (intended vs actual)",
            f"{spec.target_loop} · PV (real vs seen)",
        ],
    )
    fig1.add_trace(go.Scatter(y=sig["SP_real"], name="SP_real",
                              line=dict(color=C_ACCENT, width=1.6)), row=1, col=1)
    fig1.add_trace(go.Scatter(y=sig["SP_seen"], name="SP_seen",
                              line=dict(color=C_FAIL, width=1.6, dash="dash")), row=1, col=1)
    fig1.add_trace(go.Scatter(y=sig["CV_intended"], name="CV_intended",
                              line=dict(color=C_OK, width=1.6)), row=2, col=1)
    fig1.add_trace(go.Scatter(y=sig["CV_actual"], name="CV_actual",
                              line=dict(color=C_FAIL, width=1.6, dash="dash")), row=2, col=1)
    fig1.add_trace(go.Scatter(y=sig["PV_real"], name="PV_real",
                              line=dict(color="#c084fc", width=1.6)), row=3, col=1)
    fig1.add_trace(go.Scatter(y=sig["PV_seen"], name="PV_seen",
                              line=dict(color=C_FAIL, width=1.6, dash="dash")), row=3, col=1)
    for r in (1, 2, 3):
        if atk_end > atk_start:
            fig1.add_vrect(x0=atk_start, x1=atk_end, fillcolor=C_FAIL, opacity=0.10,
                           line_width=0, row=r, col=1)
    fig1.update_layout(
        template="plotly_dark", height=620,
        margin=dict(l=50, r=20, t=60, b=40),
        legend=dict(orientation="h", y=-0.08),
    )
    fig1.update_xaxes(title_text="time (s)", row=3, col=1)
    st.plotly_chart(fig1, width="stretch", config={"displayModeBar": False})

    # 5-PV overlay
    fig2 = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                         subplot_titles=PV_COLS)
    baseline_pv = result.baseline["pv_physical"]
    attacked_pv = result.attacked["pv_physical"]
    for i, name in enumerate(PV_COLS):
        fig2.add_trace(go.Scatter(y=baseline_pv[:, i], name="baseline" if i == 0 else None,
                                  showlegend=(i == 0),
                                  line=dict(color=C_ACCENT, width=1.3)), row=i + 1, col=1)
        fig2.add_trace(go.Scatter(y=attacked_pv[:, i], name="attacked" if i == 0 else None,
                                  showlegend=(i == 0),
                                  line=dict(color=C_FAIL, width=1.3, dash="dash")), row=i + 1, col=1)
        fig2.add_vrect(x0=atk_start, x1=atk_end, fillcolor=C_FAIL,
                       opacity=0.10, line_width=0, row=i + 1, col=1)
    fig2.update_layout(
        template="plotly_dark", height=780,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.04),
    )
    fig2.update_xaxes(title_text="time (s)", row=5, col=1)
    st.markdown("#### All 5 PVs — baseline vs attacked")
    # Hint when the attack produces near-zero divergence on plant output
    max_dpv = float(np.max(np.abs(attacked_pv - baseline_pv)))
    if max_dpv < 1e-3:
        st.warning(
            f"⚠ Plant output is essentially unchanged (max |Δ PV| = {max_dpv:.5f}). "
            "This happens when the attack lives entirely in the target window "
            f"(start_offset = {spec.start_offset}). For **{spec.injection_point.value} attack** "
            "to visibly drift the plant, set **start_offset < 0** so the controller "
            "re-plans on the spoofed input. The 3-row chart above still shows the "
            f"operator-view divergence (SP_seen / CV_actual / PV_seen) correctly."
        )
    st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

    # Metrics row
    pv_diffs = np.abs(attacked_pv - baseline_pv).max(axis=0)
    m_cols = st.columns(5)
    for i, name in enumerate(PV_COLS):
        m_cols[i].metric(f"Δ {name}", f"{pv_diffs[i]:.3f}")

    # ── Save attack ────────────────────────────────────────────────────
    st.markdown("---")
    save_cols = st.columns([3, 1])
    default_name = (
        f"{spec.target_loop}_{spec.injection_point.value}_{spec.attack_type.value}"
        f"_{spec.start_offset:+d}_{int(spec.magnitude):g}"
    )
    with save_cols[0]:
        save_name = st.text_input(
            "Name for this attack",
            value=default_name,
            key="cus_save_name",
            help="A descriptive name. Saved attacks persist on disk in cache/saved_attacks/.",
        )
    with save_cols[1]:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("💾 Save attack", key="cus_save", use_container_width=True):
            scen = int(st.session_state.get("cus_scen", 0))
            try:
                new_id = _save_attack(result, save_name, scen)
                st.success(f"Saved as #{new_id} → cache/saved_attacks/{new_id}.npz")
            except Exception as e:
                st.error(f"Save failed: {type(e).__name__}: {e}")

    # Saved attacks list (always visible if any exist)
    _render_saved_attacks()


# ============================================================================
# SAVED ATTACKS — persistent storage + replay
# ============================================================================

def _load_saved_index() -> list:
    if not SAVED_INDEX.exists():
        return []
    try:
        with open(SAVED_INDEX) as f:
            return json.load(f).get("items", [])
    except Exception:
        return []


def _save_attack(result, name: str, scenario: int) -> int:
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    items = _load_saved_index()
    new_id = max([it["id"] for it in items], default=0) + 1
    spec = result.spec
    sig = result.signals

    # Reconstruct the full 133-column physical-units trajectory so the
    # XGBoost Guardian (per-second, 133 features) can score the attack.
    bundle = get_bundle()
    scalers = bundle.scalers
    pv_idx_arr = np.asarray(scalers.pv_idx, dtype=np.int64)
    plant_in_idx_arr = np.asarray(scalers.plant_in_idx, dtype=np.int64)
    x_scaled = result.attacked["x_cv_target_used"]   # (180, 128) plant-scaled
    plant_scale = np.asarray(scalers.plant_scale)
    plant_mean  = np.asarray(scalers.plant_mean)
    plant_in_phys = (
        x_scaled * plant_scale[plant_in_idx_arr]
        + plant_mean[plant_in_idx_arr]
    )
    pv_phys = result.attacked["pv_physical"]         # (180, 5) physical
    full_133 = np.zeros((pv_phys.shape[0], len(scalers.sensor_cols)), dtype=np.float32)
    full_133[:, plant_in_idx_arr] = plant_in_phys.astype(np.float32)
    full_133[:, pv_idx_arr]       = pv_phys.astype(np.float32)

    np.savez_compressed(
        SAVED_DIR / f"{new_id}.npz",
        baseline_pv=result.baseline["pv_physical"],
        attacked_pv=result.attacked["pv_physical"],
        attack_label=result.attack_label,
        full_133=full_133,
        sensor_cols=np.array(list(scalers.sensor_cols)),
        SP_real=sig.get("SP_real", np.zeros(0)),
        SP_seen=sig.get("SP_seen", np.zeros(0)),
        CV_intended=sig.get("CV_intended", np.zeros(0)),
        CV_actual=sig.get("CV_actual", np.zeros(0)),
        PV_real=sig.get("PV_real", np.zeros(0)),
        PV_seen=sig.get("PV_seen", np.zeros(0)),
    )
    max_dpv = float(np.max(np.abs(result.attacked["pv_physical"] - result.baseline["pv_physical"])))
    items.append({
        "id": new_id,
        "name": (name or f"attack_{new_id}").strip(),
        "loop": spec.target_loop,
        "injection_point": spec.injection_point.value,
        "attack_type": spec.attack_type.value,
        "start_offset": spec.start_offset,
        "duration": spec.duration,
        "magnitude": spec.magnitude,
        "scenario": scenario,
        "max_dpv": max_dpv,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })
    with open(SAVED_INDEX, "w") as f:
        json.dump({"items": items}, f, indent=2, default=float)
    return new_id


def _delete_attack(item_id: int) -> None:
    items = [it for it in _load_saved_index() if it["id"] != item_id]
    with open(SAVED_INDEX, "w") as f:
        json.dump({"items": items}, f, indent=2, default=float)
    npz = SAVED_DIR / f"{item_id}.npz"
    if npz.exists():
        npz.unlink()


def _render_saved_attacks() -> None:
    items = _load_saved_index()
    if not items:
        return
    with st.expander(f"📁 Saved attacks ({len(items)})", expanded=False):
        st.caption(
            "Persistent across browser sessions. Stored in `dashboard/cache/saved_attacks/`. "
            "Useful for building a personal demo set you can re-show without re-running the sliders."
        )

        # Score saved attacks against the production XGBoost Hybrid Guardian
        test_cols = st.columns([3, 1, 1])
        with test_cols[0]:
            st.markdown(
                f"<div style='color:{C_MUTED};font-size:0.8rem;margin-top:6px'>"
                "Score every saved attack against the XGBoost Hybrid Guardian "
                "(per-second binary detector, F1 0.587). "
                "Shows whether each attack would be caught."
                "</div>",
                unsafe_allow_html=True,
            )
        with test_cols[1]:
            if st.button("🛡 Test all", key="saved_test_all", use_container_width=True):
                _score_all_saved(items)
        with test_cols[2]:
            # Show this only if there's at least one legacy save to clean up
            legacy_ids = [
                it["id"] for it in items
                if (SAVED_DIR / f"{it['id']}.npz").exists()
                and "full_133" not in np.load(
                    SAVED_DIR / f"{it['id']}.npz", allow_pickle=True,
                ).files
            ]
            if legacy_ids:
                if st.button(f"🗑 Delete {len(legacy_ids)} legacy",
                             key="saved_purge_legacy",
                             use_container_width=True,
                             help="Removes saves missing the full 133-column data "
                                  "(created before the Guardian-scoring update)."):
                    for _id in legacy_ids:
                        _delete_attack(_id)
                    st.session_state.pop("view_saved_id", None)
                    st.session_state.pop("saved_test_results", None)
                    st.rerun()

        results = st.session_state.get("saved_test_results")
        if results:
            _render_saved_test_table(results)

        st.markdown("---")
        for it in items:
            row = st.columns([4, 2, 1, 1])
            with row[0]:
                st.markdown(f"**#{it['id']} · {it['name']}**")
                st.caption(
                    f"{it['loop']} / {it['injection_point']} / {it['attack_type']} · "
                    f"start={it['start_offset']:+d}s · dur={it['duration']}s · "
                    f"mag={it['magnitude']:g} · scen={it['scenario']}"
                )
            with row[1]:
                st.markdown(
                    f"<div style='color:{C_MUTED};font-size:0.7rem;font-weight:700;margin-top:6px'>MAX ΔPV</div>"
                    f"<div style='color:{C_TEXT};font-size:1.1rem;font-weight:700'>{it['max_dpv']:.3f}</div>",
                    unsafe_allow_html=True,
                )
            with row[2]:
                if st.button("View", key=f"view_{it['id']}"):
                    st.session_state["view_saved_id"] = it["id"]
            with row[3]:
                if st.button("🗑", key=f"del_{it['id']}"):
                    _delete_attack(it["id"])
                    st.session_state.pop("view_saved_id", None)
                    st.rerun()

        view_id = st.session_state.get("view_saved_id")
        if view_id is not None and (SAVED_DIR / f"{view_id}.npz").exists():
            _render_saved_view(int(view_id), items)


def _score_all_saved(items: list) -> None:
    """Score every saved attack through the XGBoost Hybrid Guardian (per-second).

    The Guardian is a per-row binary detector on 133 raw sensor columns. We
    score each of the 180 seconds of the attack window and report the PEAK
    attack probability as the verdict for that attack — i.e., did *any*
    second in the attack window exceed the decision threshold?
    """
    try:
        g = get_trts_rf()           # legacy name, returns the Guardian
    except Exception as e:
        st.error(f"Could not load Guardian: {type(e).__name__}: {e}")
        return
    feature_list = g["features"]
    scaler = g["scaler"]
    model  = g["model"]
    thr    = g["threshold"]
    rows = []
    for it in items:
        npz_path = SAVED_DIR / f"{it['id']}.npz"
        spec_str = f"{it['loop']}/{it['injection_point']}/{it['attack_type']}"
        base = {
            "id": it["id"], "name": it["name"], "spec": spec_str,
            "max_dpv": it.get("max_dpv", 0.0),
            "threshold": thr, "peak_proba": None, "mean_proba": None,
        }
        if not npz_path.exists():
            rows.append({**base, "proba": None, "decision": "file missing"})
            continue
        try:
            arr = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            rows.append({**base, "proba": None, "decision": f"load failed: {type(e).__name__}"})
            continue
        if "full_133" not in arr.files or "sensor_cols" not in arr.files:
            rows.append({**base, "proba": None, "decision": "legacy save · re-run attack"})
            continue
        try:
            full = np.asarray(arr["full_133"])
            cols_in_save = [str(c) for c in np.asarray(arr["sensor_cols"]).tolist()]
            missing = [f for f in feature_list if f not in cols_in_save]
            if missing:
                rows.append({**base, "proba": None,
                             "decision": f"missing {len(missing)} guardian features"})
                continue
            col_index = [cols_in_save.index(f) for f in feature_list]
            X = full[:, col_index]
            Xs = scaler.transform(X)
            proba_per_sec = model.predict_proba(Xs)[:, 1]
            peak  = float(proba_per_sec.max())
            mean_ = float(proba_per_sec.mean())
            rows.append({**base, "proba": peak, "peak_proba": peak, "mean_proba": mean_,
                         "decision": "caught" if peak >= thr else "missed"})
        except Exception as e:
            rows.append({**base, "proba": None,
                         "decision": f"score failed: {type(e).__name__}"})
    st.session_state["saved_test_results"] = {
        "rows":           rows,
        "n_train":        g["n_train"],
        "n_train_attack": g["n_train_attack"],
        "model_name":     g["name"],
    }


def _render_saved_test_table(results: dict) -> None:
    rows = results["rows"]
    if not rows:
        return
    scorable = [r for r in rows if r.get("decision") in ("caught", "missed")]
    n_caught = sum(1 for r in scorable if r["decision"] == "caught")
    model_name = results.get("model_name", "Classifier")
    summary = (
        f"<div style='color:{C_MUTED};font-size:0.8rem;margin:8px 0'>"
        f"<b>{n_caught}/{len(scorable)}</b> caught at threshold {rows[0]['threshold']:.2f}"
        f"{' · ' + str(len(rows) - len(scorable)) + ' unscorable' if len(rows) > len(scorable) else ''} · "
        f"{model_name} trained on {results['n_train']:,} rows "
        f"({results['n_train_attack']:,} attack) · "
        f"each attack scored per-second over its 180-s window; verdict = peak attack probability."
        "</div>"
    )
    table = (
        f"<table style='width:100%;border-collapse:collapse;font-size:0.85rem;color:{C_TEXT};margin-bottom:6px'>"
        f"<tr style='background:rgba(255,255,255,0.05);color:{C_MUTED};font-weight:700;text-transform:uppercase;font-size:0.7rem'>"
        f"<th style='padding:8px;text-align:left'>#</th>"
        f"<th style='padding:8px;text-align:left'>Name</th>"
        f"<th style='padding:8px;text-align:left'>Spec</th>"
        f"<th style='padding:8px;text-align:right'>max ΔPV</th>"
        f"<th style='padding:8px;text-align:right'>Attack proba</th>"
        f"<th style='padding:8px;text-align:center'>Verdict</th></tr>"
    )
    for r in rows:
        proba = r.get("proba")
        decision = r.get("decision", "n/a")
        if proba is not None:
            is_caught = decision == "caught"
            verdict_color = C_OK if is_caught else C_FAIL
            verdict_label = "✓ CAUGHT" if is_caught else "✗ MISSED"
            bar_w = max(0, min(100, int(proba * 100)))
            bar_color = C_OK if is_caught else C_FAIL
            proba_cell = (
                f"<div style='display:inline-block;width:80px;background:rgba(255,255,255,0.05);"
                f"border-radius:3px;overflow:hidden;vertical-align:middle;margin-right:6px'>"
                f"<div style='width:{bar_w}%;height:14px;background:{bar_color}'></div></div>"
                f"<span style='font-family:monospace'>{proba:.3f}</span>"
            )
        else:
            verdict_color = C_WARN
            verdict_label = decision.upper()
            proba_cell = f"<span style='color:{C_MUTED};font-style:italic'>—</span>"
        max_dpv = r.get("max_dpv", 0.0) or 0.0
        table += (
            f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05)'>"
            f"<td style='padding:8px;color:{C_MUTED}'>#{r['id']}</td>"
            f"<td style='padding:8px;font-weight:600'>{r['name']}</td>"
            f"<td style='padding:8px;font-family:monospace;color:{C_ACCENT}'>{r['spec']}</td>"
            f"<td style='padding:8px;text-align:right'>{max_dpv:.3f}</td>"
            f"<td style='padding:8px;text-align:right'>{proba_cell}</td>"
            f"<td style='padding:8px;text-align:center;color:{verdict_color};font-weight:700;font-size:0.8rem'>{verdict_label}</td>"
            f"</tr>"
        )
    table += "</table>"
    st.markdown(summary + table, unsafe_allow_html=True)


def _render_saved_view(item_id: int, items: list) -> None:
    meta = next((it for it in items if it["id"] == item_id), None)
    if meta is None:
        return
    npz_path = SAVED_DIR / f"{item_id}.npz"
    if not npz_path.exists():
        st.error(f"Saved attack #{item_id} file is missing on disk: {npz_path}")
        return
    try:
        arr = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        st.error(f"Could not load saved attack #{item_id}: {type(e).__name__}: {e}")
        return
    if "baseline_pv" not in arr.files or "attacked_pv" not in arr.files:
        st.error(
            f"Saved attack #{item_id} is missing required arrays "
            f"(baseline_pv / attacked_pv). Delete it and re-run the attack."
        )
        return
    baseline = np.asarray(arr["baseline_pv"])
    attacked = np.asarray(arr["attacked_pv"])
    label = (np.asarray(arr["attack_label"])
             if "attack_label" in arr.files
             else np.zeros(baseline.shape[0], dtype=np.int8))

    st.markdown(
        f"<div class='pm-card' style='margin-top:10px'>"
        f"<div class='pm-card-title'>VIEWING #{meta['id']} · {meta['name']}</div>"
        f"<div style='color:{C_MUTED};font-size:0.8rem'>"
        f"{meta['loop']} / {meta['injection_point']} / {meta['attack_type']} · "
        f"start_offset={meta['start_offset']:+d}s · duration={meta['duration']}s · "
        f"magnitude={meta['magnitude']:g} · scenario={meta['scenario']} · "
        f"saved {meta['timestamp'][:19]}Z"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        subplot_titles=PV_COLS,
    )
    a_start = int(np.argmax(label)) if label.sum() > 0 else 0
    a_end = len(label) - int(np.argmax(label[::-1])) if label.sum() > 0 else 0
    for i, name in enumerate(PV_COLS):
        fig.add_trace(
            go.Scatter(y=baseline[:, i], name="baseline" if i == 0 else None,
                       showlegend=(i == 0),
                       line=dict(color=C_ACCENT, width=1.3)),
            row=i + 1, col=1,
        )
        fig.add_trace(
            go.Scatter(y=attacked[:, i], name="attacked" if i == 0 else None,
                       showlegend=(i == 0),
                       line=dict(color=C_FAIL, width=1.3, dash="dash")),
            row=i + 1, col=1,
        )
        if a_end > a_start:
            fig.add_vrect(x0=a_start, x1=a_end, fillcolor=C_FAIL,
                          opacity=0.10, line_width=0, row=i + 1, col=1)
    fig.update_layout(
        template="plotly_dark", height=600,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.04),
    )
    fig.update_xaxes(title_text="time (s)", row=5, col=1)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False},
                    key=f"saved_chart_{item_id}")
    pv_diffs = np.abs(attacked - baseline).max(axis=0)
    m_cols = st.columns(5)
    for i, name in enumerate(PV_COLS):
        m_cols[i].metric(f"Δ {name}", f"{pv_diffs[i]:.3f}")
    if st.button("✕ Close", key=f"close_{item_id}"):
        st.session_state.pop("view_saved_id", None)
        st.rerun()
