"""
PlantMirror — Attack Simulator

Interactive ICS-attack simulator on top of the frozen GRU digital twin.
Pick a target loop, an injection point (SP / CV / PV), an attack type,
onset + duration + magnitude — get baseline vs attacked trajectories,
the target-loop SP/CV/PV real-vs-seen traces, and a CSV export.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
GEN_DIR = HERE.parent / "generator"
if str(GEN_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_DIR))

from core import (  # type: ignore
    INPUT_LEN,
    LOOP_ORDER,
    LOOP_SPECS,
    PV_COLS,
    TARGET_LEN,
    default_paths,
    load_bundle,
    load_replay,
)

from attacks import (  # type: ignore
    AttackSpec,
    AttackType,
    InjectionPoint,
    run_attack_sim,
)

# ── Page setup ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PlantMirror · Attack Simulator",
    page_icon="⚠",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background: #0a1122; color: #e5e7eb; }
      h1 { color: #f87171 !important; }
      .stRadio label, .stSlider label, .stSelectbox label { color: #e5e7eb !important; }
      [data-testid="stMetricValue"] { color: #f87171; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚠ PlantMirror — Attack Simulator")
st.caption(
    "Inject SP / CV / PV attacks into the frozen GRU twin and inspect the "
    "clean-vs-attacked trajectories. Plant and controllers are frozen; only "
    "the signals at the injection point change."
)

# ── Load frozen bundle (cached) ─────────────────────────────────────────────

PATHS = default_paths()


@st.cache_resource(show_spinner="Loading GRU plant + 5 controllers…")
def _load():
    bundle = load_bundle(PATHS["ckpt_dir"], PATHS["split_dir"])
    src = load_replay(PATHS["test_csvs"][0], bundle.scalers)
    return bundle, src


try:
    bundle, src = _load()
except Exception as exc:
    st.error(f"Failed to load artifacts: {exc}")
    st.stop()

max_t_end = len(src) - TARGET_LEN - 1
min_t_end = INPUT_LEN

st.success(
    f"Ready · plant {bundle.plant.n_plant_in}-in/{bundle.plant.n_pv}-out · "
    f"{len(bundle.controllers)} controllers · replay "
    f"{len(src):,} ticks (usable cursors {min_t_end:,}–{max_t_end:,})"
)

# ── Attack configuration ────────────────────────────────────────────────────

st.markdown("### Attack spec")

cfg_row1 = st.columns([1, 1, 1, 2])
with cfg_row1[0]:
    loop = st.selectbox("Target loop", LOOP_ORDER, index=1, key="loop")
with cfg_row1[1]:
    injection = st.selectbox(
        "Injection point",
        [ip.value for ip in InjectionPoint],
        index=0,
        help=(
            "SP: spoof the setpoint the controller sees.  "
            "CV: override the controller's actuator command before it reaches the plant.  "
            "PV: spoof the PV feedback the controller sees."
        ),
        key="injection",
    )
with cfg_row1[2]:
    atk_type = st.selectbox(
        "Attack type",
        [at.value for at in AttackType],
        index=0,
        help=(
            "bias: add magnitude to the signal.  "
            "freeze: hold the signal at its onset value.  "
            "replay: substitute the value from `magnitude` seconds earlier."
        ),
        key="atk_type",
    )
with cfg_row1[3]:
    scenario = st.selectbox(
        "Plant scenario embedding",
        [0, 1, 2, 3],
        format_func=lambda s: {0: "0 · normal", 1: "1 · AP_no", 2: "2 · AP_with", 3: "3 · AE_no"}[s],
        index=0,
        key="scenario",
    )

cfg_row2 = st.columns([1, 1, 2])
with cfg_row2[0]:
    start_offset = st.slider(
        "Start offset (s, 0 = cursor)",
        -(INPUT_LEN - 1), TARGET_LEN - 1, 0,
        help=(
            "Seconds relative to t_end.  "
            "0 = attack starts at the cursor.  "
            "Negative = attacker was already active that many seconds before "
            "the cursor — history window is affected, so controllers see it "
            "and re-plan.  "
            "Positive = attack starts that many seconds into the target — "
            "controllers already planned normally; only the plant reacts "
            "(for CV/SP)."
        ),
        key="start_offset",
    )
with cfg_row2[1]:
    # Duration cap: attack can't run past the end of the target window.
    max_dur = max(1, TARGET_LEN - int(start_offset)) if int(start_offset) >= 0 \
        else INPUT_LEN + TARGET_LEN - 1 + int(start_offset) + 1
    default_dur = min(60, int(max_dur))
    duration = st.slider(
        "Duration (s)", 1, int(max_dur), int(default_dur),
        key="duration",
    )

# Magnitude ranges adapted per signal type for usable defaults
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
lo_m, hi_m, def_m = MAG_DEFAULTS.get((loop, injection), (-50.0, 50.0, 10.0))
mag_label = "Magnitude (phys units)" if atk_type != "replay" else "Replay lag (s)"
with cfg_row2[2]:
    if atk_type == "replay":
        magnitude = st.slider(mag_label, 1, 180, 30, key="magnitude")
    else:
        magnitude = st.slider(
            mag_label, float(lo_m), float(hi_m), float(def_m),
            step=float((hi_m - lo_m) / 100.0), key="magnitude",
        )

cfg_row3 = st.columns([2, 1, 1])
with cfg_row3[0]:
    cursor = st.slider(
        "Replay cursor (t_end)",
        int(min_t_end), int(max_t_end), int(min_t_end + 1000),
        step=60, key="cursor",
    )
with cfg_row3[1]:
    run_btn = st.button("▶ RUN ATTACK", use_container_width=True, type="primary")
with cfg_row3[2]:
    clear_btn = st.button("✕ CLEAR", use_container_width=True)

if clear_btn:
    st.session_state.pop("atk_result", None)

# ── Execute ─────────────────────────────────────────────────────────────────

if run_btn:
    spec = AttackSpec(
        target_loop=loop,
        injection_point=InjectionPoint(injection),
        attack_type=AttackType(atk_type),
        start_offset=int(start_offset),
        duration=int(duration),
        magnitude=float(magnitude),
    )
    t0 = time.time()
    with st.spinner("Running baseline + attacked rollouts…"):
        result = run_attack_sim(bundle, src, int(cursor), spec, scenario=int(scenario))
    if result is None:
        st.error("Cursor out of range for a full 300+180s window.")
    else:
        st.session_state["atk_result"] = result
        st.session_state["atk_t_s"] = time.time() - t0

# ── Render ──────────────────────────────────────────────────────────────────

result = st.session_state.get("atk_result")
if result is None:
    st.info("Configure an attack and hit **RUN ATTACK** to see the comparison.")
    st.stop()

spec = result.spec
sig = result.signals
label = result.attack_label
t_axis = np.arange(TARGET_LEN)
atk_start, atk_end = spec.target_window()
hist_s, hist_e = spec.history_window()

st.markdown("---")
st.subheader(
    f"{spec.target_loop} · {spec.injection_point.value} attack · "
    f"{spec.attack_type.value} · "
    f"start={spec.start_offset:+d}s · dur={spec.duration}s · "
    f"|m|={abs(spec.magnitude):g}"
)
hist_note = (
    f"history: [{hist_s - INPUT_LEN}, {hist_e - INPUT_LEN}) · "
    if hist_e > hist_s else "history: (none) · "
)
tgt_note = (
    f"target: [{atk_start}, {atk_end})" if atk_end > atk_start else "target: (none)"
)
st.caption(
    f"ran in {st.session_state.get('atk_t_s', 0):.1f}s · "
    f"cursor={result.t_end} · scenario={result.scenario} · {hist_note}{tgt_note}"
)

# ── Target-loop signal comparison (3 rows) ─────────────────────────────────

fig1 = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    subplot_titles=[
        f"{spec.target_loop} · SP (real vs seen)",
        f"{spec.target_loop} · CV (intended vs actual)",
        f"{spec.target_loop} · PV (real vs seen)",
    ],
)
# SP
fig1.add_trace(go.Scatter(y=sig["SP_real"], name="SP_real",
                          line=dict(color="#60a5fa", width=1.6)), row=1, col=1)
fig1.add_trace(go.Scatter(y=sig["SP_seen"], name="SP_seen",
                          line=dict(color="#f87171", width=1.6, dash="dash")), row=1, col=1)
# CV
fig1.add_trace(go.Scatter(y=sig["CV_intended"], name="CV_intended",
                          line=dict(color="#34d399", width=1.6)), row=2, col=1)
fig1.add_trace(go.Scatter(y=sig["CV_actual"], name="CV_actual",
                          line=dict(color="#f87171", width=1.6, dash="dash")), row=2, col=1)
# PV
fig1.add_trace(go.Scatter(y=sig["PV_real"], name="PV_real",
                          line=dict(color="#c084fc", width=1.6)), row=3, col=1)
fig1.add_trace(go.Scatter(y=sig["PV_seen"], name="PV_seen",
                          line=dict(color="#f87171", width=1.6, dash="dash")), row=3, col=1)

# Highlight attack window in each row
for r in (1, 2, 3):
    fig1.add_vrect(
        x0=atk_start, x1=atk_end, fillcolor="#f87171", opacity=0.10,
        line_width=0, row=r, col=1,
    )

fig1.update_layout(
    template="plotly_dark", height=620,
    margin=dict(l=50, r=20, t=60, b=40),
    legend=dict(orientation="h", y=-0.08),
)
fig1.update_xaxes(title_text="time (s)", row=3, col=1)
st.plotly_chart(fig1, use_container_width=True)

# ── All 5 PVs: baseline vs attacked overlay ────────────────────────────────

st.markdown("#### All 5 PVs — baseline vs attacked")
fig2 = make_subplots(
    rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035,
    subplot_titles=PV_COLS,
)
baseline_pv = result.baseline["pv_physical"]
attacked_pv = result.attacked["pv_physical"]
for i, name in enumerate(PV_COLS):
    fig2.add_trace(go.Scatter(y=baseline_pv[:, i], name="baseline" if i == 0 else None,
                              showlegend=(i == 0),
                              line=dict(color="#60a5fa", width=1.4)),
                   row=i + 1, col=1)
    fig2.add_trace(go.Scatter(y=attacked_pv[:, i], name="attacked" if i == 0 else None,
                              showlegend=(i == 0),
                              line=dict(color="#f87171", width=1.4, dash="dash")),
                   row=i + 1, col=1)
    fig2.add_vrect(x0=atk_start, x1=atk_end, fillcolor="#f87171",
                   opacity=0.10, line_width=0, row=i + 1, col=1)

fig2.update_layout(
    template="plotly_dark", height=900,
    margin=dict(l=50, r=20, t=50, b=40),
    legend=dict(orientation="h", y=-0.04),
)
fig2.update_xaxes(title_text="time (s)", row=5, col=1)
st.plotly_chart(fig2, use_container_width=True)

# ── Diff metrics ───────────────────────────────────────────────────────────

def _win(x):
    return x[atk_start:atk_end]

m_cols = st.columns(5)
pv_diffs = np.abs(attacked_pv - baseline_pv).max(axis=0)
for i, (name, col) in enumerate(zip(PV_COLS, m_cols)):
    col.metric(f"Δ {name} max", f"{pv_diffs[i]:.3f}")

# ── CSV export ─────────────────────────────────────────────────────────────

df_out = pd.DataFrame({
    "t": t_axis,
    "attack_label": label,
    f"{spec.target_loop}_SP_real": sig["SP_real"],
    f"{spec.target_loop}_SP_seen": sig["SP_seen"],
    f"{spec.target_loop}_CV_intended": sig["CV_intended"],
    f"{spec.target_loop}_CV_actual": sig["CV_actual"],
    f"{spec.target_loop}_PV_real": sig["PV_real"],
    f"{spec.target_loop}_PV_seen": sig["PV_seen"],
})
for i, name in enumerate(PV_COLS):
    df_out[f"{name}_baseline"] = baseline_pv[:, i]
    df_out[f"{name}_attacked"] = attacked_pv[:, i]

buf = io.StringIO()
df_out.to_csv(buf, index=False)
st.download_button(
    f"⬇ Download attack trace ({len(df_out)} rows × {len(df_out.columns)} cols)",
    data=buf.getvalue(),
    file_name=(
        f"attack_{spec.target_loop}_{spec.injection_point.value}_"
        f"{spec.attack_type.value}_s{spec.start_offset:+d}_d{spec.duration}_"
        f"m{spec.magnitude:g}_t{result.t_end}.csv"
    ),
    mime="text/csv",
    use_container_width=True,
)

st.markdown(
    "<hr style='margin-top:40px;opacity:0.2'/>"
    "<div style='text-align:center;opacity:0.5;font-size:0.8rem'>"
    "PlantMirror Attack Simulator · frozen GRU twin · SP/CV/PV injection"
    "</div>",
    unsafe_allow_html=True,
)
