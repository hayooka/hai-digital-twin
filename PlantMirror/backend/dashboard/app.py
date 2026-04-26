"""
app.py — PlantMirror Final Dashboard (4 tabs + chatbot sidebar).
Run:  streamlit run app.py --server.port 8504
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from components import apply_css, metric_card, render_theme_toggle, C_MUTED
from student_guide_text import LIMITATIONS
from tabs import attack_sim, rollout, loop_explorer, classifier
import chatbot_sidebar

st.set_page_config(
    page_title="PlantMirror — Digital Twin Dashboard",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_css()

# ── Persistent header (renders on every tab) ──────────────────────────────

st.markdown("<div class='pm-header'>", unsafe_allow_html=True)
cols = st.columns([1, 1, 1, 1, 1])
with cols[0]:
    st.markdown(metric_card("TRAINING DATA", "194 h", "normal-only"), unsafe_allow_html=True)
with cols[1]:
    st.markdown(metric_card("MODEL SIZE", "5.25 M", "2×512 GRU"), unsafe_allow_html=True)
with cols[2]:
    st.markdown(metric_card("NRMSE (180 s)", "0.0095", "test set"), unsafe_allow_html=True)
with cols[3]:
    st.markdown(metric_card("CLASSIFIER F1", "0.587", "XGBoost Hybrid Guardian · AUROC 0.904"), unsafe_allow_html=True)
with cols[4]:
    st.markdown(metric_card("SCENARIOS", "4", "Normal · AP_no · AP_with · AE_no"), unsafe_allow_html=True)

with st.expander("ℹ About the twin — architecture, scope, honest limitations"):
    st.markdown(
        "<div style='color:#cbd5e1;font-size:0.9rem;margin-bottom:10px'>"
        "GRU 2×512 hidden · encoder window 300 s · decoder horizon 180 s · HAI 23.05 P1 boiler only."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Limitations (from the Student Guide §5):**")
    for lim in LIMITATIONS:
        st.markdown(
            f"<div class='pm-limit'>"
            f"<span class='pm-limit-id'>§{lim['id']}</span>"
            f"<div class='pm-limit-title'>{lim['title']}</div>"
            f"<div class='pm-limit-body'>{lim['body']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
st.markdown("</div>", unsafe_allow_html=True)


# ── 4 tabs ────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "⚔ Attack Simulator",
    "📈 Rollout Tester",
    "🔄 Loop Explorer",
    "🛡 Classifier Validation",
])

with tab1:
    attack_sim.render()

with tab2:
    rollout.render()

with tab3:
    loop_explorer.render()

with tab4:
    classifier.render()


# ── Sidebar (theme toggle + chatbot) ─────────────────────────────────────

with st.sidebar:
    render_theme_toggle()
    st.markdown("---")
    chatbot_sidebar.render()

st.markdown(
    f"<div style='text-align:center;color:{C_MUTED};font-size:0.75rem;"
    f"margin-top:40px'>PlantMirror Final Dashboard · HAI 23.05 · "
    f"model: gru_scenario_weighted (val 0.000310, test NRMSE 0.00949)</div>",
    unsafe_allow_html=True,
)
