"""
chatbot_sidebar.py — PlantMirror Co-Pilot rendered in the Streamlit sidebar.

Wraps the Layer 4 chatbot (OpenAI GPT-4o-mini) so it's available on every tab.
LiveState is built from st.session_state entries that tabs can populate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# Use the existing `app/chatbot.py` so there's one source of truth for the
# SYSTEM_PROMPT. Extract just the non-Streamlit bits.
sys.path.insert(0, str(REPO / "app"))


def _build_live_state_summary() -> str:
    """Render the session_state into a compact LIVE_STATE block for the model."""
    ss = st.session_state
    lines = ["LIVE_STATE (ground any 'right now' answer in these numbers):"]
    for k, v in [
        ("current_tab", ss.get("_current_tab", "unknown")),
        ("guardian_threshold", ss.get("guardian_threshold", 0.35)),
        ("selected_loop", ss.get("selected_loop", "(none)")),
        ("last_attack_spec", ss.get("last_attack_spec", "(none run yet)")),
        ("last_alarm", ss.get("last_alarm", "(none)")),
    ]:
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def _get_api_key() -> str | None:
    """Three sources, first hit wins: st.secrets, env var, .openai_key file."""
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass
    import os
    env = os.environ.get("OPENAI_API_KEY")
    if env:
        return env.strip()
    for p in [REPO / "app" / ".openai_key", HERE / ".openai_key"]:
        if p.exists():
            txt = p.read_text(encoding="utf-8").strip()
            if txt:
                return txt
    return None


SYSTEM_PROMPT = """You are the PlantMirror Co-Pilot — an embedded educational assistant
for a digital twin of the HAI 23.05 P1 boiler process. Your job: help the operator
UNDERSTAND what they're looking at on the dashboard.

Five PVs, 1 Hz:
- P1_PIT01 pressure (bar), fast - reacts first to pump/valve attacks
- P1_LIT01 drum level (mm), slow ~10-30s lag
- P1_FT03Z feedwater flow (L/h), mid speed
- P1_TIT01 steam temperature (°C), very slow (60-120s thermal inertia)
- P1_TIT03 coolant temp (°C), coupled to TIT01

Five controllers (PC/LC/FC/TC/CC) drive valves and pumps to track setpoints.
4 scenarios: 0 Normal | 1 AP_no (single-point attack) | 2 AP_with (coordinated
multi-point) | 3 AE_no (sensor spoofing).

The Guardian (XGBoost) runs per-second binary detection (F1 0.587 @ threshold 0.35).
The GRU plant forecasts 180-s PV trajectories. The Assistive layer ranks diverged
PVs and walks the causal graph upstream for root cause.

Answer rules:
- Short, concrete, educational. Plant-engineer tone, not ML-paper tone.
- Use LIVE_STATE numbers when the user says 'right now' / 'currently' - never invent numbers.
- Connect concepts to DASHBOARD PANELS (which bar will move, which tab to open).
- End technical answers with one follow-up: "You could next ask about X."
- If unsure, say so. Don't hallucinate."""


def render() -> None:
    st.markdown("### 💬 PlantMirror Co-Pilot")
    st.caption("Ask anything about the twin, sensors, alerts, or how to read the dashboard.")

    key = _get_api_key()
    if not key:
        st.warning(
            "OpenAI API key not found. "
            "Drop a `.openai_key` file next to `chatbot_sidebar.py`, "
            "set `OPENAI_API_KEY`, or add to Streamlit secrets."
        )
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── render history ─────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── prompt suggestions when empty ─────────────────────────────────
    if not st.session_state.chat_history:
        with st.expander("💡 Quick-start questions", expanded=True):
            for q in [
                "What are the 4 attack scenarios?",
                "Which sensor reacts first in an attack?",
                "What does the XGBoost Hybrid Guardian do?",
                "What does the 4-experiment matrix prove?",
                "Why is start_offset = -30 important for attacks?",
            ]:
                if st.button("→ " + q, key=f"qs_{q}", use_container_width=True):
                    st.session_state._pending_question = q

    typed = st.chat_input("Type a question…")
    pending = st.session_state.pop("_pending_question", None)
    prompt = typed or pending

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": _build_live_state_summary()},
            ]
            messages.extend(st.session_state.chat_history[-8:])

            with st.chat_message("assistant"):
                placeholder = st.empty()
                out = ""
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.2,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    out += delta
                    placeholder.markdown(out + "▌")
                placeholder.markdown(out)
                st.session_state.chat_history.append({"role": "assistant", "content": out})
        except Exception as e:
            st.error(f"Chat error: {e}")

    if st.session_state.chat_history:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
