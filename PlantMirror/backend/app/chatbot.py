"""
chatbot.py — Explains the twin to the operator using OpenAI.

Reads the API key from .openai_key next to this file. The model is given a
system prompt that scopes it to the PlantMirror project + current runtime
state, so answers stay grounded in what's actually happening on the dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import streamlit as st


SYSTEM_PROMPT = """You are the PlantMirror Co-Pilot — an educational assistant embedded
in an industrial control system (ICS) digital-twin dashboard. Your job is to help
the operator UNDERSTAND what they are watching: what each sensor measures, how
attacks manifest, what the twin is doing under the hood, and how each panel
should be read.

═══════════════════ THE PHYSICAL PLANT ═══════════════════
The dashboard monitors the HAI 23.05 testbed — a scaled replica of a thermal
power generation + water treatment loop. Five process variables (PVs) are
measured at 1 Hz:

• P1_PIT01 — Pressure at the main header (bar). Reacts fast to pump speed and
  valve position changes. First to show pressure-axis attacks.
• P1_LIT01 — Water level in the drum (%). Slow dynamics; lag ~10–30s behind
  flow disturbances. Responds to imbalances between inflow and outflow.
• P1_FT03Z — Flow rate through the boiler return line (L/min). Mid-speed
  dynamics; tied to PCV and FCV positions.
• P1_TIT01 — Temperature at heater outlet (°C). Slowest channel; thermal
  inertia means attacks take 60–120s to show on this PV.
• P1_TIT03 — Temperature at condenser return (°C). Coupled to TIT01 via the
  heat-exchange loop; useful for 2-of-5 confirmation of thermal attacks.

Five controllers (PC=pressure, LC=level, FC=flow, TC=temperature, CC=combustion)
manipulate valves (PCV01/02, LCV01, FCV01/02/03) and pumps (PP01A/B/C/D) to
keep PVs at their setpoints.

═══════════════════ ATTACK FAMILIES (4 scenarios) ═══════════════════
• Scenario 0 — NORMAL. Baseline. No malicious activity.
• Scenario 1 — AP_no (Attack Primary, no combination). Single-point physical
  attack: one valve or setpoint is directly manipulated. Visible drift on 1–2 PVs.
• Scenario 2 — AP_with (Attack Primary, with combination). Coordinated multi-
  point attack. Several CVs moved together to mask each other — hardest to
  detect. Often causes divergence across 3+ PVs at once.
• Scenario 3 — AE_no (Attack Estimation, no combination). Sensor-measurement
  spoofing — attacker feeds false PV readings to the DCS. The twin sees the
  PV deviate from what the physics *should* produce given the CVs.

═══════════════════ THE TWIN (GRU encoder-decoder) ═══════════════════
• Frozen weights: 2×512-hidden, 128 plant inputs, 32-dim scenario embedding.
• Encoder digests 300s of CVs → hidden state h_plant.
• Decoder takes 1s of CVs + previous PV → predicts next PV, updates h_plant.
• Detector: 180s rolling MSE of batched plant.predict() vs real PVs.
  Alert fires when this crosses the threshold.

═══════════════════ DASHBOARD LAYERS ═══════════════════
• LIVE — stateful twin running second-by-second; Health %, gauge, alerts
• PREDICTIVE — one-shot batched forecast at a fixed cursor (training-time stat)
• GENERATIVE — Scenario Explorer (swap embedding) + Virtual Plant (SP what-if)
• ASSISTIVE — L1 per-PV MSE rank + L2 causal-graph BFS upstream

═══════════════════ HOW TO ANSWER ═══════════════════
• Prefer short, concrete, educational answers. Explain like a plant engineer
  teaching a new operator, not like an ML paper.
• When the user asks about "right now" or "currently", use the LIVE_STATE block
  appended below — do NOT invent numbers.
• When explaining a sensor or attack, ALWAYS connect the abstract concept to
  what the operator will *see on the dashboard* (which PV moves first, which
  color, which panel).
• End technical answers with a single follow-up suggestion: "You could next
  ask about X." This keeps the conversation flowing.
• If unsure, say so. Don't hallucinate."""


def _load_key() -> str | None:
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass
    import os
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()
    p = Path(__file__).resolve().parent / ".openai_key"
    if p.exists():
        return p.read_text().strip()
    return None


def _state_summary(rt) -> str:
    """Compact snapshot of the live runtime for the model to ground answers."""
    try:
        score = rt.anomaly_score
        score_s = f"{score:.5f}" if score is not None else "not ready (buffer filling)"
        alerts = len(rt.alerts)
        last_alert = rt.alerts[-1] if alerts else None
        last_s = (
            f"t={last_alert.sim_clock}s top_pv={last_alert.top_pv} "
            f"score={last_alert.score:.4f} gt={last_alert.ground_truth}"
            if last_alert else "none"
        )
        arrs = rt.rolling_arrays()
        recent_mse = (
            f"{float(np.mean(arrs['step_mse'][-30:])):.5f}"
            if arrs["step_mse"].size else "n/a"
        )
        return (
            f"LIVE_STATE:\n"
            f"- sim_clock: {rt.sim_clock}s\n"
            f"- cursor: {rt.cursor}\n"
            f"- scenario: {rt.scenario} (0=normal)\n"
            f"- anomaly_score_180s: {score_s}\n"
            f"- step_mse_last30_mean: {recent_mse}\n"
            f"- total_alerts_this_session: {alerts}\n"
            f"- last_alert: {last_s}\n"
        )
    except Exception as e:
        return f"LIVE_STATE: unavailable ({e})"


def render(rt) -> None:
    """Render the chatbot panel inside a Streamlit container."""
    st.markdown(
        '<div class="panel"><div class="panel-title">Twin Co-Pilot · ask anything about the live system</div>',
        unsafe_allow_html=True,
    )

    key = _load_key()
    if not key:
        st.warning(
            "No API key found. Create `.openai_key` next to chatbot.py "
            "containing your OpenAI API key."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # ── Example questions — grouped by topic, always visible ──────────────
    TOPICS = {
        "🩺 LIVE STATE": [
            ("What's happening right now?", "Explain the current state of the twin in plain language — what each panel is showing right now."),
            ("Why did the last alert fire?", "Walk me through the most recent alert: which PV drifted, by how much, and what the causal-graph suggests as the root cause."),
            ("Is the plant healthy?", "Read the current Health Index and anomaly score, and explain what they imply about plant state."),
        ],
        "📡 SENSORS": [
            ("Explain each of the 5 sensors", "Describe what each of the 5 PVs (P1_PIT01, LIT01, FT03Z, TIT01, TIT03) measures, its units, and its typical reaction speed."),
            ("Which sensor reacts first in an attack?", "Explain the dynamics: which PV typically drifts first under each attack family, and why."),
            ("How do sensors affect each other?", "Explain the physical coupling between the 5 PVs using the causal graph parents_full.json."),
        ],
        "⚔ ATTACKS": [
            ("What are the 4 attack scenarios?", "Explain scenarios 0–3 (normal, AP_no, AP_with, AE_no) with examples of what an attacker does in each."),
            ("What's the hardest attack to detect?", "Explain why AP_with (coordinated multi-point attack) is the most difficult scenario for the twin and how it manifests."),
            ("How do sensor-spoofing attacks work?", "Explain the AE_no scenario: what the attacker manipulates, and how the twin can still catch it."),
        ],
        "🧠 HOW IT WORKS": [
            ("How does the twin predict sensors?", "Explain the GRU encoder-decoder: encoder compresses 300s of history, decoder steps 1s at a time using the hidden state."),
            ("Twin vs Model — what's the difference?", "Explain how the live stateful twin differs from the raw trained GRU model."),
            ("How is an alert actually triggered?", "Walk through: 180s rolling window → MSE → compare to threshold → 60s cooldown → alert record."),
        ],
        "🔍 THE DASHBOARD": [
            ("What do the 4 tabs do?", "Explain Live / Predictive / Generative / Assistive layers and when to use each."),
            ("How does the causal graph work?", "Explain how the Assistive layer walks parents_full.json upstream from the top-drifting PV."),
            ("Why does Virtual Plant look similar?", "Explain why moving SP sliders in the Generative tab often produces similar outputs — the controller-absorption effect."),
        ],
    }
    if not st.session_state["chat_history"]:
        st.markdown(
            '<div style="font-family:var(--mono);font-size:0.7rem;color:#a855f7;'
            'letter-spacing:1px;margin:4px 0 8px 0;">💡 PICK A TOPIC TO GET STARTED</div>',
            unsafe_allow_html=True,
        )
        for topic, qs in TOPICS.items():
            with st.expander(topic, expanded=(topic == "🩺 LIVE STATE")):
                for i, (label, q) in enumerate(qs):
                    if st.button("→ " + label, key=f"ex_{topic}_{i}",
                                 use_container_width=True):
                        st.session_state["_pending_question"] = q

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Free-form chat input — ALWAYS available, user can type any question.
    typed = st.chat_input("💬 Type your own question here — anything about the twin, sensors, attacks, alerts...")
    # Pending question from an example-button click takes priority for one render.
    pending = st.session_state.pop("_pending_question", None)
    prompt = typed or pending
    if prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": _state_summary(rt)},
            ]
            messages.extend(st.session_state["chat_history"][-8:])
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
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": out}
                )
        except Exception as e:
            st.error(f"Chat error: {e}")

    if st.session_state["chat_history"]:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
