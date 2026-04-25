# How the Digital Twin is Built — GRUs + Causal Graph

## The two building blocks

```
┌─────────────────────────────┐      ┌─────────────────────────────┐
│   BLOCK 1 — GRU WEIGHTS     │      │   BLOCK 2 — CAUSAL GRAPH    │
│   (learned from data)       │      │   (hand-curated physics)    │
├─────────────────────────────┤      ├─────────────────────────────┤
│  gru_plant.pt               │      │  parents_full.json          │
│    2×512 hidden             │      │                             │
│    in : 128 plant features  │      │  P1_PIT01 ← {FCV01,PCV01}   │
│         + 32-d scenario     │      │  P1_LIT01 ← {FCV02,PP01B}   │
│    out: 5 PV predictions    │      │  P1_TIT01 ← {CC_SP,FCV03}   │
│                             │      │  P1_TIT03 ← {P1_TIT01,PCV02}│
│  gru_ctrl_CC.pt             │      │  P1_FT03Z ← (derived)       │
│  gru_ctrl_FC.pt             │      │                             │
│  gru_ctrl_LC.pt             │      │  each edge labeled with     │
│  gru_ctrl_PC.pt             │      │  (lag_seconds, level)       │
│  gru_ctrl_TC.pt             │      │                             │
│    5 small GRUs             │      └─────────────────────────────┘
│    SP + PV  →  CV command   │
└─────────────────────────────┘
```

---

## Assembly — how they snap together into a twin

```
                           ┌───────────────────┐
   Setpoints (SP) ───────► │  Controller GRUs  │ ── CV (5 valves/pumps)
   Plant state (PV_prev)─► │   5× gru_ctrl_*   │        │
                           └───────────────────┘        │
                                                        ▼
                                              ┌───────────────────┐
   Scenario id (0–3) ──► 32-d embedding ────► │   PLANT GRU       │
                                              │   gru_plant.pt    │ ──► PV_next
                               PV_prev ─────► │   (encoder+dec)   │        │
                                              └───────────────────┘        │
                                                        ▲                  │
                                                        │                  │
                                                        └───── feeds back ─┘
                                                        (autoregressive loop)
```

This closed loop **is** the digital twin: SP → Controllers → CV → Plant → PV → back into controllers, step by step at 1 Hz.

---

## Where the causal graph enters

The GRU alone predicts; the causal graph **explains**. They cooperate at two moments:

```
   Real plant PV   ──┐
                     ├──►  residual = (PV_twin − PV_real)²
   GRU twin PV    ──┘                       │
                                             ▼
                               ┌──────────────────────┐
                               │   detector fires     │
                               │   when residual > θ  │
                               └──────────┬───────────┘
                                          │ top_pv (biggest error)
                                          ▼
                               ┌──────────────────────┐
                               │  causal graph BFS    │
                               │  parents_full.json   │
                               │                      │
                               │  top_pv → parent     │
                               │        → grandparent │
                               │        → root cause  │
                               └──────────────────────┘
                                          │
                                          ▼
                                   "FCV02 is the likely
                                    originating fault"
```

The GRU tells you **that** something is wrong.
The causal graph tells you **where** it started.

---

## The full build recipe — 5 loops

```
┌────────────────────────────────────────────────────────────────────┐
│  LOOP 1 — TRAINING (offline, already done)                         │
│   train.csv ──► train GRUs ──► gru_plant.pt + 5× gru_ctrl_*.pt     │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  LOOP 2 — CALIBRATION (offline, already done)                      │
│   val.csv ──► sweep residual distribution ──► threshold θ = 0.326  │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  LOOP 3 — WARM-UP (at dashboard start)                             │
│   first 300 s of test.csv ──► encoder ──► h_plant (hidden state)   │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  LOOP 4 — LIVE STEP (every 1 s forever)                            │
│                                                                    │
│   cv_t, pv_{t-1}, h_plant                                          │
│        │                                                           │
│        ▼                                                           │
│   plant.step_once()  ──►  pv_twin_t, h_plant (updated)             │
│        │                                                           │
│        ▼                                                           │
│   residual_t = (pv_twin_t − pv_real_t)²                            │
│        │                                                           │
│        ▼                                                           │
│   rolling mean over 180 s ──► anomaly_score                        │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  LOOP 5 — DIAGNOSIS (when anomaly_score > θ)                       │
│                                                                    │
│   argmax(per-PV residual) ──► top_pv                               │
│                │                                                   │
│                ▼                                                   │
│   BFS(parents_full.json, top_pv) ──► upstream chain                │
│                │                                                   │
│                ▼                                                   │
│   root-cause candidate + lag estimate                              │
└────────────────────────────────────────────────────────────────────┘
```

---

## TL;DR — what you need, what each part does

| Artifact | Role in the twin |
|---|---|
| `gru_plant.pt` | The physics simulator — "given CVs, predict PVs" |
| `gru_ctrl_*.pt` × 5 | The control logic — "given SP+PV, predict CV" |
| Scenario embedding (inside `gru_plant.pt`) | Lets one model serve 4 regimes (normal + 3 attacks) |
| `parents_full.json` | The causal map — "when PV X drifts, who upstream caused it" |
| `scaler.pkl` + `ctrl_scaler_*.pkl` | Normalize real-world units so GRUs see the same scale as training |
| Threshold θ | Turns a continuous residual into a yes/no alert |

**Minimum to run a twin:**
`gru_plant.pt` + scalers + one test CSV.

**To run a closed-loop what-if twin (Generative):**
add the 5 controller GRUs.

**To get root-cause diagnosis (Assistive):**
add `parents_full.json`.
