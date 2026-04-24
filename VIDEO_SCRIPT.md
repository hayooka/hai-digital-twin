# HAI Digital Twin Video Script

## Title

**HAI Digital Twin: Predicting Industrial Attacks Before They Escalate**

---

## 0. Opening Hook

**Visual**
- Fast dashboard shots
- Boiler / plant diagram
- Time-series signals moving in real time
- Attack windows highlighted in red

**Voiceover**

What if an industrial control system could tell us not only that something is going wrong, but what is going wrong, when it starts, and what will happen next?

That is the problem this project solves.

In critical infrastructure, attacks do not always look like sudden shutdowns. Sometimes they begin as small manipulations of actuators, controller outputs, or process values. By the time operators notice the damage on the plant floor, the attack has already propagated through the system.

This project builds a **digital twin of the HAI industrial control system** that can forecast future plant behavior, generate realistic scenario data, and classify attack types from system dynamics.

This is not just monitoring. This is **predictive cyber-physical intelligence**.

---

## 1. Problem Definition

**Visual**
- Normal operation dashboard
- Then subtle attack begins on one actuator
- Delay before effect appears in downstream signals

**Voiceover**

Industrial control systems are hard to defend because they are **dynamic, interconnected, and safety-critical**.

An attack on one control signal can ripple through pressure, level, flow, and temperature loops before the operator sees a clear failure.

That creates three real-world problems:

1. We often detect attacks too late, after the physical process is already affected.
2. We do not always know which attack type is happening from raw sensor streams alone.
3. Real attack data is scarce and imbalanced, which makes training robust detection models difficult.

So the core question became:

**Can we build a digital twin that learns normal and attack-conditioned process behavior well enough to forecast the future, detect deviations early, generate realistic synthetic attack trajectories, and support attack classification?**

That is the mission of this project.

---

## 2. Solution

**Visual**
- High-level architecture animation
- Inputs -> controllers -> plant -> predictions -> residuals -> detection/classification

**Voiceover**

Our solution is a **closed-loop digital twin** of the HAI system.

Instead of modeling the plant as an isolated black box, we model the interaction between:

- controller setpoints,
- controller outputs,
- plant variables,
- scenario context,
- and future system evolution.

The twin does three things:

1. **Forecasts future plant variables** over a prediction horizon.
2. **Measures residuals** between predicted and actual behavior to detect abnormal activity.
3. **Generates synthetic trajectories** for rare attack scenarios, which helps train downstream classifiers.

So the twin becomes both:

- a **predictive monitoring engine**, and
- a **data generation engine** for cyber-physical security research.

---

## 3. HAI System and Dealing with Data

**Visual**
- HAI system schematic
- Labels for loops: pressure, level, flow, temperature, cooling
- Dataset timeline and train/val/test split

**Voiceover**

The project uses the **HAI 23.05 dataset**, a one-hertz industrial control system dataset with labeled attack intervals.

We focus on five main process variables and their connected control loops:

- Pressure control
- Level control
- Flow control
- Temperature control
- Cooling control

Each loop contains a relationship between:

- setpoint,
- controller output,
- and measured process variable.

To make the data usable for sequence learning, we convert the raw logs into sliding windows:

- **300 input steps** to provide system history,
- **180 target steps** to predict the future,
- with a stride of **60 steps**.

The split is designed to reflect generalization:

- training uses normal data from train1, train2, and part of train3,
- validation uses the remaining normal segment of train3,
- testing uses held-out train4, which contains attack scenarios.

That matters because the model is not just memorizing noise. It is learning how the process evolves across time and across scenarios.

And because attack data is limited, preprocessing and split discipline are essential to keep the evaluation meaningful.

---

## 4. Best Model Architecture and Training Methods

**Visual**
- Architecture diagram
- Controller models feeding plant model
- Scenario embeddings
- Training stage timeline

**Voiceover**

The best-performing approach in this project is a **GRU-based closed-loop digital twin**.

It combines:

- dedicated sequence models for controllers,
- a multi-input multi-output GRU plant model,
- scenario-aware conditioning,
- and staged fine-tuning.

Here is the idea.

The controller models learn how control actions evolve from setpoints and feedback history.

Then the plant model takes:

- non-process-variable signals,
- predicted control trajectories,
- previous plant state,
- and scenario embeddings,

and rolls the system forward autoregressively to predict future process variables.

Why GRU?

Because the system is sequential, nonlinear, and strongly time-dependent. GRUs gave the project a strong balance between:

- temporal memory,
- training stability,
- and manageable complexity.

The strongest training recipe is not a single step. It is a **multi-stage training strategy**:

1. Start from a warm-started plant checkpoint.
2. Train the **GRU Causal Plus** model with causal channels that inject extra physically relevant context into each loop.
3. Fine-tune using **scenario-weighted loss**, so harder and rarer attack cases receive more learning pressure.

Important training ideas include:

- warm-starting instead of training from scratch,
- scheduled sampling for autoregressive robustness,
- causal feature augmentation for control loops,
- and weighted losses to improve difficult scenarios.

This is what turns the model from a pure forecaster into a practical digital twin for anomaly analysis.

---

## 5. Results with Dashboard: Predicting the Future

**Visual**
- Dashboard showing actual vs predicted trajectories
- Forecast horizon
- Per-sensor residuals
- Normal vs attack scenario examples

**Voiceover**

The first result is straightforward and powerful:

**the digital twin can predict future plant behavior over the forecast horizon.**

On the dashboard, we compare:

- actual future process values,
- predicted future trajectories,
- and residual error over time.

Under normal conditions, the twin tracks the system closely.

When attacks occur, the gap between prediction and reality becomes informative. That gap is exactly what gives us a predictive signal for detection.

This is important because we are not waiting for a threshold alarm on a raw sensor. We are comparing the real plant to an internal simulation of how the plant should behave next.

That means the dashboard can answer:

- what the system is doing now,
- what the twin expected it to do,
- and where the two begin to diverge.

That is the foundation of early warning.

---

## 6. Results with Dashboard: Generating New Data

**Visual**
- Real vs synthetic trajectory comparisons
- Scenario panels
- Distribution overlays

**Voiceover**

The second result is data generation.

Because some attack classes are rare, the project uses the digital twin to generate **synthetic future trajectories** for specific scenarios.

On the dashboard, we compare:

- real attack trajectories,
- synthetic trajectories generated by the twin,
- and their mean and spread over time.

The goal is not to create random fake signals.

The goal is to create **scenario-consistent synthetic data** that preserves the structure of real process dynamics closely enough to support model training and experimentation.

This matters for industrial security because attack data is expensive, limited, and often hard to collect safely.

If the synthetic data is realistic enough, we can use it to:

- balance classes,
- stress-test classifiers,
- and explore attack behavior beyond what is available in the original dataset.

So the twin is not only a predictor of the future.

It is also a **generator of plausible cyber-physical futures**.

---

## 7. Results with Dashboard: Classification

**Visual**
- Classification dashboard
- Confusion matrix
- Per-class metrics
- TRTS and Mixed experiment comparison

**Voiceover**

The third result is attack classification.

After extracting features from predicted or generated trajectories, we train a classifier to distinguish among:

- Normal
- AP_no
- AP_with
- AE_no

This gives the project a practical response layer:

not just detecting that something is abnormal, but identifying **what type of abnormal behavior is happening**.

The classification dashboard shows:

- confusion matrices,
- macro F1 performance,
- and comparisons across training strategies.

One of the most interesting outcomes is that synthetic data is not just visually plausible.

It is also useful.

The project demonstrates that synthetic trajectory data can support classification performance on real attack windows, which makes the digital twin valuable beyond forecasting alone.

That is a major contribution:

the model becomes both the source of anomaly signals and the source of training data for downstream security analytics.

---

## Closing Pitch

**Visual**
- Return to full system dashboard
- Highlight prediction, generation, classification together

**Voiceover**

So what did we build?

We built a digital twin that learns how an industrial system behaves, predicts what it should do next, highlights when reality starts to diverge, generates realistic scenario data, and supports attack classification.

In other words:

This project moves industrial monitoring from **reactive observation** to **predictive understanding**.

It gives us a way to ask not only:

- Is the system safe right now?

but also:

- What is likely to happen next?
- Does this behavior match the physics and control logic we expect?
- And if not, what kind of attack are we dealing with?

That is the value of the HAI Digital Twin.

Not just seeing the system.

Understanding it early enough to act.

---

## Optional Final Slide Text

**On-screen text**

HAI Digital Twin  
Predict. Detect. Generate. Classify.

Closed-loop forecasting for industrial cyber-physical security.
