# HAI Digital Twin

A closed-loop **GRU-based digital twin** of an industrial water treatment plant, built on the HAI 23.05 benchmark dataset. The twin learns to simulate normal and attack-injected plant behaviour across five physical control loops, and is used downstream for anomaly detection and attack classification.

**Live dashboard:** [resonant-cobbler-711d31.netlify.app](https://resonant-cobbler-711d31.netlify.app)

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [The HAI 23.05 Dataset](#2-the-hai-2305-dataset)
3. [The Five Control Loops](#3-the-five-control-loops)
4. [Attack Taxonomy](#4-attack-taxonomy)
5. [System Architecture](#5-system-architecture)
6. [Three-Stage Training Pipeline](#6-three-stage-training-pipeline)
7. [Results](#7-results)
8. [Setup](#8-setup)
9. [Running the Full Pipeline](#9-running-the-full-pipeline)
10. [Project Structure](#10-project-structure)
11. [Using the Saved Classifier](#11-using-the-saved-classifier)
12. [Development History](#12-development-history)

---

## 1. What This Project Does

Industrial control systems (ICS) are difficult to protect because attacks can look like normal process variability. This project builds a **predictive digital twin**: a model that learns what the plant *should* be doing given its current inputs, then flags deviations between prediction and observation as potential attacks.

The full pipeline has three responsibilities:

| Stage | Question answered | Output |
|---|---|---|
| **Generation** | Given control valve inputs, what PV trajectories should the plant produce? | Simulated PV sequences per scenario |
| **Detection** | Is the plant behaving as expected right now? | Anomaly score, binary alert |
| **Classification** | If anomalous, which attack type is occurring? | One of: `Normal`, `AP_no`, `AP_with`, `AE_no` |

---

## 2. The HAI 23.05 Dataset

**HAI (HIL-based Augmented ICS)** is a publicly available benchmark dataset from KAIST, recorded at 1 Hz from a Hardware-in-the-Loop simulation of a water treatment and heating plant. Version 23.05 is the latest release.

The plant has two subsystems running simultaneously:
- **Water treatment**: pumps, pressure vessels, level tanks, and flow control valves
- **Boiler**: temperature regulation, combustion control, cooling loop

HAI injects ground-truth labelled cyber-attacks into the simulation and records all sensor and actuator signals before, during, and after each attack. This gives a rare dataset where the exact attack window is known, making it suitable for supervised anomaly detection research.

### Signal types

| Suffix | Type | Examples |
|---|---|---|
| `PIT` / `PCV` | Pressure (transmitter / control valve) | `P1_PIT01`, `P1_PCV01D` |
| `LIT` | Level transmitter | `P1_LIT01` |
| `FT` / `FCV` | Flow transmitter / valve | `P1_FT03`, `P1_FCV03D` |
| `TIT` | Temperature transmitter | `P1_TIT01`, `P1_TIT03` |
| `PP` | Pump speed | `P1_PP04D` |

Signals ending in `D` are discrete (valve open/closed or pump on/off); all others are continuous.

### Data splits used in this project

| Split | Source files | Rows | Purpose |
|---|---|---|---|
| Train | `train1.csv`, `train2.csv`, `train3.csv` (first 30%) | ~130 k | Model training — normal-operation segments only |
| Validation | `train3.csv` (last 70%) | ~50 k | Hyperparameter tuning — normal only |
| Test | `train4.csv` (100%) + 20% of each attack file | ~80 k | Final evaluation — includes labelled attack windows |

**Window parameters:** 300-step input (5 min) → 180-step target (3 min), stride 60 steps.

### How to obtain the data

Download from the [official HAI benchmark repository](https://github.com/icsdataset/hai) and place the CSV files under `00_data/processed/`:

```
00_data/processed/
├── train1.csv
├── train2.csv
├── train3.csv
├── train4.csv   ← held-out period (contains normal operation)
├── test1.csv    ← attack segments (AP_no, AP_with)
└── test2.csv    ← attack segments (AE_no)
```

---

## 3. The Five Control Loops

A **control loop** is a closed-loop feedback system that drives a **Process Variable (PV)** toward a **Setpoint (SP)** by adjusting a **Control Valve (CV)**. The digital twin models each loop with a dedicated GRU controller.

| Loop | Name | PV — what is measured | CV — what is actuated | Physical role |
|---|---|---|---|---|
| **PC** | Pressure Control | `P1_PIT01` — tank pressure (bar) | `P1_PCV01D` — pressure relief valve | Maintains system pressure within safe bounds |
| **LC** | Level Control | `P1_LIT01` — tank water level (cm) | `P1_FCV03D` — inlet flow valve | Keeps tank level stable to ensure pump supply |
| **FC** | Flow Control | `P1_FT03Z` — pipe flow rate (L/min) | `P1_PCV02D` — flow control valve | Regulates volumetric flow to downstream processes |
| **TC** | Temperature Control | `P1_TIT01` — heat exchanger outlet temperature (°C) | `P1_FT02` — hot-water feed rate | Maintains process temperature for the boiler |
| **CC** | Cooling Control | `P1_TIT03` — cooling water outlet temperature (°C) | `P1_PP04SP` — cooling pump setpoint | Removes heat from the reactor vessel |

### Why the CC loop is different

The CC loop uses a `CCSequenceModel` (a direct sequence-to-sequence network) rather than a standard GRU controller. The cooling pump is driven by a cascade setpoint signal that exhibits non-stationary periodicity; a plain GRU underfits this signal, so a dedicated architecture was used.

### Causal augmentation

Each controller's input is enriched with **3 causally-related sensor channels** derived from the HAI causal graph. For example, the FC controller (which controls flow) receives additional pressure and temperature readings that physically influence the valve dynamics. This augmentation significantly improves controller fidelity.

| Loop | Extra channels added |
|---|---|
| PC | `P1_PCV02D`, `P1_FT01`, `P1_TIT01` |
| LC | `P1_FT03`, `P1_FCV03D`, `P1_PCV01D` |
| FC | `P1_PIT01`, `P1_LIT01`, `P1_TIT03` |
| TC | `P1_FT02`, `P1_PIT02`, `P1_TIT02` |
| CC | `P1_PP04D`, `P1_FCV03D`, `P1_PCV02D` |

---

## 4. Attack Taxonomy

HAI 23.05 includes three types of cyber-attacks, all targeting actuator signals (control valves and pump setpoints). The scenarios are encoded as integer class labels throughout the codebase.

| ID | Label | Full name | What the attacker does | Effect on plant |
|---|---|---|---|---|
| 0 | `Normal` | Normal operation | No attack | Nominal PV trajectories |
| 1 | `AP_no` | Actuator Pollution — no combustion | Injects false CV commands while combustion is off | Pressure/flow/level deviate from expected trajectories |
| 2 | `AP_with` | Actuator Pollution — with combustion | Same as AP_no but while the boiler is active | More severe disturbance; thermal coupling amplifies anomaly |
| 3 | `AE_no` | Actuator Emulation — no combustion | Overwrites HAIEND function-block outputs in the PLC | Plant continues to respond plausibly, making the attack harder to detect |

**Why `AE_no` is the hardest attack:** Unlike AP attacks (which inject raw noise into CVs), AE attacks manipulate the *internal PLC state* — specifically the HAIEND signals that PLC function blocks compute from sensor readings. The plant behaves "normally" by its own sensors while actually following attacker commands. Detecting AE attacks requires the model to learn the relationship between PLC internals and observable PVs, which is why Stage 3 of training finetunes on HAIEND signals.

### Scenario weighting

Because attacks are rarer than normal operation in the dataset, the training loss uses **per-scenario weights** to prevent the model from ignoring minority classes:

| Scenario | Loss weight |
|---|---|
| Normal | 1.0× |
| AP_no | 3.0× |
| AP_with | 6.0× |
| AE_no | 2.0× |

The `P1_TIT03` (cooling temperature) channel is additionally upweighted at 2.0× because it is the primary indicator for CC-loop attacks and is underrepresented in the loss without explicit weighting.

---

## 5. System Architecture

### Overview

The digital twin has two interacting components:

1. **Five GRU Controllers** — each takes `[SP, PV]` history as input and predicts the future CV sequence for its loop.
2. **One GRU Plant** — takes non-PV sensor signals and the predicted CV sequences, and autoregressively rolls out future PV trajectories.

```
Input window (300 steps)
│
├── [SP, PV history per loop] ──► GRUController[PC] ──► CV_PC (180 steps)
│                                 GRUController[LC] ──► CV_LC
│                                 GRUController[FC] ──► CV_FC
│                                 GRUController[TC] ──► CV_TC
│                                 CCSequenceModel   ──► CV_CC
│
└── [non-PV sensor signals]   ──► GRUPlant (encoder)
                                        │
                              hidden state h (scenario-aware)
                                        │
                              GRUPlant (autoregressive decoder)
                              input_t = [ CV_targets_t ‖ pv_{t-1} ]
                                        │
                                        ▼
                          PV predictions (180 steps):
                          P1_PIT01  — pressure
                          P1_LIT01  — level
                          P1_FT03Z  — flow
                          P1_TIT01  — temperature
                          P1_TIT03  — cooling temperature
```

### GRUPlant detail

The plant model uses an **encoder–decoder GRU** architecture:

- **Encoder:** Processes the 300-step input window of non-PV signals. A learned **scenario embedding** is concatenated to every encoder input timestep, giving the model an explicit signal about which operational regime is active (Normal / AP_no / AP_with / AE_no).
- **Decoder:** Autoregressively generates the 180-step PV forecast. At each step `t`, the input is the concatenation of: the predicted CV targets at time `t` (from the five controllers above) and the PV prediction from step `t-1`. A final fully-connected block maps the GRU hidden state to the 5 PV outputs.
- **Scheduled sampling:** During training, teacher forcing (using real PV values as decoder inputs) is annealed from 100% → ~52% over the training run. This prevents exposure bias where the model never learns to recover from its own prediction errors.

### GRUController detail

Each controller is a standard GRU with:
- Input: `[SP_t, PV_t] + 3 causal channels` at each timestep of the 300-step history
- Output: predicted CV sequence for the next 180 steps (via a single linear projection of the final hidden state)

---

## 6. Three-Stage Training Pipeline

The model is trained in three sequential stages. Each stage warm-starts from the previous checkpoint. This curriculum is necessary because learning all tasks simultaneously leads to unstable training.

### Stage 0 — Base warm-start (committed to repo)

A plain GRU plant trained on normal-operation data only, without scenario embeddings or causal controller inputs. This gives a stable initialisation that has already learned the gross physical dynamics of the plant.

**Checkpoint:** `outputs/pipeline/Re__reults_of_gru_after_wight_/gru_plant.pt`

> This checkpoint is not committed to the repo (model weights are gitignored). If the file is missing, Stage 1 training will start from random initialisation rather than the warm-start — results will still converge but may take longer. The folder name contains a typo from the original training run and is preserved intentionally to avoid breaking paths.

### Stage 1 — GRU Causal Plus

Adds three improvements over the base model:

1. **Causal augmentation:** Controller inputs are enriched with 3 causally-related sensor channels per loop.
2. **Scenario embedding:** A 4-class embedding is concatenated to every encoder input, giving the plant model explicit scenario context.
3. **In-the-loop controllers:** All five GRU controllers are trained jointly with the plant, with controller CV predictions fed into the plant decoder.

The training loss at this stage is standard MSE, averaged uniformly across all PV channels and scenarios.

**Script:** `03_model/train_gru_causal_plus.py`
**Output:** `outputs/pipeline/gru_causal_plus/`

### Stage 2 — GRU Scenario Weighted

Fine-tunes Stage 1 with scenario-aware loss weighting (see [Attack Taxonomy](#4-attack-taxonomy) for the weight table). This forces the model to maintain accurate predictions under minority attack scenarios that would otherwise be suppressed by the majority Normal class.

The `P1_TIT03` channel is additionally upweighted to ensure the CC-loop dynamics are learned accurately under AE attacks.

**Script:** `03_model/train_gru_scenario_weighted.py`
**Output:** `outputs/pipeline/gru_scenario_weighted/` ← **used by all downstream evaluation**

### Stage 3 — HAIEND Fine-tune (optional)

Fine-tunes the Stage 2 plant model to additionally predict HAIEND signals — the internal PLC function-block outputs that AE attacks directly manipulate. Incorporating these as auxiliary outputs improves AE detection sensitivity.

**Script:** `03_model/finetune_haiend.py`
**Output:** `outputs/pipeline/gru_haiend/`

---

## 7. Results

All numbers are reported on the held-out test set (unseen during all training stages).

### Plant model — prediction accuracy

| Metric | Value | Interpretation |
|---|---|---|
| NRMSE (overall) | **0.0095** | Normalised RMSE across all 5 PVs and all test windows |
| NRMSE (Normal) | ~0.007 | Near-perfect tracking on normal operation |
| NRMSE (AP_with) | ~0.018 | Largest error; combustion-coupled attacks are most physically disruptive |

NRMSE < 0.10 is the target threshold. The final model comfortably achieves this.

### Anomaly detection

Residuals between predicted and observed PVs are fed into an IsolationForest + per-PV threshold ensemble:

| Metric | Value |
|---|---|
| AUROC | **0.899** |
| F1 (attack vs. normal) | ~0.82 |

### Attack classification (TRTS experiment)

A Random Forest classifier trained on **synthetic** PV trajectories from the digital twin, then evaluated on real test data:

| Experiment | Description | Macro F1 |
|---|---|---|
| TSTR | Train on real, test on synthetic | ~0.88 |
| TRTS | Train on synthetic, test on real | ~0.76 |
| Mixed | Train on 50% real + 50% synthetic, test on real | ~0.81 |

The TRTS result (~0.76) demonstrates that the synthetic data is realistic enough to train a classifier that generalises to real sensor readings — validating the quality of the digital twin as a data generator.

---

## 8. Setup

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- Git
- GPU recommended (CUDA) — CPU training is possible but slow

### Clone and install

```bash
git clone <repo-url>
cd hai-digital-twin

conda env create -f environment.yml
conda activate digital_twin
```

> **CUDA note:** The environment installs the CPU build of PyTorch by default. For GPU training, edit the `torch` line in `environment.yml` to match your CUDA version before creating the environment. See [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

### Place raw data

Download the HAI 23.05 dataset from the [official repository](https://github.com/icsdataset/hai) and place the CSV files at:

```
00_data/processed/train1.csv
00_data/processed/train2.csv
00_data/processed/train3.csv
00_data/processed/train4.csv
00_data/processed/test1.csv
00_data/processed/test2.csv
```

---

## 9. Running the Full Pipeline

All commands are run from the **repo root** with the `digital_twin` environment active.

### Step 1 — Preprocess

Normalises the raw CSVs and creates sliding-window `.npz` files under `outputs/scaled_split/`.

```bash
python 02_data_pipeline/scaled_split.py
```

Output files: `train_data.npz`, `val_data.npz`, `test_data.npz`

> Skip this step if the `.npz` files already exist.

### Step 2 — Train Stage 1 (Causal Plus)

Warm-starts from the base checkpoint already in the repo. Trains the plant model with causal controller inputs and scenario embeddings.

```bash
python 03_model/train_gru_causal_plus.py
```

Expected runtime: 2–6 hours on GPU depending on hardware.
Output: `outputs/pipeline/gru_causal_plus/`

### Step 3 — Train Stage 2 (Scenario Weighted)

Warm-starts from Stage 1. Applies scenario-weighted loss to improve attack-scenario fidelity.

```bash
python 03_model/train_gru_scenario_weighted.py
```

Expected runtime: 1–3 hours on GPU.
Output: `outputs/pipeline/gru_scenario_weighted/` ← **used by all downstream steps**

### Step 4 (optional) — Fine-tune on HAIEND signals

Adds auxiliary HAIEND output heads for improved AE detection.

```bash
python 03_model/finetune_haiend.py
```

### Step 5 — Evaluate the model

Computes NRMSE tables across all scenarios and saves `eval_results.json`.

```bash
python 04_evaluate/evaluate_model.py
```

### Step 6 — Attack detection

Runs IsolationForest + threshold-based detection. Saves ROC, PR curves, and confusion matrix.

```bash
python 05_detect/sec3_detection.py
# figures → report_plots/figures/s3/
```

### Step 7 — Attack classification

Trains the TRTS classifier on synthetic data and evaluates on real test data. Saves the classifier artifact.

```bash
python 05_detect/sec3_classification.py
# classifier → outputs/classifiers/trts_rf_classifier.pkl
# scaler     → outputs/classifiers/trts_rf_scaler.pkl
```

---

## 10. Project Structure

```
hai-digital-twin/
│
├── 00_data/
│   └── processed/               # raw HAI CSV files (not committed — download separately)
│
├── 02_data_pipeline/
│   ├── config.py                # loop definitions, column lists, path constants
│   ├── shared.py                # shared constants and helpers (SCENARIO_NAMES, CTRL_LOOPS,
│   │                            #   CTRL_HIDDEN_PER_LOOP, EXTRA_CHANNELS, augment_ctrl_data)
│   ├── scaled_split.py          # raw CSV → normalised .npz windows (step 1)
│   └── pipeline.py              # loads .npz files and splits into plant/controller arrays
│
├── 03_model/
│   ├── gru.py                           # model definitions: GRUPlant, GRUController,
│   │                                    #   CCSequenceModel
│   ├── train_gru_causal_plus.py         # stage 1 training (causal + scenario embedding)
│   ├── train_gru_scenario_weighted.py   # stage 2 training (scenario-weighted loss)
│   └── finetune_haiend.py               # stage 3 training (HAIEND auxiliary outputs)
│
├── 04_evaluate/
│   ├── evaluate_model.py        # NRMSE evaluation per scenario; saves eval_results.json
│   ├── anomaly_detector.py      # IsolationForest + per-PV threshold experiments
│   └── plot_utils.py            # shared plotting utilities and chain-prediction cache
│
├── 05_detect/
│   ├── sec3_detection.py        # attack detection: ROC, PR, confusion matrix
│   ├── sec3_classification.py   # TSTR/TRTS/Mixed RF classifier experiments
│   ├── sec3_classification_xgb.py  # same experiments with XGBoost classifier
│   ├── monitor.py               # real-time predictive monitor (WHEN / WHAT / HOW)
│   ├── evaluate_generation.py   # synthetic data quality (FID-style experiments)
│   └── code/                    # scripts that generate report figures
│       ├── sec1_1_shared.py
│       ├── sec1_6_ctrl_loops.py
│       └── sec2_generation.py
│
├── outputs/                     # generated artifacts (partially committed)
│   ├── scaled_split/            # preprocessed windows (generated by step 1)
│   ├── pipeline/
│   │   ├── Re__reults_of_gru_after_wight_/  # base plant checkpoint (in repo)
│   │   ├── gru_causal_plus/                 # stage 1 output
│   │   └── gru_scenario_weighted/           # stage 2 output (used by detection)
│   └── classifiers/
│       ├── trts_rf_classifier.pkl           # saved TRTS attack classifier
│       └── trts_rf_scaler.pkl               # paired scaler
│
├── report_plots/
│   ├── figures/                 # all generated figures
│   └── code/                    # figure-generation scripts (mirror 05_detect/code/)
│
├── trials/                      # archived experiment scripts (development history)
├── environment.yml              # conda environment specification
└── README.md                    # this file
```

---

## 11. Using the Saved Classifier

The saved TRTS classifier classifies PV-trajectory windows into 4 attack classes.

```python
import joblib
import numpy as np

clf    = joblib.load("outputs/classifiers/trts_rf_classifier.pkl")
scaler = joblib.load("outputs/classifiers/trts_rf_scaler.pkl")

# X: shape (N, T, 5) — array of PV windows (N windows, T timesteps, 5 PVs)
# Extract statistical features before classifying:
from 05_detect.sec3_classification import extract_features
X_features = extract_features(X)          # (N, 5*6) = (N, 30) statistical features

y_pred = clf.predict(scaler.transform(X_features))
# y_pred values: 0=Normal, 1=AP_no, 2=AP_with, 3=AE_no
```

**Feature extraction** (`extract_features`) computes 6 statistics per PV channel (mean, std, min, max, absolute mean, mean first-difference), yielding a 30-dimensional feature vector per window.

---

## 12. Development History

The `trials/` folder is a chronological record of every model variant tried before arriving at the final architecture.

**Chapter 1 — First sequence models**

Started with an LSTM using causal input features. Reasonable on normal data but failed to generalise across attack scenarios. A Transformer with scheduled sampling was slower to train with no meaningful improvement — dropped.

**Chapter 2 — GRU encoder–decoder, causal backbone**

Rebuilt around a GRU encoder–decoder. A plain GRU baseline confirmed the architecture could fit normal trajectories well. Adding causal graph guidance to encoder inputs gave a clear improvement in physical consistency. Adding controller-in-the-loop inputs and a richer encoder produced the direct ancestor of the final model. A parallel boiler-subsystem twin stress-tested the architecture before committing to the full HAI plant.

**Chapter 3 — Scenario awareness**

With normal prediction working, the challenge shifted to attack-scenario generalisation. Scenario embeddings with separate per-attack output heads were promising but unstable without loss weighting. Class-weighted loss helped stability, but heads collapsed on minority attack types. A two-phase curriculum (normal first, then attacks) improved stability but hurt generalisation. A redesigned loss with explicit attack/prediction separation and refined per-scenario weights gave the best attack classification up to this point.

**Final model**

Lessons from all chapters combined into the two-stage training scheme (Causal Plus → Scenario Weighted). The model predicts future PV trajectories with NRMSE = 0.0095 on held-out data. Attack detection is derived from prediction residuals (AUROC = 0.899), and attack classification uses a Random Forest trained on synthetic data generated by the twin (TRTS Macro F1 ≈ 0.76).
