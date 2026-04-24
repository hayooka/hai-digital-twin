# HAI Digital Twin

A closed-loop digital twin of the HAI 23.05 industrial control system. The twin simulates normal and attack-injected plant behaviour across five control loops, and is used for attack detection and classification.

---

## Project Structure

```
hai-digital-twin/
├── 02_data_pipeline/
│   ├── config.py                      # loop definitions, column names, hyperparameters
│   ├── scaled_split.py                # raw CSV → normalised sliding windows (.npz)
│   └── pipeline.py                    # load windows, split into plant + controller arrays
│
├── 03_model/
│   ├── gru.py                         # GRUPlant, GRUController, CCSequenceModel
│   ├── train_gru_causal_plus.py       # stage 1: warm-starts plant from Re__reults_of_gru_after_wight_
│   └── train_gru_scenario_weighted.py # stage 2: warm-starts from gru_causal_plus output
│
├── 04_evaluate/
│   ├── evaluate_model.py              # NRMSE tables + eval_results.json per checkpoint
│   ├── anomaly_detector.py            # IsolationForest + per-PV threshold experiments
│   └── plot_utils.py                  # shared plotting utilities
│
├── 05_detect/
│   ├── sec3_detection.py              # attack detection: ROC, PR, confusion matrix
│   ├── sec3_classification.py         # TRTS classifier: trains + saves final model
│   ├── monitor.py                     # full predictive monitor (WHEN / WHAT / HOW)
│   └── evaluate_generation.py        # synthetic data quality experiments
│
├── outputs/
│   ├── scaled_split/                  # preprocessed windows (already generated)
│   ├── pipeline/
│   │   ├── Re__reults_of_gru_after_wight_/  # base plant checkpoint (warm-start source for stage 1)
│   │   ├── gru_causal_plus/           # stage 1 output (warm-start source for stage 2)
│   │   └── gru_scenario_weighted/     # stage 2 output (used by detection and classification)
│   └── classifiers/
│       ├── trts_rf_classifier.pkl     # final saved TRTS attack classifier
│       └── trts_rf_scaler.pkl         # scaler paired with classifier (always use together)
│
└── report_plots/
    └── code/                          # scripts that generate report figures
```

---

## Architecture

```
[SP, PV history]  →  GRUController × 4 (PC, LC, FC, TC)  →  CV sequence
[TIT03, PP04SP]   →  CCSequenceModel                      →  pump speed
                                    │
                          GRUPlant (encoder–decoder)
                                    │
              Encoder: non-PV signals + scenario embedding → hidden h
              Decoder: autoregressive rollout
                       input_t = [ cv_target_t ‖ pv_{t-1} ]
                       pv_t    = FC( GRU(input_t, h) )
                                    │
                                    ▼
                    5 PV outputs: P1_PIT01, P1_LIT01, P1_FT03Z,
                                  P1_TIT01, P1_TIT03
```

---

## Data

**HAI 23.05** — 1 Hz industrial control system logs with labelled attack segments.

| Split | Source |
|---|---|
| Train | train1 (100%) + train2 (100%) + train3 (first 30%) — normal only |
| Val   | train3 (last 70%) — normal only |
| Test  | train4 (fully held out — contains attacks) |

Attack types: `AP_no` (Actuator Pollution, no combustion), `AP_with` (with combustion), `AE_no` (Actuator Emulation, no combustion).

Input window: 300 steps (5 min) → Target window: 180 steps (3 min), stride: 60 steps.

---

## Running the Full Pipeline

> **Preprocessed data already exists** in `outputs/scaled_split/`. Skip step 1 unless you have new raw data.

### Step 1 — Preprocess data *(skip if already done)*
```bash
python 02_data_pipeline/scaled_split.py
```

### Step 2 — Train the GRU digital twin

Training runs in three stages, each warm-starting from the previous:

```
Re__reults_of_gru_after_wight_/gru_plant.pt   ← base checkpoint (already exists)
          ↓
stage 1: train_gru_causal_plus.py       → outputs/pipeline/gru_causal_plus/
          ↓
stage 2: train_gru_scenario_weighted.py → outputs/pipeline/gru_scenario_weighted/
```

```bash
python 03_model/train_gru_causal_plus.py        # stage 1
python 03_model/train_gru_scenario_weighted.py  # stage 2
```

### Step 3 — Attack detection
Uses `gru_scenario_weighted` checkpoint. Produces ROC, PR curve, confusion matrix, and detection stats.
```bash
python 05_detect/sec3_detection.py
# outputs → report_plots/figures/s3/
```

### Step 4 — Attack classification (trains + saves final classifier)
Uses `gru_scenario_weighted` checkpoint. Runs three experiments (Baseline / TRTS / Mixed) and saves the TRTS classifier.
```bash
python 05_detect/sec3_classification.py
# outputs → report_plots/figures/s3c_*.png
#           outputs/classifiers/trts_rf_classifier.pkl
#           outputs/classifiers/trts_rf_scaler.pkl
```

---

## Final Classifier

The saved TRTS classifier (`outputs/classifiers/trts_rf_classifier.pkl`) classifies windows into 4 attack types:

| Class | Label |
|---|---|
| 0 | Normal |
| 1 | AP_no |
| 2 | AP_with |
| 3 | AE_no |

**Macro F1 = 0.937** (trained on synthetic data only, tested on real data).

To load and use it:
```python
import joblib
import numpy as np

clf    = joblib.load("outputs/classifiers/trts_rf_classifier.pkl")
scaler = joblib.load("outputs/classifiers/trts_rf_scaler.pkl")

# features: extract from (N, TARGET_LEN, N_PV) PV trajectory array
# see sec3_classification.py → extract_features()
y_pred = clf.predict(scaler.transform(X_features))
```

---

## Team Workflow

```bash
git pull                          # before starting
git add .
git commit -m "describe your work"
git push                          # after finishing
```

Always `git pull` before starting — do not edit other people's files.
