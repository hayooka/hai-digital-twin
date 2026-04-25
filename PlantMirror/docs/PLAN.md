Based on the code in the notebook, here is the **pipeline architecture** of the GRU-based surrogate model. The design is a **coupled "Controller-Plant" structure**, which is ideal for closed-loop industrial process modeling.

## High-Level Pipeline Overview

The system is divided into two main models that run sequentially in a closed loop:

```
[Setpoints (SP)] → [GRU Controllers] → [Manipulated Variables (CVs)] → [GRU Plant] → [Process Variables (PVs)] → (feedback to controllers)
```

## Detailed Component Architecture

### **1. Data Preprocessing Pipeline**
```
Raw HAI + HAIEnd CSVs → Merge by index → Remove constants → Normalize (StandardScaler) → Create sequence datasets
```
- **Input data**: 87 HAI columns + 36 HAIEnd columns
- **Normalization**: Separate scalers for each controller (5) and one for the plant
- **Sequence generation**: Sliding windows of length 300 (5 minutes) with stride 10

### **2. GRU Controller Architecture (Per Loop)**
```
Input: [SP, PV, (CV_fb)] → GRU (2 layers, 64 hidden units) → Linear(64 → 1) → CV output
```

**Key features:**
- One GRUController per loop (PC, LC, FC, TC) - 4 total
- Processes sequences and outputs single CV value (uses last timestep only)
- CC loop uses a separate **Classifier-Regressor** model (not GRU-based)

**CC Classifier-Regressor Architecture:**
```
Input: [P1_TIT03, P1_PP04SP] → Shared Linear(2→32) → ReLU → Dropout
                                                    ↓
                                    Classifier(32→1) [logit for pump on/off]
                                    Regressor(32→1) [pump speed if on]
```

### **3. GRU Plant Architecture (MIMO)**
```
Input: [CVs (5) + Auxiliaries (51)] → Concatenate with current PVs (5)
                                      ↓
                                 GRU (2 layers, 256 hidden units)
                                      ↓
                            FC (256→128→ReLU→128→5) → Next-step PVs
```

**Key features:**
- MIMO: 56 inputs → 5 outputs
- Autoregressive: Uses previous PV predictions as inputs for next timestep
- **Scheduled sampling** during training: mixes teacher forcing with model predictions

### **4. Training Pipeline**

```
For each epoch:
    ├── Controller Training (PC, LC, FC, TC)
    │   └── One-step MSE loss (predict CV from [SP, PV, CV_fb] sequence)
    │
    ├── CC Classifier-Regressor Training
    │   └── Combined loss: BCE (classification) + MSE (regression on on-samples)
    │
    └── Plant Training with Scheduled Sampling
        ├── ss_ratio = 0: Full teacher forcing (fast forward)
        ├── 0 < ss_ratio < 0.5: Mix teacher and predicted PVs
        └── Validation: Full autoregressive (ss_ratio = 1.0)
```

### **5. Closed-Loop Inference Pipeline**

For multi-horizon validation (5, 10, 15, 30 minutes):

```
For each timestep:
    1. Feed [SP(t), PV(t), CV_fb(t)] to respective GRUController → CV_pred(t)
    2. Feed [CV_pred(t), Aux(t)] + previous PV_pred(t-1) to GRUPlant → PV_pred(t)
    3. Feed PV_pred(t) back to controllers for next timestep
```

## Data Flow Summary

| Stage | Input | Output | Model |
|-------|-------|--------|-------|
| **Controller** | SP, PV, CV_fb (3-4 features) | CV (1 value) | GRU (64-dim, 2 layers) |
| **Plant** | CVs (5) + Auxiliaries (51) + previous PVs (5) | Next PVs (5) | GRU (256-dim, 2 layers) |

## Key Architectural Decisions

1. **Separate controllers per loop**: Allows each loop to have its own dynamics and input features
2. **MIMO plant model**: Captures cross-coupling between different process variables
3. **Autoregressive plant**: Essential for multi-step prediction without error accumulation
4. **Scheduled sampling**: Bridges the gap between teacher-forced training and autoregressive inference
5. **CC Classifier-Regressor**: Handles the discrete switching behavior of the cooling pump

## Dimensional Summary

| Component | Input Dim | Hidden Dim | Output Dim |
|-----------|-----------|------------|------------|
| PC Controller | 3 | 64 | 1 |
| LC Controller | 3 | 64 | 1 |
| FC Controller | 3 | 64 | 1 |
| TC Controller | 3 | 64 | 1 |
| CC Model | 2 | 32 | 1 (logit) + 1 (speed) |
| Plant Model | 56 | 256 | 5 |

This architecture is specifically designed for **closed-loop simulation** of the HAI process, where controllers generate control actions and the plant predicts the resulting process variables in an autoregressive manner.