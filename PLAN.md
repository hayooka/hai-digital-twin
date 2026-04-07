# HAI Digital Twin — Full Architecture & Plan

## Goal
Build a **generative Digital Twin** of the HAI ICS system that can:
1. Simulate realistic sensor readings under **normal operation**
2. Simulate realistic sensor readings under **attack scenarios (A, B, C...)**
3. Answer: *"What would happen in the next 180s if scenario X occurs?"*
4. Run **counterfactual analysis**: compare normal vs attack trajectories

---

## Dataset
| Source | Location | Content |
|--------|----------|---------|
| HAI raw | `C:\Users\ahmma\Desktop\farah\hai-23.05` | Physical sensors (P1-P4) |
| HAIEnd raw | `C:\Users\ahmma\Desktop\farah\haiend-23.05` | Network/control signals |
| Processed | `data/processed/train1-4.csv, test1-2.csv` | HAI + HAIEnd merged (313 cols, 133 non-constant) |
| Domain graphs | `C:\Users\ahmma\Desktop\farah\boiler\dcs_*.json` | Control system structure (PLCs) |

### Sample Counts
| Split | Normal | Attack | Total |
|-------|--------|--------|-------|
| train1-4 | 896,396 | 0 | 896,396 |
| test1 | 51,018 | 2,981 | 53,999 |
| test2 | 221,996 | 8,403 | 230,399 |
| **Total** | **1,169,410** | **11,384** | **1,180,794** |

- 52 attack episodes total (14 in test1, 38 in test2)
- Class imbalance: ~1% attack

---

## Architecture Overview

```
Input: past 300s of sensor readings + scenario label
                    ↓
         ┌──────────────────────┐
         │   Transformer Seq2Seq │
         │                      │
         │  Encoder (4L, 8H)    │  ← processes 300s context
         │       +              │
         │  Scenario Embedding  │  ← normal / attack_A / attack_B
         │       ↓              │
         │  Decoder (4L, 8H)    │  ← predicts future autoregressively
         └──────────────────────┘
                    ↓
         Predicted 180s sensor readings
                    ↓
         ┌──────────────────────┐
         │   Causal Validator   │  ← checks physics via causal graph
         └──────────────────────┘
```

### Training Loss
```
total_loss = MSE(pred, target)  +  λ × causal_loss(pred, parents.json)
```
- MSE = prediction accuracy
- Causal loss = penalizes predictions that violate physical causal relationships
- λ starts at 0.1, increases gradually

---

## 4-Phase Plan

### Phase 1 — Causal Graph  [`01_causal_graph/`]  ✅ DONE
**Goal**: Extract sensor-to-sensor causal relationships from domain knowledge

**Input**: `boiler/dcs_*.json` (control system graphs from HAI dataset authors)

**Process**:
- Parse 7 DCS JSON files (one per PLC subsystem)
- Yellow nodes = physical sensors/actuators
- Find paths: sensor → [PLC blocks] → actuator
- Map short names (PIT01) → HAI column names (P1_PIT01)

**Output**: `outputs/causal_graph/parents.json`
```json
{
  "P1_PCV01D": [{"parent": "P1_PIT01", "lag": 1}],
  "P1_FCV01D": [{"parent": "P1_FT02",  "lag": 1},
                {"parent": "P1_TIT01", "lag": 1}],
  ...
}
```
- 32 causal edges, 15 target sensors mapped

**Files**:
- `01_causal_graph/build_graph.py` — main script
- `01_causal_graph/causal_graph_discovery.ipynb` — PCMCI validation (optional)

---

### Phase 2 — Data Pipeline  [`02_data_pipeline/`]  ✅ DONE (needs scenario labels)
**Goal**: Load, preprocess, window, and label data with scenario IDs

**Process**:
1. Load HAI+HAIEnd from raw or processed CSVs
2. Drop constant columns → 133 non-constant sensors
3. Fit Z-score normalizer on train1-4 (normal data only)
4. Temporal 80/20 split for train/val (no random split — time-series)
5. Episode split: 52 attack episodes → 42 train / 10 test (seed=42)
6. Sliding windows: X=(N, 300, F), Y=(N, 180, F)
7. **[TODO]** Add scenario labels: 0=normal, 1=A101, 2=A201, ...

**Output**: numpy arrays + scenario labels
```
X_train (N, 300, 133)   Y_train (N, 180, 133)
X_val   (M, 300, 133)   Y_val   (M, 180, 133)
X_test  (K, 300, 133)   Y_test  (K, 180, 133)
scenario_train (N,)     scenario_test  (K,)
```

**Files**:
- `02_data_pipeline/data_loader.py` — load & merge HAI+HAIEnd
- `02_data_pipeline/prep.py` — windowing, normalization, episode split
- `02_data_pipeline/scaled_split.py` — train/val split
- `02_data_pipeline/for_notebook/label_attacks.py` — attack episode labels

---

### Phase 3 — Model  [`03_model/`]  🔧 IN PROGRESS
**Goal**: Transformer Seq2Seq + Scenario Conditioning + Causal Loss

#### 3A — Transformer Architecture (EXISTS in `transformer.py`)
```
Input proj:  Linear(133 → 256)
Pos encoding: sinusoidal
Encoder:     4 layers, 8 heads, FFN=1024, pre-norm
Decoder:     4 layers, 8 heads, causal mask
Output proj: Linear(256 → 133)

Training:    teacher-forcing
Inference:   autoregressive (step by step)
```

#### 3B — Scenario Embedding  [TODO]
```python
# Add to transformer:
self.scenario_emb = nn.Embedding(n_scenarios, d_model)

# In forward():
scenario_vec = self.scenario_emb(scenario_id)   # (B, d_model)
src = src + scenario_vec.unsqueeze(1)            # broadcast over time
```

#### 3C — Causal Loss  [TODO — `causal_loss.py`]
```python
# For each target sensor with known parents:
for sensor_i, parents in parents_dict.items():
    expected = weighted_sum(pred[:, :, parent_indices], coefficients)
    causal_loss += MSE(pred[:, :, sensor_i], expected)

total_loss = mse_loss + λ * causal_loss
```

**Hyperparameters**:
- d_model=256, n_heads=8, n_layers=4, ffn_dim=1024
- dropout=0.1, epochs=50, batch=64, lr=1e-4
- λ (causal weight) = start 0.1, scheduler to increase

**Files**:
- `03_model/transformer.py` — model architecture
- `03_model/train.py` — training loop
- `03_model/causal_loss.py` — causal penalty [TODO]

---

### Phase 4 — Evaluate & Counterfactual  [`04_evaluate/`]  🔧 PARTIAL
**Goal**: Measure simulation quality + run what-if scenarios

#### 4A — Evaluation (EXISTS in `eval.py`)
```
RMSE val    (normal)  → should be low
RMSE test   (normal)  → should be low
RMSE test   (attack)  → should be low (generative goal)
Separation ratio      → attack/normal RMSE ≤ 1.2x
```

#### 4B — Counterfactual Simulation  [TODO — `counterfactual.py`]
```python
past = sensor_readings[-300:]   # same 300s input

pred_normal   = model(past, scenario=0)
pred_attack_A = model(past, scenario="A201")
pred_attack_B = model(past, scenario="A202")

# Impact = difference from normal
impact_A = pred_attack_A - pred_normal   # which sensors change?
impact_B = pred_attack_B - pred_normal
```

**Files**:
- `04_evaluate/eval.py` — RMSE + separation ratio
- `04_evaluate/counterfactual.py` — what-if simulation [TODO]

---

## Folder Structure
```
hai-digital-twin/
├── 00_archive/          old notebooks (kept for reference)
├── 01_causal_graph/     Phase 1 ✅
├── 02_data_pipeline/    Phase 2 ✅ (scenario labels TODO)
├── 03_model/            Phase 3 🔧 (causal_loss + scenario emb TODO)
├── 04_evaluate/         Phase 4 🔧 (counterfactual TODO)
├── 05_notebooks/        graph_analysis + causal_graph_discovery
├── data/processed/      HAI+HAIEnd merged CSVs
└── outputs/
    ├── causal_graph/    parents.json, edges_domain.csv
    ├── models/          saved model checkpoints
    └── plots/           validation plots
```

---

## What's Done vs TODO

| Task | Status |
|------|--------|
| Domain causal graph extraction | ✅ Done |
| Data loading + preprocessing | ✅ Done |
| Transformer Seq2Seq model | ✅ Done |
| Training loop (generative mode) | ✅ Done |
| RMSE evaluation | ✅ Done |
| Scenario labels for attack episodes | 🔧 TODO |
| Scenario Embedding in Transformer | 🔧 TODO |
| Causal Loss during training | 🔧 TODO |
| Counterfactual simulation | 🔧 TODO |
| PCMCI validation of causal graph | Optional |
| HAIEnd cross-domain causal analysis | Optional |
