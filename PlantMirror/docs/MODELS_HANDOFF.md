# MODELS_HANDOFF — PlantMirror Digital Twin

Canonical reference for anyone building a dashboard, downstream pipeline, or integration on top of the PlantMirror HAI 23.05 digital twin. Every model artifact, every file path, every performance number you need to wire things up.

> **Prerequisite**: clone `github.com/hayooka/hai-digital-twin` and make sure you're on `main` (with the `farah` branch merged in, or cherry-pick the `04_evaluate/` and `05_detect/` files noted below).

---

## 0. Feature specification — the 133 active sensors

Raw CSVs hold **311 sensor columns**. Preprocessing removes **178 zero-variance columns** (flat signals carrying no information), leaving **133 active sensors** that all models consume.

### Breakdown (from `02_data_pipeline/scaled_split.py`)

**5 Process Variables — these are the model's prediction targets:**

| PV | Physical meaning |
|---|---|
| `P1_PIT01` | Boiler pressure |
| `P1_LIT01` | Drum water level |
| `P1_FT03Z` | Feedwater flow |
| `P1_TIT01` | Steam temperature |
| `P1_TIT03` | Coolant temperature |

**128 Plant inputs — model inputs:**

| Subsystem | Count | Examples |
|---|---|---|
| **P1 loop signals** | 22 | `P1_FCV01D`, `P1_FCV01Z`, `P1_FCV02D/Z`, `P1_FCV03D/Z`, `P1_FT01`, `P1_FT01Z`, `P1_FT02`, `P1_FT02Z`, `P1_FT03`, `P1_LCV01D/Z`, `P1_PCV01D/Z`, `P1_PCV02D/Z`, `P1_PIT02`, `P1_PP04`, `P1_PP04D`, `P1_PP04SP`, `P1_TIT02` |
| **P2 subsystem** | 16 | `P2_24Vdc`, `P2_ATSW_Lamp`, `P2_AutoGO`, `P2_AutoSD`, `P2_MASW`, `P2_MASW_Lamp`, `P2_ManualGO`, `P2_ManualSD`, `P2_SCO`, `P2_SCST`, `P2_SIT01`, `P2_VIBTR01-04`, `P2_VT01` |
| **P3 subsystem** | 5 | `P3_FIT01`, `P3_LCP01D`, `P3_LCV01D`, `P3_LIT01`, `P3_PIT01` |
| **P4 subsystem** | 11 | `P4_HT_FD`, `P4_HT_PO`, `P4_HT_PS`, `P4_LD`, `P4_ST_FD`, `P4_ST_GOV`, `P4_ST_LD`, `P4_ST_PO`, `P4_ST_PS`, `P4_ST_PT01`, `P4_ST_TT01` |
| **Setpoints / DCS logic** | 7 | `x1001_05_SETPOINT_OUT`, `x1001_15_ASSIGN_OUT`, `x1002_07_SETPOINT_OUT`, `x1002_08_SETPOINT_OUT`, `x1003_10_SETPOINT_OUT`, `x1003_18_SETPOINT_OUT`, `x1003_24_SUM_OUT` |
| **HAIEND hidden DCS signals** | 36 | `1001.5-OUT`, `1001.13-OUT` … `1001.20-OUT`, `1002.7-OUT` … `1002.31-OUT`, `1003.5-OUT` … `1003.30-OUT`, `1020.13-OUT` … `1020.21-OUT`, `DM-PP04-D`, `DM-PP04-AO`, `DM-AIT-DO`, `DM-AIT-PH`, `DM-TWIT-04`, `DM-TWIT-05`, `GATEOPEN`, `PP04-SP-OUT`, `1004.21-OUT`, `1004.24-OUT` |
| **Digital mirror (DM) signals** | 33 | `DM-FT01Z`, `DM-FT02Z`, `DM-FT03Z`, `DM-FCV01-D/Z`, `DM-FCV02-D/Z`, `DM-FCV03-D/Z`, `DM-FT01/02/03`, `DM-LCV01-D/Z`, `DM-LIT01`, `DM-PCV01-D/Z`, `DM-PCV02-D/Z`, `DM-PIT01/02`, `DM-PWIT-03`, `DM-TIT01/02`, `DM-TWIT-03`, `DM-HT01-D` |

*(The subsystem counts add up to 130 due to a few signals appearing in two logical groups — e.g., some DM-* entries are both "digital mirror" and "HAIEND". The canonical list is what's in `generator/scalers/metadata.pkl`.)*

### Getting the exact column list in Python

```python
import pickle
with open("hai-digital-twin/generator/scalers/metadata.pkl", "rb") as f:
    meta = pickle.load(f)
sensor_cols = meta["sensor_cols"]   # 133 columns, ordered
```

---

## 1. Plant surrogate (forecasting)

**Role**: given 300 s of real history + a CV schedule, predict the next 180 s of 5 PVs.

**Model**: `GRUPlant` — 2-layer × 512-hidden encoder-decoder GRU with 32-d scenario embedding (4 classes: Normal / AP_no / AP_with / AE_no). 5.25 M parameters.

**Primary checkpoint**: `gru_scenario_weighted`
- Path: `generator/weights/gru_plant.pt` (after extracting `for_dashboard.zip`)
- val_loss: **0.000310**
- test mean NRMSE: **0.00949**
- No HAIEND head (`n_haiend = 0`)

**Load + predict:**
```python
from generator.core import load_bundle, closed_loop_rollout, load_replay
from pathlib import Path

bundle = load_bundle(
    ckpt_dir=Path("hai-digital-twin/generator/weights"),
    split_dir=Path("hai-digital-twin/generator/scalers"),
)
src = load_replay(Path("processed/test1.csv"), bundle.scalers)

# One-shot closed-loop rollout at cursor t_end = 1000
result = closed_loop_rollout(bundle, src, t_end=1000, scenario=0)
pv_physical = result["pv_physical"]   # shape (180, 5) — 5 PVs over 180 s
```

**Training details** (see [`03_model/train_gru_scenario_weighted.py`](hai-digital-twin/03_model/train_gru_scenario_weighted.py))
- Data: 194 h of normal operation (train1/train2/train3)
- Validation: held-out cross-period (train4)
- Loss: MSE on scaled PVs, scheduled sampling ramps 0→0.5 over epochs 10-40
- Windows: 300 s input + 180 s target, stride 60 s, normal-only
- Results: [`outputs/pipeline/gru_scenario_weighted/eval_results.json`](hai-digital-twin/outputs/pipeline/gru_scenario_weighted/eval_results.json)

**Long-horizon variant** (our fine-tune): `hai-digital-twin/training/checkpoints/v2_weighted_init_best.pt`
- val_loss 0.00206 at target_len=1800 (30-min rollout)
- Trained on vast.ai A100, BATCH=256, 14 epochs, early-stopped
- Use for 30-min rollouts; use the primary weighted plant for anything ≤ 3 min

---

## 2. Controller surrogates (5 per-loop)

**Role**: predict the next 180 s of CV commands given 300 s of [SP, PV, CV, + 3 causal features] history. One model per loop.

**Models:**

| Loop | Class | Hidden | Special |
|---|---|---|---|
| PC (pressure) | `GRUController` | 64 | — |
| LC (level) | `GRUController` | 64 | — |
| FC (flow) | `GRUController` | 128 | — |
| TC (temperature) | `GRUController` | 64 | — |
| CC (cooling) | `CCSequenceModel` | 64 | Two heads: `head_cv` (speed) + `head_on` (gate sigmoid). Gated output = `sigmoid(logit) > 0.5 ? cv : 0` |

**Paths**: `generator/weights/gru_ctrl_{cc,fc,lc,pc,tc}.pt`

**Load**: all 5 controllers come bundled via `load_bundle()` above, accessible as `bundle.controllers["PC"]`, etc.

**Causal features per loop** (3 extra channels each, from `generator/core.py::LOOP_SPECS`):

| Loop | SP | PV | CV | Causal features |
|---|---|---|---|---|
| PC | `x1001_05_SETPOINT_OUT` | `P1_PIT01` | `P1_PCV01D` | `P1_PCV02D`, `P1_FT01`, `P1_TIT01` |
| LC | `x1002_07_SETPOINT_OUT` | `P1_LIT01` | `P1_LCV01D` | `P1_FT03`, `P1_FCV03D`, `P1_PCV01D` |
| FC | `x1002_08_SETPOINT_OUT` | `P1_FT03Z` | `P1_FCV03D` | `P1_PIT01`, `P1_LIT01`, `P1_TIT03` |
| TC | `x1003_18_SETPOINT_OUT` | `P1_TIT01` | `P1_FCV01D` | `P1_FT02`, `P1_PIT02`, `P1_TIT02` |
| CC | `P1_PP04SP` | `P1_TIT03` | `P1_PP04` | `P1_PP04D`, `P1_FCV03D`, `P1_PCV02D` |

---

## 3. Attack injection engine

**Role**: inject SP / CV / PV attacks on top of closed-loop rollout. Not a new model — wrapper over Layers 1 + 2.

**Module**: [`attack_sim/attacks.py`](hai-digital-twin/attack_sim/attacks.py)

```python
from attack_sim.attacks import AttackSpec, InjectionPoint, AttackType, run_attack_sim

spec = AttackSpec(
    target_loop="LC",
    injection_point=InjectionPoint.SP,        # SP / CV / PV
    attack_type=AttackType.BIAS,              # bias / freeze / replay
    start_offset=0,                            # seconds rel. to cursor (can be negative)
    duration=60,
    magnitude=20.0,
)
result = run_attack_sim(bundle, src, t_end=1000, spec=spec, scenario=0)

# result.baseline — clean closed-loop rollout
# result.attacked — rollout with the injection applied
# result.signals  — dict with SP_real/SP_seen, CV_intended/CV_actual, PV_real/PV_seen
# result.attack_label — (180,) int8 per-second label (1 during attack)
```

**Injection semantics:**
- **SP attack** = mutate the setpoint the controller sees (HMI / operator workstation compromise)
- **CV attack** = let controller compute normally, then override its output before plant sees it (controller-to-field compromise)
- **PV attack** = spoof the PV feedback the controller sees (sensor compromise)

---

## 4. Guardian — binary anomaly detector (XGBoost Hybrid)

**Role**: per-row (1 Hz) binary "attack or normal."

**Model**: `XGBClassifier` (Optuna-tuned) with upstream `StandardScaler`. Reads all 133 sensor columns.

**Artifact**: `C:/Users/PC GAMING/Desktop/AI/HAI/best_hai_classifier.pkl` (dict with keys `scaler`, `model`, `features`).

**Load + predict:**
```python
import joblib, numpy as np, pandas as pd

pipe = joblib.load("path/to/best_hai_classifier.pkl")
scaler   = pipe["scaler"]      # StandardScaler
model    = pipe["model"]       # XGBClassifier
features = pipe["features"]    # list of 133 column names

# For a DataFrame `df` with all 133 sensor columns:
X = df[features].to_numpy(dtype=np.float32)
X_scaled = scaler.transform(X)
proba = model.predict_proba(X_scaled)[:, 1]   # P(attack) per row
alert = (proba >= 0.35).astype(int)            # 1 = flag
```

**Performance on real test set** (test1+test2 combined, 284,398 rows, 4 % attack rate):

| Threshold | F1 | Precision | Recall | AUROC |
|---|---|---|---|---|
| 0.50 (default) | 0.559 | 0.812 | 0.427 | 0.904 |
| **0.35 (peak F1) — recommended** | **0.587** | 0.648 | 0.536 | 0.904 |
| 0.60 (precision mode) | 0.511 | 0.893 | 0.358 | 0.904 |

**Trained on**: real train split (70 % of test1+test2) + `synthetic_attacks.csv` (36 k rows generated by the plant surrogate — Mixed experiment design).

**Training script**: [`C:/Users/PC GAMING/Desktop/AI/HAI/model_builder.py`](../AI/HAI/model_builder.py)

---

## 5. TRTS Attributor — multiclass attack-type classifier

**Role**: after Guardian fires, classify attack type (Normal / AP_no / AP_with / AE_no) over a 180-s window.

**Model**: `RandomForestClassifier(n_estimators=500, random_state=42)` with upstream `StandardScaler`. Reads **30 engineered features** (5 PVs × 6 stats).

**Artifact**: `~/Downloads/trts_rf_classifier.pkl` + `~/Downloads/trts_rf_scaler.pkl`

**Feature extractor** — defined in [`05_detect/sec3_classification.py::extract_features`](hai-digital-twin/05_detect/sec3_classification.py) (on `origin/farah`):

```python
def extract_features(traj):
    """(N, T, K) → (N, K*6) statistical features."""
    feats = []
    for k in range(traj.shape[-1]):
        r = traj[:, :, k]
        feats += [
            r.mean(1), r.std(1), r.min(1), r.max(1),
            np.abs(r).mean(1),
            np.diff(r, axis=1).mean(1),
        ]
    return np.stack(feats, axis=1)
```

**Input**: `(N, 180, 5)` — N windows × 180 timesteps × 5 PVs.
**Output**: `(N, 30)` — 5 PVs × 6 stats (mean, std, min, max, abs-mean, diff-mean).

**Use:**
```python
import joblib
rf     = joblib.load("trts_rf_classifier.pkl")
scaler = joblib.load("trts_rf_scaler.pkl")
from sec3_classification import extract_features

X = extract_features(windowed_pvs)    # (N, 30)
X_s = scaler.transform(X)
cls = rf.predict(X_s)                 # class ID 0-3
```

**Performance** (teammate-reported, from `report_plots/figures/s3_experement/s3c_confusion_trts.png`):
- Macro F1 = **0.937** on 3,377 windowed test samples
- Normal precision = 1.000 (zero false positives on normal windows)

**Trained on**: synthetic trajectories generated by the plant surrogate (TRTS = "Train on Synthetic, Test on Real"). Training script: [`05_detect/sec3_classification.py`](hai-digital-twin/05_detect/sec3_classification.py) on `origin/farah`.

---

## 6. Baselines (for context)

| Model | F1 | Script |
|---|---|---|
| IsolationForest (raw features) | 0.301 | [`05_classifier/train_bench.py`](hai-digital-twin/05_classifier/train_bench.py) |
| One-Class SVM | 0.392 | same |
| Autoencoder | 0.389 | same |
| Residual-MSE threshold (threshold=0.326) | — | `app/twin_runtime.py` (live use) |
| IsolationForest on GRU residuals (NEW on farah) | TBD | [`04_evaluate/anomaly_detector.py`](hai-digital-twin/04_evaluate/anomaly_detector.py) |

**Features**: 50 engineered stats (5 PVs × 10 — mean, std, min, max, p05, p95, slope, lag-1 autocorr, FFT peak freq/power) over 180-s windows, stride 60 s. From [`05_classifier/features.py`](hai-digital-twin/05_classifier/features.py).

**Numbers**: `outputs/classifier/bench.json`.

---

## 7. Causal graph (for root-cause)

**Role**: directed graph of inter-signal dependencies with lag estimates.

**Artifact**: [`outputs/causal_graph/parents_full.json`](hai-digital-twin/outputs/causal_graph/parents_full.json)

**Structure**:
```json
{
  "P1_PCV02D": [
    {"parent": "P1_PIT01", "lag": 2, "level": 0, "dynamics": 1, "via": "dcs_1001h", "lag_method": "tdmi"},
    {"parent": "P1_PP01AD", "lag": 3, "level": 0, "dynamics": 1, "via": "dcs_1001h", "lag_method": "constant_fallback"}
  ],
  "P1_PCV01D": [ ... ]
}
```

**19 PVs with explicit parent lists. Lag range: 2-26 s. Lag methods: TDMI (Transfer-entropy Time-Delayed Mutual Information), constant_fallback, xcorr.**

**Built once offline** ([`01_causal_graph/build_graph_full.py`](hai-digital-twin/01_causal_graph/build_graph_full.py)); do not recompute per run.

---

## 8. Assistive reasoner (root-cause trace)

**Role**: when Guardian / MSE detector fires, rank which upstream signals likely caused the divergence.

**Module**: [`app/assistive.py`](hai-digital-twin/app/assistive.py). Not ML — pure algorithm:

1. **L1**: rank 5 PVs by per-PV MSE between predicted (plant output) and observed (sensor) → top-K hot signals
2. **L2**: BFS upstream through `parents_full.json`, following edges by lag → candidate root-cause list with path + cumulative lag

**Use:**
```python
from app.assistive import rank_residuals_l1, trace_upstream_l2
import json

with open("outputs/causal_graph/parents_full.json") as f:
    graph = json.load(f)

hot_pvs = rank_residuals_l1(pv_pred, pv_true)      # list of (col, mse), sorted desc
root_causes = trace_upstream_l2(hot_pvs, graph)    # list of candidate upstream signals
```

---

## 9. LLM chatbot

**Role**: conversational explanation over live twin state.

**Model**: OpenAI GPT-4o-mini via the API. No fine-tuning — prompt-engineered with a `LIVE_STATE` JSON blob appended each turn.

**Module**: [`app/chatbot.py`](hai-digital-twin/app/chatbot.py)

**Live-state schema** (injected into system prompt):
- `current_t`: simulated second index
- `alerts`: list of `{t, pv, cause, score}` from the assistive layer
- `scenario`: current scenario (`normal` / `AP_no` / `AP_with` / `AE_no`)
- `loops`: per-loop current SP / PV / CV values

**API key**: from env var `OPENAI_API_KEY` or `st.secrets["OPENAI_API_KEY"]`.

---

## 10. One-page summary (copy-paste)

```
LAYER                        MODEL                       PATH
─────────────────────────────────────────────────────────────────────────────────────
0. Feature spec              133 active sensors          generator/scalers/metadata.pkl
1. Plant surrogate           GRUPlant weighted (2×512)   generator/weights/gru_plant.pt
                             (long horizon: v2)          training/checkpoints/v2_weighted_init_best.pt
2. Controllers ×5            GRU + CCSequenceModel       generator/weights/gru_ctrl_*.pt
3. Attack sandbox            wrapper over 1+2            attack_sim/attacks.py
4. Guardian (binary)         XGBoost Hybrid (Optuna)     C:/AI/HAI/best_hai_classifier.pkl
5. Attributor (multiclass)   RandomForest TRTS           ~/Downloads/trts_rf_classifier.pkl
                             feature extractor           05_detect/sec3_classification.py
6. Baselines                 IsoForest/OCSVM/AE          outputs/classifier/bench.json
7. Causal graph              pre-computed DAG            outputs/causal_graph/parents_full.json
8. Assistive reasoner        L1+L2 algorithm             app/assistive.py
9. LLM chatbot               GPT-4o-mini (API)           app/chatbot.py
```

**Nothing in this list needs to be retrained for a sibling dashboard.** All artifacts are frozen and importable. If you bump any model, update this file.

---

## 11. Environment

- Python 3.10+ (3.11 recommended)
- PyTorch 2.2+ with CUDA (for GRU inference speed; CPU also works)
- `pip install torch numpy pandas scikit-learn joblib xgboost plotly streamlit`
- Full env: [`hai-digital-twin/environment.yml`](hai-digital-twin/environment.yml)

## 12. Datasets

- Training: HAI 23.05 `processed/train{1,2,3,4}.csv` (1.5 GB total)
- Testing: HAI 23.05 `processed/test{1,2}.csv` with per-second labels
- Synthetic: `C:/AI/HAI/synthetic_attacks.csv` (36 k rows generated by plant surrogate)

## 13. Known limitations (from Student Guide §5)

1. Not a physics simulation — learned approximation only; extrapolation past training envelope is unreliable.
2. Doesn't discover new attack types — only propagates user-defined perturbations.
3. Doesn't simulate internal DCS attacks (AE01-AE08) as outputs — HAIEND is input only.
4. P1 boiler only — no P2 turbine, P3 water treatment, P4 HIL simulation.
5. Three of five loops (LC, TC, CC) fail KS distributional fidelity test — use with caution.
6. Attack-response fidelity is qualitative, not quantitatively validated against real attack data.

## Change log

- 2026-04-24: initial version, covering the 9-layer inventory + weighted plant swap + v2 long-horizon fine-tune.
