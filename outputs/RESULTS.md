# HAI Digital Twin — Experiment Results

## Model Performance Summary

All models are evaluated on the HAI dataset. "Causal" variants inject causal graph structure as an inductive bias during training.

| Model | Variant | RMSE Val | RMSE Test (Normal) | RMSE Test (Attack) | Separation Ratio | Causal Violation % |
|-------|---------|----------|-------------------|-------------------|-----------------|-------------------|
| LSTM | No Causal | 0.1721 | 0.2001 | 0.8039 | 4.018 | 96.33% |
| LSTM | Causal | 0.2597 | 0.2839 | 1.2136 | **4.275** | **13.10%** |
| Transformer | No Causal | 0.1372 | 0.1450 | 0.3069 | 2.117 | 96.62% |
| Transformer | Causal | 0.2383 | 0.2509 | 0.9231 | 3.679 | **13.02%** |

### Metric Definitions

- **RMSE Val / Test Normal**: Reconstruction error on clean (non-attack) data — lower is better.
- **RMSE Test Attack**: Reconstruction error on attack data — higher means the model is more sensitive to anomalies.
- **Separation Ratio**: `RMSE_attack / RMSE_normal` — higher means better discrimination between normal and attack traffic.
- **Causal Violation %**: Percentage of predictions that violate causal constraints from the graph — lower is better for causal models.

---

## Key Observations

### Effect of Causal Constraints
- Adding causal constraints **dramatically reduces causal violations**: from ~96% down to ~13% for both architectures.
- The trade-off is a modest increase in reconstruction error on normal data (RMSE val/test normal increases ~0.04–0.10).
- Despite higher normal RMSE, causal LSTM achieves the **best separation ratio (4.275)**, meaning it better separates attack from normal behavior.

### LSTM vs Transformer
- Transformer (no causal) achieves the **lowest reconstruction error** overall (RMSE val = 0.137), outperforming LSTM on clean data.
- LSTM (causal) achieves the **highest separation ratio (4.275)** — best attack discrimination.
- Both architectures benefit similarly from causal constraints (violation rate drops ~83 percentage points).

### Recommendation
- For **anomaly detection sensitivity**: LSTM + Causal (separation ratio 4.275, violation 13.1%)
- For **reconstruction fidelity on normal data**: Transformer + No Causal (RMSE val 0.137)
- For **balanced performance**: Transformer + Causal (reasonable separation 3.68, low violation 13.0%, lower normal RMSE than LSTM causal)

---

## Causal Graph Summary

The causal graph (`outputs/causal_graph/parents_full.json`) encodes 19 target variables and their causal parents, derived from process knowledge and lag estimation.

### Graph Statistics
- **Nodes (targets)**: 19
- **Lag estimation methods**: TDMI (time-delayed mutual information), xcorr fallback, constant fallback
- **Causality levels**: 0 (direct DCS control loop), 1 (direct physical), 2 (indirect via physical subsystem)
- **Subsystems referenced**: `dcs_1001h`, `dcs_1002h`, `dcs_1003h`, `dcs_1004h`, `dcs_1010h`, `dcs_1020h`, `phy_boiler`, `phy_boiler:TK01`, `phy_boiler:TK02`, `phy_boiler:TK03`, `phy_boiler:HEX01T`

### Notable Causal Relationships

| Target | Key Parents | Max Lag | Notes |
|--------|------------|---------|-------|
| `P1_PCV02D` | `P1_PIT01`, `P1_PP01AD`, `P1_PP01BD` | 3 | Pressure control valve |
| `P1_LCV01D` | `P1_LIT01`, `P1_FT03`, `P1_PP01AD`, `P1_PCV01D`, `P1_FCV03D` | 4 | Level control valve |
| `P1_PP01AD` | 8 parents (pumps, valves) | 10 | Central pump, many dependencies |
| `P1_SOL01D` | `P1_PIT01_HH` + 5 via TK01 | 10 | Solenoid valve, long lag paths |
| `P1_TIT03` | 5 parents via `phy_boiler:TK01` | 26 | Temperature sensor, slowest dynamics |
| `P1_TIT02` | `P1_PP02D`, `P4_HT_PO`, `P1_FCV02D` | 22 | Cross-process heat transfer |
