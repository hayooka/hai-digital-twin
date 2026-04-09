"""
run_eval_lstm_causal.py — Evaluate a saved LSTM checkpoint (with causal loss).

Loads:
    outputs/lstm/causal/models/lstm.pt   — saved model weights + metadata
    outputs/scaled_split/val_data.npz    — val windows
    outputs/scaled_split/test_data.npz   — test windows

Run:
    python 04_evaluate/run_eval_lstm_causal.py
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

ROOT     = Path(__file__).parent.parent
EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT / "03_model"))
sys.path.insert(0, str(EVAL_DIR))

from lstm import LSTMSeq2Seq, plot_val_predictions
from eval import evaluate_twin

BATCH       = 64
CHECKPOINT  = ROOT / "outputs/lstm/causal/models/lstm.pt"
VAL_DATA    = ROOT / "outputs/scaled_split/val_data.npz"
TEST_DATA   = ROOT / "outputs/scaled_split/test_data.npz"

# ── Load checkpoint ───────────────────────────────────────────────────────────
print(f"Loading checkpoint: {CHECKPOINT}")
ckpt = torch.load(CHECKPOINT, map_location="cpu")

n_feat      = ckpt["n_feat"]
n_scenarios = ckpt["n_scenarios"]
sensor_cols = ckpt["sensor_cols"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}  |  features: {n_feat}  |  scenarios: {n_scenarios}")

model = LSTMSeq2Seq(
    n_features=n_feat, n_scenarios=n_scenarios,
    hidden_size=256, num_layers=4, dropout=0.1, output_len=180,
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\nLoading val data:  {VAL_DATA}")
val  = np.load(VAL_DATA)
X_val, Y_val = val["X"], val["y"]
print(f"  X_val: {X_val.shape}  Y_val: {Y_val.shape}")

print(f"Loading test data: {TEST_DATA}")
test = np.load(TEST_DATA)
X_test, Y_test = test["X"], test["y"]
y_test         = test["attack_labels"]   # binary: 0=normal, 1=attack
print(f"  X_test: {X_test.shape}  Y_test: {Y_test.shape}")

# ── predict_fn ────────────────────────────────────────────────────────────────
def predict_fn(X: np.ndarray, Y: np.ndarray,
               scenario: np.ndarray | None = None) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            src    = torch.tensor(X[i:i+BATCH]).float().to(device)
            tgt    = torch.tensor(Y[i:i+BATCH]).float().to(device)
            sc     = torch.tensor(scenario[i:i+BATCH]).long().to(device) if scenario is not None else None
            dec_in = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
            preds.append(model(src, dec_in, sc).cpu().numpy())
    return np.concatenate(preds, axis=0)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\nEvaluating...")
metrics = evaluate_twin(
    predict_fn=lambda X, Y: predict_fn(X, Y, np.zeros(len(X), dtype=np.int32)),
    X_val=X_val, Y_val=Y_val,
    X_test=X_test, Y_test=Y_test,
    y_test=y_test,
    save_path=str(ROOT / "outputs/lstm/causal/lstm_metrics.json"),
)

# ── Causal violation % (absolute threshold, same as Transformer) ──────────────
print("\nChecking causal violations...")

CAUSAL_THRESHOLD = 0.1   # absolute threshold (normalized space)

parents_data = json.loads((ROOT / "outputs/causal_graph/parents_full.json").read_text())
col_idx = {col: i for i, col in enumerate(sensor_cols)}

relationships = []
for target, plist in parents_data.items():
    if target not in col_idx:
        continue
    t_idx = col_idx[target]
    parents = [(col_idx[p["parent"]], int(p.get("lag", 1)))
               for p in plist if p["parent"] in col_idx]
    if parents:
        relationships.append((target, t_idx, parents))

# Run predictions on val set (normal behavior — strictest test)
Y_pred_val = predict_fn(X_val, Y_val, np.zeros(len(X_val), dtype=np.int32))

total_checks = 0
total_violations = 0
per_edge = []

for sensor_name, t_idx, parents in relationships:
    max_lag = max(lag for _, lag in parents)
    T = Y_pred_val.shape[1]
    if max_lag >= T:
        continue

    target_vals = Y_pred_val[:, max_lag:, t_idx]

    parent_signals = []
    for p_idx, lag in parents:
        offset = max_lag - lag
        parent_signals.append(Y_pred_val[:, offset:T - lag, p_idx])
    expected = np.stack(parent_signals, axis=0).mean(axis=0)

    diff = np.abs(target_vals - expected)
    violations = (diff > CAUSAL_THRESHOLD).sum()
    checks = diff.size
    pct = 100.0 * violations / checks

    total_checks += checks
    total_violations += violations
    per_edge.append((sensor_name, pct))

overall_pct = 100.0 * total_violations / total_checks if total_checks > 0 else float("nan")

print(f"\n{'='*55}")
print(f"  CAUSAL VIOLATION REPORT  (threshold={CAUSAL_THRESHOLD})")
print(f"{'='*55}")
for name, pct in sorted(per_edge, key=lambda x: -x[1]):
    flag = "  *** FAIL" if pct >= 20 else ""
    print(f"  {name:<22} {pct:6.2f}%{flag}")
print(f"{'='*55}")
print(f"  Overall causal violation: {overall_pct:.2f}%  {'PASS (<20%)' if overall_pct < 20 else 'FAIL (>=20%)'}")
print(f"{'='*55}")

metrics["causal_violation_pct"] = round(overall_pct, 4)
with open(ROOT / "outputs/lstm/causal/lstm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  Updated outputs/lstm/causal/lstm_metrics.json")

# ── Causal violation plot ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sorted_edges = sorted(per_edge, key=lambda x: x[1])
names  = [e[0] for e in sorted_edges]
values = [e[1] for e in sorted_edges]
colors = ["#d62728" if v >= 20 else "#2ca02c" for v in values]

fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))
bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.7)
ax.axvline(20, color="black", linestyle="--", linewidth=1.2, label="20% threshold")
ax.axvline(overall_pct, color="#ff7f0e", linestyle="-", linewidth=1.5,
           label=f"Overall: {overall_pct:.1f}%")

for bar, val in zip(bars, values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=8)

ax.set_xlabel("Causal Violation %", fontsize=11)
ax.set_title(f"Causal Violation per Sensor  (threshold={CAUSAL_THRESHOLD})", fontsize=12)
ax.set_xlim(0, max(values) * 1.12)
ax.legend(fontsize=9)
fig.tight_layout()

plot_path = ROOT / "outputs/lstm/causal/plots/causal_violations.png"
Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"  Saved: {plot_path}")

# ── Scenario prediction plots ─────────────────────────────────────────────────
print("\nPlotting scenario prediction plots...")

STRIDE   = 60
PAST_LEN = 1000

SCENARIO_LABELS = {
    0: "Normal",
    1: "AP (no combination)",
    2: "AP (with combination)",
    3: "AE (no combination)",
}

def _pick(cols, keywords, n=1):
    result = []
    for kw in keywords:
        for i, c in enumerate(cols):
            if kw in c and i not in result:
                result.append(i)
                if len(result) == n:
                    return result
    return result

SCENARIO_SENSORS = {
    0: [_pick(sensor_cols, ["P1_PIT01"], 1),
        _pick(sensor_cols, ["P1_FT01"], 1),
        _pick(sensor_cols, ["P1_FCV01D"], 1),
        _pick(sensor_cols, ["P1_TIT01"], 1)],
    1: [_pick(sensor_cols, ["P1_PIT02"], 1),
        _pick(sensor_cols, ["P1_LIT01"], 1),
        _pick(sensor_cols, ["P1_LCV01D"], 1),
        _pick(sensor_cols, ["P1_PP01AD"], 1)],
    2: [_pick(sensor_cols, ["P1_FT02"], 1),
        _pick(sensor_cols, ["P1_FCV02D"], 1),
        _pick(sensor_cols, ["P1_PCV01D"], 1),
        _pick(sensor_cols, ["P1_TIT02"], 1)],
    3: [_pick(sensor_cols, ["P1_TIT03"], 1),
        _pick(sensor_cols, ["P1_FT03"], 1),
        _pick(sensor_cols, ["P1_FCV03D"], 1),
        _pick(sensor_cols, ["P1_SOL01D"], 1)],
}

def reconstruct_past(X_windows, win_idx, stride, past_len):
    enc_len = X_windows.shape[1]
    current = X_windows[win_idx]
    extra   = past_len - enc_len
    if extra <= 0:
        return current[-past_len:]
    n_back = int(np.ceil(extra / stride))
    parts = []
    for j in range(max(0, win_idx - n_back), win_idx):
        parts.append(X_windows[j][:stride])
    parts.append(current)
    past = np.concatenate(parts, axis=0)
    return past[-past_len:]

test_scenario_labels = test["scenario_labels"]

for scenario_id, scenario_name in SCENARIO_LABELS.items():

    if scenario_id == 0:
        X_src = X_val
        Y_src = Y_val
        mask  = np.arange(len(X_val))
    else:
        mask = np.where(test_scenario_labels == scenario_id)[0]
        X_src = X_test
        Y_src = Y_test

    if len(mask) == 0:
        print(f"  Scenario {scenario_id}: no windows found, skipping")
        continue

    win_local = mask[len(mask) // 2]
    past_data  = reconstruct_past(X_src, win_local, STRIDE, PAST_LEN)
    true_future = Y_src[win_local]

    src_t  = torch.tensor(X_src[win_local:win_local+1]).float().to(device)
    sc_t   = torch.tensor([scenario_id]).long().to(device)
    pred_future = model.predict(src_t, dec_len=180, scenario=sc_t).cpu().numpy()[0]

    sensor_indices = SCENARIO_SENSORS[scenario_id]
    sensor_names   = [sensor_cols[idx[0]] if idx else "?" for idx in sensor_indices]
    n_sensors      = len(sensor_indices)

    fig, axes = plt.subplots(n_sensors, 1, figsize=(12, 4 * n_sensors), sharey=False)
    fig.suptitle(f"Scenario {scenario_id}: {scenario_name}", fontsize=13, fontweight="bold")

    past_t   = np.arange(-PAST_LEN, 0)
    future_t = np.arange(0, 180)

    for ax, idx_list, sname in zip(axes, sensor_indices, sensor_names):
        if not idx_list:
            ax.set_visible(False)
            continue
        sidx = idx_list[0]

        ax.plot(past_t,   past_data[:, sidx],   color="black", lw=1.2, label="Past (true)")
        ax.plot(future_t, true_future[:, sidx],  color="green", lw=1.5, label="Future (true)")
        ax.plot(future_t, pred_future[:, sidx],  color="red",   lw=1.5,
                linestyle="--", label="Predicted")

        ax.axvline(0, color="gray", linestyle=":", lw=1)
        ax.set_title(sname[:30], fontsize=9)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("normalised value")
        ax.legend(fontsize=7)

    fig.tight_layout()
    out_path = ROOT / f"outputs/lstm/causal/plots/scenario_{scenario_id}_predictions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out_path}")

# ── Val predictions overview ──────────────────────────────────────────────────
print("\nPlotting val predictions...")
Path(ROOT / "outputs/lstm/causal/plots").mkdir(parents=True, exist_ok=True)
plot_val_predictions(model, X_val, Y_val, sensor_cols, device,
                     dec_len=180, model_name="HAI Digital Twin LSTM (causal)",
                     out_path=str(ROOT / "outputs/lstm/causal/plots/val_predictions_comparison.png"))