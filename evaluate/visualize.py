"""
Digital Twin — Visualization

Plots:
    1. Predicted vs Actual sensor values (normal vs attack window)
    2. Reconstruction error over time (RMSE per window)
    3. Per-sensor error heatmap (which sensors spike during attacks)
    4. ISO Forest anomaly score distribution

Run:
    python evaluate/visualize.py
"""
from __future__ import annotations

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # no display needed — saves PNGs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.transformer_model import TransformerSeq2Seq
from utils.prep import twin

Path("outputs/plots").mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
ENC_LEN  = 60
DEC_LEN  = 180
N_FEAT   = 277
D_MODEL  = 256
N_HEADS  = 8
N_LAYERS = 4
FFN_DIM  = 1024
DROPOUT  = 0.1
BATCH    = 64

# ── Load model + data ─────────────────────────────────────────────────────────
print("Loading checkpoint...")
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("outputs/transformer_twin.pt", map_location=device, weights_only=False)

model = TransformerSeq2Seq(
    n_features=N_FEAT, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT,
).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print("Loading test data...")
data    = twin(input_len=ENC_LEN, target_len=DEC_LEN, stride=60)
X_test  = data["X_test"]          # (K, 60,  277)
Y_test  = data["Y_test"]          # (K, 180, 277)
y_test  = data["y_test_labels"]   # (K,)  0/1

train_errors = checkpoint["train_errors"]
test_errors  = checkpoint["test_errors"]
train_errors = np.nan_to_num(np.clip(train_errors, 0, np.percentile(train_errors, 99.9)))
test_errors  = np.nan_to_num(np.clip(test_errors,  0, np.percentile(train_errors, 99.9)))


# ── Helper: run model on a batch ──────────────────────────────────────────────

def predict_batch(X: np.ndarray, Y: np.ndarray):
    """Returns (pred, rmse_per_window, per_sensor_mse)."""
    preds, rmses, sensor_errs = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            src    = torch.tensor(X[i:i+BATCH]).float().to(device)
            tgt    = torch.tensor(Y[i:i+BATCH]).float().to(device)
            dec_in = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
            pred   = model(src, dec_in)
            mse    = ((pred - tgt) ** 2)
            rmses.append(mse.mean(dim=(1, 2)).sqrt().cpu().numpy())
            sensor_errs.append(mse.mean(dim=1).cpu().numpy())   # (B, 277)
            preds.append(pred.cpu().numpy())
    return (np.concatenate(preds),
            np.concatenate(rmses),
            np.concatenate(sensor_errs))

print("Running predictions on test2...")
pred_all, rmse_all, sensor_err_all = predict_batch(X_test, Y_test)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Predicted vs Actual (normal window vs attack window, top 3 sensors)
# ═══════════════════════════════════════════════════════════════════════════════

normal_idx = np.where(y_test == 0)[0]
attack_idx = np.where(y_test == 1)[0]

# Pick the attack window with the highest RMSE
best_attack = attack_idx[np.argmax(rmse_all[attack_idx])]
best_normal = normal_idx[np.argmin(rmse_all[normal_idx])]

# Pick 3 sensors with highest error in the attack window
top_sensors = np.argsort(sensor_err_all[best_attack])[-3:][::-1]

fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharey=False)
fig.suptitle("Transformer: Predicted vs Actual Sensor Values", fontsize=14, fontweight="bold", y=1.01)

timesteps = np.arange(DEC_LEN)
colors     = ["#1D9E75", "#534AB7", "#D85A30"]

for row, sensor in enumerate(top_sensors):
    for col, (idx, label, color_act, color_pred) in enumerate([
        (best_normal, "Normal window",  "#1D9E75", "#0F6E56"),
        (best_attack, "Attack window",  "#E24B4A", "#8B0000"),
    ]):
        ax = axes[row][col]
        actual    = Y_test[idx, :, sensor]
        predicted = pred_all[idx, :, sensor]

        ax.plot(timesteps, actual,    label="Actual",    color=color_act,  linewidth=1.5)
        ax.plot(timesteps, predicted, label="Predicted", color=color_pred, linewidth=1.2,
                linestyle="--", alpha=0.85)

        ax.fill_between(timesteps, actual, predicted,
                        alpha=0.15, color="#E24B4A" if col == 1 else "#888780",
                        label="Error")

        rmse_val = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        ax.set_title(f"{label} · Sensor {sensor} · RMSE={rmse_val:.4f}",
                     fontsize=10, color="#2C2C2A")
        ax.set_xlabel("Future timestep", fontsize=9)
        ax.set_ylabel("Normalized value", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/plots/01_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/plots/01_predicted_vs_actual.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 — RMSE over time (all test2 windows, color-coded normal/attack)
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 4))

window_idx = np.arange(len(rmse_all))
ax.scatter(window_idx[y_test == 0], rmse_all[y_test == 0],
           s=8, color="#1D9E75", alpha=0.5, label=f"Normal ({(y_test==0).sum()})")
ax.scatter(window_idx[y_test == 1], rmse_all[y_test == 1],
           s=14, color="#E24B4A", alpha=0.8, label=f"Attack ({(y_test==1).sum()})", zorder=5)

normal_mean = rmse_all[y_test == 0].mean()
ax.axhline(normal_mean, color="#888780", linewidth=1, linestyle="--",
           label=f"Normal mean={normal_mean:.4f}")
ax.axhline(normal_mean * 5, color="#BA7517", linewidth=1, linestyle="--",
           label=f"5× threshold={normal_mean*5:.4f}")

ax.set_title("Transformer Reconstruction RMSE — test2 windows", fontsize=13, fontweight="bold")
ax.set_xlabel("Window index (time →)", fontsize=10)
ax.set_ylabel("RMSE", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/plots/02_rmse_over_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/plots/02_rmse_over_time.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Per-sensor error heatmap (top 30 sensors, normal vs attack)
# ═══════════════════════════════════════════════════════════════════════════════

mean_normal_err = sensor_err_all[y_test == 0].mean(axis=0)   # (277,)
mean_attack_err = sensor_err_all[y_test == 1].mean(axis=0)   # (277,)

# Top 30 sensors by attack error
top30 = np.argsort(mean_attack_err)[-30:][::-1]

heatmap_data = np.stack([mean_normal_err[top30], mean_attack_err[top30]], axis=0)

fig, ax = plt.subplots(figsize=(14, 3))
im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd",
               vmin=0, vmax=heatmap_data.max())

ax.set_yticks([0, 1])
ax.set_yticklabels(["Normal", "Attack"], fontsize=11)
ax.set_xticks(range(30))
ax.set_xticklabels([f"S{s}" for s in top30], rotation=45, ha="right", fontsize=8)
ax.set_title("Per-Sensor MSE Heatmap — Top 30 attacked sensors (test2)",
             fontsize=12, fontweight="bold")

plt.colorbar(im, ax=ax, label="Mean MSE")
plt.tight_layout()
plt.savefig("outputs/plots/03_sensor_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/plots/03_sensor_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4 — ISO Forest anomaly score distribution
# ═══════════════════════════════════════════════════════════════════════════════

cap          = np.percentile(train_errors, 99.9)
train_errors = np.clip(train_errors, 0, cap)
test_errors  = np.clip(test_errors,  0, cap)

pca           = PCA(n_components=20, random_state=42)
train_reduced = pca.fit_transform(np.log1p(train_errors))
test_reduced  = pca.transform(np.log1p(test_errors))

attack_rate = float((y_test == 1).sum()) / len(y_test)
iso = IsolationForest(n_estimators=200, contamination=attack_rate,
                      random_state=42, n_jobs=-1)
iso.fit(train_reduced)
scores = -iso.score_samples(test_reduced)   # higher = more anomalous

fig, ax = plt.subplots(figsize=(10, 4))

ax.hist(scores[y_test == 0], bins=60, color="#1D9E75", alpha=0.6,
        label=f"Normal ({(y_test==0).sum()})", density=True)
ax.hist(scores[y_test == 1], bins=30, color="#E24B4A", alpha=0.75,
        label=f"Attack ({(y_test==1).sum()})", density=True)

threshold = np.percentile(scores[y_test == 0], 99)
ax.axvline(threshold, color="#BA7517", linewidth=1.5, linestyle="--",
           label=f"99th pct normal = {threshold:.3f}")

ax.set_title("ISO Forest Anomaly Score Distribution — test2", fontsize=13, fontweight="bold")
ax.set_xlabel("Anomaly score (higher = more anomalous)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/plots/04_iso_score_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/plots/04_iso_score_distribution.png")


print("\nAll plots saved to outputs/plots/")
print("  01_predicted_vs_actual.png  — Predicted vs actual sensor values")
print("  02_rmse_over_time.png       — RMSE per window over time")
print("  03_sensor_heatmap.png       — Which sensors are attacked")
print("  04_iso_score_distribution.png — ISO Forest separation")
