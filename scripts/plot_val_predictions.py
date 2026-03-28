"""
Evaluate transformer on each train file — stock-price-style plot.

For each train file (train1–4):
    - Blue  : actual sensor readings  (100% of the file)
    - Orange: model predictions       (last 20% val only)
    - Dashed vertical line at the 80/20 split

No test data is used anywhere.

Run after training transformer_model.py:
    python scripts/plot_val_predictions.py

Output: outputs/val_predictions.png
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer_model import (
    TransformerSeq2Seq, N_FEAT, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, DROPOUT,
)
from utils.data_loader import load_merged, identify_common_constants
from utils.prep import twin

plt.style.use("fivethirtyeight")

CHECKPOINT = "outputs/transformer_twin.pt"
ENC_LEN    = 300
DEC_LEN    = 180
TRAIN_FRAC = 0.8
META_COLS  = {"timestamp", "attack", "label", "attack_p1", "attack_p2", "attack_p3"}

Path("outputs").mkdir(exist_ok=True)


# ── Load trained model ────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ckpt  = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model = TransformerSeq2Seq(
    n_features=N_FEAT, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT,
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("Loaded checkpoint.")

# ── Get normalizer (no test data — twin() fits on train only) ─────────────────
print("Loading normalizer ...")
norm = twin(input_len=ENC_LEN, target_len=DEC_LEN, stride=60)["norm"]

const_hai, const_hiend, hiend_dups = identify_common_constants()


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_and_normalise(num: int) -> tuple[np.ndarray, list[str]]:
    """Load train{num} (full file), normalise → (T, F) array + col names."""
    df   = load_merged("train", num,
                       drop_constants=True, keep_hai_duplicates=True,
                       const_cols_hai=const_hai, const_cols_hiend=const_hiend,
                       hiend_dup_cols=hiend_dups)
    cols = [c for c in df.columns if c not in META_COLS and df[c].dtype != object]
    arr  = norm.transform(df)[cols].values.astype(np.float32)
    return arr, cols


@torch.no_grad()
def predict_val(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Predict the val portion of arr (last 20%) contiguously.

    Uses the last ENC_LEN rows of the train portion as context for the first
    window, then steps forward in non-overlapping DEC_LEN blocks.

    Returns
    -------
    predictions : (P,) mean over sensors — stitched continuous prediction
    val_start   : row index where val begins (the 80% mark)
    """
    T         = len(arr)
    val_start = int(T * TRAIN_FRAC)

    all_preds = []
    # Step through val in non-overlapping DEC_LEN blocks
    pos = val_start
    while pos + DEC_LEN <= T:
        enc_start = max(0, pos - ENC_LEN)
        enc       = arr[enc_start:pos]                       # up to ENC_LEN rows

        # Pad with zeros if val starts before ENC_LEN context is available
        if len(enc) < ENC_LEN:
            pad = np.zeros((ENC_LEN - len(enc), arr.shape[1]), dtype=np.float32)
            enc = np.concatenate([pad, enc], axis=0)

        src  = torch.tensor(enc[np.newaxis]).float().to(device)  # (1, ENC_LEN, F)
        pred = model.predict(src, dec_len=DEC_LEN)               # (1, DEC_LEN, F)
        all_preds.append(pred[0].cpu().numpy())                  # (DEC_LEN, F)
        pos += DEC_LEN

    if not all_preds:
        return np.array([]), val_start

    predictions = np.concatenate(all_preds, axis=0)   # (P*DEC_LEN, F)
    return predictions, val_start


def pick_sensor(arr: np.ndarray, cols: list[str]) -> tuple[int, str]:
    """Return index + name of the sensor with highest variance."""
    idx = int(arr.var(axis=0).argmax())
    return idx, cols[idx]


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(16, 20))
fig.suptitle(
    "Transformer Digital Twin — val predictions vs actual\n"
    "(blue = actual full file · orange = predicted last 20% · no test data)",
    fontsize=13,
)

for i, num in enumerate(range(1, 5)):
    print(f"\ntrain{num} ...")
    arr, cols        = load_and_normalise(num)
    predictions, val_start = predict_val(arr)

    sidx, sname = pick_sensor(arr, cols)

    ax = axes[i]

    # ── Actual: full 100% of the file ────────────────────────────────────────
    ax.plot(arr[:, sidx],
            color="#1f77b4", lw=1, label="Actual sensor readings (train file)")

    # ── Predicted: val portion (stitched) ────────────────────────────────────
    if len(predictions) > 0:
        pred_x = np.arange(val_start, val_start + len(predictions))
        ax.plot(pred_x, predictions[:, sidx],
                color="#ff7f0e", lw=1.2, label="Predicted (val 20%)")
        rmse = np.sqrt(((predictions[:, sidx] -
                         arr[val_start:val_start + len(predictions), sidx]) ** 2).mean())

    # ── Split line ────────────────────────────────────────────────────────────
    ax.axvline(val_start, color="grey", linestyle="--", lw=1.2)
    ax.text(val_start + 20, ax.get_ylim()[1] * 0.92,
            "← train 80%  |  val 20% →", fontsize=8, color="grey")

    ax.set_title(
        f"train{num}  |  sensor: {sname[:30]}"
        + (f"  |  val RMSE = {rmse:.4f}" if len(predictions) > 0 else ""),
        fontsize=10,
    )
    ax.set_xlabel("timestep")
    ax.set_ylabel("normalised value")
    ax.legend(fontsize=9)

    print(f"  total rows={len(arr)}  val_start={val_start}"
          + (f"  RMSE={rmse:.4f}" if len(predictions) > 0 else "  (no predictions)"))

plt.tight_layout()
out1 = "outputs/val_predictions_context.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out1}")


# ── Figure 2: IBM-style — actual (red) vs predicted (blue), val only ──────────
# Mirrors the stock-price plot: both lines on the same axis, val portion only.

fig2, axes2 = plt.subplots(4, 1, figsize=(14, 18))
fig2.suptitle(
    "Transformer Digital Twin — Real vs Predicted (val 20% only)",
    fontsize=13,
)

# Reuse the data collected in the first loop
results = {}   # num → (arr, cols, predictions, val_start)
for num in range(1, 5):
    arr, cols              = load_and_normalise(num)
    predictions, val_start = predict_val(arr)
    results[num]           = (arr, cols, predictions, val_start)

for i, num in enumerate(range(1, 5)):
    arr, cols, predictions, val_start = results[num]
    sidx, sname = pick_sensor(arr, cols)
    ax = axes2[i]

    if len(predictions) == 0:
        ax.set_title(f"train{num} — no val predictions")
        continue

    n        = len(predictions)
    actual   = arr[val_start : val_start + n, sidx]
    pred     = predictions[:, sidx]
    t        = np.arange(n)
    rmse     = np.sqrt(((pred - actual) ** 2).mean())

    ax.plot(t, actual, color="red",  lw=1.2, label=f"Real sensor value (train{num} val)")
    ax.plot(t, pred,   color="blue", lw=1.2, label=f"Predicted sensor value (train{num} val)")
    ax.set_title(
        f"train{num} val prediction  |  sensor: {sname[:35]}  |  RMSE = {rmse:.4f}",
        fontsize=10,
    )
    ax.set_xlabel("Time (val timesteps)")
    ax.set_ylabel("Normalised sensor value")
    ax.legend(fontsize=9)

plt.tight_layout()
out2 = "outputs/val_predictions_comparison.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
