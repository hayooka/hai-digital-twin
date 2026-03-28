"""
Transformer Seq2Seq — Layer 1: Physical Model / Digital Twin (primary)

Predicts NORMAL sensor readings given 300s of context.
Trained on benign data only (80% of each train1-4, temporal split).
High reconstruction error on test2 = anomaly signal → passed to Layer 2.

Data split (temporal 80/20 per file — see utils/prep.py twin()):
    Train : first 80% of train1-4  → windowed per file (no gap crossing)
    Val   : last  20% of train1-4  → same per-file windowing
    Test  : test2 only (held-out — never seen during training)

Shape: X (N, 300, F) → Y (N, 180, F)   where F = actual sensor count after
       constant column deletion (determined at runtime, ~112 features).

Window size = 300s  (from window_size_analysis notebook).
"""

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # add project root to path
from utils.prep import twin
from utils.eval  import evaluate_twin
from utils.data_loader import load_merged, identify_common_constants

# ── Data ──────────────────────────────────────────────────────────────────────
# stride=60 keeps memory manageable; stride=1 → OOM on 300s windows
data    = twin(input_len=300, target_len=180, stride=60)
X_train = data["X_train"]        # (N, 300, F)  encoder input — normal only
Y_train = data["Y_train"]        # (N, 180, F)  decoder target
X_val   = data["X_val"]          # (M, 300, F)  last 20% of each train file
Y_val   = data["Y_val"]          # (M, 180, F)
X_test  = data["X_test"]         # (K, 300, F)  test2 held-out
Y_test  = data["Y_test"]         # (K, 180, F)
y_test  = data["y_test_labels"]  # (K,)  0=normal 1=attack
norm    = data["norm"]           # fitted normalizer — pass to ISO Forest

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_FEAT   = X_train.shape[2]   # dynamic: actual sensor count after constant deletion
D_MODEL  = 256
N_HEADS  = 8
N_LAYERS = 4
FFN_DIM  = 1024
DROPOUT  = 0.1
EPOCHS   = 50
BATCH    = 64
LR       = 1e-4

Path("outputs").mkdir(exist_ok=True)


# ── Positional Encoding ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── Model ─────────────────────────────────────────────────────────────────────

class TransformerSeq2Seq(nn.Module):
    """
    Encoder-Decoder Transformer for sensor forecasting.

    Encoder: processes 300s of past sensor readings → context (memory)
    Decoder: predicts 180s of future readings from context
    Loss:    MSE on sensor values

    No attack-type conditioning — learns purely from sensor values.
    Anomaly detection via reconstruction error: high error on test2 = attack.
    """
    def __init__(self, n_features=N_FEAT, d_model=128, n_heads=8,
                 n_layers=3, ffn_dim=512, dropout=0.1):
        super().__init__()
        self.input_proj  = nn.Linear(n_features, d_model)
        self.output_proj = nn.Linear(d_model, n_features)
        self.pos_enc     = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn_dim, dropout,
            batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, ffn_dim, dropout,
            batch_first=True, norm_first=True)

        self.encoder = nn.TransformerEncoder(enc_layer, n_layers,
                                             enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec_layer, n_layers)

    def encode(self, src):
        """src: (B, enc_len, n_features) → memory: (B, enc_len, d_model)"""
        return self.encoder(self.pos_enc(self.input_proj(src)))

    def forward(self, src, tgt_in):
        """
        src:    (B, 300, F)  encoder input (300s context window)
        tgt_in: (B, 180, F)  teacher-forced decoder input
        returns (B, 180, F)
        """
        memory = self.encode(src)
        tgt_h  = self.pos_enc(self.input_proj(tgt_in))
        T      = tgt_in.size(1)
        mask   = nn.Transformer.generate_square_subsequent_mask(
            T, device=src.device)
        out = self.decoder(tgt_h, memory, tgt_mask=mask, tgt_is_causal=True)
        return self.output_proj(out)                    # (B, 180, 277)

    @torch.no_grad()
    def predict(self, src, dec_len=180):
        """
        Autoregressive inference — no teacher forcing.
        src: (B, 300, F) → returns (B, dec_len, F)
        """
        self.eval()
        memory = self.encode(src)
        pred   = src[:, -1:, :]   # start token = last encoder step
        outs   = []
        for _ in range(dec_len):
            tgt_h = self.pos_enc(self.input_proj(pred))
            T     = pred.size(1)
            mask  = nn.Transformer.generate_square_subsequent_mask(
                T, device=src.device)
            out   = self.decoder(tgt_h, memory, tgt_mask=mask, tgt_is_causal=True)
            step  = self.output_proj(out[:, -1:, :])
            outs.append(step)
            pred  = torch.cat([pred, step], dim=1)
        return torch.cat(outs, dim=1)                   # (B, 180, 277)

    def reconstruction_errors(self, X, Y, batch_size=64):
        """
        Per-sensor MSE errors for each window → (N, 277).
        Pass results to ISO Forest for anomaly detection.
        """
        device = next(self.parameters()).device
        self.eval()
        errors = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                src  = torch.tensor(X[i:i+batch_size]).float().to(device)
                tgt  = torch.tensor(Y[i:i+batch_size]).float().to(device)
                pred = self.predict(src, dec_len=tgt.size(1))
                err  = ((pred - tgt) ** 2).mean(dim=1)  # (B, 277)
                errors.append(err.cpu().numpy())
        return np.concatenate(errors, axis=0)           # (N, 277)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, X_tr, Y_tr, X_v, Y_v,
          epochs=EPOCHS, batch=BATCH, lr=LR):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    N = len(X_tr)
    best_val, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        idx   = np.random.permutation(N)
        total = 0.0
        for i in range(0, N, batch):
            b      = idx[i:i+batch]
            src    = torch.tensor(X_tr[b]).float().to(device)
            tgt    = torch.tensor(Y_tr[b]).float().to(device)
            # teacher forcing: decoder sees [last encoder step, Y[0..T-2]]
            dec_in = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
            loss   = criterion(model(src, dec_in), tgt)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        # ── validate ──
        model.eval()
        val_loss, n_b = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(X_v), batch):
                src    = torch.tensor(X_v[i:i+batch]).float().to(device)
                tgt    = torch.tensor(Y_v[i:i+batch]).float().to(device)
                dec_in = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
                val_loss += criterion(model(src, dec_in), tgt).item()
                n_b += 1
        val_loss /= max(1, n_b)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train={total/max(1,N//batch):.5f}  val={val_loss:.5f}")

    model.load_state_dict(best_state)
    print(f"Best val loss: {best_val:.5f}")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def make_predict_fn(model, batch_size=BATCH):
    """
    Wrap a trained model into the predict_fn(X, Y) → Y_pred interface
    expected by evaluate_twin() in utils/eval.py.
    Teacher-forced: uses Y as decoder input (same as training).
    """
    device = next(model.parameters()).device
    model.eval()

    def predict_fn(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                src    = torch.tensor(X[i:i+batch_size]).float().to(device)
                tgt    = torch.tensor(Y[i:i+batch_size]).float().to(device)
                dec_in = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
                pred   = model(src, dec_in)
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds, axis=0)

    return predict_fn


# ── Val prediction plots ───────────────────────────────────────────────────────

META_COLS_PLOT = {"timestamp", "attack", "label", "attack_p1", "attack_p2", "attack_p3"}
TRAIN_FRAC_PLOT = 0.8

def plot_val_predictions(model: "TransformerSeq2Seq", fitted_norm, device):
    """
    Generate two evaluation plots using only the val 20% of each train file.
    No test data used.

    Outputs:
        outputs/val_predictions_context.png    — full file + predicted val overlay
        outputs/val_predictions_comparison.png — IBM-style actual (red) vs predicted (blue)
    """
    plt.style.use("fivethirtyeight")
    const_hai, const_hiend, hiend_dups = identify_common_constants()

    def _load_file(num):
        df   = load_merged("train", num, drop_constants=True, keep_hai_duplicates=True,
                           const_cols_hai=const_hai, const_cols_hiend=const_hiend,
                           hiend_dup_cols=hiend_dups)
        cols = [c for c in df.columns if c not in META_COLS_PLOT and df[c].dtype != object]
        arr  = fitted_norm.transform(df)[cols].values.astype(np.float32)
        return arr, cols

    enc_len = X_train.shape[1]   # 300 — reuse the module-level loaded data shapes
    dec_len = Y_train.shape[1]   # 180

    @torch.no_grad()
    def _predict_val(arr):
        T         = len(arr)
        val_start = int(T * TRAIN_FRAC_PLOT)
        preds, pos = [], val_start
        while pos + dec_len <= T:
            enc_start = max(0, pos - enc_len)
            enc = arr[enc_start:pos]
            if len(enc) < enc_len:
                pad = np.zeros((enc_len - len(enc), arr.shape[1]), dtype=np.float32)
                enc = np.concatenate([pad, enc], axis=0)
            src  = torch.tensor(enc[np.newaxis]).float().to(device)
            pred = model.predict(src, dec_len=dec_len)
            preds.append(pred[0].cpu().numpy())
            pos += dec_len
        return (np.concatenate(preds, axis=0) if preds else np.array([])), val_start

    # Collect per-file results
    results = {}
    for num in range(1, 5):
        print(f"  Plotting train{num} val ...")
        arr, cols              = _load_file(num)
        predictions, val_start = _predict_val(arr)
        sidx                   = int(arr.var(axis=0).argmax())
        results[num]           = (arr, cols, predictions, val_start, sidx)

    # ── Figure 1: full-file context ───────────────────────────────────────────
    fig1, axes1 = plt.subplots(4, 1, figsize=(16, 20))
    fig1.suptitle("Transformer — val predictions in context\n"
                  "(blue = actual full file · orange = predicted last 20%)", fontsize=13)
    for i, num in enumerate(range(1, 5)):
        arr, cols, preds, val_start, sidx = results[num]
        ax = axes1[i]
        ax.plot(arr[:, sidx], color="#1f77b4", lw=1, label="Actual (full file)")
        if len(preds):
            px   = np.arange(val_start, val_start + len(preds))
            rmse = np.sqrt(((preds[:, sidx] - arr[val_start:val_start+len(preds), sidx])**2).mean())
            ax.plot(px, preds[:, sidx], color="#ff7f0e", lw=1.2,
                    label=f"Predicted val  RMSE={rmse:.4f}")
        ax.axvline(val_start, color="grey", linestyle="--", lw=1)
        ax.text(val_start + 20, ax.get_ylim()[1] * 0.92,
                "← train 80%  |  val 20% →", fontsize=8, color="grey")
        ax.set_title(f"train{num}  |  sensor: {cols[sidx][:35]}", fontsize=10)
        ax.set_xlabel("timestep"); ax.set_ylabel("normalised value")
        ax.legend(fontsize=9)
    fig1.tight_layout()
    fig1.savefig("outputs/val_predictions_context.png", dpi=150, bbox_inches="tight")
    print("  Saved: outputs/val_predictions_context.png")

    # ── Figure 2: IBM stock-price style — val only ────────────────────────────
    fig2, axes2 = plt.subplots(4, 1, figsize=(14, 18))
    fig2.suptitle("Transformer — Real vs Predicted (val 20% only)", fontsize=13)
    for i, num in enumerate(range(1, 5)):
        arr, cols, preds, val_start, sidx = results[num]
        ax = axes2[i]
        if len(preds) == 0:
            ax.set_title(f"train{num} — no predictions"); continue
        n      = len(preds)
        actual = arr[val_start:val_start + n, sidx]
        rmse   = np.sqrt(((preds[:, sidx] - actual)**2).mean())
        ax.plot(actual,         color="red",  lw=1.2, label=f"Real sensor value (train{num} val)")
        ax.plot(preds[:, sidx], color="blue", lw=1.2, label=f"Predicted (train{num} val)")
        ax.set_title(f"train{num}  |  sensor: {cols[sidx][:35]}  |  RMSE = {rmse:.4f}", fontsize=10)
        ax.set_xlabel("Time (val timesteps)"); ax.set_ylabel("Normalised sensor value")
        ax.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig("outputs/val_predictions_comparison.png", dpi=150, bbox_inches="tight")
    print("  Saved: outputs/val_predictions_comparison.png")
    plt.close("all")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 1. Build model ────────────────────────────────────────────────────────
    model = TransformerSeq2Seq(
        n_features=N_FEAT, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 2. Train ──────────────────────────────────────────────────────────────
    model = train(model, X_train, Y_train, X_val, Y_val)

    # ── 3. Evaluate as simulator (RMSE) — eTaPR belongs to ISO Forest ─────────
    results = evaluate_twin(
        predict_fn=make_predict_fn(model),
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test, y_test=y_test,
        label="Transformer Seq2Seq",
        save_path="outputs/transformer_metrics.json",
    )

    # ── 4. Compute per-sensor errors → for ISO Forest ─────────────────────────
    print("\nComputing per-sensor reconstruction errors for ISO Forest...")
    train_errors = model.reconstruction_errors(X_train, Y_train)   # (N, 277)
    test_errors  = model.reconstruction_errors(X_test,  Y_test)    # (K, 277)
    print(f"  train_errors {train_errors.shape}")
    print(f"  test_errors  {test_errors.shape}")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    torch.save({
        "model_state":  model.state_dict(),
        "train_errors": train_errors,
        "test_errors":  test_errors,
        "y_test":       y_test,
        "metrics":      results,
    }, "outputs/transformer_twin.pt")

    print("\nSaved:")
    print("  outputs/transformer_twin.pt      (model + errors)")
    print("  outputs/transformer_metrics.json (RMSE metrics)")

    # ── 6. Plot val predictions ────────────────────────────────────────────────
    print("\nGenerating val prediction plots ...")
    plot_val_predictions(model, norm, device)
