"""
Transformer Seq2Seq — Digital Twin (primary)

Data:   train1+2+3 (benign) → windowed per file, no gap crossing
        Val:  train4 (benign)
        Test: test1+2 (all rows)
Shape:  X (N, 60, 277) → Y (N, 180, 277)
"""

import sys
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # add project root to path
from utils.prep import twin

# ── Data ──────────────────────────────────────────────────────────────────────
data    = twin(input_len=60, target_len=180, stride=180)  # stride=1 → OOM; 180=non-overlapping
X_train = data["X_train"]        # (N, 60,  277)  encoder input
Y_train = data["Y_train"]        # (N, 180, 277)  decoder target
X_val   = data["X_val"]          # (M, 60,  277)
Y_val   = data["Y_val"]          # (M, 180, 277)
X_test  = data["X_test"]         # (K, 60,  277)
Y_test  = data["Y_test"]         # (K, 180, 277)
y_test  = data["y_test_labels"]  # (K,)  0=normal 1=attack
norm    = data["norm"]           # pass to ISO Forest later

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_FEAT   = X_train.shape[2]   # 277
D_MODEL  = 128
N_HEADS  = 8
N_LAYERS = 3
FFN_DIM  = 512
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

    Encoder: processes 60 past timesteps → context (memory)
    Decoder: predicts 180 future timesteps from context
    Loss:    MSE on sensor values
    """
    def __init__(self, n_features=277, d_model=128, n_heads=8,
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
        src:    (B, 60,  277)  encoder input
        tgt_in: (B, 180, 277)  teacher-forced decoder input
        returns (B, 180, 277)
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
        src: (B, 60, 277) → returns (B, dec_len, 277)
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
        self.eval()
        errors = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                src  = torch.tensor(X[i:i+batch_size]).float()
                tgt  = torch.tensor(Y[i:i+batch_size]).float()
                pred = self.predict(src, dec_len=tgt.size(1))
                err  = ((pred - tgt) ** 2).mean(dim=1)  # (B, 277)
                errors.append(err.cpu().numpy())
        return np.concatenate(errors, axis=0)           # (N, 277)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, X_tr, Y_tr, X_v, Y_v,
          epochs=EPOCHS, batch=BATCH, lr=LR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True)
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
            src    = torch.tensor(X_tr[b]).float()
            tgt    = torch.tensor(Y_tr[b]).float()
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
                src    = torch.tensor(X_v[i:i+batch]).float()
                tgt    = torch.tensor(Y_v[i:i+batch]).float()
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

def score_windows(model, X, Y, batch=BATCH):
    """
    Scalar MSE per window using teacher forcing (fast).
    Returns (N,) anomaly scores — higher = more anomalous.
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            src    = torch.tensor(X[i:i+batch]).float()
            tgt    = torch.tensor(Y[i:i+batch]).float()
            dec_in = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
            pred   = model(src, dec_in)
            mse    = ((pred - tgt) ** 2).mean(dim=(1, 2))   # (B,)
            scores.append(mse.cpu().numpy())
    return np.concatenate(scores)   # (N,)


def evaluate(model, X_val, Y_val, X_test, Y_test, y_test):
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score, confusion_matrix)

    print("\nScoring windows...")
    val_scores  = score_windows(model, X_val,  Y_val)    # benign only → sets normal baseline
    test_scores = score_windows(model, X_test, Y_test)   # mix of normal + attack

    print(f"  val  scores: mean={val_scores.mean():.5f}  p95={np.percentile(val_scores, 95):.5f}")
    print(f"  test scores: mean={test_scores.mean():.5f}  p95={np.percentile(test_scores, 95):.5f}")

    # ── Threshold from val_scores only (no data leakage) ─────────────────────
    # val is benign-only → 95th percentile = upper bound of normal reconstruction error
    best_thr = float(np.percentile(val_scores, 95))

    # ── Final metrics ─────────────────────────────────────────────────────────
    pred = (test_scores >= best_thr).astype(int)
    f1   = f1_score(y_test,  pred, zero_division=0)
    pre  = precision_score(y_test, pred, zero_division=0)
    rec  = recall_score(y_test,  pred, zero_division=0)
    try:   auc = roc_auc_score(y_test, test_scores)
    except: auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    print("\n" + "=" * 50)
    print("  TRANSFORMER SEQ2SEQ — EVALUATION RESULTS")
    print("=" * 50)
    print(f"  F1        = {f1:.4f}")
    print(f"  Precision = {pre:.4f}")
    print(f"  Recall    = {rec:.4f}")
    print(f"  ROC-AUC   = {auc:.4f}")
    print(f"  Threshold = {best_thr:.6f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print("=" * 50)

    return {
        "f1": float(f1), "precision": float(pre), "recall": float(rec),
        "roc_auc": float(auc), "threshold": best_thr,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "val_scores": val_scores, "test_scores": test_scores,
    }


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    # ── 1. Build model ────────────────────────────────────────────────────────
    model = TransformerSeq2Seq(
        n_features=N_FEAT, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 2. Train ──────────────────────────────────────────────────────────────
    model = train(model, X_train, Y_train, X_val, Y_val)

    # ── 3. Evaluate (anomaly detection via reconstruction error) ──────────────
    results = evaluate(model, X_val, Y_val, X_test, Y_test, y_test)

    # ── 4. Compute per-sensor errors → for ISO Forest ─────────────────────────
    print("\nComputing per-sensor reconstruction errors for ISO Forest...")
    train_errors = model.reconstruction_errors(X_train, Y_train)   # (N, 277)
    test_errors  = model.reconstruction_errors(X_test,  Y_test)    # (K, 277)
    print(f"  train_errors {train_errors.shape}")
    print(f"  test_errors  {test_errors.shape}")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    torch.save({
        "model_state":   model.state_dict(),
        "train_errors":  train_errors,
        "test_errors":   test_errors,
        "y_test":        y_test,
        "metrics":       results,
    }, "outputs/transformer_twin.pt")

    with open("outputs/transformer_metrics.json", "w") as f:
        json.dump({k: v for k, v in results.items()
                   if not isinstance(v, np.ndarray)}, f, indent=2)

    print("\nSaved:")
    print("  outputs/transformer_twin.pt      (model + errors)")
    print("  outputs/transformer_metrics.json (evaluation results)")
