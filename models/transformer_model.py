"""
Transformer Seq2Seq — Digital Twin (primary)

Data:   train1+2+3 (benign) + test1 normal rows → windowed per file
        Val:  train4 (benign)
        Test: test2 only (held-out)
Shape:  X (N, 60, 277) → Y (N, 180, 277)

No attack-type labels used — model learns purely from sensor values.
High reconstruction error on test2 = anomaly signal.
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
data    = twin(input_len=60, target_len=180, stride=60)   # stride=1 → OOM; 60=3× more data than stride=180
X_train = data["X_train"]        # (N, 60,  277)  encoder input  — normal only
Y_train = data["Y_train"]        # (N, 180, 277)  decoder target
X_val   = data["X_val"]          # (M, 60,  277)
Y_val   = data["Y_val"]          # (M, 180, 277)
X_test  = data["X_test"]         # (K, 60,  277)  — test2 only
Y_test  = data["Y_test"]         # (K, 180, 277)
y_test  = data["y_test_labels"]  # (K,)  0=normal 1=attack
norm    = data["norm"]           # pass to ISO Forest later

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_FEAT   = X_train.shape[2]   # 277
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

    Encoder: processes 60 past timesteps → context (memory)
    Decoder: predicts 180 future timesteps from context
    Loss:    MSE on sensor values

    No attack-type conditioning — model learns purely from sensor values.
    Anomaly detection via reconstruction error: high error = attack.
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


def evaluate(model, X_val, Y_val, X_test, Y_test, y_test):
    """
    Evaluates the Transformer as a SIMULATOR — using RMSE only.
    F1/ROC-AUC belong to ISO Forest, not here.

    Reports:
        RMSE on val (normal)         → baseline simulator accuracy
        RMSE on test normal windows  → simulator on unseen normal
        RMSE on test attack windows  → simulator on known attacks
    """
    device = next(model.parameters()).device
    model.eval()

    def _rmse(X, Y, mask=None):
        errs = []
        with torch.no_grad():
            for i in range(0, len(X), BATCH):
                src    = torch.tensor(X[i:i+BATCH]).float().to(device)
                tgt    = torch.tensor(Y[i:i+BATCH]).float().to(device)
                dec_in = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
                pred   = model(src, dec_in)
                mse    = ((pred - tgt) ** 2).mean(dim=(1, 2)).cpu().numpy()
                errs.append(mse)
        errs = np.concatenate(errs)
        if mask is not None:
            errs = errs[mask]
        return float(np.sqrt(np.mean(errs)))

    normal_mask = (y_test == 0)
    attack_mask = (y_test == 1)

    rmse_val    = _rmse(X_val,  Y_val)                      # normal baseline
    rmse_normal = _rmse(X_test, Y_test, mask=normal_mask)   # test normal
    rmse_attack = _rmse(X_test, Y_test, mask=attack_mask)   # test known attacks

    print("\n" + "=" * 50)
    print("  TRANSFORMER SEQ2SEQ — SIMULATOR EVALUATION")
    print("=" * 50)
    print(f"  RMSE val    (normal)         = {rmse_val:.5f}")
    print(f"  RMSE test   (normal windows) = {rmse_normal:.5f}")
    print(f"  RMSE test   (attack windows) = {rmse_attack:.5f}")
    print(f"  Attack/Normal ratio          = {rmse_attack/rmse_normal:.2f}x")
    print("  → Generalization Gap will be computed after Diffusion")
    print("=" * 50)

    return {
        "rmse_val":    rmse_val,
        "rmse_normal": rmse_normal,
        "rmse_attack": rmse_attack,
    }


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

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
        "model_state":  model.state_dict(),
        "train_errors": train_errors,
        "test_errors":  test_errors,
        "y_test":       y_test,
        "metrics":      results,
    }, "outputs/transformer_twin.pt")

    with open("outputs/transformer_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:")
    print("  outputs/transformer_twin.pt      (model + errors)")
    print("  outputs/transformer_metrics.json (RMSE metrics)")
