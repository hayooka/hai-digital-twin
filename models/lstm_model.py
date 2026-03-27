"""
LSTM Seq2Seq — Layer 1: Physical Model / Digital Twin (baseline)

Same role as transformer_model.py but using an LSTM encoder-decoder.
Trained on benign data only (80% of each train1-4, temporal split).

Shape: X (N, 300, F) → Y (N, 180, F)   F = actual sensor count at runtime.
Window size = 300s  (from window_size_analysis notebook).
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.prep import twin
from utils.eval  import evaluate_twin

# ── Data ──────────────────────────────────────────────────────────────────────
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
N_FEAT    = X_train.shape[2]   # dynamic: actual sensor count
HIDDEN    = 256
N_LAYERS  = 2
DROPOUT   = 0.1
EPOCHS    = 50
BATCH     = 64
LR        = 1e-3

Path("outputs").mkdir(exist_ok=True)


# ── Model ─────────────────────────────────────────────────────────────────────

class LSTMSeq2Seq(nn.Module):
    """
    LSTM Encoder-Decoder for sensor forecasting.

    Encoder: LSTM processes 300s of past readings → hidden state
    Decoder: LSTM generates 180s of future readings step-by-step
    Loss:    MSE on sensor values

    Teacher forcing during training; autoregressive at inference.
    """

    def __init__(self, n_features=N_FEAT, hidden=HIDDEN,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.hidden   = hidden
        self.n_layers = n_layers

        self.encoder = nn.LSTM(
            n_features, hidden, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            n_features, hidden, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden, n_features)

    def forward(self, src, tgt_in):
        """
        src:    (B, 300, F)  encoder input
        tgt_in: (B, 180, F)  teacher-forced decoder input
        returns (B, 180, F)
        """
        _, (h, c) = self.encoder(src)
        out, _    = self.decoder(tgt_in, (h, c))
        return self.proj(out)

    @torch.no_grad()
    def predict(self, src, dec_len=180):
        """Autoregressive inference. src: (B, 300, F) → (B, dec_len, F)"""
        self.eval()
        _, (h, c) = self.encoder(src)
        step  = src[:, -1:, :]   # start token = last encoder output
        outs  = []
        for _ in range(dec_len):
            out, (h, c) = self.decoder(step, (h, c))
            step = self.proj(out)
            outs.append(step)
        return torch.cat(outs, dim=1)

    def reconstruction_errors(self, X, Y, batch_size=64):
        """Per-sensor MSE per window → (N, F). Pass to ISO Forest."""
        device = next(self.parameters()).device
        self.eval()
        errors = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                src  = torch.tensor(X[i:i+batch_size]).float().to(device)
                tgt  = torch.tensor(Y[i:i+batch_size]).float().to(device)
                pred = self.predict(src, dec_len=tgt.size(1))
                err  = ((pred - tgt) ** 2).mean(dim=1)  # (B, F)
                errors.append(err.cpu().numpy())
        return np.concatenate(errors, axis=0)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, X_tr, Y_tr, X_v, Y_v,
          epochs=EPOCHS, batch=BATCH, lr=LR):
    device    = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    N         = len(X_tr)
    best_val, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        model.train()
        idx   = np.random.permutation(N)
        total = 0.0
        for i in range(0, N, batch):
            b      = idx[i:i+batch]
            src    = torch.tensor(X_tr[b]).float().to(device)
            tgt    = torch.tensor(Y_tr[b]).float().to(device)
            dec_in = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
            loss   = criterion(model(src, dec_in), tgt)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

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


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"N_FEAT = {N_FEAT}  (after constant deletion)")

    model = LSTMSeq2Seq(
        n_features=N_FEAT, hidden=HIDDEN,
        n_layers=N_LAYERS, dropout=DROPOUT,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train(model, X_train, Y_train, X_val, Y_val)

    # ── Evaluate as simulator (RMSE) ──────────────────────────────────────────
    def _predict_fn(X, Y):
        preds = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X), BATCH):
                src    = torch.tensor(X[i:i+BATCH]).float().to(device)
                tgt    = torch.tensor(Y[i:i+BATCH]).float().to(device)
                dec_in = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
                preds.append(model(src, dec_in).cpu().numpy())
        return np.concatenate(preds, axis=0)

    results = evaluate_twin(
        predict_fn=_predict_fn,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test, y_test=y_test,
        label="LSTM Seq2Seq",
        save_path="outputs/lstm_metrics.json",
    )

    # ── Per-sensor errors → for ISO Forest ────────────────────────────────────
    train_errors = model.reconstruction_errors(X_train, Y_train)
    test_errors  = model.reconstruction_errors(X_test,  Y_test)

    torch.save({
        "model_state":  model.state_dict(),
        "train_errors": train_errors,
        "test_errors":  test_errors,
        "y_test":       y_test,
        "metrics":      results,
    }, "outputs/lstm_twin.pt")

    print("\nSaved:")
    print("  outputs/lstm_twin.pt      (model + errors)")
    print("  outputs/lstm_metrics.json (RMSE metrics)")
