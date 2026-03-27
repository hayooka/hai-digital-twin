"""
LSTM Autoencoder — Layer 2 (baseline detector) / Layer 3 (generator alternative)

ROLE A — Anomaly Detector (Layer 2 baseline)
    Reconstructs the SAME window (autoencoder, not seq2seq).
    High reconstruction error = anomaly.
    Evaluated with eTaPR (segment-level) + standard metrics.
    Shape: (N, 300, F) → (N, 300, F)

ROLE B — Attack Generator (Layer 3 alternative)
    Trains on real attack windows from test1.
    Generates synthetic attacks for Guided Generation pipeline.
    Shape: (N, 300, F)

Window size = 300s  (from window_size_analysis notebook).
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.eval import evaluate_detector

Path("outputs").mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROLE A — Anomaly Detector (Layer 2 baseline)
# ─────────────────────────────────────────────────────────────────────────────

from utils.prep import detect

data    = detect(window_size=300, stride=60)
X_train = data["X_train"]        # (N, 300, F)  benign — input == target
X_val   = data["X_val"]          # (M, 300, F)  benign val
X_test  = data["X_test"]         # (K, 300, F)  test1+test2 all rows
y_test  = data["y_test_labels"]  # (K,)  0=normal 1=attack
norm    = data["norm"]

N_FEAT   = X_train.shape[2]   # dynamic
HIDDEN   = 128
N_LAYERS = 2
DROPOUT  = 0.1
EPOCHS   = 50
BATCH    = 64
LR       = 1e-3


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for anomaly detection.

    Encoder compresses the window into a fixed-size hidden state.
    Decoder reconstructs the original window from that state.
    High MSE at test time = anomaly.
    """

    def __init__(self, n_features=N_FEAT, hidden=HIDDEN,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.encoder = nn.LSTM(
            n_features, hidden, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            hidden, hidden, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden, n_features)

    def forward(self, x):
        """x: (B, T, F) → reconstructed (B, T, F)"""
        B, T, _ = x.shape
        _, (h, c) = self.encoder(x)
        # Repeat bottleneck T times as decoder input
        dec_in     = h[-1].unsqueeze(1).expand(B, T, -1)
        out, _     = self.decoder(dec_in, (h, c))
        return self.proj(out)

    @torch.no_grad()
    def anomaly_scores(self, X, batch_size=64):
        """Per-window mean MSE → (N,) anomaly score. Higher = more anomalous."""
        device = next(self.parameters()).device
        self.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                x    = torch.tensor(X[i:i+batch_size]).float().to(device)
                pred = self(x)
                mse  = ((pred - x) ** 2).mean(dim=(1, 2))  # (B,)
                scores.append(mse.cpu().numpy())
        return np.concatenate(scores)


def train_ae(model, X_tr, X_v, epochs=EPOCHS, batch=BATCH, lr=LR):
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
            b    = idx[i:i+batch]
            x    = torch.tensor(X_tr[b]).float().to(device)
            loss = criterion(model(x), x)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        model.eval()
        val_loss, n_b = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(X_v), batch):
                xv = torch.tensor(X_v[i:i+batch]).float().to(device)
                val_loss += criterion(model(xv), xv).item()
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


# ── Run (Role A) ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"N_FEAT = {N_FEAT}  (after constant deletion)")

    model = LSTMAutoencoder(
        n_features=N_FEAT, hidden=HIDDEN,
        n_layers=N_LAYERS, dropout=DROPOUT,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train_ae(model, X_train, X_val)

    # ── Anomaly scores on test ─────────────────────────────────────────────────
    scores = model.anomaly_scores(X_test)   # (K,)

    # ── Evaluate with eTaPR (Layer 2 evaluation) ──────────────────────────────
    results = evaluate_detector(
        y_true=y_test,
        scores=scores,
        label="LSTM Autoencoder",
        theta_p=0.5,
        theta_r=0.1,
        save_path="outputs/lstm_ae_metrics.json",
    )

    torch.save({
        "model_state": model.state_dict(),
        "scores":      scores,
        "y_test":      y_test,
        "metrics":     results,
    }, "outputs/lstm_ae.pt")

    print("\nSaved:")
    print("  outputs/lstm_ae.pt          (model + scores)")
    print("  outputs/lstm_ae_metrics.json (eTaPR metrics)")


# ─────────────────────────────────────────────────────────────────────────────
# ROLE B — Attack Generator alternative (Layer 3)
# Uncomment and run separately if using LSTM-AE as the generator
# ─────────────────────────────────────────────────────────────────────────────

# from utils.prep import generate
#
# gen_data        = generate(norm=None, window_len=300, stride=60)
# attack_windows  = gen_data["attack_windows"]   # (N, 300, F)  test1 attack windows
# normal_windows  = gen_data["normal_windows"]   # (M, 300, F)  train1 normal windows
# gen_norm        = gen_data["norm"]
#
# # Flatten windows to rows for a row-level generative AE
# F = attack_windows.shape[2]
# attack_rows = attack_windows.reshape(-1, F)    # (N*300, F)
# normal_rows = normal_windows.reshape(-1, F)    # (M*300, F)
#
# # ── Build and train LSTM-AE generator here ────────────────────────────────
