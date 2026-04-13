"""
lstm.py — LSTM Seq2Seq with scenario conditioning (causal loss compatible).

Matches Transformer capabilities:
    - Scenario embedding (4 classes: normal, AP_no, AP_comb, AE_no)
    - Teacher forcing via [last encoder step, target[:, :-1, :]]
    - Autoregressive inference with scenario conditioning
    - Compatible with CausalLoss from causal_loss.py

Architecture:
    Encoder: 4‑layer BiLSTM (256 per direction) → bridge to 256
    Decoder: 4‑layer LSTM (256), input = concat(prev_output, scenario_emb)
    Output:  Linear(256 → n_features)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


class _Encoder(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(
            input_size    = n_features,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )
        self.h_bridge = nn.Linear(hidden_size * 2, hidden_size)
        self.c_bridge = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        _, (h, c) = self.lstm(x)
        h = h.view(self.num_layers, 2, x.size(0), self.hidden_size)
        h = torch.cat([h[:, 0], h[:, 1]], dim=-1)
        h = torch.tanh(self.h_bridge(h))
        c = c.view(self.num_layers, 2, x.size(0), self.hidden_size)
        c = torch.cat([c[:, 0], c[:, 1]], dim=-1)
        c = torch.tanh(self.c_bridge(c))
        return h, c


class _Decoder(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout,
                 scenario_emb_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = n_features + scenario_emb_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout  = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, n_features)

    def step(self, x, scenario_vec, h, c):
        sc_exp = scenario_vec.unsqueeze(1)
        lstm_input = torch.cat([x, sc_exp], dim=-1)
        out, (h, c) = self.lstm(lstm_input, (h, c))
        pred = self.out_proj(self.dropout(out))
        return pred, h, c


class LSTMSeq2Seq(nn.Module):
    """
    LSTM Seq2Seq with scenario conditioning.
    """
    def __init__(
        self,
        n_features:           int,
        n_scenarios:          int = 4,
        hidden_size:          int = 256,
        num_layers:           int = 4,
        dropout:              float = 0.1,
        output_len:           int = 180,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_scenarios = n_scenarios
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_len = output_len

        self.scenario_emb = nn.Embedding(n_scenarios, hidden_size)
        self.encoder = _Encoder(n_features, hidden_size, num_layers, dropout)
        self.decoder = _Decoder(n_features, hidden_size, num_layers, dropout,
                                scenario_emb_dim=hidden_size)

    def forward(self, src, tgt_in, scenario):
        """
        Teacher‑forced forward pass.
        src: (B, 300, F)
        tgt_in: (B, 180, F)  # already includes start token + target[:-1]
        scenario: (B,) int
        """
        h, c = self.encoder(src)
        sc_emb = self.scenario_emb(scenario)
        outputs = []
        for t in range(self.output_len):
            x_t = tgt_in[:, t:t+1, :]
            pred, h, c = self.decoder.step(x_t, sc_emb, h, c)
            outputs.append(pred)
        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def predict(self, src, dec_len=180, scenario=None):
        self.eval()
        if scenario is None:
            scenario = torch.zeros(src.size(0), dtype=torch.long, device=src.device)
        h, c = self.encoder(src)
        sc_emb = self.scenario_emb(scenario)
        dec_input = src[:, -1:, :]
        outputs = []
        for _ in range(dec_len):
            pred, h, c = self.decoder.step(dec_input, sc_emb, h, c)
            outputs.append(pred)
            dec_input = pred
        return torch.cat(outputs, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Helper functions (mirroring transformer.py) ───────────────────────────────

def make_predict_fn(model, batch_size=64):
    device = next(model.parameters()).device
    model.eval()
    def predict_fn(X: np.ndarray, Y: np.ndarray, scenario: np.ndarray | None = None) -> np.ndarray:
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                src = torch.tensor(X[i:i+batch_size]).float().to(device)
                tgt = torch.tensor(Y[i:i+batch_size]).float().to(device)
                if scenario is not None:
                    sc = torch.tensor(scenario[i:i+batch_size]).long().to(device)
                else:
                    sc = torch.zeros(len(src), dtype=torch.long, device=device)
                dec_in = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
                pred = model(src, dec_in, sc)
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds, axis=0)
    return predict_fn



# ── LSTM Controller ──────────────────────────────────────────────────────────

class LSTMController(nn.Module):
    """
    Per-loop LSTM seq2seq controller. Mirrors GRUController but uses LSTM
    (hidden state h + cell state c).

    Encoder : [SP, PV] (or [SP, PV, CV_fb]) over input_len steps → (h, c)
    Decoder : predict CV for target_len steps
              input at each decoder step = previous CV (teacher-forced or predicted)
    """

    def __init__(self, n_inputs: int = 2, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.1, output_len: int = 180):
        super().__init__()
        self.output_len = output_len
        drop = dropout if layers > 1 else 0.0

        self.encoder = nn.LSTM(n_inputs, hidden, layers,
                               batch_first=True, dropout=drop)
        self.decoder = nn.LSTM(1, hidden, layers,
                               batch_first=True, dropout=drop)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor,
                y_cv_teacher: Optional[torch.Tensor] = None,
                ss_ratio: float = 0.0) -> torch.Tensor:
        B = x.size(0)
        _, (h, c) = self.encoder(x)
        prev = torch.zeros(B, 1, 1, device=x.device)
        outputs = []

        for t in range(self.output_len):
            out, (h, c) = self.decoder(prev, (h, c))
            cv_pred = self.out(out)
            outputs.append(cv_pred)

            if y_cv_teacher is not None and ss_ratio > 0.0:
                use_pred = (torch.rand(B, device=x.device) < ss_ratio
                            ).view(B, 1, 1).float()
                prev = use_pred * cv_pred + (1 - use_pred) * y_cv_teacher[:, t:t+1, :]
            elif y_cv_teacher is not None:
                prev = y_cv_teacher[:, t:t+1, :]
            else:
                prev = cv_pred

        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor,
                target_len: Optional[int] = None) -> torch.Tensor:
        self.eval()
        orig = self.output_len
        if target_len is not None:
            self.output_len = target_len
        out = self.forward(x)
        self.output_len = orig
        return out


# ── LSTM CC Classifier-Regressor ──────────────────────────────────────────────

class LSTMCCClassifierRegressor(nn.Module):
    """
    CC loop model using LSTM. Mirrors CCClassifierRegressor but with an LSTM
    trunk instead of a simple linear layer, for consistency with the LSTM model.

    Shared trunk  : [P1_TIT03, P1_PP04SP] last timestep → Linear → ReLU → Dropout
    Classifier    : Linear(hidden→1)  logit for pump on/off
    Regressor     : Linear(hidden→1)  pump speed
    """

    def __init__(self, n_inputs: int = 2, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, 1)
        self.regressor  = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x[:, -1, :]
        h = self.trunk(x)
        return self.classifier(h), self.regressor(h)

    @torch.no_grad()
    def predict(self, x: torch.Tensor,
                threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        logit, speed = self.forward(x)
        pump_on = torch.sigmoid(logit) > threshold
        pump_cv = pump_on.float() * speed
        return pump_on.squeeze(-1), pump_cv


# ── LSTM Plant (MIMO, autoregressive) ────────────────────────────────────────

class LSTMPlant(nn.Module):
    """
    MIMO LSTM plant model — identical interface to GRUPlant.

    Encoder : x_cv input window + scenario embedding → (h, c) hidden/cell states
    Decoder : step-by-step concat(x_cv_target[:, t, :], pv_prev) → LSTM → FC → pv_next

    LSTM retains both hidden state (h) and cell state (c), giving it stronger
    long-range memory than GRU at the cost of ~33% more parameters.
    Scheduled sampling mixes teacher forcing with model predictions.
    """

    def __init__(self, n_plant_in: int, n_pv: int,
                 hidden: int = 256, layers: int = 2,
                 n_scenarios: int = 4, emb_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.n_plant_in  = n_plant_in
        self.n_pv        = n_pv
        self.n_scenarios = n_scenarios
        drop = dropout if layers > 1 else 0.0

        self.scenario_emb = nn.Embedding(n_scenarios, emb_dim)

        self.encoder = nn.LSTM(n_plant_in + emb_dim, hidden, layers,
                               batch_first=True, dropout=drop)
        self.decoder = nn.LSTM(n_plant_in + n_pv, hidden, layers,
                               batch_first=True, dropout=drop)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_pv),
        )

    def forward(self, x_cv: torch.Tensor, x_cv_target: torch.Tensor,
                pv_init: torch.Tensor, scenario: torch.Tensor,
                pv_teacher: Optional[torch.Tensor] = None,
                ss_ratio: float = 0.0) -> torch.Tensor:
        """
        x_cv:        (B, input_len,  n_plant_in)
        x_cv_target: (B, target_len, n_plant_in)
        pv_init:     (B, n_pv)
        scenario:    (B,) int  0–3
        pv_teacher:  (B, target_len, n_pv)  optional
        ss_ratio:    scheduled sampling ratio
        → (B, target_len, n_pv)
        """
        B, T_target, _ = x_cv_target.shape

        emb = self.scenario_emb(scenario).unsqueeze(1).expand(-1, x_cv.size(1), -1)
        _, (h, c) = self.encoder(torch.cat([x_cv, emb], dim=-1))

        pv = pv_init
        outputs = []

        for t in range(T_target):
            dec_in = torch.cat([x_cv_target[:, t, :], pv], dim=-1).unsqueeze(1)
            out, (h, c) = self.decoder(dec_in, (h, c))
            pv_pred = self.fc(out.squeeze(1))
            outputs.append(pv_pred)

            if pv_teacher is not None and ss_ratio > 0.0:
                use_pred = (torch.rand(B, device=x_cv.device) < ss_ratio
                            ).float().unsqueeze(-1)
                pv = use_pred * pv_pred + (1 - use_pred) * pv_teacher[:, t, :]
            elif pv_teacher is not None:
                pv = pv_teacher[:, t, :]
            else:
                pv = pv_pred

        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def predict(self, x_cv: torch.Tensor, x_cv_target: torch.Tensor,
                pv_init: torch.Tensor, scenario: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x_cv, x_cv_target, pv_init, scenario)


def plot_val_predictions(model, X_val, Y_val, sensor_cols, device,
                         dec_len=180, model_name="LSTM Seq2Seq",
                         out_path="outputs/plots/val_predictions_comparison.png"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("fivethirtyeight")
    sidx = int(Y_val.var(axis=(0, 1)).argmax())
    sensor_name = sensor_cols[sidx] if sidx < len(sensor_cols) else str(sidx)
    n = min(200, len(X_val))
    src = torch.tensor(X_val[:n]).float().to(device)
    pred = model.predict(src, dec_len=dec_len, scenario=torch.zeros(n, dtype=torch.long, device=device)).cpu().numpy()
    actual = Y_val[:n]
    pred_flat = pred[:, :, sidx].flatten()
    actual_flat = actual[:, :, sidx].flatten()
    rmse = np.sqrt(((pred_flat - actual_flat) ** 2).mean())
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual_flat, color="red", lw=1, label="Actual")
    ax.plot(pred_flat, color="blue", lw=1, label=f"Predicted  RMSE={rmse:.4f}")
    ax.set_title(f"{model_name} — Val predictions | sensor: {sensor_name[:40]}")
    ax.set_xlabel("timestep")
    ax.set_ylabel("normalised value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out_path}")