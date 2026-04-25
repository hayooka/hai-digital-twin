"""
Transformer Seq2Seq — Layer 1: Physical Model / Digital Twin (primary)
Shape: X (N, 300, F) → Y (N, 180, F)   where F = actual sensor count after
       constant column deletion (determined at runtime, ~112 features).

Window size = 300s  (from window_size_analysis notebook).
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Legacy hyperparameters (used only by TransformerSeq2Seq / train.py) ───────
# TransformerPlant, TransformerController, TransformerCCClassifierRegressor
# receive their hyperparameters explicitly from train_transformer.py.
_D_MODEL  = 256
_N_HEADS  = 8
_N_LAYERS = 4
_FFN_DIM  = 1024
_DROPOUT  = 0.1
_EPOCHS   = 50
_BATCH    = 64
_LR       = 1e-4

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
        self.pe: torch.Tensor
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── Model ─────────────────────────────────────────────────────────────────────

class TransformerSeq2Seq(nn.Module):
    """
    Encoder-Decoder Transformer for sensor forecasting.

    Encoder: processes 300s of past sensor readings → context (memory)
    Decoder: predicts 180s of future readings from context
    Loss:    MSE + λ * causal_loss

    Scenario conditioning: a learned embedding for each scenario class
    (0=normal, 1=AP_no_combination, 2=AP_with_combination, 3=AE_no_combination)

    so the model can simulate different attack trajectories.
    """
    def __init__(self, n_features, d_model=128, n_heads=8,
                 n_layers=3, ffn_dim=512, dropout=0.1, n_scenarios=4):
        super().__init__()
        self.input_proj  = nn.Linear(n_features, d_model)
        self.output_proj = nn.Linear(d_model, n_features)
        self.pos_enc     = PositionalEncoding(d_model, max_len=512, dropout=dropout)
        # Type-annotated so Pyright resolves __call__ correctly (fixes __getitem__ error)
        self.scenario_emb: nn.Embedding = nn.Embedding(n_scenarios, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn_dim, dropout,
            batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, ffn_dim, dropout,
            batch_first=True, norm_first=True)

        self.encoder = nn.TransformerEncoder(enc_layer, n_layers,
                                             enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec_layer, n_layers)

    def encode(self, src, scenario: torch.Tensor | None = None):
        """
        src:      (B, enc_len, n_features)
        scenario: (B,) int tensor with class 0-3, or None → treated as class 0
        → memory: (B, enc_len, d_model)
        """
        h = self.input_proj(src)                          # (B, T, d_model)
        if scenario is not None:
            h = h + self.scenario_emb(scenario).unsqueeze(1)  # broadcast over time
        return self.encoder(self.pos_enc(h))

    def forward(self, src, tgt_in, scenario: torch.Tensor | None = None):
        """
        src:      (B, 300, F)  encoder input
        tgt_in:   (B, 180, F)  teacher-forced decoder input
        scenario: (B,) int tensor, values in {0,1,2,3}  (optional)
        returns   (B, 180, F)
        """
        memory = self.encode(src, scenario)
        tgt_h  = self.pos_enc(self.input_proj(tgt_in))
        T      = tgt_in.size(1)
        mask   = nn.Transformer.generate_square_subsequent_mask(
            T, device=src.device)
        out = self.decoder(tgt_h, memory, tgt_mask=mask, tgt_is_causal=True)
        return self.output_proj(out)                    # (B, 180, 277)

    @torch.no_grad()
    def predict(self, src, dec_len=180, scenario: torch.Tensor | None = None):
        """
        Autoregressive inference — no teacher forcing.
        src:      (B, 300, F)
        scenario: (B,) optional scenario class
        → returns (B, dec_len, F)
        """
        self.eval()
        memory = self.encode(src, scenario)
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
          epochs=_EPOCHS, batch=_BATCH, lr=_LR):
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

def make_predict_fn(model, batch_size=_BATCH):
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

def plot_val_predictions(model: "TransformerSeq2Seq", X_val: np.ndarray, Y_val: np.ndarray,
                         sensor_cols: list, device,
                         dec_len: int = 180, model_name: str = "Transformer Seq2Seq",
                         out_path: str = "outputs/plots/val_predictions_comparison.png"):
    """
    Plot actual vs predicted for a sample of val windows.
    X_val: (N, 300, F), Y_val: (N, 180, F)
    """
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)
    plt.style.use("fivethirtyeight")

    # Pick the sensor with highest variance in val
    sidx = int(Y_val.var(axis=(0, 1)).argmax())
    sensor_name = sensor_cols[sidx] if sidx < len(sensor_cols) else str(sidx)

    # Predict on first 200 val windows
    n = min(200, len(X_val))
    src  = torch.tensor(X_val[:n]).float().to(device)
    pred = model.predict(src, dec_len=dec_len).cpu().numpy()   # (n, 180, F)
    actual = Y_val[:n]                                          # (n, 180, F)

    # Flatten across windows for a time-series view
    pred_flat   = pred[:, :, sidx].flatten()
    actual_flat = actual[:, :, sidx].flatten()
    rmse = np.sqrt(((pred_flat - actual_flat) ** 2).mean())

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual_flat, color="red",  lw=1,   label="Actual")
    ax.plot(pred_flat,   color="blue", lw=1,   label=f"Predicted  RMSE={rmse:.4f}")
    ax.set_title(f"{model_name} — Val predictions | sensor: {sensor_name[:40]}", fontsize=11)
    ax.set_xlabel("timestep"); ax.set_ylabel("normalised value")
    ax.legend(fontsize=9)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out_path}")


# ── Transformer Plant (MIMO, autoregressive) ──────────────────────────────────

class TransformerPlant(nn.Module):
    """
    MIMO Transformer plant model — identical interface to GRUPlant / LSTMPlant.

    Encoder : attends over [x_cv + scenario_emb] input window → memory
    Decoder : at each target step cross-attends to memory + causal self-attention
              over accumulated [x_cv_target, pv_prev] tokens

    Training  (ss_ratio=0, pv_teacher provided) : fully parallel — O(T) not O(T²)
    Inference (no pv_teacher)                   : autoregressive step-by-step
    Scheduled sampling (ss_ratio > 0)           : step-by-step with mixed input

    """

    def __init__(self, n_plant_in: int, n_pv: int,
                 d_model: int = 128, n_heads: int = 8, n_layers: int = 3,
                 n_scenarios: int = 4, emb_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.n_plant_in  = n_plant_in
        self.n_pv        = n_pv
        self.d_model     = d_model
        self.n_scenarios = n_scenarios

        self.scenario_emb = nn.Embedding(n_scenarios, emb_dim)
        # Encoder projection: x_cv + scenario embedding → d_model
        self.enc_proj = nn.Linear(n_plant_in + emb_dim, d_model)
        # Decoder projection: x_cv_target + pv_prev → d_model
        self.dec_proj = nn.Linear(n_plant_in + n_pv, d_model)
        self.pos_enc  = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout,
            batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_model * 4, dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers,
                                             enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec_layer, n_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_pv),
        )

    def _encode(self, x_cv: torch.Tensor, scenario: torch.Tensor) -> torch.Tensor:
        emb = self.scenario_emb(scenario).unsqueeze(1).expand(-1, x_cv.size(1), -1)
        h   = self.pos_enc(self.enc_proj(torch.cat([x_cv, emb], dim=-1)))
        return self.encoder(h)                              # (B, input_len, d_model)

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
        memory = self._encode(x_cv, scenario)               # (B, enc_len, d_model)

        # ── Teacher-forcing: fully parallel ──────────────────────────────────
        if ss_ratio == 0.0 and pv_teacher is not None:
            pv_in  = torch.cat([pv_init.unsqueeze(1), pv_teacher[:, :-1, :]], dim=1)
            dec_in = self.pos_enc(self.dec_proj(
                torch.cat([x_cv_target, pv_in], dim=-1)))   # (B, T, d_model)
            mask   = nn.Transformer.generate_square_subsequent_mask(
                T_target, device=x_cv.device)
            out = self.decoder(dec_in, memory, tgt_mask=mask, tgt_is_causal=True)
            return self.fc(out)                             # (B, T, n_pv)

        # ── Autoregressive / scheduled sampling ──────────────────────────────
        pv = pv_init
        dec_tokens: list[torch.Tensor] = []
        outputs:    list[torch.Tensor] = []

        for t in range(T_target):
            token = self.dec_proj(
                torch.cat([x_cv_target[:, t, :], pv], dim=-1).unsqueeze(1))
            dec_tokens.append(token)
            dec_seq = self.pos_enc(torch.cat(dec_tokens, dim=1))
            T_cur   = dec_seq.size(1)
            mask    = nn.Transformer.generate_square_subsequent_mask(
                T_cur, device=x_cv.device)
            out     = self.decoder(dec_seq, memory, tgt_mask=mask, tgt_is_causal=True)
            pv_pred = self.fc(out[:, -1, :])               # (B, n_pv)
            outputs.append(pv_pred)

            if pv_teacher is not None and ss_ratio > 0.0:
                use_pred = (torch.rand(B, device=x_cv.device) < ss_ratio
                            ).float().unsqueeze(-1)
                pv = use_pred * pv_pred + (1 - use_pred) * pv_teacher[:, t, :]
            elif pv_teacher is not None:
                pv = pv_teacher[:, t, :]
            else:
                pv = pv_pred

        return torch.stack(outputs, dim=1)                  # (B, target_len, n_pv)

    @torch.no_grad()
    def predict(self, x_cv: torch.Tensor, x_cv_target: torch.Tensor,
                pv_init: torch.Tensor, scenario: torch.Tensor) -> torch.Tensor:
        """Full autoregressive rollout (no teacher forcing)."""
        self.eval()
        return self.forward(x_cv, x_cv_target, pv_init, scenario)


# ── Transformer Controller ────────────────────────────────────────────────────

class TransformerController(nn.Module):
    """
    Per-loop Transformer seq2seq controller. Mirrors GRUController/LSTMController
    but uses a Transformer encoder-decoder.

    Encoder : [SP, PV] (or [SP, PV, CV_fb]) over input_len steps → memory
    Decoder : predict CV for target_len steps autoregressively
    """

    def __init__(self, n_inputs: int = 2, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, ffn_dim: int = 128, dropout: float = 0.1,
                 output_len: int = 180):
        super().__init__()
        self.output_len = output_len
        self.d_model = d_model

        self.src_proj  = nn.Linear(n_inputs, d_model)
        self.tgt_proj  = nn.Linear(1, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        encoder_layer  = nn.TransformerEncoderLayer(d_model, n_heads, ffn_dim,
                                                    dropout, batch_first=True)
        decoder_layer  = nn.TransformerDecoderLayer(d_model, n_heads, ffn_dim,
                                                    dropout, batch_first=True)
        self.encoder   = nn.TransformerEncoder(encoder_layer, n_layers)
        self.decoder   = nn.TransformerDecoder(decoder_layer, n_layers)
        self.out       = nn.Linear(d_model, 1)

    def _causal_mask(self, sz: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, x: torch.Tensor,
                y_cv_teacher: Optional[torch.Tensor] = None,
                ss_ratio: float = 0.0) -> torch.Tensor:
        B = x.size(0)
        memory = self.encoder(self.pos_enc(self.src_proj(x)))

        # ── Fast parallel path (teacher forcing, no scheduled sampling) ────────
        # Used during training and open-loop validation — O(1) decoder passes
        if y_cv_teacher is not None and ss_ratio == 0.0:
            start   = torch.zeros(B, 1, 1, device=x.device)
            tgt_in  = torch.cat([start, y_cv_teacher[:, :-1, :]], dim=1)  # (B, T, 1)
            tgt_emb = self.pos_enc(self.tgt_proj(tgt_in))
            T       = tgt_in.size(1)
            mask    = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
            out     = self.decoder(tgt_emb, memory, tgt_mask=mask, tgt_is_causal=True)
            return self.out(out)                                            # (B, T, 1)

        # ── Autoregressive path (inference or scheduled sampling) ─────────────
        prev = torch.zeros(B, 1, 1, device=x.device)
        dec_tokens = [prev]
        outputs = []

        for t in range(self.output_len):
            tgt_seq = torch.cat(dec_tokens, dim=1)           # (B, t+1, 1)
            tgt_emb = self.pos_enc(self.tgt_proj(tgt_seq))
            mask    = self._causal_mask(tgt_seq.size(1), x.device)
            out     = self.decoder(tgt_emb, memory, tgt_mask=mask)
            cv_pred = self.out(out[:, -1:, :])               # (B, 1, 1)
            outputs.append(cv_pred)

            if y_cv_teacher is not None and ss_ratio > 0.0:
                use_pred = (torch.rand(B, device=x.device) < ss_ratio
                            ).view(B, 1, 1).float()
                next_tok = use_pred * cv_pred + (1 - use_pred) * y_cv_teacher[:, t:t+1, :]
            else:
                next_tok = cv_pred
            dec_tokens.append(next_tok)

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


# ── Transformer CC Classifier-Regressor ──────────────────────────────────────

class TransformerCCClassifierRegressor(nn.Module):
    """
    CC loop model — full Transformer encoder over [SP, PV] history.

    Replaces the previous single-timestep MLP with a proper sequence encoder
    so the pump on/off decision and speed regression use the full 300-step
    context, consistent with the rest of the Transformer model stack.

    Architecture:
        Input projection : (B, T, n_inputs) → (B, T, d_model)
        Positional enc   : adds position info
        TransformerEncoder (n_layers)
        Pooling          : last token → (B, d_model)
        Classifier head  : Linear → 1 logit  (pump on/off, BCE loss)
        Regressor head   : Linear → 1 value  (pump speed, MSE loss on pump-on samples)
    """

    def __init__(self, n_inputs: int = 2, d_model: int = 32, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj    = nn.Linear(n_inputs, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout,
            batch_first=True, norm_first=True)
        self.encoder    = nn.TransformerEncoder(enc_layer, n_layers,
                                                enable_nested_tensor=False)
        self.classifier = nn.Linear(d_model, 1)
        self.regressor  = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:                      # (B, n_inputs) → add time dim
            x = x.unsqueeze(1)
        h = self.pos_enc(self.proj(x))        # (B, T, d_model)
        h = self.encoder(h)                   # (B, T, d_model)
        h = h[:, -1, :]                       # last token  (B, d_model)
        return self.classifier(h), self.regressor(h)

    @torch.no_grad()
    def predict(self, x: torch.Tensor,
                threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        logit, speed = self.forward(x)
        pump_on = torch.sigmoid(logit) > threshold
        pump_cv = pump_on.float() * speed
        return pump_on.squeeze(-1), pump_cv
