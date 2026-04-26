"""
gru.py — GRU-based models for the HAI Digital Twin closed-loop system.

Closed-loop structure:
    [SP] → GRUController (×4) → [CV] → GRUPlant → [PV] → (feedback to controllers)
                                         ↑
                         CCClassifierRegressor (CC loop, pump on/off + speed)

Components:
    GRUController        one per loop: PC, LC, FC, TC
    CCClassifierRegressor single model for the CC cooling loop
    GRUPlant             MIMO plant, autoregressive, scheduled sampling
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


# ── GRU Controller ────────────────────────────────────────────────────────────

class GRUController(nn.Module):
    """
    Per-loop GRU seq2seq controller.

    Encoder : [SP, PV] (or [SP, PV, CV_fb]) over input_len steps → hidden state h
    Decoder : predict CV for target_len steps
              input at each decoder step = previous CV (teacher-forced or predicted)

    One instance per loop: PC, LC, FC, TC.
    n_inputs = 2  → [SP, PV]
    n_inputs = 3  → [SP, PV, CV_fb]  (loops that expose CV feedback)
    """

    def __init__(self, n_inputs: int = 2, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.1, output_len: int = 180):
        super().__init__()
        self.output_len = output_len
        drop = dropout if layers > 1 else 0.0

        self.encoder = nn.GRU(n_inputs, hidden, layers,
                              batch_first=True, dropout=drop)
        # Decoder takes previous CV as the only input token
        self.decoder = nn.GRU(1, hidden, layers,
                              batch_first=True, dropout=drop)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor,
                y_cv_teacher: Optional[torch.Tensor] = None,
                ss_ratio: float = 0.0) -> torch.Tensor:
        """
        x:            (B, input_len, n_inputs)  [SP, PV] history
        y_cv_teacher: (B, target_len, 1)        teacher CV (None → autoregressive)
        ss_ratio:     scheduled-sampling ratio  [0 = teacher-forced, 1 = autoregressive]
        → (B, target_len, 1)
        """
        B = x.size(0)
        _, h = self.encoder(x)                              # h: (L, B, hidden)
        prev = torch.zeros(B, 1, 1, device=x.device)       # start token
        outputs = []

        for t in range(self.output_len):
            out, h = self.decoder(prev, h)                  # (B, 1, hidden)
            cv_pred = self.out(out)                         # (B, 1, 1)
            outputs.append(cv_pred)

            if y_cv_teacher is not None and ss_ratio > 0.0:
                use_pred = (torch.rand(B, device=x.device) < ss_ratio
                            ).view(B, 1, 1).float()
                prev = use_pred * cv_pred + (1 - use_pred) * y_cv_teacher[:, t:t+1, :]
            elif y_cv_teacher is not None:
                prev = y_cv_teacher[:, t:t+1, :]           # pure teacher forcing
            else:
                prev = cv_pred                              # pure autoregressive

        return torch.cat(outputs, dim=1)                    # (B, target_len, 1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor,
                target_len: Optional[int] = None) -> torch.Tensor:
        """Autoregressive inference from [SP, PV] history."""
        self.eval()
        orig = self.output_len
        if target_len is not None:
            self.output_len = target_len
        out = self.forward(x)
        self.output_len = orig
        return out


# ── LSTM Controller ───────────────────────────────────────────────────────────

class LSTMController(nn.Module):
    """
    Identical interface to GRUController but uses LSTM cells.
    LSTM's cell state gives it longer memory — useful for slow dynamics (PC, LC).
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


# ── CC Classifier-Regressor ───────────────────────────────────────────────────

class CCClassifierRegressor(nn.Module):
    """
    CC loop (cooling control) model.

    Shared trunk  : [P1_TIT03, P1_PP04SP] → Linear(2→32) → ReLU → Dropout
    Classifier    : Linear(32→1)  logit for pump on/off
    Regressor     : Linear(32→1)  pump speed (meaningful only when pump is on)

    Accepts a sequence (B, T, 2) — uses the last timestep — or a flat (B, 2).
    """

    def __init__(self, n_inputs: int = 2, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, 1)   # pump on/off logit
        self.regressor  = nn.Linear(hidden, 1)   # pump speed

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, n_inputs) or (B, T, n_inputs)
        → pump_logit (B, 1),  pump_speed (B, 1)
        """
        if x.dim() == 3:
            x = x[:, -1, :]                      # last timestep only
        h = self.trunk(x)
        return self.classifier(h), self.regressor(h)

    @torch.no_grad()
    def predict(self, x: torch.Tensor,
                threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns pump_on (B,) bool, pump_cv (B, 1) float.
        pump_cv = pump_speed when on, 0 when off.
        """
        self.eval()
        logit, speed = self.forward(x)
        pump_on = torch.sigmoid(logit) > threshold
        pump_cv = pump_on.float() * speed
        return pump_on.squeeze(-1), pump_cv


# ── CC Sequence Model (full history + two-head output) ───────────────────────

class CCSequenceModel(nn.Module):
    """
    CC loop model with full sequence history and two-head output.

    Inputs  : [SP, PV, CV, TIT03] over input_len steps  (n_inputs=4)
    Encoder : GRU over full sequence → hidden state
    Decoder : autoregressive, two heads per step:
                - on/off logit  (BCE auxiliary loss — helps learn transitions)
                - cv_value      (MSE loss — predicts full CV in normalised space)

    NOTE: cv_value predicts the full normalised CV directly (including zeros).
          BCE is auxiliary only — not used to mask output at inference.
          This avoids normalisation mismatch (0 in raw ≠ 0 in normalised space).
    """

    def __init__(self, n_inputs: int = 4, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.1,
                 output_len: int = 180):
        super().__init__()
        self.output_len = output_len
        drop = dropout if layers > 1 else 0.0

        self.encoder  = nn.GRU(n_inputs, hidden, layers,
                               batch_first=True, dropout=drop)
        self.decoder  = nn.GRU(1, hidden, layers,
                               batch_first=True, dropout=drop)
        self.head_on  = nn.Linear(hidden, 1)   # on/off logit (auxiliary)
        self.head_cv  = nn.Linear(hidden, 1)   # CV value in normalised space

    def forward(self, x: torch.Tensor,
                y_teacher: Optional[torch.Tensor] = None,
                ss_ratio: float = 0.0
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x:         (B, input_len, n_inputs)
        y_teacher: (B, target_len, 1)  true CV for teacher forcing
        → logits (B, target_len, 1),  cv_preds (B, target_len, 1)
        """
        B = x.size(0)
        _, h = self.encoder(x)
        prev = torch.zeros(B, 1, 1, device=x.device)
        logits, cv_preds = [], []

        for t in range(self.output_len):
            out, h  = self.decoder(prev, h)
            logit   = self.head_on(out)    # (B, 1, 1)
            cv_pred = self.head_cv(out)    # (B, 1, 1)
            logits.append(logit)
            cv_preds.append(cv_pred)

            if y_teacher is not None and ss_ratio > 0.0:
                use_pred = (torch.rand(B, device=x.device) < ss_ratio
                            ).view(B, 1, 1).float()
                prev = use_pred * cv_pred + (1 - use_pred) * y_teacher[:, t:t+1, :]
            elif y_teacher is not None:
                prev = y_teacher[:, t:t+1, :]
            else:
                prev = cv_pred

        return torch.cat(logits, dim=1), torch.cat(cv_preds, dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor,
                target_len: Optional[int] = None) -> torch.Tensor:
        """Returns predicted CV sequence (B, output_len, 1) in normalised space."""
        self.eval()
        orig = self.output_len
        if target_len is not None:
            self.output_len = target_len
        _, cv_preds = self.forward(x)
        self.output_len = orig
        return cv_preds


# ── GRU Plant (MIMO, autoregressive, scheduled sampling) ─────────────────────

class GRUPlant(nn.Module):
    """
    MIMO GRU plant model.

    Encoder : x_cv input window (B, input_len, n_plant_in) with scenario embedding
              → hidden state h
    Decoder : for each target step t:
                dec_input = concat(x_cv_target[:, t, :], pv_prev)
                          = (B, n_plant_in + n_pv)
              → GRU step → FC(hidden → 128 → ReLU → 128 → n_pv) → pv_next

    Scheduled sampling: at each step randomly selects between teacher PV and
    model-predicted PV with probability ss_ratio.

    All three plant variants (GRU / LSTM / Transformer) share this interface:
        forward(x_cv, x_cv_target, pv_init, scenario, pv_teacher, ss_ratio)
        predict(x_cv, x_cv_target, pv_init, scenario)
    """

    def __init__(self, n_plant_in: int, n_pv: int,
                 hidden: int = 256, layers: int = 2,
                 n_scenarios: int = 4, emb_dim: int = 32,
                 dropout: float = 0.1,
                 scenario_heads: bool = False,
                 n_haiend: int = 0):
        super().__init__()
        self.n_plant_in     = n_plant_in
        self.n_pv           = n_pv
        self.n_scenarios    = n_scenarios
        self.scenario_heads = scenario_heads
        self.n_haiend       = n_haiend
        drop = dropout if layers > 1 else 0.0

        self.scenario_emb = nn.Embedding(n_scenarios, emb_dim)

        self.encoder = nn.GRU(n_plant_in + emb_dim, hidden, layers,
                              batch_first=True, dropout=drop)
        self.decoder = nn.GRU(n_plant_in + n_pv, hidden, layers,
                              batch_first=True, dropout=drop)

        if scenario_heads:
            self.fc_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden, 128), nn.ReLU(),
                    nn.Dropout(dropout), nn.Linear(128, n_pv),
                ) for _ in range(n_scenarios)
            ])
        else:
            self.fc = nn.Sequential(
                nn.Linear(hidden, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_pv),
            )

        # Second output head: predicts internal PLC function block signals (HAIEND).
        # These are the exact channels AE attacks manipulate, so residuals here
        # expose AE attack footprints that are invisible in PV residuals alone.
        if n_haiend > 0:
            self.haiend_head = nn.Sequential(
                nn.Linear(hidden, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_haiend),
            )
        else:
            self.haiend_head = None

    def forward(self, x_cv: torch.Tensor, x_cv_target: torch.Tensor,
                pv_init: torch.Tensor, scenario: torch.Tensor,
                pv_teacher: Optional[torch.Tensor] = None,
                ss_ratio: float = 0.0):
        """
        x_cv:        (B, input_len,  n_plant_in)  CV/aux input window
        x_cv_target: (B, target_len, n_plant_in)  CV/aux target window
        pv_init:     (B, n_pv)                    PV at last input step
        scenario:    (B,) int  0–3
        pv_teacher:  (B, target_len, n_pv)        teacher PV (optional)
        ss_ratio:    scheduled sampling ratio
        → pv_out      (B, target_len, n_pv)
          haiend_out  (B, target_len, n_haiend)  or None if n_haiend == 0
        """
        B, T_target, _ = x_cv_target.shape

        # Encode input window (scenario embedding broadcast over time)
        emb = self.scenario_emb(scenario).unsqueeze(1).expand(-1, x_cv.size(1), -1)
        _, h = self.encoder(torch.cat([x_cv, emb], dim=-1))

        pv = pv_init
        pv_outputs     = []
        haiend_outputs = []

        for t in range(T_target):
            dec_in = torch.cat([x_cv_target[:, t, :], pv], dim=-1).unsqueeze(1)
            out, h  = self.decoder(dec_in, h)              # (B, 1, hidden)
            h_out   = out.squeeze(1)                       # (B, hidden)
            if self.scenario_heads:
                pv_pred = torch.zeros(B, self.n_pv, device=x_cv.device)
                for sc_id in range(self.n_scenarios):
                    mask = (scenario == sc_id)
                    if mask.any():
                        pv_pred[mask] = self.fc_heads[sc_id](h_out[mask])
            else:
                pv_pred = self.fc(h_out)
            pv_outputs.append(pv_pred)

            if self.haiend_head is not None:
                haiend_outputs.append(self.haiend_head(h_out))

            if pv_teacher is not None and ss_ratio > 0.0:
                use_pred = (torch.rand(B, device=x_cv.device) < ss_ratio
                            ).float().unsqueeze(-1)
                pv = use_pred * pv_pred + (1 - use_pred) * pv_teacher[:, t, :]
            elif pv_teacher is not None:
                pv = pv_teacher[:, t, :]                   # pure teacher forcing
            else:
                pv = pv_pred                               # pure autoregressive

        pv_out = torch.stack(pv_outputs, dim=1)            # (B, target_len, n_pv)
        haiend_out = (torch.stack(haiend_outputs, dim=1)   # (B, target_len, n_haiend)
                      if haiend_outputs else None)
        return pv_out, haiend_out

    @torch.no_grad()
    def predict(self, x_cv: torch.Tensor, x_cv_target: torch.Tensor,
                pv_init: torch.Tensor, scenario: torch.Tensor):
        """Full autoregressive rollout. Returns (pv_out, haiend_out)."""
        self.eval()
        return self.forward(x_cv, x_cv_target, pv_init, scenario)
