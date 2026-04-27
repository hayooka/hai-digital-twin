"""
plant.py — PlantLSTM MIMO model.

Takes [CVs + aux] + current PVs + scenario_embedding → next PVs.

Training uses scheduled sampling (same as GRU notebook):
  epochs 0-9:   ss_ratio = 0.0  (pure teacher forcing)
  epochs 10-99: ss_ratio ramps 0 → 0.5
  epochs 100+:  ss_ratio = 0.5

Validation always uses ss_ratio = 1.0 (fully autoregressive).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class PlantLSTM(nn.Module):
    """
    MIMO LSTM plant model.

    At each timestep t:
      input  = concat([cv_aux_t (n_in), pv_current (n_out), scenario_emb])
      LSTM   → hidden
      output = FC → pv_pred (n_out)

    pv_current is either ground-truth (teacher forcing) or own prediction
    depending on ss_ratio.
    """

    def __init__(
        self,
        n_in:        int,          # CVs + aux signals
        n_out:       int,          # PVs
        hidden_dim:  int  = 256,
        num_layers:  int  = 2,
        dropout:     float = 0.1,
        n_scenarios: int  = 5,
        emb_dim:     int  = 32,
    ):
        super().__init__()
        self.n_in       = n_in
        self.n_out      = n_out
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.scenario_emb = nn.Embedding(n_scenarios, emb_dim)

        # Input: cv_aux + pv_current + scenario_emb
        self.lstm = nn.LSTM(
            n_in + n_out + emb_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_out),
        )

    def forward(
        self,
        x_cv:       torch.Tensor,                           # (B, seq_len, n_in)
        pv_init:    torch.Tensor,                           # (B, n_out)
        scenario:   torch.Tensor,                           # (B,)
        teacher_pv: Optional[torch.Tensor] = None,          # (B, seq_len, n_out)
        ss_ratio:   float = 0.0,
    ) -> torch.Tensor:
        """
        Returns pv_pred: (B, seq_len, n_out)

        ss_ratio = 0.0  → pure teacher forcing (fast path if teacher_pv given)
        ss_ratio = 1.0  → fully autoregressive
        0 < ss_ratio < 1 → scheduled sampling (Bernoulli mix)
        """
        B, seq_len, _ = x_cv.shape
        emb        = self.scenario_emb(scenario)        # (B, emb_dim)
        pv_current = pv_init
        h          = None
        outputs    = []

        # Fast vectorised path for pure teacher forcing
        if ss_ratio == 0.0 and teacher_pv is not None:
            return self._fast_teacher_forward(x_cv, pv_init, scenario, teacher_pv)

        for t in range(seq_len):
            x_t  = torch.cat([x_cv[:, t, :], pv_current, emb], dim=-1).unsqueeze(1)
            out, h = self.lstm(x_t, h)
            pv_pred = self.fc(out[:, 0, :])             # (B, n_out)
            outputs.append(pv_pred)

            if t < seq_len - 1:
                if teacher_pv is not None and ss_ratio < 1.0:
                    use_pred = (torch.rand(B, device=x_cv.device) < ss_ratio).float().unsqueeze(-1)
                    pv_current = use_pred * pv_pred.detach() + (1 - use_pred) * teacher_pv[:, t, :]
                else:
                    pv_current = pv_pred.detach()

        return torch.stack(outputs, dim=1)               # (B, seq_len, n_out)

    def _fast_teacher_forward(
        self,
        x_cv:       torch.Tensor,
        pv_init:    torch.Tensor,
        scenario:   torch.Tensor,
        teacher_pv: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorised full-sequence forward — 50-100x faster than per-step loop."""
        emb      = self.scenario_emb(scenario).unsqueeze(1)             # (B,1,emb_dim)
        emb_full = emb.expand(-1, x_cv.size(1), -1)                    # (B,T,emb_dim)
        pv_input = torch.cat([pv_init.unsqueeze(1),
                               teacher_pv[:, :-1, :]], dim=1)           # (B,T,n_out)
        x_full   = torch.cat([x_cv, pv_input, emb_full], dim=-1)       # (B,T,n_in+n_out+emb)
        out, _   = self.lstm(x_full)
        return self.fc(out)                                              # (B,T,n_out)

    @torch.no_grad()
    def step(
        self,
        x_cv_t:      torch.Tensor,                        # (B, n_in)
        pv_current:  torch.Tensor,                        # (B, n_out)
        scenario:    torch.Tensor,                        # (B,)
        h:           Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Single-timestep step for closed-loop rollout inference.
        Returns: pv_pred (B, n_out), h_new
        """
        emb   = self.scenario_emb(scenario)               # (B, emb_dim)
        x_t   = torch.cat([x_cv_t, pv_current, emb], dim=-1).unsqueeze(1)
        out, h_new = self.lstm(x_t, h)
        pv_pred    = self.fc(out[:, 0, :])
        return pv_pred, h_new


if __name__ == "__main__":
    B, T, N_IN, N_OUT = 4, 300, 19, 5
    model = PlantLSTM(n_in=N_IN, n_out=N_OUT)
    x_cv  = torch.randn(B, T, N_IN)
    pv_i  = torch.randn(B, N_OUT)
    sc    = torch.zeros(B, dtype=torch.long)
    tpv   = torch.randn(B, T, N_OUT)

    out_tf   = model(x_cv, pv_i, sc, teacher_pv=tpv, ss_ratio=0.0)
    out_auto = model(x_cv, pv_i, sc, teacher_pv=None, ss_ratio=1.0)
    print(f"Teacher-forced output : {out_tf.shape}")
    print(f"Autoregressive output : {out_auto.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
