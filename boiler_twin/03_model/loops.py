"""
loops.py — ControllerLSTM for each control loop.

Each controller takes:
  [SP, PV, (CV_fb)] + scenario_embedding → CV prediction

Architecture:
  Input projection: n_ctrl_in + emb_dim → hidden
  LSTM: 2 layers, hidden=128
  Output: Linear → 1 (CV)

Scenario embedding is injected by adding it to the input projection output,
only for the targeted loop (attack-conditioned).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LoopConfig:
    name:   str
    sp:     str
    pv:     str
    cv:     str
    cv_fb:  Optional[str] = None

    @property
    def n_inputs(self) -> int:
        """SP + PV + CV_fb (if exists)."""
        return 3 if self.cv_fb else 2


LOOPS = {
    "PC": LoopConfig("PC", "x1001_05_SETPOINT_OUT", "P1_PIT01",  "P1_PCV01D", "P1_PCV01Z"),
    "LC": LoopConfig("LC", "x1002_07_SETPOINT_OUT", "P1_LIT01",  "P1_LCV01D", "P1_LCV01Z"),
    "FC": LoopConfig("FC", "x1002_08_SETPOINT_OUT", "P1_FT03Z",  "P1_FCV03D", "P1_FCV03Z"),
    "TC": LoopConfig("TC", "x1003_18_SETPOINT_OUT", "P1_TIT01",  "P1_FCV01D", "P1_FCV01Z"),
    "CC": LoopConfig("CC", "P1_PP04SP",             "P1_TIT03",  "P1_PP04",   None),
}


class ControllerLSTM(nn.Module):
    """
    Single-loop LSTM controller.

    Forward (training / teacher-forced):
      inputs   : (B, seq_len, n_ctrl_in)  — [SP, PV, CV_fb] scaled
      scenario : (B,)                     — int scenario id 0..N
      h        : optional hidden state

    Returns:
      cv_pred  : (B, seq_len)   — predicted CV at each step
      h        : new hidden state
    """

    def __init__(
        self,
        n_ctrl_in:   int,
        hidden_dim:  int = 128,
        num_layers:  int = 2,
        dropout:     float = 0.1,
        n_scenarios: int = 5,
        emb_dim:     int = 16,
    ):
        super().__init__()
        self.n_ctrl_in  = n_ctrl_in
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.emb_dim    = emb_dim

        self.scenario_emb = nn.Embedding(n_scenarios, emb_dim)
        self.input_proj   = nn.Linear(n_ctrl_in + emb_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def _embed_input(self, x: torch.Tensor, scenario: torch.Tensor) -> torch.Tensor:
        """
        x        : (B, seq_len, n_ctrl_in)
        scenario : (B,)
        returns  : (B, seq_len, hidden_dim)
        """
        emb = self.scenario_emb(scenario)           # (B, emb_dim)
        emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, seq_len, emb_dim)
        return self.input_proj(torch.cat([x, emb], dim=-1))  # (B, seq_len, hidden)

    def forward(
        self,
        inputs:   torch.Tensor,
        scenario: torch.Tensor,
        h:        Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        x_proj, _ = self.lstm(self._embed_input(inputs, scenario), h)
        cv_pred    = self.fc(x_proj).squeeze(-1)    # (B, seq_len)
        # return new hidden state for rollout
        _, h_new = self.lstm(self._embed_input(inputs, scenario), h)
        return cv_pred, h_new

    @torch.no_grad()
    def step(
        self,
        x_t:      torch.Tensor,
        scenario: torch.Tensor,
        h:        Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Single-timestep inference for closed-loop rollout.
        x_t      : (B, n_ctrl_in)
        scenario : (B,)
        returns  : cv_pred (B,), h_new
        """
        x_t   = x_t.unsqueeze(1)                   # (B, 1, n_ctrl_in)
        emb   = self.scenario_emb(scenario).unsqueeze(1)  # (B, 1, emb_dim)
        x_in  = self.input_proj(torch.cat([x_t, emb], dim=-1))
        out, h_new = self.lstm(x_in, h)
        cv_pred    = self.fc(out[:, 0, :]).squeeze(-1)
        return cv_pred, h_new


def build_controllers(
    n_scenarios: int = 5,
    hidden_dim:  int = 128,
    num_layers:  int = 2,
    dropout:     float = 0.1,
    emb_dim:     int = 16,
) -> dict[str, ControllerLSTM]:
    """Build one ControllerLSTM per loop."""
    return {
        name: ControllerLSTM(
            n_ctrl_in   = loop.n_inputs,
            hidden_dim  = hidden_dim,
            num_layers  = num_layers,
            dropout     = dropout,
            n_scenarios = n_scenarios,
            emb_dim     = emb_dim,
        )
        for name, loop in LOOPS.items()
    }


if __name__ == "__main__":
    controllers = build_controllers()
    for name, model in controllers.items():
        n_params = sum(p.numel() for p in model.parameters())
        loop = LOOPS[name]
        print(f"  {name}: inputs={loop.n_inputs}  params={n_params:,}")
