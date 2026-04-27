"""
causal_loss.py — Causal regularization for PlantLSTM using phy_boiler.json.

Idea:
  For each PV, phy_boiler.json tells us which CVs physically cause it.
  We penalise the plant model if its predictions show strong correlation
  in the WRONG direction (i.e., effect → cause).

  penalty = mean of off-diagonal correlations that violate causal direction
  loss    = MSE + lambda_causal * causal_penalty
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_causal"))
from parse_dcs import build_causal_adjacency, PV_COLS, CV_COLS


def build_causal_mask(pv_cols: list[str], cv_cols: list[str]) -> torch.Tensor:
    """
    Build binary mask of shape (n_pv, n_cv).
    mask[i, j] = 1  if cv_cols[j] causally affects pv_cols[i]  (allowed direction)
    mask[i, j] = 0  if no physical edge (penalise strong correlation)

    Uses phy_boiler.json adjacency.
    """
    adj    = build_causal_adjacency()              # {pv: [cvs]}
    n_pv   = len(pv_cols)
    n_cv   = len(cv_cols)
    mask   = torch.zeros(n_pv, n_cv, dtype=torch.float32)

    for i, pv in enumerate(pv_cols):
        for j, cv in enumerate(cv_cols):
            if cv in adj.get(pv, []):
                mask[i, j] = 1.0

    return mask                                    # (n_pv, n_cv)


class CausalLoss(nn.Module):
    """
    Penalises plant predictions that violate the physical causal graph.

    For each (PV_i, CV_j) pair where no physical edge exists,
    we compute the temporal cross-correlation between predicted PV_i
    and the corresponding CV_j input, and penalise high values.

    This encourages the model to respect the structure from phy_boiler.json.
    """

    def __init__(
        self,
        pv_cols:       list[str] = PV_COLS,
        cv_cols:       list[str] = CV_COLS,
        lambda_causal: float     = 0.1,
    ):
        super().__init__()
        self.lambda_causal = lambda_causal
        self.pv_cols       = pv_cols
        self.cv_cols       = cv_cols

        mask = build_causal_mask(pv_cols, cv_cols)  # (n_pv, n_cv)
        # non_causal_mask[i,j] = 1 where NO physical edge → penalise
        self.register_buffer("non_causal_mask", 1.0 - mask)

    def forward(
        self,
        pv_pred: torch.Tensor,    # (B, seq_len, n_pv)
        cv_seqs: torch.Tensor,    # (B, seq_len, n_cv_total)  ← first n_cv are loop CVs
    ) -> torch.Tensor:
        """
        Returns scalar causal penalty.

        We compute normalised cross-correlation between each predicted PV
        and each CV over the sequence dimension, then sum violations.
        """
        n_cv = len(self.cv_cols)
        cv   = cv_seqs[:, :, :n_cv]              # (B, T, n_cv) — loop CVs only

        # Normalise sequences to zero mean, unit std
        def _norm(x):
            mu  = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
            return (x - mu) / std

        pv_n = _norm(pv_pred)                    # (B, T, n_pv)
        cv_n = _norm(cv)                         # (B, T, n_cv)

        # Cross-correlation: (B, n_pv, n_cv)
        corr = torch.einsum("btp,btc->bpc", pv_n, cv_n) / pv_pred.size(1)
        corr = corr.abs()                        # magnitude only

        # Apply non-causal mask and average
        mask    = self.non_causal_mask.to(corr.device)   # (n_pv, n_cv)
        penalty = (corr * mask.unsqueeze(0)).sum() / (mask.sum() * corr.size(0) + 1e-8)

        return self.lambda_causal * penalty


def combined_loss(
    pv_pred:       torch.Tensor,
    pv_target:     torch.Tensor,
    cv_seqs:       torch.Tensor,
    causal_loss_fn: CausalLoss,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (total_loss, mse_loss, causal_penalty).
    Caller can log both components separately.
    """
    mse     = nn.functional.mse_loss(pv_pred, pv_target)
    causal  = causal_loss_fn(pv_pred, cv_seqs)
    return mse + causal, mse, causal


if __name__ == "__main__":
    causal_fn = CausalLoss(lambda_causal=0.1)
    print("Non-causal mask (n_pv × n_cv):")
    print(f"  PVs: {causal_fn.pv_cols}")
    print(f"  CVs: {causal_fn.cv_cols}")
    print(causal_fn.non_causal_mask)

    B, T = 8, 300
    n_pv, n_cv_total = len(PV_COLS), len(CV_COLS) + 14
    pv_pred  = torch.randn(B, T, n_pv)
    pv_tgt   = torch.randn(B, T, n_pv)
    cv_seqs  = torch.randn(B, T, n_cv_total)

    total, mse, causal = combined_loss(pv_pred, pv_tgt, cv_seqs, causal_fn)
    print(f"\n  MSE={mse:.4f}  Causal={causal:.4f}  Total={total:.4f}")
