"""
causal_loss.py
Causal penalty: predicted sensor values should follow parent relationships
defined in outputs/causal_graph/parents_full.json (built from boiler/dcs_*.json).

Usage:
    causal = CausalLoss("outputs/causal_graph/parents_full.json", sensor_cols)
    loss   = mse_loss + lambda_ * causal(pred)   # pred: (B, T, F)
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path


class CausalLoss:
    """
    For each target sensor with known causal parents (from domain knowledge):
        pred[target, t] ≈ mean(pred[parent_i, t-lag_i])

    This penalises predictions that violate the known physical relationships
    extracted from the HAI boiler DCS graphs.
    """

    def __init__(self, parents_json: str, sensor_cols: list):
        data    = json.loads(Path(parents_json).read_text())
        col_idx = {col: i for i, col in enumerate(sensor_cols)}

        # relationships: list of (target_idx, [(parent_idx, lag), ...])
        self.relationships = []
        skipped = []

        for target, plist in data.items():
            if target not in col_idx:
                skipped.append(target)
                continue
            t_idx = col_idx[target]
            parents = []
            for p in plist:
                name = p["parent"]
                lag  = int(p.get("lag", 1))
                if name in col_idx:
                    parents.append((col_idx[name], lag))
            if parents:
                self.relationships.append((t_idx, parents))

        print(f"CausalLoss ready: {len(self.relationships)} sensors with parents "
              f"({len(skipped)} targets not in sensor_cols: {skipped})")

    def __call__(self, pred: torch.Tensor) -> torch.Tensor:
        """
        pred: (B, T, F)
        Returns scalar causal penalty (mean over all relationships).
        """
        if not self.relationships:
            return pred.sum() * 0.0

        losses = []
        T = pred.size(1)

        for t_idx, parents in self.relationships:
            # Find the max lag so we can align all parents to the same window
            max_lag = max(lag for _, lag in parents)
            if max_lag >= T:
                continue

            # target aligned: pred[:, max_lag:, t_idx]  shape (B, T-max_lag)
            target = pred[:, max_lag:, t_idx]

            # Each parent shifted so parent[t-lag] aligns with target[t]
            parent_signals = []
            for p_idx, lag in parents:
                offset = max_lag - lag          # extra shift to align
                parent_signals.append(pred[:, offset: T - lag, p_idx])

            # expected = mean of aligned parent signals  (B, T-max_lag)
            expected = torch.stack(parent_signals, dim=0).mean(dim=0)

            losses.append(F.mse_loss(target, expected))

        if not losses:
            return pred.sum() * 0.0

        return torch.stack(losses).mean()
