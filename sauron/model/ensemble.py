"""Foundation model ensemble logic.

Combines predictions from Chronos-2 and MOIRAI-2 with learned or fixed weights.
"""

import numpy as np
import torch
import torch.nn as nn


class LearnedEnsemble(nn.Module):
    """Learned weighting of foundation model predictions.

    Instead of fixed weights, learn per-sector, per-horizon weights
    that determine how much to trust each backbone.
    """

    def __init__(self, num_backbones: int = 2, num_sectors: int = 12):
        super().__init__()
        # Per-sector weights for each backbone
        self.logits = nn.Parameter(torch.zeros(num_sectors, num_backbones))

    def forward(self, predictions: list[torch.Tensor]) -> torch.Tensor:
        """Combine predictions from multiple backbones.

        Args:
            predictions: list of (batch, num_sectors) tensors, one per backbone

        Returns:
            (batch, num_sectors) weighted ensemble prediction
        """
        weights = torch.softmax(self.logits, dim=-1)  # (num_sectors, num_backbones)
        stacked = torch.stack(predictions, dim=-1)      # (batch, num_sectors, num_backbones)
        return (stacked * weights.unsqueeze(0)).sum(dim=-1)

    def get_weights(self) -> dict[str, np.ndarray]:
        """Get per-sector backbone weights for interpretability."""
        weights = torch.softmax(self.logits, dim=-1).detach().cpu().numpy()
        return {"weights": weights}
