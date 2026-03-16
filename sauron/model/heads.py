"""Per-sector output heads.

Each sector gets a lightweight MLP head that maps from the shared representation
to tendency, confidence, and volatility scores.
"""

import torch
import torch.nn as nn

from sauron.sectors import SECTORS


class SectorHead(nn.Module):
    """Output head for a single sector."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # tendency, confidence, volatility
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(x)
        return {
            "tendency": torch.tanh(out[..., 0]),      # [-1, 1]
            "confidence": torch.sigmoid(out[..., 1]),  # [0, 1]
            "volatility": torch.sigmoid(out[..., 2]),  # [0, 1]
        }


class MultiSectorHead(nn.Module):
    """Multi-sector output: one head per sector, shared input representation."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.heads = nn.ModuleDict({
            token: SectorHead(input_dim, hidden_dim)
            for token in SECTORS
        })

    def forward(self, sector_representations: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        """Apply per-sector heads.

        Args:
            sector_representations: (batch, num_sectors, input_dim)
                Ordered to match SECTORS keys.

        Returns:
            dict of sector_token -> {tendency, confidence, volatility}
        """
        sector_names = list(SECTORS.keys())
        results = {}
        for i, token in enumerate(sector_names):
            if i < sector_representations.shape[1]:
                results[token] = self.heads[token](sector_representations[:, i])
        return results
