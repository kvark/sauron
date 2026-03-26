"""Unified Sauron model: temporal encoder → sector graph → output heads."""

import torch
import torch.nn as nn

from sauron.model.heads import MultiSectorHead
from sauron.model.sector_graph import SectorInteractionGraph
from sauron.sectors import SECTORS


class TransformerTemporalEncoder(nn.Module):
    """Encode (batch, lookback, num_features) into (batch, hidden_dim) via Transformer.

    Uses learned positional encoding and mean-pooling over the time dimension.
    Designed to be small and heavily regularized for limited data (~4K samples).
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        max_len: int = 512,
    ):
        super().__init__()
        self.proj = nn.Linear(num_features, hidden_dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, lookback, num_features)
            mask: (batch, lookback, num_features) optional

        Returns:
            (batch, hidden_dim)
        """
        if mask is not None:
            x = x * mask
        seq_len = x.shape[1]
        h = self.proj_dropout(self.proj(x))  # (batch, lookback, hidden_dim)
        h = self.layer_norm(h + self.pos_encoding[:, :seq_len, :])
        h = self.transformer(h)  # (batch, lookback, hidden_dim)
        h = h.mean(dim=1)  # mean-pool over time → (batch, hidden_dim)
        return self.output_dropout(h)


class TemporalEncoder(nn.Module):
    """Encode (batch, lookback, num_features) into (batch, hidden_dim).

    Replaces the foundation model backbone for training. Once Chronos/MOIRAI
    embeddings are available, they can be concatenated or substituted here.
    """

    def __init__(self, num_features: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(num_features, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, lookback, num_features)
            mask: (batch, lookback, num_features) optional

        Returns:
            (batch, hidden_dim)
        """
        if mask is not None:
            x = x * mask
        h = self.proj(x)
        _, hidden = self.gru(h)
        return hidden[-1]  # last layer hidden state


class SauronModel(nn.Module):
    """Full Sauron model for training.

    Pipeline:
        features (batch, lookback, F) → TemporalEncoder → (batch, H)
        → expand to (batch, num_sectors, H) with learned sector queries
        → SectorInteractionGraph → (batch, num_sectors, H)
        → MultiSectorHead → {sector: {tendency, confidence, volatility}}
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_graph_layers: int = 2,
        num_graph_heads: int = 4,
        dropout: float = 0.1,
        encoder_type: str = "transformer",
    ):
        super().__init__()
        self.num_sectors = len(SECTORS)

        if encoder_type == "transformer":
            self.encoder = TransformerTemporalEncoder(
                num_features, hidden_dim, dropout=dropout,
            )
        elif encoder_type == "gru":
            self.encoder = TemporalEncoder(num_features, hidden_dim)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type!r}. Use 'transformer' or 'gru'.")
        self.encoder_dropout = nn.Dropout(dropout)

        # Learned sector query vectors — each sector attends to the shared encoding differently
        self.sector_queries = nn.Parameter(torch.randn(self.num_sectors, hidden_dim) * 0.02)

        # Gate: combine shared encoding with sector-specific query
        self.sector_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        self.graph = SectorInteractionGraph(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_graph_heads,
            num_layers=num_graph_layers,
            num_sectors=self.num_sectors,
            dropout=dropout,
        )

        self.heads = MultiSectorHead(input_dim=hidden_dim)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            features: (batch, lookback, num_features)
            mask: (batch, lookback, num_features) optional

        Returns:
            dict of sector_token -> {tendency, confidence, volatility}
        """
        batch = features.shape[0]

        # Encode temporal features
        encoding = self.encoder_dropout(self.encoder(features, mask))  # (batch, H)

        # Expand to per-sector representations via learned queries
        encoding_expanded = encoding.unsqueeze(1).expand(-1, self.num_sectors, -1)
        queries_expanded = self.sector_queries.unsqueeze(0).expand(batch, -1, -1)
        sector_features = self.sector_gate(
            torch.cat([encoding_expanded, queries_expanded], dim=-1)
        )  # (batch, num_sectors, H)

        # Cross-sector interaction
        sector_repr = self.graph(sector_features)  # (batch, num_sectors, H)

        # Per-sector output heads
        return self.heads(sector_repr)
