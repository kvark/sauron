"""Layer 2: Event-Driven Regime Encoder.

Encodes GDELT geopolitical events into dense vectors and detects regime shifts
(sanctions, wars, coups, trade agreements) that modulate forecasts.

This is where Sauron diverges from generic forecasters — it understands that
"Russia invades Ukraine" is a regime change, not just a data point.
"""

import torch
import torch.nn as nn


class EventEncoder(nn.Module):
    """Encode geopolitical events into dense representations.

    Input features per timestep:
    - event_type: CAMEO code (categorical, ~300 types)
    - goldstein_scale: impact score [-10, 10]
    - num_mentions: media coverage volume
    - avg_tone: sentiment [-100, 100]
    - actor_country_1, actor_country_2: country codes (categorical)
    """

    def __init__(
        self,
        num_event_types: int = 300,
        num_countries: int = 250,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.event_type_emb = nn.Embedding(num_event_types, embedding_dim // 2)
        self.country_emb = nn.Embedding(num_countries, embedding_dim // 4)

        # Continuous features: goldstein, mentions, tone
        self.continuous_proj = nn.Linear(3, embedding_dim // 4)

        # Combine all event features: type(dim/2) + country(dim/4) + continuous(dim/4) = dim
        self.event_fusion = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Temporal aggregation: attention over events within a day
        self.day_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        self.output_dim = embedding_dim

    def forward(
        self,
        event_types: torch.Tensor,       # (batch, max_events)
        goldstein: torch.Tensor,          # (batch, max_events)
        mentions: torch.Tensor,           # (batch, max_events)
        tone: torch.Tensor,              # (batch, max_events)
        actor1_country: torch.Tensor,     # (batch, max_events)
        actor2_country: torch.Tensor,     # (batch, max_events)
        event_mask: torch.Tensor,         # (batch, max_events) bool
    ) -> torch.Tensor:
        """Encode events for a single timestep.

        Returns: (batch, embedding_dim) aggregated event representation.
        """
        # Embed categorical features
        type_emb = self.event_type_emb(event_types)
        c1_emb = self.country_emb(actor1_country)
        c2_emb = self.country_emb(actor2_country)
        country_emb = c1_emb + c2_emb  # symmetric combination

        # Project continuous features
        continuous = torch.stack([goldstein, mentions, tone], dim=-1)
        cont_emb = self.continuous_proj(continuous)

        # Fuse all features
        combined = torch.cat([type_emb, country_emb, cont_emb], dim=-1)
        event_emb = self.event_fusion(combined)

        # Self-attention over events in the day, with masking
        key_padding_mask = ~event_mask
        attended, _ = self.day_attention(
            event_emb, event_emb, event_emb,
            key_padding_mask=key_padding_mask,
        )

        # Masked mean pooling
        mask_expanded = event_mask.unsqueeze(-1).float()
        pooled = (attended * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)

        return pooled


class RegimeDetector(nn.Module):
    """Detect regime shifts from event sequences.

    Looks at a window of daily event embeddings and classifies whether
    a structural break is occurring (sanctions, war, major policy shift).
    """

    def __init__(self, embedding_dim: int = 128, window_size: int = 7):
        super().__init__()
        self.window_size = window_size

        self.temporal = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        self.regime_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.regime_embedding = nn.Sequential(
            nn.Linear(embedding_dim + 1, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, event_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect regime shifts from a sequence of daily event embeddings.

        Args:
            event_sequence: (batch, days, embedding_dim)

        Returns:
            regime_prob: (batch, 1) probability of regime shift
            regime_emb: (batch, embedding_dim) regime-aware representation
        """
        _, hidden = self.temporal(event_sequence)
        last_hidden = hidden[-1]  # (batch, embedding_dim)

        regime_prob = self.regime_head(last_hidden)

        # Combine hidden state with regime probability for a regime-aware embedding
        regime_emb = self.regime_embedding(torch.cat([last_hidden, regime_prob], dim=-1))

        return regime_prob, regime_emb
