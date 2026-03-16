"""Layer 3: Cross-Sector Interaction Graph.

A graph attention network where nodes are sectors and edges represent learned
causal/correlational relationships. Propagates shocks across sectors:
energy disruption → trade impact → financial stress, with learned lag structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SectorGraphAttention(nn.Module):
    """Single graph attention layer over sector nodes.

    Each sector is a node. Edges are fully connected (all-to-all) with
    learned attention weights that capture which sectors influence which.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)

        # Learned adjacency bias: which sectors tend to influence which
        # This is a soft prior that attention can override
        self.adjacency_bias = nn.Parameter(torch.zeros(num_heads, 12, 12))

        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply graph attention over sector nodes.

        Args:
            x: (batch, num_sectors, dim) sector representations

        Returns:
            (batch, num_sectors, out_dim) updated representations
        """
        batch, num_sectors, _ = x.shape

        q = self.W_q(x).view(batch, num_sectors, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch, num_sectors, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch, num_sectors, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores + learned adjacency bias
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        adj_bias = self.adjacency_bias[:, :num_sectors, :num_sectors].unsqueeze(0)
        attn = attn + adj_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, num_sectors, -1)
        out = self.out_proj(out)

        # Residual + norm
        if x.shape[-1] == out.shape[-1]:
            out = self.norm(out + x)
        else:
            out = self.norm(out)

        return out


class SectorInteractionGraph(nn.Module):
    """Multi-layer graph attention network for cross-sector interaction modeling.

    Captures:
    - Direct sector-to-sector influence (energy → trade)
    - Multi-hop propagation (energy → trade → finance)
    - Sector-specific responses to regime events
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_sectors: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_sectors = num_sectors

        # Project sector inputs to hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stacked graph attention layers (each hop = one layer of propagation)
        self.layers = nn.ModuleList([
            SectorGraphAttention(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Regime modulation: how does a regime shift affect each sector differently?
        self.regime_gate = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sector_features: torch.Tensor,
        regime_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Propagate information through the sector interaction graph.

        Args:
            sector_features: (batch, num_sectors, input_dim)
            regime_embedding: (batch, hidden_dim) from RegimeDetector, optional

        Returns:
            (batch, num_sectors, hidden_dim) updated sector representations
        """
        x = self.input_proj(sector_features)

        # If regime shift detected, modulate sector representations
        if regime_embedding is not None:
            regime_expanded = regime_embedding.unsqueeze(1).expand(-1, self.num_sectors, -1)
            gate = self.regime_gate(torch.cat([x, regime_expanded], dim=-1))
            x = x * gate  # selective amplification/dampening per sector

        # Multi-hop message passing through graph attention
        for layer in self.layers:
            x = layer(x)

        return x

    def get_adjacency_weights(self) -> torch.Tensor:
        """Extract learned sector-to-sector influence weights for interpretability.

        Returns: (num_layers, num_heads, num_sectors, num_sectors) attention biases
        """
        weights = []
        for layer in self.layers:
            weights.append(layer.adjacency_bias.detach())
        return torch.stack(weights)
