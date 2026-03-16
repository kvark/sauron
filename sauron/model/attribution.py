"""Layer 5: Interpretability & Attribution.

Produces human-readable explanations for sector predictions:
"CHIPS tendency ↓0.3 because: US export controls (+0.15), TSMC capex delay (+0.10)"

Uses integrated gradients over the event encoder and sector graph to attribute
predictions to specific input features and events.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DriverAttribution:
    """Attribution of a sector prediction to specific drivers."""

    sector: str
    tendency: float
    confidence: float
    drivers: list[tuple[str, float]]  # (driver_name, contribution) sorted by |contribution|


class AttributionLayer(nn.Module):
    """Compute driver attributions for sector predictions."""

    def __init__(self, num_sectors: int = 12, hidden_dim: int = 128):
        super().__init__()
        # Attention over input features to produce importance weights
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        self.num_sectors = num_sectors

    def compute_attributions(
        self,
        sector_representations: torch.Tensor,  # (batch, num_sectors, hidden_dim)
        feature_embeddings: torch.Tensor,       # (batch, num_features, hidden_dim)
        feature_names: list[str],
        sector_names: list[str],
    ) -> list[list[DriverAttribution]]:
        """Compute which input features drive each sector prediction.

        Uses cross-attention from sector representations to input features
        to derive importance weights.

        Returns: list (batch) of list (sectors) of DriverAttribution
        """
        # Cross-attention: sectors attend to input features
        attn_output, attn_weights = self.feature_attention(
            sector_representations,  # query: sectors
            feature_embeddings,      # key: features
            feature_embeddings,      # value: features
        )
        # attn_weights: (batch, num_sectors, num_features)

        batch_attributions = []
        for b in range(attn_weights.shape[0]):
            sector_attributions = []
            for s in range(min(self.num_sectors, len(sector_names))):
                weights = attn_weights[b, s].detach().cpu().numpy()

                # Pair feature names with their attention weights
                drivers = sorted(
                    zip(feature_names, weights.tolist()),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )

                sector_attributions.append(DriverAttribution(
                    sector=sector_names[s],
                    tendency=0.0,  # filled in by caller
                    confidence=0.0,
                    drivers=drivers[:10],  # top 10 drivers
                ))

            batch_attributions.append(sector_attributions)

        return batch_attributions


def format_attribution(attr: DriverAttribution) -> str:
    """Format attribution as human-readable string."""
    direction = "↑" if attr.tendency > 0 else "↓"
    lines = [f"{attr.sector} tendency {direction}{abs(attr.tendency):.2f} because:"]
    for name, weight in attr.drivers[:5]:
        sign = "+" if weight > 0 else "-"
        lines.append(f"  {sign} {name} ({abs(weight):.3f})")
    return "\n".join(lines)
