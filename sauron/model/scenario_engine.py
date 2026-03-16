"""Layer 4: Scenario Simulation Engine.

Accepts hypothetical event injections and produces counterfactual forecasts.
"What if EU imposes full sanctions on Russia?" → inject event → propagate through
event encoder → sector graph → compare with baseline forecast.

This capability is completely absent from all existing foundation models.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ScenarioEvent:
    """A hypothetical geopolitical event for scenario simulation."""

    event_type: int          # CAMEO code
    goldstein_scale: float   # expected impact [-10, 10]
    actor1_country: int      # country code index
    actor2_country: int      # country code index
    num_mentions: float      # expected media coverage (proxy for magnitude)
    avg_tone: float          # expected sentiment
    description: str = ""    # human-readable description


@dataclass
class ScenarioResult:
    """Result of a counterfactual scenario simulation."""

    baseline_forecast: dict[str, float]      # sector -> tendency without event
    counterfactual_forecast: dict[str, float] # sector -> tendency with event
    delta: dict[str, float]                  # sector -> change caused by event
    regime_shift_prob: float                 # probability this event triggers regime change
    most_affected_sectors: list[str]         # sectors ranked by impact magnitude


class ScenarioEngine(nn.Module):
    """Counterfactual scenario simulation via event injection.

    Pipeline:
    1. Generate baseline forecast (no hypothetical event)
    2. Encode hypothetical event via EventEncoder
    3. Inject event embedding into the event sequence
    4. Re-run RegimeDetector and SectorInteractionGraph
    5. Generate counterfactual forecast
    6. Compute delta (counterfactual - baseline)
    """

    def __init__(self, event_encoder, regime_detector, sector_graph, output_heads):
        super().__init__()
        self.event_encoder = event_encoder
        self.regime_detector = regime_detector
        self.sector_graph = sector_graph
        self.output_heads = output_heads

    def simulate(
        self,
        baseline_state: dict,
        scenario_events: list[ScenarioEvent],
        sector_names: list[str],
    ) -> ScenarioResult:
        """Run a counterfactual scenario.

        Args:
            baseline_state: dict with 'event_sequence', 'sector_features', 'backbone_forecast'
            scenario_events: hypothetical events to inject
            sector_names: names of sectors in order

        Returns:
            ScenarioResult with baseline vs counterfactual comparison
        """
        with torch.no_grad():
            # 1. Baseline forecast (already computed, passed in)
            baseline_sector_repr = baseline_state["sector_representations"]
            baseline_forecast = self._to_sector_dict(
                self.output_heads(baseline_sector_repr), sector_names
            )

            # 2. Encode hypothetical events
            hyp_embeddings = self._encode_scenario_events(scenario_events)

            # 3. Inject into event sequence
            event_seq = baseline_state["event_sequence"].clone()
            # Append hypothetical event embeddings to the most recent day
            event_seq[:, -1, :] = event_seq[:, -1, :] + hyp_embeddings.mean(dim=0)

            # 4. Re-run regime detection with injected events
            regime_prob, regime_emb = self.regime_detector(event_seq)

            # 5. Re-run sector graph with new regime state
            sector_features = baseline_state["sector_features"]
            counterfactual_repr = self.sector_graph(sector_features, regime_emb)

            # 6. Generate counterfactual forecast
            counterfactual_forecast = self._to_sector_dict(
                self.output_heads(counterfactual_repr), sector_names
            )

            # 7. Compute deltas
            delta = {
                s: counterfactual_forecast[s] - baseline_forecast[s]
                for s in sector_names
            }

            most_affected = sorted(sector_names, key=lambda s: abs(delta[s]), reverse=True)

            return ScenarioResult(
                baseline_forecast=baseline_forecast,
                counterfactual_forecast=counterfactual_forecast,
                delta=delta,
                regime_shift_prob=regime_prob.mean().item(),
                most_affected_sectors=most_affected,
            )

    def _encode_scenario_events(self, events: list[ScenarioEvent]) -> torch.Tensor:
        """Encode hypothetical events into embeddings."""
        batch = len(events)
        event_types = torch.tensor([e.event_type for e in events]).unsqueeze(0)
        goldstein = torch.tensor([e.goldstein_scale for e in events]).unsqueeze(0).float()
        mentions = torch.tensor([e.num_mentions for e in events]).unsqueeze(0).float()
        tone = torch.tensor([e.avg_tone for e in events]).unsqueeze(0).float()
        actor1 = torch.tensor([e.actor1_country for e in events]).unsqueeze(0)
        actor2 = torch.tensor([e.actor2_country for e in events]).unsqueeze(0)
        mask = torch.ones(1, len(events), dtype=torch.bool)

        return self.event_encoder(event_types, goldstein, mentions, tone, actor1, actor2, mask)

    def _to_sector_dict(self, tensor: torch.Tensor, sector_names: list[str]) -> dict[str, float]:
        """Convert output tensor to sector name -> value dict."""
        values = tensor.squeeze().cpu().numpy()
        return {name: float(values[i]) for i, name in enumerate(sector_names)}
