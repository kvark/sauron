"""Run predictions using a trained Sauron model."""

from dataclasses import dataclass
from datetime import datetime

from sauron.sectors import SECTORS


@dataclass
class SectorPrediction:
    sector: str
    horizon: str
    tendency: float
    confidence: float
    volatility: float
    drivers: list[tuple[str, float]]

    def to_dict(self) -> dict:
        return {
            "sector": self.sector,
            "horizon": self.horizon,
            "tendency": round(self.tendency, 4),
            "confidence": round(self.confidence, 4),
            "volatility": round(self.volatility, 4),
            "drivers": [(name, round(w, 4)) for name, w in self.drivers[:5]],
        }


@dataclass
class WorldState:
    """Full prediction output: tendency vectors for all sectors."""

    timestamp: str
    predictions: list[SectorPrediction]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "sectors": {p.sector: p.to_dict() for p in self.predictions},
        }

    def summary(self) -> str:
        lines = [f"Sauron World State @ {self.timestamp}", "=" * 50]
        for p in sorted(self.predictions, key=lambda x: abs(x.tendency), reverse=True):
            arrow = "▲" if p.tendency > 0 else "▼" if p.tendency < 0 else "─"
            bar_len = int(abs(p.tendency) * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(
                f"  {p.sector:10s} {arrow} {p.tendency:+.2f} [{bar}] "
                f"conf={p.confidence:.2f} vol={p.volatility:.2f}"
            )
            if p.drivers:
                top = p.drivers[0]
                lines.append(f"             └─ top driver: {top[0]} ({top[1]:+.3f})")
        return "\n".join(lines)
