"""Training configuration."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    # Data
    lookback_days: int = 90
    horizons: list[int] = field(default_factory=lambda: [30, 90, 180])
    data_start: str = "2015-01-01"

    # Model
    backbone: str = "chronos-2"
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_graph_layers: int = 2
    num_graph_heads: int = 4
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 10
    grad_clip: float = 1.0
    val_split: float = 0.15
    test_split: float = 0.15

    # Loss weights
    direction_loss_weight: float = 0.3
    confidence_loss_weight: float = 0.1

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        flat = {}
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in flat.items() if k in known_fields}
        return cls(**filtered)
