#!/usr/bin/env python3
"""Train the Sauron geo-economic model."""

import argparse
import sys

from sauron.data.pipeline import SauronDataset
from sauron.model.sauron_model import SauronModel
from sauron.training.config import TrainingConfig
from sauron.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Sauron")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    parser.add_argument("--hf-only", action="store_true", help="Only use HuggingFace data sources")
    parser.add_argument("--start", default=None, help="Data start date (YYYY-MM-DD)")
    args = parser.parse_args()

    config = TrainingConfig.from_yaml(args.config)
    start = args.start or config.data_start

    # Build dataset
    dataset = SauronDataset(args.config)

    if args.synthetic:
        print("[Train] Generating synthetic data...")
        samples = _make_synthetic_samples(config)
    else:
        print(f"[Train] Fetching data from {start}...")
        samples = dataset.build(start=start, horizon_days=config.horizons[1], hf_only=args.hf_only)

    if len(samples) < 10:
        print(f"[Train] Only {len(samples)} samples — not enough to train. "
              "Try --synthetic or check data sources.")
        sys.exit(1)

    # Split into train / val / test
    n = len(samples)
    n_test = int(n * config.test_split)
    n_val = int(n * config.val_split)
    n_train = n - n_val - n_test

    # Chronological split — no shuffling across time
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    print(f"[Train] Samples: {n_train} train / {n_val} val / {n_test} test")

    # Build model
    num_features = samples[0]["features"].shape[1]
    model = SauronModel(
        num_features=num_features,
        hidden_dim=config.hidden_dim,
        num_graph_layers=config.num_graph_layers,
        num_graph_heads=config.num_graph_heads,
        dropout=config.dropout,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Model: {num_params:,} parameters")

    # Train
    trainer = Trainer(model, config)
    history = trainer.train(train_samples, val_samples)

    print(f"[Train] Done. Best val loss: {trainer.best_val_loss:.4f}")
    print(f"[Train] Checkpoint saved to {config.checkpoint_dir}/best.pt")


def _make_synthetic_samples(config: TrainingConfig, n_samples: int = 500) -> list[dict]:
    """Generate synthetic samples for pipeline testing."""
    import numpy as np
    from sauron.sectors import SECTORS

    num_features = 40  # approximate real feature count
    lookback = config.lookback_days
    sector_names = list(SECTORS.keys())

    samples = []
    for _ in range(n_samples):
        features = np.random.randn(lookback, num_features).astype(np.float32)
        mask = np.ones_like(features)
        # Randomly mask ~5% of values
        mask[np.random.rand(*mask.shape) < 0.05] = 0.0

        labels = {}
        for sector in sector_names:
            # Synthetic tendency correlated with feature means
            signal = features.mean() * 0.3 + np.random.randn() * 0.5
            labels[sector] = float(np.tanh(signal))

        samples.append({
            "features": features,
            "mask": mask,
            "labels": labels,
            "date": None,
        })

    return samples


if __name__ == "__main__":
    main()
