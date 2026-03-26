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
        samples = dataset.build(start=start, horizon_days=config.horizons, hf_only=args.hf_only)

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
        encoder_type=config.encoder_type,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Model: {num_params:,} parameters")

    # Train
    trainer = Trainer(model, config)
    history = trainer.train(train_samples, val_samples)

    print(f"[Train] Done. Best val loss: {trainer.best_val_loss:.4f}")
    print(f"[Train] Checkpoint saved to {config.checkpoint_dir}/best.pt")

    # Evaluate on test set
    _evaluate(model, config, test_samples)


def _evaluate(model, config, test_samples):
    """Run evaluation on the held-out test set."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from sauron.sectors import SECTORS
    from sauron.training.evaluate import (
        calibration_error,
        directional_accuracy,
        tendency_mse,
    )
    from sauron.training.trainer import SauronDatasetTorch

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    # Load best checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(f"{config.checkpoint_dir}/best.pt", map_location=device))
    model.eval()

    loader = DataLoader(SauronDatasetTorch(test_samples), batch_size=config.batch_size)
    sector_names = list(SECTORS.keys())

    # Collect predictions per sector
    all_preds = {s: [] for s in sector_names}
    all_confs = {s: [] for s in sector_names}
    all_targets = {s: [] for s in sector_names}

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            output = model(features)

            # Use primary horizon (index 0) for evaluation
            primary_labels = labels[:, 0, :]
            for i, sector in enumerate(sector_names):
                if sector not in output:
                    continue
                valid = ~primary_labels[:, i].isnan()
                if valid.sum() == 0:
                    continue
                all_preds[sector].append(output[sector]["tendency"][valid].cpu().numpy())
                all_confs[sector].append(output[sector]["confidence"][valid].cpu().numpy())
                all_targets[sector].append(primary_labels[:, i][valid].cpu().numpy())

    # Compute and print metrics
    print(f"\n{'Sector':<12} {'DirAcc':>8} {'MSE':>8} {'CalErr':>8} {'N':>6}")
    print("-" * 46)

    total_correct = 0
    total_n = 0

    for sector in sector_names:
        if not all_preds[sector]:
            continue
        preds = np.concatenate(all_preds[sector])
        targets = np.concatenate(all_targets[sector])
        confs = np.concatenate(all_confs[sector])

        da = directional_accuracy(preds, targets)
        mse = tendency_mse(preds, targets)
        ce = calibration_error(preds, targets, confs)

        n = len(preds)
        total_correct += da * n
        total_n += n

        print(f"{sector:<12} {da:>8.1%} {mse:>8.4f} {ce:>8.4f} {n:>6}")

    if total_n > 0:
        overall_da = total_correct / total_n
        print("-" * 46)
        print(f"{'OVERALL':<12} {overall_da:>8.1%}")

    # Baseline: predict 0 for everything (no tendency)
    baseline_mse_vals = []
    for sector in sector_names:
        if all_targets[sector]:
            targets = np.concatenate(all_targets[sector])
            baseline_mse_vals.append((targets ** 2).mean())
    if baseline_mse_vals:
        print(f"\nBaseline MSE (predict 0): {np.mean(baseline_mse_vals):.4f}")


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
