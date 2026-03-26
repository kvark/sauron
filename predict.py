#!/usr/bin/env python3
"""Generate predictions using a trained Sauron model."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from sauron.data.pipeline import SauronDataset, engineer_features, normalize_features
from sauron.inference.predict import SectorPrediction, WorldState
from sauron.model.sauron_model import SauronModel
from sauron.sectors import SECTORS
from sauron.training.config import TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Sauron predictions")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon in days")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of summary")
    args = parser.parse_args()

    config = TrainingConfig.from_yaml(args.config)

    # Fetch latest features
    print("[Predict] Fetching latest data...")
    dataset = SauronDataset(args.config)
    features = dataset.fetch_all_features(start=config.data_start)
    features = engineer_features(features)
    features_norm, _ = normalize_features(features)

    # Take the last lookback window as our context
    lookback = config.lookback_days
    if len(features_norm) < lookback:
        print(f"[Predict] Only {len(features_norm)} days of data, need {lookback}")
        return

    latest_window = features_norm.iloc[-lookback:].values.astype(np.float32)
    latest_window = np.nan_to_num(latest_window, nan=0.0)
    latest_date = features_norm.index[-1]

    # Load model
    num_features = latest_window.shape[1]
    model = SauronModel(
        num_features=num_features,
        hidden_dim=config.hidden_dim,
        num_graph_layers=config.num_graph_layers,
        num_graph_heads=config.num_graph_heads,
        dropout=config.dropout,
        encoder_type=config.encoder_type,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    # Run inference
    with torch.no_grad():
        x = torch.tensor(latest_window).unsqueeze(0)  # (1, lookback, features)
        output = model(x)

    # Build predictions
    sector_names = list(SECTORS.keys())
    predictions = []
    for sector in sector_names:
        if sector not in output:
            continue
        predictions.append(SectorPrediction(
            sector=sector,
            horizon=f"{args.horizon}d",
            tendency=output[sector]["tendency"].item(),
            confidence=output[sector]["confidence"].item(),
            volatility=output[sector]["volatility"].item(),
            drivers=[],  # attribution not wired yet
        ))

    world = WorldState(
        timestamp=datetime.now().isoformat(),
        predictions=predictions,
    )

    if args.json:
        print(json.dumps(world.to_dict(), indent=2))
    else:
        print(f"\nData through: {latest_date.date()}")
        print(f"Forecast horizon: {args.horizon} days\n")
        print(world.summary())


if __name__ == "__main__":
    main()
