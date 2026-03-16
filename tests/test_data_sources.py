"""Tests for data source modules (offline/unit tests only)."""

import numpy as np
import pandas as pd

from sauron.data.pipeline import align_to_daily, build_dataset, normalize_features


def test_align_to_daily():
    idx1 = pd.date_range("2020-01-01", periods=10, freq="D")
    idx2 = pd.date_range("2020-01-03", periods=10, freq="D")

    df1 = pd.DataFrame({"a": range(10)}, index=idx1)
    df2 = pd.DataFrame({"b": range(10)}, index=idx2)

    merged = align_to_daily(df1, df2)
    assert "a" in merged.columns
    assert "b" in merged.columns
    assert merged.index.freq == "D" or len(merged) >= 10


def test_normalize_zscore():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [10, 20, 30, 40, 50]})
    normed, stats = normalize_features(df, method="zscore")

    assert abs(normed["x"].mean()) < 0.1
    assert "mean" in stats["x"]
    assert "std" in stats["x"]


def test_normalize_minmax():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    normed, stats = normalize_features(df, method="minmax")

    assert normed["x"].min() >= -0.01
    assert normed["x"].max() <= 1.01


def test_build_dataset():
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    features = pd.DataFrame(
        np.random.randn(200, 5),
        index=dates,
        columns=[f"f{i}" for i in range(5)],
    )
    labels = pd.DataFrame(
        np.random.randn(200, 2),
        index=dates,
        columns=["CHIPS_90d", "GREEN_90d"],
    )

    samples = build_dataset(features, labels, lookback_days=30, horizon_days=90)
    assert len(samples) > 0
    assert samples[0]["features"].shape == (30, 5)
    assert "CHIPS" in samples[0]["labels"]
