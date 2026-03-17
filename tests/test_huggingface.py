"""Tests for HuggingFace data source connectors (offline/unit tests).

These tests mock the HuggingFace datasets library to avoid network calls.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_wdi_wide_df():
    """Create a mock WDI dataset in wide format (Indicator Code + year columns)."""
    rows = []
    indicators = {
        "NY.GDP.MKTP.CD": 1e12,
        "NY.GDP.MKTP.KD.ZG": 2.5,
        "SP.POP.TOTL": 330e6,
    }
    for country in ["USA", "CHN"]:
        for ind_code, base_val in indicators.items():
            row = {
                "Country Name": f"{country} Name",
                "Country Code": country,
                "Indicator Name": f"{ind_code} name",
                "Indicator Code": ind_code,
            }
            for yr in range(2018, 2024):
                row[str(yr)] = base_val * (1 + 0.01 * (yr - 2018))
            rows.append(row)
    return pd.DataFrame(rows)


def _make_gdelt_events_df():
    """Create a mock GDELT events DataFrame."""
    rng = np.random.default_rng(42)
    n = 100
    dates = pd.date_range("2025-05-01", periods=10, freq="D")
    return pd.DataFrame({
        "SQLDATE": [pd.Timestamp(d).strftime("%Y%m%d") for d in rng.choice(dates, n)],
        "EventCode": rng.choice(["190", "163", "061", "050", "14"], n).tolist(),
        "EventRootCode": rng.choice(["19", "16", "06", "05", "14"], n).tolist(),
        "GoldsteinScale": rng.uniform(-10, 10, n),
        "NumMentions": rng.integers(1, 100, n),
        "AvgTone": rng.uniform(-10, 10, n),
        "NumSources": rng.integers(1, 20, n),
        "Actor1CountryCode": rng.choice(["USA", "CHN", "RUS", "GBR"], n).tolist(),
        "Actor2CountryCode": rng.choice(["USA", "CHN", "RUS", "GBR"], n).tolist(),
        "ActionGeo_CountryCode": rng.choice(["US", "CH", "RS", "UK"], n).tolist(),
    })


def _make_etf_prices_df():
    """Create a mock ETF daily prices DataFrame in long format."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    tickers = ["XLE", "SOXX", "XBI", "ICLN", "ITA"]
    rows = []
    for ticker in tickers:
        base = np.random.default_rng(hash(ticker) % 2**32).uniform(50, 200)
        for i, d in enumerate(dates):
            rows.append({
                "Date": d,
                "ticker": ticker,
                "close": base * (1 + 0.001 * i),
            })
    return pd.DataFrame(rows)


def _make_crude_oil_df():
    """Create a mock crude oil prices DataFrame."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "date": dates,
        "WTI First Purchase Price": np.random.default_rng(42).uniform(50, 80, 100),
        "US Imports Volume": np.random.default_rng(43).uniform(5000, 8000, 100),
    })


class MockDataset:
    """Mock HuggingFace Dataset that returns a DataFrame."""

    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df.copy()


# ---- Tests ----


@patch("sauron.data.sources.huggingface._load_hf_dataset")
def test_fetch_wdi_hf_wide_format(mock_load):
    mock_load.return_value = _make_wdi_wide_df()

    from sauron.data.sources.huggingface import fetch_wdi_hf

    result = fetch_wdi_hf(
        indicators={
            "NY.GDP.MKTP.CD": "gdp_current_usd",
            "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
            "SP.POP.TOTL": "population",
        },
        countries=["USA", "CHN"],
        start_year=2018,
        end_year=2023,
    )

    assert "country" in result.columns
    assert "year" in result.columns
    assert "gdp_current_usd" in result.columns
    assert set(result["country"].unique()) == {"USA", "CHN"}
    assert len(result) > 0


@patch("sauron.data.sources.huggingface._load_hf_dataset")
def test_wdi_to_daily_features(mock_load):
    mock_load.return_value = _make_wdi_wide_df()

    from sauron.data.sources.huggingface import fetch_wdi_hf, wdi_to_daily_features

    wb = fetch_wdi_hf(
        indicators={"NY.GDP.MKTP.CD": "gdp_current_usd"},
        countries=["USA"],
        start_year=2020,
        end_year=2023,
    )
    daily = wdi_to_daily_features(wb)

    assert isinstance(daily.index, pd.DatetimeIndex)
    assert any("USA_" in c for c in daily.columns)
    assert len(daily) > 365  # should have daily data for multiple years


@patch("sauron.data.sources.huggingface._load_hf_dataset")
def test_fetch_gdelt_hf(mock_load):
    mock_load.return_value = _make_gdelt_events_df()

    from sauron.data.sources.huggingface import fetch_gdelt_hf

    result = fetch_gdelt_hf(min_goldstein_abs=3.0)

    assert "date" in result.columns
    assert "GoldsteinScale" in result.columns
    assert "EventCode" in result.columns
    assert (result["GoldsteinScale"].abs() >= 3.0).all()
    assert len(result) > 0


@patch("sauron.data.sources.huggingface._load_hf_dataset")
def test_gdelt_hf_feeds_into_aggregation(mock_load):
    mock_load.return_value = _make_gdelt_events_df()

    from sauron.data.sources.gdelt import aggregate_daily_sector_features
    from sauron.data.sources.huggingface import fetch_gdelt_hf

    raw = fetch_gdelt_hf(min_goldstein_abs=0.0)
    agg = aggregate_daily_sector_features(raw)

    assert isinstance(agg.index, pd.DatetimeIndex)
    # Should have sector feature columns
    sector_cols = [c for c in agg.columns if "_event_count" in c]
    assert len(sector_cols) > 0


@patch("sauron.data.sources.huggingface._load_hf_dataset")
def test_fetch_etf_prices_hf(mock_load):
    mock_load.return_value = _make_etf_prices_df()

    from sauron.data.sources.huggingface import fetch_etf_prices_hf

    result = fetch_etf_prices_hf()

    assert isinstance(result.index, pd.DatetimeIndex)
    assert "XLE" in result.columns or "SOXX" in result.columns
    assert len(result) > 0


@patch("sauron.data.sources.huggingface._load_hf_dataset")
def test_fetch_crude_oil_hf(mock_load):
    mock_load.return_value = _make_crude_oil_df()

    from sauron.data.sources.huggingface import fetch_crude_oil_hf

    result = fetch_crude_oil_hf()

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.name == "date"
    assert len(result) >= 100
    # Should have renamed WTI column
    assert "wti_crude_daily" in result.columns


@patch("sauron.data.sources.huggingface._load_hf_dataset")
def test_fetch_source_hf_worldbank(mock_load):
    """Test that fetch_source routes hf_worldbank correctly."""
    mock_load.return_value = _make_wdi_wide_df()

    from sauron.data.fetch import fetch_source

    result = fetch_source("hf_worldbank", "2018-01-01")

    assert result is not None
    assert isinstance(result.index, pd.DatetimeIndex)


@patch("sauron.data.sources.huggingface._load_hf_dataset")
def test_fetch_source_hf_gdelt(mock_load):
    """Test that fetch_source routes hf_gdelt correctly."""
    mock_load.return_value = _make_gdelt_events_df()

    from sauron.data.fetch import fetch_source

    result = fetch_source("hf_gdelt", "2025-05-01")

    assert result is not None
    assert isinstance(result.index, pd.DatetimeIndex)
