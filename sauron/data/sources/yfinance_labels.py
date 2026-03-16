"""ETF-based label generation for sector tendency scores.

Downloads historical ETF prices, computes rolling returns, and normalizes
to [-1, 1] tendency scores using z-scoring against historical distribution.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


def load_baskets(path: str | Path = "data/sector_etf_baskets.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_etf_prices(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch adjusted close prices for a list of ETF tickers."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    return prices


def compute_tendency_labels(
    baskets: dict,
    horizons_days: list[int] = [30, 90, 180],
    start: str = "2010-01-01",
    z_window: int = 252,  # ~1 year for z-score normalization
) -> pd.DataFrame:
    """Compute sector tendency labels for all sectors and horizons.

    Returns a DataFrame with columns like CHIPS_30d, CHIPS_90d, etc.
    Values are in [-1, 1] representing bearish to bullish tendency.
    """
    # Collect all unique tickers
    all_tickers = set()
    for sector_data in baskets.values():
        all_tickers.update(sector_data["etfs"])

    prices = fetch_etf_prices(sorted(all_tickers), start=start)

    results = {}
    for sector, cfg in baskets.items():
        etfs = cfg["etfs"]
        weights = np.array(cfg["weights"])

        # Get available ETFs (some may not have data)
        available = [t for t in etfs if t in prices.columns]
        if not available:
            print(f"[Labels] Warning: no price data for sector {sector}")
            continue

        # Reindex weights to match available ETFs
        avail_weights = np.array([w for t, w in zip(etfs, weights) if t in available])
        avail_weights = avail_weights / avail_weights.sum()  # renormalize

        sector_prices = prices[available]

        for horizon in horizons_days:
            # Forward returns (what happens in the NEXT n days)
            returns = sector_prices.pct_change(horizon).shift(-horizon)

            # Weighted average across basket
            weighted_return = (returns * avail_weights).sum(axis=1)

            # Z-score against rolling window
            rolling_mean = weighted_return.rolling(z_window, min_periods=60).mean()
            rolling_std = weighted_return.rolling(z_window, min_periods=60).std()
            z_scored = (weighted_return - rolling_mean) / (rolling_std + 1e-8)

            # Clip to [-1, 1] using tanh-like squashing
            tendency = np.tanh(z_scored / 2)

            results[f"{sector}_{horizon}d"] = tendency

    df = pd.DataFrame(results, index=prices.index)
    df.index.name = "date"
    return df
