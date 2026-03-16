"""FRED (Federal Reserve Economic Data) connector.

Fetches macroeconomic time series: GDP, inflation, rates, employment, etc.
Requires FRED_API_KEY environment variable.
"""

import os
from datetime import datetime

import pandas as pd
from fredapi import Fred


def get_client() -> Fred:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=key)


def fetch_series(
    series_ids: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch multiple FRED series and merge into a single DataFrame.

    Returns a DataFrame with DatetimeIndex and one column per series.
    Missing values are forward-filled then back-filled for initial NaNs.
    """
    client = get_client()
    end = end or datetime.now().strftime("%Y-%m-%d")
    frames = {}

    for sid in series_ids:
        try:
            s = client.get_series(sid, observation_start=start, observation_end=end)
            frames[sid] = s
        except Exception as e:
            print(f"[FRED] Warning: failed to fetch {sid}: {e}")

    if not frames:
        raise RuntimeError("No FRED series fetched successfully")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # Resample to daily (many FRED series are monthly/quarterly)
    df = df.resample("D").ffill()
    df = df.bfill()  # fill leading NaNs from later-starting series
    return df


def fetch_default(start: str = "2010-01-01") -> pd.DataFrame:
    """Fetch the default FRED series defined in config."""
    default_series = [
        "GDP", "CPIAUCSL", "FEDFUNDS", "UNRATE", "DGS10",
        "DTWEXBGS", "M2SL", "INDPRO", "HOUST", "DCOILWTICO",
    ]
    return fetch_series(default_series, start=start)
