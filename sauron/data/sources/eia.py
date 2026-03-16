"""EIA (U.S. Energy Information Administration) connector.

Fetches energy production, consumption, and price data.
Requires EIA_API_KEY environment variable (free key).
"""

import os

import pandas as pd
import requests

BASE_URL = "https://api.eia.gov/v2"

# Key series for energy sector modeling
DEFAULT_SERIES = {
    "PET.RWTC.D": "wti_crude_daily",
    "PET.RBRTE.D": "brent_crude_daily",
    "NG.RNGWHHD.D": "henry_hub_natgas_daily",
    "ELEC.GEN.ALL-US-99.M": "us_electricity_gen_monthly",
    "ELEC.GEN.SUN-US-99.M": "us_solar_gen_monthly",
    "ELEC.GEN.WND-US-99.M": "us_wind_gen_monthly",
    "INTL.57-1-WORL-TBPD.M": "world_oil_production_monthly",
}


def get_api_key() -> str:
    key = os.environ.get("EIA_API_KEY")
    if not key:
        raise EnvironmentError(
            "EIA_API_KEY not set. Get a free key at https://www.eia.gov/opendata/register.php"
        )
    return key


def fetch_series(
    series_id: str,
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.Series:
    """Fetch a single EIA series."""
    api_key = get_api_key()

    # v2 API uses a different URL structure
    params = {
        "api_key": api_key,
        "frequency": "daily" if ".D" in series_id else "monthly",
        "start": start,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }

    # Parse the series ID into route and facets
    parts = series_id.split(".")
    route = parts[0].lower()
    url = f"{BASE_URL}/{route}/data/"

    if end:
        params["end"] = end

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    records = data.get("response", {}).get("data", [])
    if not records:
        return pd.Series(dtype=float, name=series_id)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].sort_index().rename(series_id)


def fetch_default(start: str = "2010-01-01") -> pd.DataFrame:
    """Fetch default EIA energy series, merge into a daily DataFrame."""
    frames = {}
    for series_id, friendly_name in DEFAULT_SERIES.items():
        try:
            s = fetch_series(series_id, start=start)
            frames[friendly_name] = s
        except Exception as e:
            print(f"[EIA] Warning: failed to fetch {series_id}: {e}")

    if not frames:
        raise RuntimeError("No EIA series fetched")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.resample("D").ffill()
    return df
