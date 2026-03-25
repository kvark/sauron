"""EIA (U.S. Energy Information Administration) connector.

Fetches energy production, consumption, and price data via the v2 API.
Requires EIA_API_KEY environment variable (free key).
"""

import os

import pandas as pd
import requests

BASE_URL = "https://api.eia.gov/v2"


# v2 API series definitions: (route, frequency, facets, data_field, friendly_name)
DEFAULT_SERIES = [
    # Petroleum spot prices
    {
        "route": "petroleum/pri/spt",
        "frequency": "daily",
        "facets": {"series": ["RWTC"]},
        "data_field": "value",
        "name": "wti_crude_daily",
    },
    {
        "route": "petroleum/pri/spt",
        "frequency": "daily",
        "facets": {"series": ["RBRTE"]},
        "data_field": "value",
        "name": "brent_crude_daily",
    },
    # Natural gas spot price
    {
        "route": "natural-gas/pri/fut",
        "frequency": "daily",
        "facets": {"series": ["RNGWHHD"]},
        "data_field": "value",
        "name": "henry_hub_natgas_daily",
    },
    # Electricity generation (monthly)
    {
        "route": "electricity/electric-power-operational-data",
        "frequency": "monthly",
        "facets": {"fueltypeid": ["ALL"], "location": ["US"], "sectorid": ["99"]},
        "data_field": "generation",
        "name": "us_electricity_gen_monthly",
    },
    {
        "route": "electricity/electric-power-operational-data",
        "frequency": "monthly",
        "facets": {"fueltypeid": ["SUN"], "location": ["US"], "sectorid": ["99"]},
        "data_field": "generation",
        "name": "us_solar_gen_monthly",
    },
    {
        "route": "electricity/electric-power-operational-data",
        "frequency": "monthly",
        "facets": {"fueltypeid": ["WND"], "location": ["US"], "sectorid": ["99"]},
        "data_field": "generation",
        "name": "us_wind_gen_monthly",
    },
]


def get_api_key() -> str:
    key = os.environ.get("EIA_API_KEY")
    if not key:
        raise EnvironmentError(
            "EIA_API_KEY not set. Get a free key at https://www.eia.gov/opendata/register.php"
        )
    return key


def fetch_series(
    route: str,
    frequency: str,
    facets: dict[str, list[str]],
    data_field: str,
    name: str,
    start: str = "2010-01-01",
) -> pd.Series:
    """Fetch a single EIA v2 series."""
    api_key = get_api_key()

    # Build query params
    params = {
        "api_key": api_key,
        "frequency": frequency,
        "data[0]": data_field,
        "start": start if frequency == "daily" else start[:7],  # YYYY-MM for monthly
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }

    # Add facet filters
    for facet_key, facet_values in facets.items():
        for val in facet_values:
            params[f"facets[{facet_key}][]"] = val

    url = f"{BASE_URL}/{route}/data"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    records = data.get("response", {}).get("data", [])
    if not records:
        return pd.Series(dtype=float, name=name)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["period"])
    df[data_field] = pd.to_numeric(df[data_field], errors="coerce")
    return df.set_index("date")[data_field].sort_index().rename(name)


def fetch_default(start: str = "2010-01-01") -> pd.DataFrame:
    """Fetch default EIA energy series, merge into a daily DataFrame."""
    frames = {}
    for series_def in DEFAULT_SERIES:
        try:
            s = fetch_series(
                route=series_def["route"],
                frequency=series_def["frequency"],
                facets=series_def["facets"],
                data_field=series_def["data_field"],
                name=series_def["name"],
                start=start,
            )
            if not s.empty:
                frames[series_def["name"]] = s
        except Exception as e:
            print(f"[EIA] Warning: failed to fetch {series_def['name']}: {e}")

    if not frames:
        raise RuntimeError("No EIA series fetched")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.resample("D").ffill()
    return df
