"""HuggingFace dataset connectors.

Alternative data sources using HuggingFace datasets library.
These can replace or supplement the direct API fetchers when API keys
are unavailable or for faster offline development.

Available HF datasets:
- datonic/world_development_indicators: replaces worldbank.py (wbgapi)
- dwb2023/gdelt-event-2025-v4: supplements gdelt.py (CSV/BigQuery)
- paperswithbacktest/ETFs-Daily-Price: supplements yfinance_labels.py
- MaxPrestige/CRUDE_OIL_PRICES: partial EIA replacement (crude oil only)
"""

from __future__ import annotations

import pandas as pd


def _load_hf_dataset(repo_id: str, split: str = "train", **kwargs) -> pd.DataFrame:
    """Load a HuggingFace dataset and return as pandas DataFrame."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets huggingface_hub")

    ds = load_dataset(repo_id, split=split, **kwargs)
    return ds.to_pandas()


def _load_hf_parquet(repo_id: str, filename: str) -> pd.DataFrame:
    """Download and load a single parquet file from a HuggingFace dataset repo."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, filename, repo_type="dataset")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# World Development Indicators (replaces worldbank.py)
# ---------------------------------------------------------------------------

# Map WDI indicator codes to our friendly names (same as worldbank.py)
WDI_INDICATORS = {
    "NY.GDP.MKTP.CD": "gdp_current_usd",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "NE.TRD.GNFS.ZS": "trade_pct_gdp",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_net_pct_gdp",
    "BN.CAB.XOKA.GD.ZS": "current_account_pct_gdp",
    "FP.CPI.TOTL.ZG": "inflation_cpi_pct",
    "GB.XPD.RSDV.GD.ZS": "rd_expenditure_pct_gdp",
    "MS.MIL.XPND.GD.ZS": "military_expenditure_pct_gdp",
    "EG.USE.PCAP.KG.OE": "energy_use_per_capita",
    "IT.NET.USER.ZS": "internet_users_pct",
    "SP.POP.TOTL": "population",
}

DEFAULT_COUNTRIES = [
    "USA", "CHN", "DEU", "JPN", "GBR",
]


def fetch_wdi_hf(
    indicators: dict[str, str] | None = None,
    countries: list[str] | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Fetch World Development Indicators from HuggingFace.

    Returns a DataFrame with columns [country, year, <indicator_friendly_names>],
    matching the output format of worldbank.fetch_indicators().

    Dataset: datonic/world_development_indicators
    """
    indicators = indicators or WDI_INDICATORS
    countries = countries or DEFAULT_COUNTRIES

    print("[HF-WDI] Loading datonic/world_development_indicators...")
    raw = _load_hf_parquet(
        "datonic/world_development_indicators",
        "data/world_development_indicators.parquet",
    )

    # Dataset is long format: country_code, indicator_code, year, indicator_value
    indicator_codes = set(indicators.keys())
    filtered = raw[
        (raw["indicator_code"].isin(indicator_codes))
        & (raw["country_code"].isin(countries))
    ].copy()

    filtered["year"] = pd.to_numeric(filtered["year"], errors="coerce")
    filtered = filtered[
        (filtered["year"] >= start_year) & (filtered["year"] <= end_year)
    ].dropna(subset=["year"])
    filtered["year"] = filtered["year"].astype(int)

    filtered["friendly_name"] = filtered["indicator_code"].map(indicators)
    filtered["indicator_value"] = pd.to_numeric(filtered["indicator_value"], errors="coerce")

    result = filtered.pivot_table(
        index=["country_code", "year"],
        columns="friendly_name",
        values="indicator_value",
        aggfunc="first",
    ).reset_index()
    result = result.rename(columns={"country_code": "country"})
    result.columns.name = None

    if result.empty:
        raise RuntimeError("No WDI data matched filters")

    print(f"[HF-WDI] Loaded {len(result)} rows for {result['country'].nunique()} countries")
    return result.sort_values(["country", "year"]).reset_index(drop=True)


def wdi_to_daily_features(wb_data: pd.DataFrame) -> pd.DataFrame:
    """Expand annual WDI data to daily frequency (same as worldbank.to_daily_features)."""
    frames = []
    for country in wb_data["country"].unique():
        cdf = wb_data[wb_data["country"] == country].copy()
        cdf["date"] = pd.to_datetime(cdf["year"].astype(str) + "-01-01")
        cdf = cdf.set_index("date").drop(columns=["country", "year"])
        cdf = cdf.resample("D").ffill()
        cdf = cdf.rename(columns={c: f"{country}_{c}" for c in cdf.columns})
        frames.append(cdf)

    return pd.concat(frames, axis=1)


# ---------------------------------------------------------------------------
# GDELT Events (supplements gdelt.py)
# ---------------------------------------------------------------------------

def fetch_gdelt_hf(
    min_goldstein_abs: float = 3.0,
) -> pd.DataFrame:
    """Fetch GDELT event data from HuggingFace.

    Dataset: dwb2023/gdelt-event-2025-v4
    Downloads individual parquet files (one per day).

    Note: This dataset covers a limited date range (May 1-11, 2025).
    For historical data, use fetch_gdelt_csv() or fetch_gdelt_bigquery().
    """
    from huggingface_hub import list_repo_files

    print("[HF-GDELT] Loading dwb2023/gdelt-event-2025-v4...")
    files = [
        f for f in list_repo_files("dwb2023/gdelt-event-2025-v4", repo_type="dataset")
        if f.endswith(".parquet")
    ]

    frames = []
    for f in files:
        frames.append(_load_hf_parquet("dwb2023/gdelt-event-2025-v4", f))
    df = pd.concat(frames, ignore_index=True)

    # This dataset uses 'Day' instead of 'SQLDATE'
    df["date"] = pd.to_datetime(df["Day"].astype(str), format="%Y%m%d", errors="coerce")

    # Ensure numeric types
    for col in ["GoldsteinScale", "NumMentions", "AvgTone", "NumSources"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure EventCode and EventRootCode are strings
    for col in ["EventCode", "EventRootCode"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Filter by impact threshold
    df = df[df["GoldsteinScale"].abs() >= min_goldstein_abs]

    print(f"[HF-GDELT] Loaded {len(df)} events across {df['date'].nunique()} days")
    return df


# ---------------------------------------------------------------------------
# ETF Daily Prices (supplements yfinance_labels.py)
# ---------------------------------------------------------------------------

def fetch_etf_prices_hf() -> pd.DataFrame:
    """Fetch ETF daily prices from HuggingFace.

    Dataset: paperswithbacktest/ETFs-Daily-Price
    Returns DataFrame indexed by date with ticker columns containing adjusted close prices.
    """
    print("[HF-ETF] Loading paperswithbacktest/ETFs-Daily-Price...")
    raw = _load_hf_dataset("paperswithbacktest/ETFs-Daily-Price")

    # Detect format: could be wide (date + ticker columns) or long (date, ticker, price)
    if "date" in raw.columns or "Date" in raw.columns:
        date_col = "date" if "date" in raw.columns else "Date"

        if "ticker" in raw.columns or "Ticker" in raw.columns or "symbol" in raw.columns:
            # Long format
            ticker_col = next(
                c for c in ["ticker", "Ticker", "symbol", "Symbol"] if c in raw.columns
            )
            price_col = next(
                (c for c in ["adj_close", "Adj Close", "close", "Close", "price"]
                 if c in raw.columns),
                raw.columns[-1],
            )
            raw[date_col] = pd.to_datetime(raw[date_col])
            raw[price_col] = pd.to_numeric(raw[price_col], errors="coerce")
            df = raw.pivot_table(
                index=date_col, columns=ticker_col, values=price_col, aggfunc="last"
            )
        else:
            # Wide format: date column + ticker columns
            raw[date_col] = pd.to_datetime(raw[date_col])
            df = raw.set_index(date_col)
            # Convert all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        raise RuntimeError(f"Cannot parse ETF format. Columns: {list(raw.columns)}")

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()

    print(f"[HF-ETF] Loaded {len(df)} days, {len(df.columns)} tickers")
    return df


# ---------------------------------------------------------------------------
# Crude Oil Prices (partial EIA replacement)
# ---------------------------------------------------------------------------

def fetch_crude_oil_hf() -> pd.DataFrame:
    """Fetch crude oil price data from HuggingFace.

    Dataset: MaxPrestige/CRUDE_OIL_PRICES
    Returns DataFrame indexed by date with price columns.
    Only covers crude oil — not natural gas, electricity, etc.
    """
    print("[HF-Oil] Loading MaxPrestige/CRUDE_OIL_PRICES...")
    raw = _load_hf_dataset("MaxPrestige/CRUDE_OIL_PRICES")

    date_col = next(
        (c for c in raw.columns if "date" in c.lower() or "period" in c.lower()),
        raw.columns[0],
    )
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    df = raw.set_index(date_col).sort_index()
    df.index.name = "date"

    # Convert all value columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rename to friendly names if possible
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "wti" in col_lower or "first purchase" in col_lower:
            rename_map[col] = "wti_crude_daily"
        elif "brent" in col_lower:
            rename_map[col] = "brent_crude_daily"
        elif "import" in col_lower:
            rename_map[col] = "crude_imports_daily"
    if rename_map:
        df = df.rename(columns=rename_map)

    df = df.resample("D").ffill()
    print(f"[HF-Oil] Loaded {len(df)} days, columns: {list(df.columns)}")
    return df
