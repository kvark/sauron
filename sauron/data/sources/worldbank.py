"""World Bank Open Data connector.

Fetches development indicators: GDP, trade, FDI, R&D spending, etc.
Uses wbgapi (official World Bank Python library) — no API key needed.
"""

import pandas as pd
import wbgapi as wb

# Key indicators for geo-economic modeling
DEFAULT_INDICATORS = {
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

# Major economies to track
DEFAULT_COUNTRIES = [
    "USA", "CHN", "JPN", "DEU", "GBR", "FRA", "IND", "ITA", "BRA", "CAN",
    "RUS", "KOR", "AUS", "MEX", "IDN", "SAU", "TUR", "NLD", "CHE", "TWN",
]


def fetch_indicators(
    indicators: dict[str, str] | None = None,
    countries: list[str] | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Fetch World Bank indicators for specified countries and years.

    Returns a DataFrame with MultiIndex (country, year) and one column
    per indicator using friendly names.
    """
    indicators = indicators or DEFAULT_INDICATORS
    countries = countries or DEFAULT_COUNTRIES

    frames = []
    for wb_code, friendly_name in indicators.items():
        try:
            df = wb.data.DataFrame(
                wb_code,
                economy=countries,
                time=range(start_year, end_year + 1),
                labels=False,
                numericTimeKeys=True,
            )
            # wb returns countries as rows, years as columns
            df = df.stack().reset_index()
            df.columns = ["country", "year", friendly_name]
            frames.append(df)
        except Exception as e:
            print(f"[WorldBank] Warning: failed to fetch {wb_code}: {e}")

    if not frames:
        raise RuntimeError("No World Bank indicators fetched")

    # Merge all indicators on (country, year)
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["country", "year"], how="outer")

    result["year"] = result["year"].astype(int)
    return result.sort_values(["country", "year"]).reset_index(drop=True)


def to_daily_features(wb_data: pd.DataFrame) -> pd.DataFrame:
    """Expand annual World Bank data to daily frequency.

    Annual values are forward-filled across the year. This is appropriate
    since WB indicators change slowly and represent annual snapshots.
    Returns a DataFrame indexed by date with country-prefixed columns.
    """
    frames = []
    for country in wb_data["country"].unique():
        cdf = wb_data[wb_data["country"] == country].copy()
        cdf["date"] = pd.to_datetime(cdf["year"].astype(str) + "-01-01")
        cdf = cdf.set_index("date").drop(columns=["country", "year"])
        cdf = cdf.resample("D").ffill()
        # Prefix columns with country code
        cdf = cdf.rename(columns={c: f"{country}_{c}" for c in cdf.columns})
        frames.append(cdf)

    return pd.concat(frames, axis=1)
