"""SIPRI military expenditure data connector.

Downloads military spending data from SIPRI's public datasets.
No API key required — data is available as downloadable files.
"""

import io

import pandas as pd
import requests

# SIPRI military expenditure database (current USD)
MILEX_URL = (
    "https://milex.sipri.org/sipri_milex/sipri_milex.php?"
    "c=&g=&y1=2010&y2=2024&d=current&ct=csv"
)

# Fallback: World Bank mirror of SIPRI data
WB_MILEX_INDICATOR = "MS.MIL.XPND.CD"


def fetch_milex_direct() -> pd.DataFrame:
    """Fetch military expenditure directly from SIPRI.

    Returns DataFrame with countries as rows and years as columns.
    Values in current USD millions.
    """
    resp = requests.get(MILEX_URL, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text), skiprows=5)
    df = df.rename(columns={df.columns[0]: "country"})

    # Melt years into rows
    year_cols = [c for c in df.columns if c != "country" and c.isdigit()]
    df = df.melt(id_vars=["country"], value_vars=year_cols, var_name="year", value_name="milex_usd")
    df["year"] = df["year"].astype(int)
    df["milex_usd"] = pd.to_numeric(
        df["milex_usd"].astype(str).str.replace(",", "").str.strip(),
        errors="coerce",
    )
    return df.dropna(subset=["milex_usd"])


def fetch_milex_worldbank(
    countries: list[str] | None = None,
    start_year: int = 2010,
) -> pd.DataFrame:
    """Fallback: fetch SIPRI data via World Bank API."""
    import wbgapi as wb

    countries = countries or [
        "USA", "CHN", "RUS", "GBR", "FRA", "DEU", "JPN", "IND",
        "SAU", "KOR", "AUS", "ISR", "BRA", "ITA", "TUR",
    ]

    df = wb.data.DataFrame(
        WB_MILEX_INDICATOR,
        economy=countries,
        time=range(start_year, 2025),
        labels=False,
        numericTimeKeys=True,
    )
    df = df.stack().reset_index()
    df.columns = ["country", "year", "milex_usd"]
    df["year"] = df["year"].astype(int)
    return df


def fetch_milex(countries: list[str] | None = None) -> pd.DataFrame:
    """Fetch military expenditure, trying SIPRI direct then World Bank fallback."""
    try:
        return fetch_milex_direct()
    except Exception as e:
        print(f"[SIPRI] Direct fetch failed ({e}), falling back to World Bank")
        return fetch_milex_worldbank(countries)


def to_daily_features(milex: pd.DataFrame) -> pd.DataFrame:
    """Expand annual military expenditure to daily frequency.

    Computes year-over-year change rate as the primary feature.
    """
    frames = []
    for country in milex["country"].unique():
        cdf = milex[milex["country"] == country].sort_values("year").copy()
        cdf["milex_yoy_change"] = cdf["milex_usd"].pct_change()
        cdf["date"] = pd.to_datetime(cdf["year"].astype(str) + "-01-01")
        cdf = cdf.set_index("date")[["milex_usd", "milex_yoy_change"]]
        cdf = cdf.resample("D").ffill()
        cdf = cdf.rename(columns={c: f"{country}_{c}" for c in cdf.columns})
        frames.append(cdf)

    return pd.concat(frames, axis=1) if frames else pd.DataFrame()
