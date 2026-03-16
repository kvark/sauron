"""GDELT (Global Database of Events, Language, and Tone) connector.

Fetches geopolitical events and aggregates them into daily sector-relevant features.
Uses BigQuery (free tier) for efficient querying of the massive GDELT dataset.

GDELT fields we care about:
- SQLDATE: event date
- EventCode: CAMEO event code (e.g., 14 = protest, 19 = military force)
- GoldsteinScale: [-10, 10] impact score
- NumMentions: media coverage volume
- AvgTone: [-100, 100] sentiment
- Actor1CountryCode, Actor2CountryCode: countries involved
- ActionGeo_Lat/Long: where it happened
"""

from datetime import datetime, timedelta

import pandas as pd

# CAMEO root codes mapped to our sector relevance
CAMEO_SECTOR_MAP = {
    # Economic events
    "03": ["FINANCE", "SOFTWARE"],    # Express intent to cooperate economically
    "06": ["FINANCE", "INFRA"],       # Material cooperation
    "061": ["FINANCE"],               # Economic cooperation
    "0612": ["FINANCE"],              # Provide economic aid
    # Conflict events
    "14": ["WEAPONS", "NATRES"],      # Protest
    "17": ["WEAPONS"],                # Coerce
    "18": ["WEAPONS", "NATRES"],      # Assault
    "19": ["WEAPONS"],                # Fight / military force
    "190": ["WEAPONS", "NATRES"],     # Use conventional military force
    # Diplomatic events
    "04": ["FINANCE", "SOFTWARE"],    # Consult
    "05": ["FINANCE", "GREEN"],       # Diplomatic cooperation
    "050": ["GREEN", "NATRES"],       # Engage in diplomatic cooperation on environment
    # Trade events
    "0831": ["CHIPS", "SOFTWARE"],    # Impose trade restrictions (tech)
    "0832": ["NATRES", "AGRI"],       # Impose trade restrictions (commodities)
    # Sanctions
    "163": ["FINANCE", "NATRES", "CHIPS"],  # Impose sanctions
}

# High-impact CAMEO codes for regime detection
REGIME_SHIFT_CODES = {
    "190", "191", "192", "193", "194", "195",   # Military force
    "163", "164",                                 # Sanctions
    "0841", "0842",                               # Trade embargo
    "160", "161", "162",                          # Reduce/stop relations
}


def fetch_gdelt_bigquery(
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    min_goldstein_abs: float = 3.0,
    project_id: str | None = None,
) -> pd.DataFrame:
    """Fetch GDELT events from BigQuery.

    Requires google-cloud-bigquery and a GCP project (free tier is sufficient).
    Returns raw events filtered by impact threshold.
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        raise ImportError("pip install google-cloud-bigquery")

    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    start_int = start_date.replace("-", "")
    end_int = end_date.replace("-", "")

    query = f"""
    SELECT
        SQLDATE,
        EventCode,
        EventRootCode,
        GoldsteinScale,
        NumMentions,
        NumSources,
        AvgTone,
        Actor1CountryCode,
        Actor2CountryCode,
        ActionGeo_CountryCode,
        ActionGeo_Lat,
        ActionGeo_Long
    FROM `gdelt-bq.gdeltv2.events`
    WHERE SQLDATE BETWEEN {start_int} AND {end_int}
      AND ABS(GoldsteinScale) >= {min_goldstein_abs}
      AND NumMentions >= 5
    """

    client = bigquery.Client(project=project_id)
    df = client.query(query).to_dataframe()

    df["date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d")
    return df


def fetch_gdelt_csv(
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    min_goldstein_abs: float = 3.0,
) -> pd.DataFrame:
    """Fetch GDELT events via CSV download (no GCP account needed).

    Downloads daily event files from GDELT's public URL.
    Slower than BigQuery but requires no setup.
    """
    import io

    import requests

    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    cols = [
        "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
        "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
        "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
        "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
        "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
        "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
        "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
        "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
        "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles",
        "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
        "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long",
        "Actor1Geo_FeatureID",
        "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
        "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat", "Actor2Geo_Long",
        "Actor2Geo_FeatureID",
        "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
        "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long",
        "ActionGeo_FeatureID",
        "DATEADDED", "SOURCEURL",
    ]

    frames = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV.zip"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                import zipfile
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    for name in z.namelist():
                        with z.open(name) as f:
                            df = pd.read_csv(
                                f, sep="\t", header=None, names=cols,
                                dtype=str, on_bad_lines="skip",
                            )
                            frames.append(df)
        except Exception as e:
            print(f"[GDELT] Warning: failed to fetch {date_str}: {e}")

        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
    df["NumMentions"] = pd.to_numeric(df["NumMentions"], errors="coerce")
    df["AvgTone"] = pd.to_numeric(df["AvgTone"], errors="coerce")
    df["date"] = pd.to_datetime(df["Day"], format="%Y%m%d", errors="coerce")

    # Filter by impact
    df = df[df["GoldsteinScale"].abs() >= min_goldstein_abs]
    return df


def aggregate_daily_sector_features(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw GDELT events into daily per-sector features.

    Output columns per sector:
    - {SECTOR}_event_count: number of relevant events
    - {SECTOR}_goldstein_mean: average impact score
    - {SECTOR}_tone_mean: average sentiment
    - {SECTOR}_mentions_sum: total media coverage
    - {SECTOR}_regime_flag: binary flag for regime-shift-level events
    """
    from sauron.sectors import SECTORS

    events = events.copy()
    events["EventRootCode"] = events.get("EventRootCode", events.get("EventBaseCode", ""))

    records = []
    for date, day_events in events.groupby("date"):
        row = {"date": date}

        for sector_token in SECTORS:
            # Find events relevant to this sector via CAMEO mapping
            mask = pd.Series(False, index=day_events.index)
            for cameo, sectors in CAMEO_SECTOR_MAP.items():
                if sector_token in sectors:
                    code_col = "EventRootCode" if len(cameo) <= 2 else "EventCode"
                    mask |= day_events[code_col].astype(str).str.startswith(cameo)

            sector_events = day_events[mask]

            row[f"{sector_token}_event_count"] = len(sector_events)
            row[f"{sector_token}_goldstein_mean"] = (
                sector_events["GoldsteinScale"].mean() if len(sector_events) > 0 else 0
            )
            row[f"{sector_token}_tone_mean"] = (
                sector_events["AvgTone"].mean() if len(sector_events) > 0 else 0
            )
            row[f"{sector_token}_mentions_sum"] = (
                sector_events["NumMentions"].sum() if len(sector_events) > 0 else 0
            )

            # Regime shift detection: any high-impact event codes?
            regime_events = sector_events[
                sector_events["EventCode"].astype(str).isin(REGIME_SHIFT_CODES)
            ]
            row[f"{sector_token}_regime_flag"] = 1 if len(regime_events) > 0 else 0

        records.append(row)

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    return df
