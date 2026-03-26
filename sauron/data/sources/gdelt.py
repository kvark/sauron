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

from datetime import date, datetime, timedelta

import pandas as pd


def _to_str(d) -> str:
    """Convert date/datetime to string if needed."""
    if isinstance(d, (date, datetime)):
        return d.strftime("%Y-%m-%d")
    return str(d)

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
    """Fetch GDELT events pre-aggregated to daily sector features via BigQuery.

    Aggregation happens server-side to avoid downloading hundreds of millions of
    raw events. Returns a DataFrame with the same schema as
    aggregate_daily_sector_features() — one row per day, with per-sector columns.

    Requires google-cloud-bigquery and a GCP project (free tier is sufficient).
    """
    from google.cloud import bigquery

    start_date = _to_str(start_date)
    end_date = _to_str(end_date) if end_date else datetime.now().strftime("%Y-%m-%d")
    start_int = start_date.replace("-", "")
    end_int = end_date.replace("-", "")

    # Build sector CASE expressions from CAMEO_SECTOR_MAP
    # Invert: for each sector, collect the CAMEO prefixes
    from sauron.sectors import SECTORS
    sector_cameos: dict[str, list[str]] = {s: [] for s in SECTORS}
    for cameo, sectors in CAMEO_SECTOR_MAP.items():
        for s in sectors:
            if s in sector_cameos:
                sector_cameos[s].append(cameo)

    # Build regime CASE
    regime_codes_str = ", ".join(f"'{c}'" for c in REGIME_SHIFT_CODES)

    # For each sector, build SQL aggregation columns
    sector_sql_parts = []
    for sector, cameos in sector_cameos.items():
        if not cameos:
            continue
        # Build OR conditions matching CAMEO prefixes against EventRootCode and EventCode
        conditions = []
        for cameo in cameos:
            if len(cameo) <= 2:
                conditions.append(f"EventRootCode = '{cameo}'")
            else:
                conditions.append(f"STARTS_WITH(EventCode, '{cameo}')")
        where = " OR ".join(conditions)

        sector_sql_parts.append(f"""
    COUNTIF({where}) AS {sector}_event_count,
    AVG(IF({where}, GoldsteinScale, NULL)) AS {sector}_goldstein_mean,
    AVG(IF({where}, AvgTone, NULL)) AS {sector}_tone_mean,
    SUM(IF({where}, NumMentions, 0)) AS {sector}_mentions_sum,
    MAX(IF(({where}) AND EventCode IN ({regime_codes_str}), 1, 0)) AS {sector}_regime_flag""")

    sector_columns = ",".join(sector_sql_parts)

    query = f"""
    SELECT
        PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) AS date,
        COUNT(*) AS total_events,
        {sector_columns}
    FROM `gdelt-bq.gdeltv2.events`
    WHERE SQLDATE BETWEEN {start_int} AND {end_int}
      AND ABS(GoldsteinScale) >= {min_goldstein_abs}
      AND NumMentions >= 5
    GROUP BY SQLDATE
    ORDER BY SQLDATE
    """

    print(f"[GDELT-BQ] Querying {start_date} to {end_date}...")
    client = bigquery.Client(project=project_id)
    df = client.query(query).to_dataframe()

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Convert nullable integer types to float and fill NaN with 0
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float).fillna(0)

    # Drop the total_events helper column
    df = df.drop(columns=["total_events"], errors="ignore")

    print(f"[GDELT-BQ] Loaded {len(df)} days of sector features")
    return df


# GKG theme-to-sector mapping for news sentiment
GKG_SECTOR_THEMES = {
    "NATRES": [
        "ENV_OIL", "ENV_MINING", "ENV_GAS", "NATURAL_DISASTER",
        "UNGP_FORESTS_RIVERS_OCEANS", "TAX_ECON_PRICE",
    ],
    "GREEN": [
        "ENV_CLIMATECHANGE", "ENV_GREEN", "ENV_SOLAR", "ENV_WIND",
        "ENV_NUCLEARPOWER", "RENEWABLE",
    ],
    "CHIPS": [
        "WB_130_SCIENCE_AND_TECHNOLOGY", "WB_133_INFORMATION_AND_COMMUNICATION_TECHNOLOGIES",
    ],
    "SOFTWARE": [
        "WB_133_INFORMATION_AND_COMMUNICATION_TECHNOLOGIES", "WB_678_DIGITAL_GOVERNMENT",
        "CYBER_ATTACK",
    ],
    "QUANTUM": [
        "WB_130_SCIENCE_AND_TECHNOLOGY",
    ],
    "WEAPONS": [
        "ARMEDCONFLICT", "MILITARY", "SECURITY_SERVICES", "KILL",
        "WB_2432_FRAGILITY_CONFLICT_AND_VIOLENCE", "WB_2433_CONFLICT_AND_VIOLENCE",
        "EPU_CATS_NATIONAL_SECURITY",
    ],
    "EDUCATION": [
        "EDUCATION", "WB_470_EDUCATION", "SOC_POINTSOFINTEREST_SCHOOL",
    ],
    "BIOTECH": [
        "GENERAL_HEALTH", "MEDICAL", "WB_621_HEALTH_NUTRITION_AND_POPULATION",
        "CRISISLEX_C03_WELLBEING_HEALTH",
    ],
    "FINANCE": [
        "EPU_ECONOMY", "EPU_ECONOMY_HISTORIC", "ECON_BANKRUPTCY",
        "WB_2670_JOBS", "TAX_ECON_PRICE",
    ],
    "INFRA": [
        "WB_135_TRANSPORT", "WB_137_WATER", "INFRASTRUCTURE",
    ],
    "AGRI": [
        "FAMINE", "FOOD_SECURITY", "WB_477_AGRICULTURE",
    ],
    "SPACE": [
        "WB_130_SCIENCE_AND_TECHNOLOGY",
    ],
}


def fetch_gkg_sentiment(
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    project_id: str | None = None,
) -> pd.DataFrame:
    """Fetch daily news sentiment per sector from GDELT GKG via BigQuery.

    Aggregates V2Tone (article-level sentiment) grouped by sector-relevant themes.
    Returns daily features: {sector}_news_tone, {sector}_news_volume, {sector}_news_polarity.

    Note: GKG scans ~130 GB/year. Default start is 2020 to stay within free tier.
    """
    from google.cloud import bigquery

    # GKG is expensive — limit to a reasonable range
    start_date = _to_str(start_date)
    end_date = _to_str(end_date) if end_date else datetime.now().strftime("%Y-%m-%d")
    # Clamp start to no earlier than 2020 to avoid blowing BQ quota
    if start_date < "2020-01-01":
        start_date = "2020-01-01"

    # Build CASE WHEN for sector classification based on themes
    sector_cases = []
    for sector, themes in GKG_SECTOR_THEMES.items():
        theme_conditions = " OR ".join(
            f"theme_name LIKE '%{t}%'" for t in themes
        )
        sector_cases.append(f"WHEN {theme_conditions} THEN '{sector}'")

    sector_case_sql = "CASE " + " ".join(sector_cases) + " ELSE NULL END"

    query = f"""
    WITH articles AS (
        SELECT
            PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS date,
            CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
            CAST(SPLIT(V2Tone, ',')[OFFSET(3)] AS FLOAT64) AS polarity,
            V2Themes
        FROM `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE _PARTITIONTIME BETWEEN '{start_date}' AND '{end_date}'
          AND V2Tone IS NOT NULL
          AND V2Themes IS NOT NULL
    ),
    article_themes AS (
        SELECT
            a.date,
            a.tone,
            a.polarity,
            {sector_case_sql} AS sector
        FROM articles a,
        UNNEST(SPLIT(a.V2Themes, ';')) AS raw_theme
        CROSS JOIN UNNEST([SPLIT(raw_theme, ',')[OFFSET(0)]]) AS theme_name
    )
    SELECT
        date,
        sector,
        AVG(tone) AS avg_tone,
        COUNT(*) AS volume,
        AVG(polarity) AS avg_polarity
    FROM article_themes
    WHERE sector IS NOT NULL
    GROUP BY date, sector
    ORDER BY date, sector
    """

    print(f"[GKG-BQ] Querying news sentiment {start_date} to {end_date}...")
    client = bigquery.Client(project=project_id)
    df = client.query(query).to_dataframe()

    if df.empty:
        print("[GKG-BQ] No data returned")
        return pd.DataFrame()

    # Pivot: one row per day, columns = {sector}_{metric}
    df["date"] = pd.to_datetime(df["date"])
    pivoted = df.pivot_table(
        index="date",
        columns="sector",
        values=["avg_tone", "volume", "avg_polarity"],
        aggfunc="first",
    )
    # Flatten multi-level columns
    pivoted.columns = [f"{sector}_news_{metric}" for metric, sector in pivoted.columns]
    pivoted = pivoted.sort_index()

    # Fill missing days with 0
    for col in pivoted.columns:
        pivoted[col] = pd.to_numeric(pivoted[col], errors="coerce").fillna(0).astype(float)

    print(f"[GKG-BQ] Loaded {len(pivoted)} days, {len(pivoted.columns)} news features")
    return pivoted


def fetch_gdelt_csv(
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    min_goldstein_abs: float = 3.0,
    max_days: int | None = None,
) -> pd.DataFrame:
    """Fetch GDELT events via CSV download (no GCP account needed).

    Downloads daily event files from GDELT's public URL.
    Slower than BigQuery but requires no setup.

    Args:
        max_days: If set, only fetch the most recent N days instead of the full range.
    """
    import io

    import requests

    start_date = _to_str(start_date)
    end_date = _to_str(end_date) if end_date else datetime.now().strftime("%Y-%m-%d")
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    if max_days is not None:
        start = max(start, end - timedelta(days=max_days))

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
