"""Data pipeline: merge, align, and normalize all data sources into training-ready datasets."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def align_to_daily(
    *dataframes: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Merge multiple DataFrames with DatetimeIndex into a single daily-aligned frame.

    - Forward-fills missing values (appropriate for economic data that updates infrequently)
    - Clips to common date range unless start/end specified
    """
    if not dataframes:
        return pd.DataFrame()

    merged = dataframes[0]
    for df in dataframes[1:]:
        merged = merged.join(df, how="outer")

    # Ensure daily frequency
    if not merged.empty:
        merged = merged.resample("D").ffill()

    if start:
        merged = merged[merged.index >= pd.Timestamp(start)]
    if end:
        merged = merged[merged.index <= pd.Timestamp(end)]

    return merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features (momentum, volatility, GDELT smoothing) to the raw feature DataFrame.

    Operates on pre-normalization data. Only computes rolling features for columns
    with >50% non-null values. Keeps all original columns alongside derived ones.

    Derived columns:
    - {col}_mom7:  7-day percentage change (momentum)
    - {col}_mom30: 30-day percentage change (momentum)
    - {col}_vol14: 14-day rolling standard deviation (volatility)
    - {col}_smooth7: 7-day rolling mean (GDELT columns only)
    """
    original_cols = list(df.columns)
    n_rows = len(df)
    result = df.copy()

    # Identify numeric columns with enough data (>80% non-null)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    eligible_cols = [c for c in numeric_cols if df[c].notna().sum() > 0.8 * n_rows]

    # GDELT sector feature suffixes — these are the noisiest, benefit most from smoothing
    gdelt_suffixes = ("_event_count", "_goldstein_mean", "_tone_mean", "_mentions_sum")
    gdelt_cols = [c for c in eligible_cols if c.endswith(gdelt_suffixes)]

    # Non-GDELT columns: FRED, EIA macro series (high-quality, daily)
    macro_cols = [c for c in eligible_cols if c not in gdelt_cols and not c.endswith("_regime_flag")]

    added = 0

    # Momentum: 7-day for macro columns only (not GDELT — too noisy)
    if macro_cols:
        pct = df[macro_cols].pct_change(periods=7, fill_method=None)
        # Replace inf values from division by zero
        pct = pct.replace([np.inf, -np.inf], np.nan)
        pct.columns = [f"{c}_mom7" for c in macro_cols]
        result = pd.concat([result, pct], axis=1)
        added += len(macro_cols)

    # Volatility: 14-day rolling std for macro columns
    if macro_cols:
        vol = df[macro_cols].rolling(window=14, min_periods=7).std()
        vol.columns = [f"{c}_vol14" for c in macro_cols]
        result = pd.concat([result, vol], axis=1)
        added += len(macro_cols)

    # GDELT smoothing: 7-day rolling mean to reduce daily noise
    if gdelt_cols:
        smooth = df[gdelt_cols].rolling(window=7, min_periods=3).mean()
        smooth.columns = [f"{c}_smooth7" for c in gdelt_cols]
        result = pd.concat([result, smooth], axis=1)
        added += len(gdelt_cols)

    print(f"[Pipeline] Feature engineering: {len(original_cols)} original + {added} derived "
          f"= {len(result.columns)} total features")
    return result


def normalize_features(df: pd.DataFrame, method: str = "zscore") -> tuple[pd.DataFrame, dict]:
    """Normalize features, returning the normalized df and stats for inverse transform.

    Args:
        df: DataFrame to normalize
        method: 'zscore' or 'minmax'

    Returns:
        (normalized_df, stats_dict) where stats_dict has keys per column
    """
    stats = {}
    result = df.copy()

    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            stats[col] = {"method": method, "mean": 0, "std": 1}
            continue

        if method == "zscore":
            mean = series.mean()
            std = series.std()
            if std < 1e-8:
                std = 1.0
            result[col] = (df[col] - mean) / std
            stats[col] = {"method": "zscore", "mean": float(mean), "std": float(std)}
        elif method == "minmax":
            lo = series.min()
            hi = series.max()
            rng = hi - lo
            if rng < 1e-8:
                rng = 1.0
            result[col] = (df[col] - lo) / rng
            stats[col] = {"method": "minmax", "min": float(lo), "range": float(rng)}

    return result, stats


def create_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Create a binary mask: 1 where data is observed, 0 where it was missing/filled."""
    return df.notna().astype(np.float32)


def build_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    lookback_days: int = 90,
    horizon_days: int | list[int] = 90,
) -> list[dict]:
    """Build windowed samples for training.

    Each sample is a dict with:
    - 'features': (lookback_days, num_features) array
    - 'mask': (lookback_days, num_features) binary mask
    - 'labels': dict of sector -> tendency score (primary horizon)
    - 'multi_labels': dict of horizon -> dict of sector -> tendency (all horizons)
    - 'date': the prediction date

    If horizon_days is a list, trains on all horizons (multi-horizon mode).
    The primary 'labels' uses the first horizon.
    """
    horizons = [horizon_days] if isinstance(horizon_days, int) else horizon_days
    primary_horizon = horizons[0]

    # Align features and labels to same dates
    common_dates = features.index.intersection(labels.index)
    features = features.loc[common_dates]
    labels = labels.loc[common_dates]

    # Collect columns per horizon
    horizon_col_map = {}
    for h in horizons:
        horizon_col_map[h] = [c for c in labels.columns if c.endswith(f"_{h}d")]

    primary_cols = horizon_col_map[primary_horizon]

    samples = []
    for i in range(lookback_days, len(features)):
        date = features.index[i]

        if date not in labels.index:
            continue

        label_row = labels.loc[date, primary_cols]
        if label_row.isna().all():
            continue

        feat_window = features.iloc[i - lookback_days : i]

        # Primary labels (for backward compat)
        primary_labels = {
            col.replace(f"_{primary_horizon}d", ""): float(label_row[col])
            for col in primary_cols
            if not pd.isna(label_row[col])
        }

        # Multi-horizon labels
        multi = {}
        for h in horizons:
            h_row = labels.loc[date, horizon_col_map[h]]
            multi[h] = {
                col.replace(f"_{h}d", ""): float(h_row[col])
                for col in horizon_col_map[h]
                if not pd.isna(h_row[col])
            }

        samples.append({
            "features": np.array(feat_window.values, dtype=np.float32),
            "mask": create_mask(feat_window).values,
            "labels": primary_labels,
            "multi_labels": multi,
            "date": date,
        })

    return samples


class SauronDataset:
    """Full pipeline: fetch all sources, merge, normalize, and build samples."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = load_config(config_path)

    def fetch_all_features(
        self, start: str = "2015-01-01", hf_only: bool = False,
    ) -> pd.DataFrame:
        """Fetch and merge all data sources into a single feature DataFrame.

        Args:
            start: Start date for data fetching.
            hf_only: If True, skip API sources and only use HuggingFace datasets.
        """
        frames = []

        if not hf_only:
            # FRED macro data
            try:
                from sauron.data.sources.fred import fetch_default as fetch_fred
                frames.append(fetch_fred(start=start))
                print("[Pipeline] FRED data loaded")
            except Exception as e:
                print(f"[Pipeline] FRED skipped: {e}")

            # EIA energy data
            try:
                from sauron.data.sources.eia import fetch_default as fetch_eia
                frames.append(fetch_eia(start=start))
                print("[Pipeline] EIA data loaded")
            except Exception as e:
                print(f"[Pipeline] EIA skipped: {e}")

        # GDELT event features — prefer BigQuery (full history), fall back to HF/CSV
        gdelt_loaded = False
        if not hf_only:
            try:
                from sauron.data.sources.gdelt import fetch_gdelt_bigquery
                gdelt_features = fetch_gdelt_bigquery(start_date=start)
                if not gdelt_features.empty:
                    frames.append(gdelt_features)
                    print(f"[Pipeline] GDELT data loaded (BigQuery)")
                    gdelt_loaded = True
            except Exception as e:
                print(f"[Pipeline] GDELT BigQuery failed: {e}")

        if not gdelt_loaded:
            try:
                from sauron.data.sources.gdelt import aggregate_daily_sector_features
                from sauron.data.sources.huggingface import fetch_gdelt_hf
                raw_gdelt = fetch_gdelt_hf()
                if not raw_gdelt.empty:
                    frames.append(aggregate_daily_sector_features(raw_gdelt))
                    print("[Pipeline] GDELT data loaded (HuggingFace)")
                    gdelt_loaded = True
            except Exception as e:
                print(f"[Pipeline] GDELT HuggingFace failed: {e}")

        # World Bank development indicators — prefer HuggingFace, fall back to wbgapi
        wb_loaded = False
        try:
            from sauron.data.sources.huggingface import fetch_wdi_hf, wdi_to_daily_features
            wb_data = fetch_wdi_hf(start_year=int(start[:4]))
            frames.append(wdi_to_daily_features(wb_data))
            print("[Pipeline] World Bank data loaded (HuggingFace)")
            wb_loaded = True
        except Exception as e:
            print(f"[Pipeline] World Bank HuggingFace failed: {e}")

        if not wb_loaded and not hf_only:
            try:
                from sauron.data.sources.worldbank import fetch_indicators, to_daily_features
                wb_data = fetch_indicators(start_year=int(start[:4]))
                frames.append(to_daily_features(wb_data))
                print("[Pipeline] World Bank data loaded (wbgapi)")
            except Exception as e:
                print(f"[Pipeline] World Bank wbgapi skipped: {e}")

        # SIPRI military expenditure
        try:
            from sauron.data.sources.sipri import fetch_milex, to_daily_features as sipri_daily
            milex = fetch_milex()
            sipri_df = sipri_daily(milex)
            if not sipri_df.empty:
                frames.append(sipri_df)
                print("[Pipeline] SIPRI data loaded")
        except Exception as e:
            print(f"[Pipeline] SIPRI skipped: {e}")

        if not frames:
            raise RuntimeError("No data sources available. Check API keys and network.")

        return align_to_daily(*frames, start=start)

    def fetch_labels(self, start: str = "2015-01-01") -> pd.DataFrame:
        """Fetch ETF-based sector tendency labels."""
        from sauron.data.sources.yfinance_labels import compute_tendency_labels, load_baskets

        baskets = load_baskets()
        horizons = self.config["data"]["horizons"]
        return compute_tendency_labels(baskets, horizons_days=horizons, start=start)

    def build(
        self,
        start: str = "2015-01-01",
        horizon_days: int | list[int] = 90,
        hf_only: bool = False,
    ) -> list[dict]:
        """Full pipeline: fetch everything and build training samples."""
        print("[Pipeline] Fetching features...")
        features = self.fetch_all_features(start=start, hf_only=hf_only)

        print("[Pipeline] Fetching labels...")
        labels = self.fetch_labels(start=start)

        print("[Pipeline] Engineering derived features...")
        features = engineer_features(features)

        print("[Pipeline] Normalizing features...")
        features_norm, self.feature_stats = normalize_features(features)

        lookback = self.config["data"]["lookback_days"]
        print(f"[Pipeline] Building samples (lookback={lookback}, horizon={horizon_days})...")
        samples = build_dataset(features_norm, labels, lookback, horizon_days)

        print(f"[Pipeline] Built {len(samples)} samples")
        return samples
