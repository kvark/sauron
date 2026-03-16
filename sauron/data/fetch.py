"""CLI entrypoint for data fetching.

Usage:
    python -m sauron.data.fetch                  # fetch all sources
    python -m sauron.data.fetch --sources fred eia  # specific sources only
    python -m sauron.data.fetch --start 2020-01-01  # custom start date
    python -m sauron.data.fetch --synthetic         # generate synthetic data (offline testing)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

ALL_SOURCES = ["fred", "eia", "gdelt", "worldbank", "sipri", "yfinance"]


def fetch_source(name: str, start: str) -> pd.DataFrame | None:
    """Fetch a single data source by name. Returns DataFrame or None on failure."""
    try:
        if name == "fred":
            from sauron.data.sources.fred import fetch_default
            return fetch_default(start=start)

        elif name == "eia":
            from sauron.data.sources.eia import fetch_default
            return fetch_default(start=start)

        elif name == "gdelt":
            from sauron.data.sources.gdelt import aggregate_daily_sector_features, fetch_gdelt_csv
            raw = fetch_gdelt_csv(start_date=start)
            if raw.empty:
                print(f"[{name}] No events returned")
                return None
            return aggregate_daily_sector_features(raw)

        elif name == "worldbank":
            from sauron.data.sources.worldbank import fetch_indicators, to_daily_features
            wb = fetch_indicators(start_year=int(start[:4]))
            return to_daily_features(wb)

        elif name == "sipri":
            from sauron.data.sources.sipri import fetch_milex, to_daily_features
            milex = fetch_milex()
            df = to_daily_features(milex)
            return df if not df.empty else None

        elif name == "yfinance":
            from sauron.data.sources.yfinance_labels import compute_tendency_labels, load_baskets
            baskets = load_baskets()
            return compute_tendency_labels(baskets, start=start)

        else:
            print(f"[{name}] Unknown source")
            return None

    except Exception as e:
        print(f"[{name}] FAILED: {e}")
        return None


def generate_synthetic(start: str = "2020-01-01", days: int = 1500) -> dict[str, pd.DataFrame]:
    """Generate synthetic data matching the schema of real sources.

    Useful for offline testing when APIs are unavailable.
    """
    from sauron.sectors import SECTORS

    rng = np.random.default_rng(42)
    dates = pd.date_range(start, periods=days, freq="D")

    # Synthetic macro features (FRED-like)
    fred_cols = {
        "GDP": 22_000 + np.cumsum(rng.normal(5, 20, days)),
        "CPIAUCSL": 280 + np.cumsum(rng.normal(0.02, 0.3, days)),
        "FEDFUNDS": np.clip(3.0 + np.cumsum(rng.normal(0, 0.01, days)), 0, 10),
        "UNRATE": np.clip(4.0 + np.cumsum(rng.normal(0, 0.05, days)), 2, 15),
        "DGS10": np.clip(3.5 + np.cumsum(rng.normal(0, 0.01, days)), 0.5, 8),
        "DTWEXBGS": 110 + np.cumsum(rng.normal(0, 0.1, days)),
        "M2SL": 20_000 + np.cumsum(rng.normal(3, 15, days)),
        "INDPRO": 100 + np.cumsum(rng.normal(0.01, 0.2, days)),
        "HOUST": np.clip(1400 + np.cumsum(rng.normal(0, 10, days)), 500, 2500),
        "DCOILWTICO": np.clip(70 + np.cumsum(rng.normal(0, 0.5, days)), 20, 150),
    }
    fred_df = pd.DataFrame(fred_cols, index=dates)
    fred_df.index.name = "date"

    # Synthetic energy features (EIA-like)
    eia_cols = {
        "wti_crude_daily": np.clip(70 + np.cumsum(rng.normal(0, 0.5, days)), 20, 150),
        "brent_crude_daily": np.clip(75 + np.cumsum(rng.normal(0, 0.5, days)), 25, 155),
        "henry_hub_natgas_daily": np.clip(3.0 + np.cumsum(rng.normal(0, 0.02, days)), 1, 10),
        "us_electricity_gen_monthly": 350 + rng.normal(0, 20, days),
        "us_solar_gen_monthly": 15 + np.cumsum(rng.normal(0.01, 0.1, days)),
        "us_wind_gen_monthly": 30 + np.cumsum(rng.normal(0.01, 0.1, days)),
        "world_oil_production_monthly": 100 + rng.normal(0, 1, days),
    }
    eia_df = pd.DataFrame(eia_cols, index=dates)
    eia_df.index.name = "date"

    # Synthetic GDELT-like sector features
    gdelt_cols = {}
    for sector in SECTORS:
        gdelt_cols[f"{sector}_event_count"] = rng.poisson(5, days).astype(float)
        gdelt_cols[f"{sector}_goldstein_mean"] = rng.normal(0, 3, days)
        gdelt_cols[f"{sector}_tone_mean"] = rng.normal(-1, 2, days)
        gdelt_cols[f"{sector}_mentions_sum"] = rng.poisson(50, days).astype(float)
        gdelt_cols[f"{sector}_regime_flag"] = (rng.random(days) < 0.02).astype(float)
    gdelt_df = pd.DataFrame(gdelt_cols, index=dates)
    gdelt_df.index.name = "date"

    # Synthetic labels (yfinance-like)
    label_cols = {}
    for sector in SECTORS:
        for horizon in [30, 90, 180]:
            label_cols[f"{sector}_{horizon}d"] = np.tanh(rng.normal(0, 0.5, days))
    labels_df = pd.DataFrame(label_cols, index=dates)
    labels_df.index.name = "date"

    return {
        "fred": fred_df,
        "eia": eia_df,
        "gdelt": gdelt_df,
        "yfinance": labels_df,
    }


def save_dataframe(df: pd.DataFrame, name: str, output_dir: Path) -> Path:
    """Save DataFrame to parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.parquet"
    df.to_parquet(path)
    print(f"  Saved {name}: {df.shape} -> {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Sauron data fetcher")
    parser.add_argument(
        "--sources", nargs="+", default=ALL_SOURCES,
        help=f"Sources to fetch (default: all). Choices: {ALL_SOURCES}",
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data instead")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.synthetic:
        print("Generating synthetic data...")
        datasets = generate_synthetic(start=args.start)
        for name, df in datasets.items():
            save_dataframe(df, name, output_dir)
        print(f"\nSynthetic data saved to {output_dir}/")

        # Also build a merged feature set + labels
        from sauron.data.pipeline import align_to_daily, normalize_features
        feature_frames = [df for n, df in datasets.items() if n != "yfinance"]
        features = align_to_daily(*feature_frames, start=args.start)
        features_norm, stats = normalize_features(features)
        save_dataframe(features_norm, "features_normalized", PROCESSED_DIR)
        save_dataframe(datasets["yfinance"], "labels", PROCESSED_DIR)

        # Save normalization stats
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        stats_path = PROCESSED_DIR / "norm_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved norm stats -> {stats_path}")
        print(f"\nProcessed data saved to {PROCESSED_DIR}/")
        return

    print(f"Fetching data from {len(args.sources)} sources (start={args.start})...")
    results = {}
    for source in args.sources:
        print(f"\n--- {source.upper()} ---")
        df = fetch_source(source, args.start)
        if df is not None and not df.empty:
            results[source] = df
            save_dataframe(df, source, output_dir)
        else:
            print(f"  No data returned for {source}")

    if not results:
        print("\nNo data fetched from any source. Try --synthetic for offline testing.")
        sys.exit(1)

    print(f"\n=== Summary ===")
    for name, df in results.items():
        print(f"  {name}: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Total sources: {len(results)}/{len(args.sources)}")


if __name__ == "__main__":
    main()
