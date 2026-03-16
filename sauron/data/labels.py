"""Label utilities — convenience re-exports and validation."""

from sauron.data.sources.yfinance_labels import (
    compute_tendency_labels,
    fetch_etf_prices,
    load_baskets,
)

__all__ = ["load_baskets", "fetch_etf_prices", "compute_tendency_labels"]


def validate_labels(labels_df, min_coverage: float = 0.5) -> dict[str, float]:
    """Check label quality: coverage per sector/horizon.

    Returns dict of column_name -> fraction of non-NaN values.
    Warns if any column has less than min_coverage.
    """
    coverage = {}
    for col in labels_df.columns:
        frac = labels_df[col].notna().mean()
        coverage[col] = frac
        if frac < min_coverage:
            print(f"[Labels] Warning: {col} has only {frac:.1%} coverage")
    return coverage
