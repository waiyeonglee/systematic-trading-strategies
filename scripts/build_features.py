#!/usr/bin/env python3
"""Build the feature panel for trend-following-strategy-v2.

Reads ``data/raw/weekly/*.csv`` (16 tradables) plus SPY and
``data/additional/fear-greed.csv``, computes the 21 features + the forward-return
label, and writes:

    data/features/<frequency>/features.csv                  (label rows kept, NaN allowed)
    data/features/<frequency>/feature_label_correlation.csv
    data/features/<frequency>/feature_label_correlation.png

Defaults encode the v2 conventions (weekly bars, Fear & Greed on, H = 12 weeks).
The dataset is frozen at the fixed end date 2026-01-01 (see src/utils/config.py).

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --no-fear-greed
    python scripts/build_features.py --horizon 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # src modules import each other by bare name

import pandas as pd

from features import (build_feature_frame, feature_columns, feature_label_correlation,
                      save_correlation_heatmap)
from labels import label_column
from utils.config import BENCHMARKS, TRADABLE_ETFS
from utils.checks import require_columns
from utils.paths import FEAR_GREED_PATH, FEATURES_DIR, raw_data_dir

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def load_ohlcv(folder: Path, tickers: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Load ticker CSVs previously written by the fetch step."""
    folder = Path(folder)
    prices = {}
    for ticker in tickers:
        path = folder / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run scripts/fetch_data.py first")
        bars = pd.read_csv(path, parse_dates=["Date"]).drop_duplicates("Date", keep="last")
        bars = bars.set_index("Date").sort_index()
        require_columns(bars, OHLCV, str(path))
        prices[ticker] = bars[OHLCV].astype(float)
    return prices


def load_fear_greed_index(path: Path = FEAR_GREED_PATH) -> pd.Series:
    """Load the dated numeric Fear & Greed Index (0-100)."""
    if not path.exists():
        raise FileNotFoundError(f"missing Fear & Greed data: {path}")
    data = pd.read_csv(path, parse_dates=["Date"])
    require_columns(data, ["Date", "Fear Greed"], str(path))
    data = data.drop_duplicates("Date", keep="last").sort_values("Date")
    values = pd.to_numeric(data["Fear Greed"], errors="coerce")
    return pd.Series(values.to_numpy(), index=pd.DatetimeIndex(data["Date"]), name="fear_greed_index")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", default="weekly", choices=("daily", "weekly"),
                        help="bar frequency of the raw CSVs (default: weekly)")
    parser.add_argument("--no-fear-greed", action="store_true",
                        help="omit the Fear & Greed index feature (20 features)")
    parser.add_argument("--horizon", type=int, default=12,
                        help="label look-ahead in bars (default: 12)")
    args = parser.parse_args()

    frequency = args.frequency
    use_fear_greed = not args.no_fear_greed
    horizon = args.horizon
    columns = feature_columns(use_fear_greed)
    label_col = label_column(horizon)

    fear_greed = load_fear_greed_index() if use_fear_greed else None
    raw_dir = raw_data_dir(frequency)
    prices = load_ohlcv(raw_dir, TRADABLE_ETFS)
    benchmark = load_ohlcv(raw_dir, BENCHMARKS)[BENCHMARKS[0]]

    features = build_feature_frame(prices, frequency=frequency, sectors=TRADABLE_ETFS,
                                   fear_greed_index=fear_greed,
                                   benchmark_prices=benchmark,
                                   label_horizon=horizon)
    correlation, sample_end = feature_label_correlation(features, columns, label_col=label_col)

    output_dir = FEATURES_DIR / frequency
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "features.csv")
    correlation.to_csv(output_dir / "feature_label_correlation.csv")
    save_correlation_heatmap(
        correlation,
        output_dir / "feature_label_correlation.png",
        f"{frequency.title()} Feature/Label Correlation "
        f"(All Labeled Samples through {sample_end.date()})",
    )
    print(f"{frequency.title()}: {features.shape[0]:,} rows x {len(columns)} features; "
          f"labels through {sample_end.date()}; saved to {output_dir}")


if __name__ == "__main__":
    main()
