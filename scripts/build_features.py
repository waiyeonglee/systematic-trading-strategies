#!/usr/bin/env python3
"""Build both the micro and the macro feature panels in one run.

Reads ``data/raw/weekly/*.csv`` (16 tradables) plus SPY and the Welch-Goyal workbook,
computes the firm features + the forward-return labels, and writes:

    data/features/<frequency>/features.csv                  (label rows kept, NaN allowed)
    data/features/<frequency>/feature_label_correlation.csv
    data/features/<frequency>/feature_label_correlation.png
    data/features/weekly/features_macro.csv                 (114 interaction columns)

``features_macro.csv`` is the macro block: 19 firm features x (5 macro + const) = 114
interaction columns (named ``<firm>__x__<macro>``), the five raw macro levels
(``macro_*``), ``fwd_ret`` and the 1/4/12-bar labels. The ``micro`` block reads the 19
firm features from the same frame. It is ~30 MB and git-ignored: the study rebuilds
the panel in memory (:mod:`macro_panel`), so the CSV is an inspection artifact. Pass
``--no-macro`` to skip it.

Defaults encode the project conventions (weekly bars, H = 12 weeks).
The dataset is frozen at the fixed end date 2026-01-01 (see src/utils/config.py).

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --horizon 4
    python scripts/build_features.py --no-macro            # features.csv only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # src modules import each other by bare name
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # macro_panel imports its siblings bare

import pandas as pd

from features import (build_feature_frame, feature_columns, feature_label_correlation,
                      save_correlation_heatmap)
from labels import MODEL_TARGETS, label_column, make_labels
from utils.config import BENCHMARKS, TRADABLE_ETFS
from utils.checks import require_columns
from utils.paths import FEATURES_DIR, raw_data_dir

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


def write_macro_panel(out: Path) -> None:
    """Write the 114-column macro-interaction panel (inspection artifact)."""
    from macro_panel import build_panel, macro_levels

    panel, _, feature_cols = build_panel()
    frame = pd.concat([panel, macro_levels(panel)], axis=1).reset_index()
    frame = frame[["date", "ticker", *feature_cols,
                   *[column for column in frame.columns if column.startswith("macro_")],
                   "fwd_ret", "fwd_ret_1bar", "fwd_ret_4bar", "fwd_ret_12bar"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    dates = panel.index.get_level_values("date")
    print(f"macro: {len(panel):,} rows | dates {dates.min():%Y-%m-%d}..{dates.max():%Y-%m-%d} "
          f"({dates.nunique()}) | tickers {panel.index.get_level_values('ticker').nunique()} "
          f"| {len(feature_cols)} interaction columns + 5 macro levels + fwd_ret + 3 labels; "
          f"saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frequency", default="weekly", choices=("daily", "weekly"),
                        help="bar frequency of the raw CSVs (default: weekly)")
    parser.add_argument("--horizon", type=int, default=12, choices=(1, 4, 12),
                        help="label holding horizon in bars (default: 12)")
    parser.add_argument("--no-macro", action="store_true",
                        help="skip the 114-column features_macro.csv panel")
    parser.add_argument("--macro-out", type=Path, default=FEATURES_DIR / "weekly" / "features_macro.csv",
                        help="destination of the macro panel "
                             "(default: data/features/weekly/features_macro.csv)")
    args = parser.parse_args()

    frequency = args.frequency
    horizon = args.horizon
    columns = feature_columns()
    target_col = MODEL_TARGETS[f"absolute{horizon}"]
    label_col = label_column(horizon)

    raw_dir = raw_data_dir(frequency)
    prices = load_ohlcv(raw_dir, TRADABLE_ETFS)
    benchmark = load_ohlcv(raw_dir, BENCHMARKS)[BENCHMARKS[0]]

    features = build_feature_frame(prices, frequency=frequency, sectors=TRADABLE_ETFS,
                                   benchmark_prices=benchmark)
    closes = pd.DataFrame(
        {ticker: prices[ticker]["Close"] for ticker in TRADABLE_ETFS}
    ).sort_index()
    labels = make_labels(closes)[target_col].rename(label_col)
    features = features.join(labels, how="left")
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

    if args.no_macro:
        print("macro: skipped (--no-macro)")
    else:
        write_macro_panel(args.macro_out)


if __name__ == "__main__":
    main()
