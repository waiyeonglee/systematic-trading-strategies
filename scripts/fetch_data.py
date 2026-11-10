#!/usr/bin/env python3
"""Regenerate the weekly OHLCV dataset for this project.

Regeneration tool only. The pipeline's normal path imports the CSVs already
present in ``data/raw/weekly/``; this script exists so the dataset can be rebuilt
from Yahoo Finance when a re-fetch is ever needed.

Fixed dataset window (recorded in agents/current_state.md):
    - start date (inclusive) : 2004-01-01 (or instrument inception, whichever is
                               later); pre-2011 history is feature-warmup only
    - end date (inclusive)   : 2026-01-01  (fixed end of the dataset)
    - fetch end (exclusive)  : 2026-01-02  <-- the setting below

Usage:
    python scripts/fetch_data.py            # full re-fetch of all 17 tickers
    python scripts/fetch_data.py --dry-run  # print settings and exit (no network)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Fixed settings (the dataset contract). Change only after an explicit project
# decision -- agents/current_state.md records the convention.
# --------------------------------------------------------------------------- #
START_DATE = "2004-01-01"         # inclusive
END_DATE_INCLUSIVE = "2026-01-01"  # fixed end of the dataset
END_DATE_EXCLUSIVE = "2026-01-02"  # yfinance exclusive end (inclusive + 1 day)

CORE_ETFS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
CROSS_ASSET_ETFS = ("TLT", "IEF", "GLD", "DBC", "UUP", "EFA", "EEM")
BENCHMARKS = ("SPY",)
TICKERS = CORE_ETFS + CROSS_ASSET_ETFS + BENCHMARKS  # 16 tradable + 1 benchmark

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_WEEKLY_DIR = PROJECT_ROOT / "data" / "raw" / "weekly"


def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily OHLCV bars into Friday week-ending bars.

    Open=first, High=max, Low=min, Close=last, Volume=sum. Buckets without a
    complete OHLC bar (e.g. a trailing partial week) are dropped.
    """
    weekly = daily.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return weekly.dropna(subset=["Open", "High", "Low", "Close"])


def trim_to_window(bars: pd.DataFrame) -> pd.DataFrame:
    """Keep only bars on or before the fixed inclusive end date (2026-01-01)."""
    return bars[bars.index <= pd.Timestamp(END_DATE_INCLUSIVE)]


def fetch_and_save(dry_run: bool = False) -> None:
    """Download all tickers and write one weekly OHLCV CSV per ticker."""
    RAW_WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"settings: start={START_DATE} (inclusive) | "
        f"end={END_DATE_EXCLUSIVE} (exclusive) | tickers={len(TICKERS)}"
    )
    if dry_run:
        print("dry-run: no download performed")
        return

    # Imported lazily so `--dry-run` works without the package installed.
    import yfinance as yf

    for position, ticker in enumerate(TICKERS, start=1):
        print(f"[{position}/{len(TICKERS)}] downloading {ticker}...")
        daily = yf.download(
            ticker, start=START_DATE, end=END_DATE_EXCLUSIVE, interval="1d",
            auto_adjust=True, actions=False, progress=False, group_by="column",
            threads=False,
        )
        if daily.empty:
            raise RuntimeError(f"no data returned for {ticker}")
        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = daily.columns.get_level_values(0)
        missing = [col for col in OHLCV if col not in daily.columns]
        if missing:
            raise RuntimeError(f"{ticker}: download missing columns {missing}")
        daily = daily[OHLCV].copy()
        daily.index = pd.to_datetime(daily.index)
        daily = daily[~daily.index.duplicated(keep="last")].sort_index()
        daily.index.name = "Date"

        bars = trim_to_window(resample_weekly(daily))
        bars.to_csv(RAW_WEEKLY_DIR / f"{ticker}.csv")
        print(f"    saved {len(bars):,} rows to {RAW_WEEKLY_DIR / f'{ticker}.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the fixed settings and exit without downloading",
    )
    args = parser.parse_args()
    fetch_and_save(dry_run=args.dry_run)
