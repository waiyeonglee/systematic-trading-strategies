"""Download adjusted OHLCV data for sector ETFs and SPY using yfinance."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf


CORE_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
BENCHMARKS = ["SPY"]
TICKERS = CORE_ETFS + BENCHMARKS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download daily or weekly adjusted OHLCV data for sector ETFs."
    )
    parser.add_argument(
        "--frequency",
        choices=("daily", "weekly"),
        required=True,
        help="Output frequency: daily or Friday week-ending weekly data.",
    )
    return parser.parse_args()


def load_date_range() -> tuple[str, str]:
    """Load the shared inclusive data date range from config.yaml."""
    with CONFIG_PATH.open() as file:
        config = yaml.safe_load(file) or {}

    data_config = config.get("data", {})
    start_date = data_config.get("start_date")
    end_date = data_config.get("end_date")
    if not start_date or not end_date:
        raise ValueError("config.yaml must define data.start_date and data.end_date.")

    return str(start_date), str(end_date)


def yfinance_exclusive_end_date(end_date: str) -> str:
    """Convert an inclusive config end date to yfinance's exclusive end date."""
    return (pd.to_datetime(end_date).date() + timedelta(days=1)).isoformat()


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download auto-adjusted daily OHLCV data for one ticker."""
    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=False,
    )

    if data.empty:
        raise RuntimeError(f"No data returned for {ticker}.")

    # Some yfinance versions return a two-level column index even for one ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise RuntimeError(
            f"Missing expected columns for {ticker}: {', '.join(missing_columns)}"
        )

    data = data[required_columns].copy()
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"
    return data.sort_index()


def resample_weekly(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily OHLCV bars into Friday week-ending bars."""
    weekly = data.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )

    # Drop empty holiday-only periods while retaining valid shortened weeks.
    return weekly.dropna(subset=["Open", "High", "Low", "Close"])


def raw_data_dir(frequency: str) -> Path:
    """Return the frequency-specific raw data directory."""
    return PROJECT_ROOT / "data" / f"stocks_{frequency}" / "raw"


def main() -> None:
    """Download, transform, and save ticker-level and combined price data."""
    args = parse_args()
    start_date, end_date = load_date_range()
    download_end_date = yfinance_exclusive_end_date(end_date)
    output_dir = raw_data_dir(args.frequency)
    output_dir.mkdir(parents=True, exist_ok=True)

    adjusted_closes: dict[str, pd.Series] = {}

    for position, ticker in enumerate(TICKERS, start=1):
        print(f"[{position}/{len(TICKERS)}] Downloading {ticker}...")
        daily_data = download_ticker(ticker, start_date, download_end_date)
        output_data = (
            daily_data if args.frequency == "daily" else resample_weekly(daily_data)
        )

        output_path = output_dir / f"{ticker}.csv"
        output_data.to_csv(output_path)
        adjusted_closes[ticker] = output_data["Close"].rename(ticker)
        print(f"    Saved {len(output_data):,} rows to {output_path}")

    combined_close = pd.concat(adjusted_closes.values(), axis=1).sort_index()
    combined_close.index.name = "Date"
    combined_path = output_dir / "adj_close.csv"
    combined_close.to_csv(combined_path)

    print(f"Saved combined adjusted-close data to {combined_path}")


if __name__ == "__main__":
    main()
