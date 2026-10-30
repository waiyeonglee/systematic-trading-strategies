"""Market data: download prices, load saved CSVs, and assemble the modelling panel."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from features import build_feature_frame, feature_columns
from labels import make_labels
from utils.checks import require_columns
from utils.config import BENCHMARKS, CORE_ETFS, TICKERS
from utils.paths import FEAR_GREED_PATH, raw_data_dir

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def yfinance_exclusive_end_date(end_date: str) -> str:
    """Convert an inclusive end date into yfinance's exclusive end date."""
    return (pd.to_datetime(end_date).date() + timedelta(days=1)).isoformat()


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download auto-adjusted daily OHLCV bars for one ticker.

    Args:
        ticker: Yahoo Finance symbol, e.g. ``"XLK"``.
        start: Inclusive start date.
        end: Exclusive end date.

    Returns:
        Date-indexed frame with ``Open``, ``High``, ``Low``, ``Close`` and ``Volume``.

    Raises:
        RuntimeError: if no data or no expected columns are returned.
    """
    data = yf.download(
        ticker, start=start, end=end, interval="1d", auto_adjust=True, actions=False,
        progress=False, group_by="column", threads=False,
    )
    if data.empty:
        raise RuntimeError(f"no data returned for {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    require_columns(data, OHLCV, f"{ticker} download")
    data = data[OHLCV].copy()
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"
    return data.sort_index()


def resample_weekly(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily OHLCV bars into Friday week-ending bars."""
    weekly = data.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return weekly.dropna(subset=["Open", "High", "Low", "Close"])


def fetch_and_save(frequency: str, start_date: str, end_date: str,
                   output_dir: Path | None = None,
                   tickers: tuple[str, ...] = TICKERS) -> None:
    """Download every ticker and write one OHLCV CSV per ticker.

    Args:
        frequency: ``"daily"`` or ``"weekly"``.
        start_date: Inclusive start date from the config.
        end_date: Inclusive end date from the config.
        output_dir: Destination folder; defaults to ``data/raw/<frequency>``.
        tickers: Symbols to download.
    """
    output_dir = Path(output_dir) if output_dir is not None else raw_data_dir(frequency)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_end = yfinance_exclusive_end_date(end_date)

    for position, ticker in enumerate(tickers, start=1):
        print(f"[{position}/{len(tickers)}] downloading {ticker}...")
        daily = download_ticker(ticker, start_date, download_end)
        bars = daily if frequency == "daily" else resample_weekly(daily)
        if frequency == "weekly":
            # Drop a trailing week whose Friday falls after the requested end date: it
            # would only contain the first few daily bars of that week.
            bars = bars[bars.index <= pd.Timestamp(end_date)]
        bars.to_csv(output_dir / f"{ticker}.csv")
        print(f"    saved {len(bars):,} rows to {output_dir / f'{ticker}.csv'}")


def load_ohlcv(folder: str | Path, tickers: tuple[str, ...] = CORE_ETFS) -> dict[str, pd.DataFrame]:
    """Load ticker CSVs previously written by :func:`fetch_and_save`.

    Args:
        folder: Folder holding ``<ticker>.csv`` files.
        tickers: Symbols to load.

    Returns:
        Ticker -> OHLCV frame.

    Raises:
        FileNotFoundError: if a ticker CSV is missing.
        ValueError: if a CSV lacks an OHLCV column.
    """
    folder = Path(folder)
    prices = {}
    for ticker in tickers:
        path = folder / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; fetch the requested frequency first")
        bars = pd.read_csv(path, parse_dates=["Date"]).drop_duplicates("Date", keep="last")
        bars = bars.set_index("Date").sort_index()
        require_columns(bars, OHLCV, str(path))
        prices[ticker] = bars[OHLCV].astype(float)
    return prices


def load_fear_greed_index(path: str | Path = FEAR_GREED_PATH) -> pd.Series:
    """Load the dated numeric Fear & Greed Index (0-100).

    Raises:
        FileNotFoundError: if the CSV is missing.
        ValueError: if required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing Fear & Greed data: {path}")

    data = pd.read_csv(path, parse_dates=["Date"])
    require_columns(data, ["Date", "Fear Greed"], str(path))
    data = data.drop_duplicates("Date", keep="last").sort_values("Date")
    values = pd.to_numeric(data["Fear Greed"], errors="coerce")
    return pd.Series(values.to_numpy(), index=pd.DatetimeIndex(data["Date"]), name="fear_greed_index")


def load_panel(frequency: str, sectors: tuple[str, ...] = CORE_ETFS,
               benchmark: str = BENCHMARKS[0],
               fear_greed_path: str | Path = FEAR_GREED_PATH,
               use_fear_greed: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Build the feature-plus-label panel and the benchmark close series.

    Args:
        frequency: ``"daily"`` or ``"weekly"``.
        sectors: Sector tickers to model.
        benchmark: Benchmark ticker whose close series and return are used.
        fear_greed_path: Fear & Greed CSV path.
        use_fear_greed: include the Fear & Greed index as a feature.

    Returns:
        Tuple ``(panel, benchmark_close)`` where the panel is indexed by
        ``(date, ticker)`` with the active features plus ``fwd_ret`` and ``excess_ret``.

    Raises:
        FileNotFoundError: if the raw CSVs have not been fetched.
        ValueError: if the joined panel is empty.
    """
    raw_dir = raw_data_dir(frequency)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"missing {raw_dir}; run `uv run python main.py fetch --frequency {frequency}` first"
        )

    prices = load_ohlcv(raw_dir, tuple(sectors))
    benchmark_prices = load_ohlcv(raw_dir, (benchmark,))[benchmark]
    closes = pd.DataFrame({ticker: prices[ticker]["Close"] for ticker in sectors}).sort_index()

    features = build_feature_frame(
        prices,
        frequency=frequency,
        sectors=tuple(sectors),
        fear_greed_index=load_fear_greed_index(fear_greed_path) if use_fear_greed else None,
        benchmark_prices=benchmark_prices,
    )
    panel = (
        features[feature_columns(use_fear_greed)]
        .join(make_labels(closes), how="inner")
        .dropna()
        .sort_index()
    )
    if panel.empty:
        raise ValueError("panel is empty after joining features and labels")

    return panel, benchmark_prices["Close"].rename(benchmark)
