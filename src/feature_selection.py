"""Build daily/weekly technical features and analyze their correlations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import talib

from fetch_data import BENCHMARKS, CORE_ETFS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES = [
    "return_1bar", "return_2bar", "return_3bar", "return_4bar",
    "return_12bar", "return_26bar", "return_52bar",
    "sma_ratio_4_26bar", "sma_ratio_12_52bar",
    "volatility_12bar", "volatility_26bar", "drawdown_12bar",
    "rsi_14bar", "macd", "macd_signal", "bollinger_position", "volume_change_4bar",
    "fear_greed_index", "spy_return_1bar",
]
LABEL = "label_forward_return_1bar"


def load_ohlcv(folder: str | Path, tickers: tuple[str, ...] = tuple(CORE_ETFS)) -> dict[str, pd.DataFrame]:
    """Load daily or weekly ticker CSVs previously saved by ``fetch_data.py``."""
    folder = Path(folder)
    prices = {}
    for ticker in tickers:
        path = folder / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; fetch the requested frequency before building features.")
        bars = pd.read_csv(path, parse_dates=["Date"]).drop_duplicates("Date", keep="last")
        bars = bars.set_index("Date").sort_index()
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = set(required).difference(bars.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        prices[ticker] = bars[required].astype(float)
    return prices


def load_fear_greed_index(path: str | Path) -> pd.Series:
    """Load the dated numeric Fear & Greed Index (0–100)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing Fear & Greed data: {path}")
    data = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Fear Greed"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    data = data.drop_duplicates("Date", keep="last").sort_values("Date")
    values = pd.to_numeric(data["Fear Greed"], errors="coerce")
    return pd.Series(values.to_numpy(), index=pd.DatetimeIndex(data["Date"]), name="fear_greed_index")


def build_feature_frame(
    prices: dict[str, pd.DataFrame],
    frequency: str = "weekly",
    sectors: tuple[str, ...] = tuple(CORE_ETFS),
    fear_greed_index: pd.Series | None = None,
    benchmark_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a feature row per asset/bar at daily or weekly frequency.

    Daily features consume daily bars; weekly features consume Friday-ending weekly
    bars produced by ``fetch_data.py`` from the daily source data. Indicator
    lookbacks count bars, so each feature uses the same number of daily or weekly
    observations, depending on the input frequency. The Fear & Greed value and
    SPY market trend are aligned using information dated on or before each bar.
    The SPY feature is its one-bar return on the corresponding daily or weekly bar.
    """
    if frequency not in {"daily", "weekly"}:
        raise ValueError("frequency must be 'daily' or 'weekly'.")
    periods_per_year = 252 if frequency == "daily" else 52
    if fear_greed_index is not None:
        fear_greed_index = fear_greed_index.copy()
        fear_greed_index.index = pd.to_datetime(fear_greed_index.index)
        fear_greed_index = pd.to_numeric(fear_greed_index, errors="coerce")
        fear_greed_index = fear_greed_index[~fear_greed_index.index.duplicated(keep="last")].sort_index()
    if benchmark_prices is not None:
        benchmark_prices = benchmark_prices.sort_index()
        spy_close = benchmark_prices["Close"].astype(float)
        spy_return = spy_close.pct_change(1, fill_method=None)
    else:
        spy_return = None

    parts = []
    for ticker in sectors:
        if ticker not in prices:
            continue
        bars = prices[ticker].sort_index()
        close, volume = bars["Close"], bars["Volume"]
        close_values = close.to_numpy(dtype=float)
        volume_values = volume.to_numpy(dtype=float)
        bar_return = close.pct_change(fill_method=None)
        macd_values, macd_signal_values, _ = talib.MACD(
            close_values, fastperiod=12, slowperiod=26, signalperiod=9
        )
        upper_band, middle_band, lower_band = talib.BBANDS(
            close_values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=talib.MA_Type.SMA
        )
        band_width = upper_band - lower_band
        return_periods = (1, 2, 3, 4, 12, 26, 52)

        features = pd.DataFrame({
            **{f"return_{n}bar": close.pct_change(n, fill_method=None) for n in return_periods},
            "sma_ratio_4_26bar": pd.Series(talib.SMA(close_values, timeperiod=4), index=bars.index)
            / pd.Series(talib.SMA(close_values, timeperiod=26), index=bars.index) - 1,
            "sma_ratio_12_52bar": pd.Series(talib.SMA(close_values, timeperiod=12), index=bars.index)
            / pd.Series(talib.SMA(close_values, timeperiod=52), index=bars.index) - 1,
            "volatility_12bar": pd.Series(talib.STDDEV(bar_return.to_numpy(dtype=float), timeperiod=12), index=bars.index)
            * np.sqrt(periods_per_year),
            "volatility_26bar": pd.Series(talib.STDDEV(bar_return.to_numpy(dtype=float), timeperiod=26), index=bars.index)
            * np.sqrt(periods_per_year),
            "drawdown_12bar": close / pd.Series(talib.MAX(close_values, timeperiod=12), index=bars.index) - 1,
            "rsi_14bar": pd.Series(talib.RSI(close_values, timeperiod=14), index=bars.index) / 100,
            "macd": pd.Series(macd_values, index=bars.index) / close,
            "macd_signal": pd.Series(macd_signal_values, index=bars.index) / close,
            "bollinger_position": pd.Series(
                np.divide(close_values - middle_band, band_width,
                          out=np.full_like(close_values, np.nan), where=band_width != 0),
                index=bars.index,
            ),
            "volume_change_4bar": pd.Series(volume_values, index=bars.index).pct_change(4, fill_method=None),
            "fear_greed_index": (
                fear_greed_index.reindex(bars.index, method="ffill")
                if fear_greed_index is not None
                else np.nan
            ),
            "spy_return_1bar": (
                spy_return.reindex(bars.index, method="ffill")
                if spy_return is not None
                else np.nan
            ),
            LABEL: close.pct_change(1, fill_method=None).shift(-1),
        }, index=bars.index)
        features = features.replace([np.inf, -np.inf], np.nan)
        features["ticker"] = ticker
        features.index.name = "date"
        parts.append(features.reset_index())

    if not parts:
        raise ValueError("No sector price series were supplied.")
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["date", "ticker"])
        .set_index(["date", "ticker"])[FEATURES + [LABEL]]
    )


def feature_label_correlation(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Correlate features and forward-return label across all labeled dates for EDA."""
    required = set(FEATURES + [LABEL])
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"Missing feature/label columns: {sorted(missing)}")
    if not isinstance(dataset.index, pd.MultiIndex) or "date" not in dataset.index.names:
        raise ValueError("dataset must have a MultiIndex with a 'date' level.")

    row_dates = pd.DatetimeIndex(dataset.index.get_level_values("date"))
    labeled_rows = dataset[LABEL].notna()
    label_dates = pd.DatetimeIndex(row_dates[labeled_rows].unique()).sort_values()
    if len(label_dates) < 2:
        raise ValueError("At least two dates with available labels are needed.")
    sample_end = label_dates[-1]
    columns = FEATURES + [LABEL]
    return dataset.loc[labeled_rows, columns].corr(method="pearson", min_periods=2), sample_end


def save_correlation_heatmap(
    correlation: pd.DataFrame, output_path: Path, title: str
) -> None:
    """Save a labeled Pearson-correlation heatmap."""
    size = max(11, len(correlation.columns) * 0.72)
    fig, ax = plt.subplots(figsize=(size, size * 0.82))
    sns.heatmap(
        correlation, ax=ax, cmap="vlag", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", square=True, linewidths=0.4,
        cbar_kws={"label": "Pearson correlation"},
    )
    ax.set_title(title, pad=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    """Build feature tables and correlation heatmaps for selected frequencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--frequency", choices=("daily", "weekly", "both"), default="both",
        help="Which pre-fetched frequency to analyze (default: both).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "reports/feature_selection"
    )
    parser.add_argument(
        "--fear-greed-path", type=Path,
        default=PROJECT_ROOT / "data/fear_and_greed_index/fear-greed.csv",
        help="CSV containing dated Fear & Greed Index values.",
    )
    args = parser.parse_args()

    frequencies = ("daily", "weekly") if args.frequency == "both" else (args.frequency,)
    fear_greed_index = load_fear_greed_index(args.fear_greed_path)
    for frequency in frequencies:
        data_dir = args.data_root / f"stocks_{frequency}" / "raw"
        prices = load_ohlcv(data_dir)
        benchmark_prices = load_ohlcv(data_dir, tickers=tuple(BENCHMARKS))[BENCHMARKS[0]]
        features = build_feature_frame(
            prices, frequency=frequency, fear_greed_index=fear_greed_index,
            benchmark_prices=benchmark_prices,
        )
        label_correlation, sample_end = feature_label_correlation(features)
        output_dir = args.output_dir / frequency
        output_dir.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_dir / "features.csv")
        label_correlation.to_csv(output_dir / "feature_label_correlation.csv")
        save_correlation_heatmap(
            label_correlation,
            output_dir / "feature_label_correlation.png",
            f"{frequency.title()} Feature/Label Correlation (All Labeled Samples through {sample_end.date()})",
        )
        print(
            f"{frequency.title()}: {features.shape[0]:,} rows × "
            f"{len(FEATURES)} features + target; label correlation uses all labeled dates "
            f"through {sample_end.date()}; saved to {output_dir}"
        )


if __name__ == "__main__":
    main()
