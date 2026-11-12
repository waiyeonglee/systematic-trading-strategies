"""Technical features and the feature/label correlation report.

Semantics match the validated reference doc ``agents/feature_engineering.md``:
every indicator lookback counts bars, so a 14-bar RSI is 14 weekly observations
on weekly data. ``FEATURES`` is the raw inspection feature list: 17 asset-level
columns plus the market-wide helper column ``spy_return_1bar``. The current
micro/macro study removes the market-wide column from the firm block; see
``scripts/macro_panel.py``. Fear & Greed is only used as monthly macro state.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import talib

from labels import LABEL
from utils.checks import require_columns
from utils.config import CORE_ETFS, periods_per_year

# Raw feature-frame columns. ``spy_return_1bar`` is a market-wide helper, not part of
# the current study's 19-column ``micro`` firm block.
FEATURES = [
    "return_1bar", "return_2bar", "return_3bar", "return_4bar",
    "return_12bar", "return_26bar", "return_52bar",
    "sma_ratio_4_26bar", "sma_ratio_12_52bar",
    "volatility_12bar", "volatility_26bar", "drawdown_12bar",
    "rsi_14bar", "macd", "macd_signal", "bollinger_position", "volume_change_4bar",
    "spy_return_1bar",
]
# Trend-following inputs: absolute, per-ticker, from past prices only. They are
# model features in the 19-column firm block, not a standalone traded rule.
TREND_FEATURES = [
    "trend_sma40_ratio",   # close / SMA(40 weekly bars) - 1  (≈ the 200-day MA rule)
    "trend_mom52",         # trailing 52-bar (≈12-month) close return
]
RETURN_PERIODS = (1, 2, 3, 4, 12, 26, 52)


def feature_columns() -> list[str]:
    """Return the raw feature list used by the single-panel feature builders.

    This helper is used by ``scripts/run_backtest.py`` and the inspection
    ``features.csv`` path. The current joint study uses
    ``scripts.macro_panel.FIRM_COLUMNS`` for its ``micro`` definition, where
    ``spy_return_1bar`` is excluded from the firm block.

    Returns:
        :data:`FEATURES` plus :data:`TREND_FEATURES`. The trend features are always on.
    """
    return FEATURES + TREND_FEATURES


def build_feature_frame(
    prices: dict[str, pd.DataFrame],
    frequency: str = "weekly",
    sectors: tuple[str, ...] = CORE_ETFS,
    benchmark_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one feature row per asset and bar.

    Every indicator lookback counts bars. The SPY return is aligned using information
    dated on or before each bar.

    Args:
        prices: Ticker -> OHLCV frame (``Date`` index, OHLCV columns).
        frequency: ``"daily"`` or ``"weekly"``; sets the annualisation factor.
        sectors: Tickers to include, in order.
        benchmark_prices: Benchmark OHLCV frame used for the market-return feature.

    Returns:
        Feature-only raw panel indexed by ``(date, ticker)`` with ``FEATURES`` and
        ``TREND_FEATURES``. Downstream study code may select a narrower firm block.

    Raises:
        ValueError: if ``frequency`` is unsupported or no price series is supplied.
    """
    annualisation = periods_per_year(frequency)

    if benchmark_prices is not None:
        spy_close = benchmark_prices.sort_index()["Close"].astype(float)
        spy_return = spy_close.pct_change(1, fill_method=None)
    else:
        spy_return = None

    parts = []
    for ticker in sectors:
        if ticker not in prices:
            continue
        bars = prices[ticker].sort_index()
        close = bars["Close"]
        close_values = close.to_numpy(dtype=float)
        bar_return = close.pct_change(fill_method=None)
        macd_values, macd_signal_values, _ = talib.MACD(
            close_values, fastperiod=12, slowperiod=26, signalperiod=9
        )
        upper_band, middle_band, lower_band = talib.BBANDS(
            close_values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=talib.MA_Type.SMA
        )
        band_width = upper_band - lower_band

        def smoothed(series: np.ndarray, window: int) -> pd.Series:
            return pd.Series(talib.SMA(series, timeperiod=window), index=bars.index)

        features = pd.DataFrame({
            **{f"return_{n}bar": close.pct_change(n, fill_method=None) for n in RETURN_PERIODS},
            "sma_ratio_4_26bar": smoothed(close_values, 4) / smoothed(close_values, 26) - 1,
            "sma_ratio_12_52bar": smoothed(close_values, 12) / smoothed(close_values, 52) - 1,
            "volatility_12bar": pd.Series(
                talib.STDDEV(bar_return.to_numpy(dtype=float), timeperiod=12), index=bars.index
            ) * np.sqrt(annualisation),
            "volatility_26bar": pd.Series(
                talib.STDDEV(bar_return.to_numpy(dtype=float), timeperiod=26), index=bars.index
            ) * np.sqrt(annualisation),
            "drawdown_12bar": close / pd.Series(talib.MAX(close_values, timeperiod=12), index=bars.index) - 1,
            "rsi_14bar": pd.Series(talib.RSI(close_values, timeperiod=14), index=bars.index) / 100,
            "macd": pd.Series(macd_values, index=bars.index) / close,
            "macd_signal": pd.Series(macd_signal_values, index=bars.index) / close,
            "bollinger_position": pd.Series(
                np.divide(close_values - middle_band, band_width,
                          out=np.full_like(close_values, np.nan), where=band_width != 0),
                index=bars.index,
            ),
            "volume_change_4bar": bars["Volume"].pct_change(4, fill_method=None),
            # Trend-following inputs: computed from past closes only.
            "trend_sma40_ratio": close / smoothed(close_values, 40) - 1,
            "trend_mom52": close.pct_change(52, fill_method=None),
            "spy_return_1bar": (
                spy_return.reindex(bars.index, method="ffill")
                if spy_return is not None else np.nan
            ),
        }, index=bars.index)

        features = features.replace([np.inf, -np.inf], np.nan)
        features["ticker"] = ticker
        features.index.name = "date"
        parts.append(features.reset_index())

    if not parts:
        raise ValueError("no sector price series were supplied")

    panel = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["date", "ticker"])
        .set_index(["date", "ticker"])
    )
    return panel[FEATURES + TREND_FEATURES]


def feature_label_correlation(dataset: pd.DataFrame,
                              columns: list[str] | None = None,
                              label_col: str = LABEL) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Correlate every feature with the forward-return label across all labelled dates.

    Args:
        dataset: Panel with the requested feature columns and the label column.
        columns: Feature columns to correlate; defaults to :data:`FEATURES`.
        label_col: Label column to correlate against; defaults to the 12-week label.

    Returns:
        Tuple ``(correlation, last_labelled_date)``.

    Raises:
        ValueError: if columns are missing, the index is not a dated panel, or fewer than
            two labelled dates exist.
    """
    features = list(FEATURES if columns is None else columns)
    require_columns(dataset, features + [label_col], "feature dataset")
    if not isinstance(dataset.index, pd.MultiIndex) or "date" not in dataset.index.names:
        raise ValueError("dataset must have a MultiIndex with a 'date' level")

    row_dates = pd.DatetimeIndex(dataset.index.get_level_values("date"))
    labelled = dataset[label_col].notna()
    label_dates = pd.DatetimeIndex(row_dates[labelled].unique()).sort_values()
    if len(label_dates) < 2:
        raise ValueError("at least two dates with available labels are needed")

    selected = features + [label_col]
    return dataset.loc[labelled, selected].corr(method="pearson", min_periods=2), label_dates[-1]


def save_correlation_heatmap(correlation: pd.DataFrame, output_path: Path, title: str) -> Path:
    """Save a labelled Pearson-correlation heatmap next to the feature table.

    Args:
        correlation: Square correlation matrix.
        output_path: Destination ``.png`` path.
        title: Figure title.

    Returns:
        The path that was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
