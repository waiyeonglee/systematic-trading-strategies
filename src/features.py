"""Technical features and the feature/label correlation report (v2-owned).

Semantics match the validated reference doc ``agents/feature_engineering.md``:
every indicator lookback counts bars, so a 14-bar RSI is 14 weekly observations
on weekly data. The feature set is the 19 base + 2 trend-following columns
(21 in total with Fear & Greed enabled).
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

FEATURES = [
    "return_1bar", "return_2bar", "return_3bar", "return_4bar",
    "return_12bar", "return_26bar", "return_52bar",
    "sma_ratio_4_26bar", "sma_ratio_12_52bar",
    "volatility_12bar", "volatility_26bar", "drawdown_12bar",
    "rsi_14bar", "macd", "macd_signal", "bollinger_position", "volume_change_4bar",
    "fear_greed_index", "spy_return_1bar",
]
# Trend-following inputs: absolute, per-ticker, from past prices only. The rule
# reads these as 0/1 signals (value > 0 -> signal on).
TREND_FEATURES = [
    "trend_sma40_ratio",   # close / SMA(40 weekly bars) - 1  (≈ the 200-day MA rule)
    "trend_mom52",         # trailing 52-bar (≈12-month) close return
]
TREND_SIGNAL_COLUMNS = {"sma40": "trend_sma40_ratio", "mom52": "trend_mom52"}
ALTERNATIVE_FEATURES = {"fear_greed": ("fear_greed_index",)}
RETURN_PERIODS = (1, 2, 3, 4, 12, 26, 52)


def feature_columns(use_fear_greed: bool = True) -> list[str]:
    """Return the feature list, minus any alternative column that is switched off.

    Args:
        use_fear_greed: include the Fear & Greed index feature.

    Returns:
        The subset of :data:`FEATURES` (+ :data:`TREND_FEATURES`) the model / rule may
        use. The trend features are always on: they are the trend-following inputs of
        this project.
    """
    excluded = () if use_fear_greed else ALTERNATIVE_FEATURES["fear_greed"]
    base = [name for name in FEATURES if name not in excluded]
    return base + TREND_FEATURES


def trend_rule_signals(panel: pd.DataFrame, rule: str = "sma40",
                       shorting: bool = False) -> pd.DataFrame:
    """Turn a trend feature into the signed rule signal the trend book trades.

    This is the rule path's decision step (design §3): the configured trend
    feature's value at bar ``t`` above zero means "long", otherwise "flat" (or
    "short" when ``shorting``). There is no fitted model and no estimation window,
    so the signal exists at every dated row and the book's one-bar execution lag is
    the only delay.

    Args:
        panel: Feature panel indexed by ``(date, ticker)``, e.g. from
            ``scripts/run_backtest.build_panel``.
        rule: ``"sma40"`` (``close / SMA(40 weekly bars) - 1``, ≈ the 200-day MA
            rule) or ``"mom52"`` (trailing 52-bar return, ≈ 12-month momentum).
        shorting: off-signal becomes ``-1`` (for ``trend_long_short``) instead of
            ``0`` (for ``trend_long_flat``, the documented reference).

    Returns:
        Copy of ``panel`` with ``pred`` = +1.0 on-signal / -1.0 or 0.0 off-signal.

    Raises:
        ValueError: if ``rule`` is unknown or the panel lacks the trend column.
    """
    if rule not in TREND_SIGNAL_COLUMNS:
        raise ValueError(f"unknown trend rule {rule!r}; use one of "
                         f"{sorted(TREND_SIGNAL_COLUMNS)}")
    column = TREND_SIGNAL_COLUMNS[rule]
    require_columns(panel, [column], "trend panel")
    off = -1.0 if shorting else 0.0
    out = panel.copy()
    out["pred"] = np.where(panel[column] > 0, 1.0, off)
    return out


def build_feature_frame(
    prices: dict[str, pd.DataFrame],
    frequency: str = "weekly",
    sectors: tuple[str, ...] = CORE_ETFS,
    fear_greed_index: pd.Series | None = None,
    benchmark_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one feature row per asset and bar.

    Every indicator lookback counts bars. The Fear & Greed value and the SPY return
    are aligned using information dated on or before each bar.

    Args:
        prices: Ticker -> OHLCV frame (``Date`` index, OHLCV columns).
        frequency: ``"daily"`` or ``"weekly"``; sets the annualisation factor.
        sectors: Tickers to include, in order.
        fear_greed_index: Dated Fear & Greed values (0-100), or ``None`` to leave the
            column missing.
        benchmark_prices: Benchmark OHLCV frame used for the market-return feature.

    Returns:
        Feature-only panel indexed by ``(date, ticker)`` with ``FEATURES`` and
        ``TREND_FEATURES``.

    Raises:
        ValueError: if ``frequency`` is unsupported or no price series is supplied.
    """
    annualisation = periods_per_year(frequency)

    if fear_greed_index is not None:
        fear_greed_index = fear_greed_index.copy()
        fear_greed_index.index = pd.to_datetime(fear_greed_index.index)
        fear_greed_index = pd.to_numeric(fear_greed_index, errors="coerce")
        fear_greed_index = fear_greed_index[~fear_greed_index.index.duplicated(keep="last")].sort_index()

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
            "fear_greed_index": (
                fear_greed_index.reindex(bars.index, method="ffill")
                if fear_greed_index is not None else np.nan
            ),
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
