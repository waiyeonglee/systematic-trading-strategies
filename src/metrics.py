"""Performance, benchmark and forecast metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from labels import TARGET_COL, target_horizon
from utils.checks import require_columns


def spy_buy_and_hold(spy_close: pd.Series, dates: pd.Index) -> pd.Series:
    """Benchmark returns aligned to the strategy's dates and timing.

    The value on date ``t`` is the benchmark's return from the close of ``t`` to ``t+1``.

    Args:
        spy_close: Benchmark adjusted close covering ``dates`` plus one bar.
        dates: Dates to return, e.g. ``portfolio.index``.

    Returns:
        Returns named ``"SPY B&H"``, with missing dates filled by zero.
    """
    return (spy_close.pct_change(fill_method=None).shift(-1)
            .reindex(dates).fillna(0.0).rename("SPY B&H"))


def equal_weight_universe(preds: pd.DataFrame) -> pd.Series:
    """Equal-weight every ticker, rebalanced each period, at zero cost."""
    return preds["fwd_ret"].groupby(level="date").mean().rename("Equal-weight")


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough fall of the cumulative wealth curve (``<= 0``)."""
    wealth = (1 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def sharpe(returns: pd.Series, ppy: int = 252) -> float:
    """Annualised Sharpe ratio with a zero risk-free rate; NaN when volatility is zero."""
    returns = returns.dropna()
    deviation = returns.std(ddof=1)
    return float(np.sqrt(ppy) * returns.mean() / deviation) if deviation > 0 else np.nan


def perf_stats(returns: pd.Series, ppy: int = 252) -> dict:
    """Annualised return, volatility, Sharpe, drawdown, total return and hit rate.

    Args:
        returns: Periodic simple returns; NaNs are dropped.
        ppy: Periods per year (252 daily, 52 weekly).
    """
    returns = returns.dropna()
    return {
        "ann_return": (1 + returns).prod() ** (ppy / len(returns)) - 1,
        "ann_vol": returns.std(ddof=1) * np.sqrt(ppy),
        "sharpe": sharpe(returns, ppy),
        "max_drawdown": max_drawdown(returns),
        "total_return": (1 + returns).prod() - 1,
        "hit_rate": (returns > 0).mean(),
    }


def strategy_report(port: pd.DataFrame, ppy: int = 252) -> pd.Series:
    """Gross versus net statistics plus turnover and cost drag for one strategy.

    Args:
        port: :func:`portfolio.build_forecast_vol_portfolio` output.
        ppy: Periods per year.
    """
    net = perf_stats(port["net_ret"], ppy)
    gross = perf_stats(port["gross_ret"], ppy)
    out = {f"net_{key}": value for key, value in net.items()}
    out.update({f"gross_{key}": value for key, value in gross.items()})
    out.update({
        "avg_turnover": port["turnover"].mean(),
        "ann_turnover": port["turnover"].mean() * ppy,
        "ann_cost_drag": port["cost"].mean() * ppy,
    })
    return pd.Series(out)


def comparison_table(series: dict[str, pd.Series], ppy: int = 252) -> pd.DataFrame:
    """Side-by-side :func:`perf_stats` for several strategies and benchmarks.

    Args:
        series: Label -> returns series; all should cover the same dates.
        ppy: Periods per year.
    """
    return pd.DataFrame({name: perf_stats(values, ppy) for name, values in series.items()}).T


def period_stats(returns: pd.Series, ppy: int = 252,
                 periods: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """Performance split by calendar year, or by named date ranges.

    Args:
        returns: Returns series.
        ppy: Periods per year.
        periods: ``{name: (start, end)}``; ``None`` splits by calendar year.

    Returns:
        One row per period with :func:`perf_stats` columns; periods with a single
        observation are skipped.
    """
    if periods is None:
        groups = {str(year): group for year, group in returns.groupby(returns.index.year)}
    else:
        groups = {name: returns.loc[start:end] for name, (start, end) in periods.items()}
    return pd.DataFrame(
        {name: perf_stats(group, ppy) for name, group in groups.items() if len(group) > 1}
    ).T


def oos_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Out-of-sample R² against a zero forecast: ``1 - SSE / sum(actual²)``."""
    return float(1 - np.sum((actual - predicted) ** 2) / np.sum(actual ** 2))


def forecast_metrics(preds: pd.DataFrame, target_col: str = TARGET_COL) -> dict:
    """Quality of the return forecasts themselves, independent of trading.

    The rank IC compares the prediction against the target column the model was
    trained on (``fwd_ret_4bar`` or ``fwd_ret_12bar``): the forward return realised
    from ``t+1 -> t+H`` for features dated at ``t``.

    Args:
        preds: Predictions panel with ``pred``, ``fwd_ret`` and ``target_col``.
        target_col: Label the model was trained on.

    Returns:
        ``oos_r2``, ``rmse``, ``direction_acc``, ``corr``, ``rank_ic_mean`` and
        ``rank_ic_t``.
    """
    require_columns(preds, ["pred", "fwd_ret", target_col], "predictions panel")
    actual, predicted = preds[target_col].values, preds["pred"].values
    nonzero = actual != 0
    rank_ic = preds.groupby(level="date").apply(
        lambda group: group["pred"].corr(group[target_col], method="spearman")
    )
    return {
        "oos_r2": oos_r2(actual, predicted),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "direction_acc": float(np.mean(np.sign(predicted[nonzero]) == np.sign(actual[nonzero]))),
        "corr": float(np.corrcoef(predicted, actual)[0, 1]),
        "rank_ic_mean": rank_ic.mean(),
        "rank_ic_t": rank_ic.mean() / rank_ic.std(ddof=1) * np.sqrt(rank_ic.count()),
    }


def oos_r2_by_period(preds: pd.DataFrame, target_col: str = TARGET_COL) -> pd.Series:
    """Out-of-sample R² per calendar year."""
    years = preds.index.get_level_values("date").year
    return preds.groupby(years).apply(
        lambda group: oos_r2(group[target_col].values, group["pred"].values)
    )


def oos_r2_benchmarks(preds: pd.DataFrame, benchmark_close: pd.Series,
                      target_col: str = TARGET_COL) -> pd.DataFrame:
    """Model OOS R² relative to SPY / Equal-weight-16 as the benchmark forecast.

    ``R² = 1 - SSE_model / SSE_bench`` on the identical ``(date, ticker)`` panel,
    where ``SSE_model = sum((y - pred)²)`` uses the pipeline's own predictions
    and ``SSE_bench`` uses the benchmark's forecast over the same label window:

    * ``spy`` — SPY's realised return over the label window
      (``price[t+H+1] / price[t+1] - 1`` for an ``H``-bar target), applied to
      every ticker on each date;
    * ``ew16`` — the cross-sectional mean of the realised target per date (the
      equal-weight-16 portfolio's return over the same window).

    Positive values mean the model's forecasts have lower squared error than the
    benchmark forecast; negative values mean the benchmark forecast is better.

    Returns one row per period — ``pooled`` plus each calendar year in the panel —
    with one column per benchmark.
    """
    require_columns(preds, ["pred", target_col], "predictions panel")
    preds = preds.sort_index()
    dates = preds.index.get_level_values("date")
    actual = preds[target_col].to_numpy(dtype=float)
    model = preds["pred"].to_numpy(dtype=float)
    horizon = target_horizon(target_col)

    # Benchmark forecasts over the same label window as the target.
    spy_fwd = benchmark_close.shift(-(horizon + 1)) / benchmark_close.shift(-1) - 1
    spy = spy_fwd.reindex(dates).to_numpy(dtype=float)
    ew = preds[target_col].groupby(level="date").mean().reindex(dates).to_numpy(dtype=float)

    years = dates.year.to_numpy()
    out = {}
    for name, bench in (("oos_r2_vs_spy", spy), ("oos_r2_vs_ew16", ew)):
        finite = np.isfinite(actual) & np.isfinite(bench)
        sse_model = np.sum((actual[finite] - model[finite]) ** 2)
        sse_bench = np.sum((actual[finite] - bench[finite]) ** 2)
        rows = {"pooled": float(1 - sse_model / sse_bench)}
        for year in sorted(set(years)):
            mask = (years == year) & finite
            sse_model_y = np.sum((actual[mask] - model[mask]) ** 2)
            sse_bench_y = np.sum((actual[mask] - bench[mask]) ** 2)
            rows[str(year)] = float(1 - sse_model_y / sse_bench_y)
        out[name] = rows
    return pd.DataFrame(out).rename_axis("period")
