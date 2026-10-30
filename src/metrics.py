"""Performance, benchmark and forecast metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from labels import TARGET_COL
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
        port: :func:`portfolio.build_portfolio` output.
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


def capm_alpha(returns: pd.Series, market: pd.Series, ppy: int = 252) -> dict:
    """OLS regression ``r = alpha + beta * market + e``.

    Args:
        returns: Strategy returns, e.g. ``portfolio["net_ret"]``.
        market: Market returns on the same dates, e.g. SPY buy-and-hold.
        ppy: Periods per year, used to annualise alpha.

    Returns:
        ``{"alpha_ann", "beta", "alpha_t"}`` with plain (non-HAC) standard errors.
    """
    frame = pd.concat([returns, market], axis=1).dropna()
    y = frame.iloc[:, 0].to_numpy()
    design = np.column_stack([np.ones(len(frame)), frame.iloc[:, 1].to_numpy()])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    standard_errors = np.sqrt(
        np.diag(residuals @ residuals / (len(y) - 2) * np.linalg.inv(design.T @ design))
    )
    return {"alpha_ann": coefficients[0] * ppy, "beta": coefficients[1],
            "alpha_t": coefficients[0] / standard_errors[0]}


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
        lambda group: group["pred"].corr(group["fwd_ret"], method="spearman")
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
