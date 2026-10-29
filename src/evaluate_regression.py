"""
evaluate_regression.py — DSA5205 Project 2: evaluation metrics, benchmarks and plots (REGRESSION).
(Classification version: evaluate.py)

Purpose
-------
Takes the outputs of train_regression.py and produces the §4.9 numbers and figures.
Trading metrics (Sharpe, drawdown, turnover, CAPM, benchmarks) are identical to the
classification case, so they are imported from evaluate.py. What is regression-specific:
    - forecast_metrics: out-of-sample R², RMSE, direction accuracy, rank IC
    - oos_r2_by_period / plot_oos_r2: the "out-of-sample R²" plot from the brief, §4.9 (i)
    - evaluate_all: same as evaluate.py but with the above, saving to ./figures_regression/

Data formats used here
----------------------
returns series   : pd.Series, index = DatetimeIndex, values = periodic simple returns
portfolio frame  : output of train_regression.build_portfolio — index = date,
                   columns gross_ret, turnover, cost, net_ret, n_holdings, holdings
predictions panel: output of train_regression.walk_forward_predict — MultiIndex (date, ticker),
                   columns include pred, excess_ret, fwd_ret
ppy              : periods per year — 252 daily, 52 weekly

Quick test
----------
    >>> y = np.array([0.01, -0.01]); oos_r2(y, y / 2)     # 0.75
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluate as _ev
from evaluate import (capm_alpha, comparison_table, max_drawdown, perf_stats,  # noqa: F401
                      period_stats, spy_buy_and_hold, equal_weight_universe, strategy_report)

FIG_DIR = Path("figures_regression")


def _in_reg_dir(fn):
    """Wrap a plot function from evaluate.py so it saves into FIG_DIR (figures_regression/).

    Input  : fn — a plot function from evaluate.py
    Output : wrapped function with the same inputs and output (a matplotlib Figure)
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        old, _ev.FIG_DIR = _ev.FIG_DIR, FIG_DIR
        try:
            return fn(*args, **kwargs)
        finally:
            _ev.FIG_DIR = old
    return wrapper


# Shared plots, redirected to figures_regression/ (see evaluate.py for their docstrings)
plot_cumulative = _in_reg_dir(_ev.plot_cumulative)
plot_sharpe_by_model = _in_reg_dir(_ev.plot_sharpe_by_model)
plot_gross_vs_net = _in_reg_dir(_ev.plot_gross_vs_net)
plot_tech_vs_trends = _in_reg_dir(_ev.plot_tech_vs_trends)
plot_period_sharpe = _in_reg_dir(_ev.plot_period_sharpe)
plot_drawdown = _in_reg_dir(_ev.plot_drawdown)
plot_coefficients = _in_reg_dir(_ev.plot_coefficients)


def plot_turnover_vs_gate(sweep: dict[float, pd.DataFrame], ppy: int = 252):
    """Annual turnover and net Sharpe against the gate (gate in predicted-return units).

    Inputs : sweep — train_regression.gate_sweep output, dict[gate -> portfolio frame]; ppy
    Output : Figure, saved as figures_regression/turnover_vs_gate.png

    Test   : >>> plot_turnover_vs_gate(gate_sweep(res["val_preds"][res["best_spec"].name]))
    """
    fig = _in_reg_dir(_ev.plot_turnover_vs_gate)(sweep, ppy)
    fig.axes[0].set_xlabel("Gate (predicted-return gap)")
    fig.savefig(FIG_DIR / "turnover_vs_gate.png", dpi=150)
    return fig


# --------------------------------------------------------------------------------------
# Regression forecast metrics
# --------------------------------------------------------------------------------------
def oos_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """Out-of-sample R² against a zero forecast: 1 - sum((y - yhat)^2) / sum(y^2).

    Zero is the natural naive forecast for excess returns (return minus median).
    Positive = beats predicting zero; negative (common for daily returns) = worse than zero.

    Inputs : y, yhat — np.ndarray of realised and predicted returns, same length
    Output : float (at most 1)

    Test   : >>> oos_r2(np.array([0.01, -0.01]), np.array([0.005, -0.005]))   # 0.75
    """
    return float(1 - np.sum((y - yhat) ** 2) / np.sum(y ** 2))


def forecast_metrics(preds: pd.DataFrame, target_col: str = "excess_ret") -> dict:
    """How good are the return forecasts themselves (independent of trading)?

    Inputs
    ------
    preds      : predictions panel with "pred", "fwd_ret" and target_col
    target_col : str — the label the model was trained on (train_regression.TARGET_COL)

    Output
    ------
    dict with float values:
        oos_r2        — out-of-sample R² vs a zero forecast (see oos_r2)
        rmse          — root mean squared error, in return units (0.01 = 1%)
        direction_acc — share of rows where sign(pred) == sign(actual)
                        (rows where actual is exactly 0, i.e. the median asset, are skipped)
        corr          — Pearson correlation of pred vs actual, pooled over all rows
        rank_ic_mean  — average per-date Spearman correlation of pred vs fwd_ret
                        (does the model rank assets correctly? This is what the strategy uses)
        rank_ic_t     — t-stat of rank_ic_mean (|t| > 2 suggests real signal)

    Test
    ----
        >>> fm = forecast_metrics(res["test_preds"])
        >>> fm["oos_r2"] <= 1                           # True
    """
    y, p = preds[target_col].values, preds["pred"].values
    nz = y != 0
    rank_ic = preds.groupby(level="date").apply(
        lambda g: g["pred"].corr(g["fwd_ret"], method="spearman"))
    return {"oos_r2": oos_r2(y, p),
            "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "direction_acc": float(np.mean(np.sign(p[nz]) == np.sign(y[nz]))),
            "corr": float(np.corrcoef(p, y)[0, 1]),
            "rank_ic_mean": rank_ic.mean(),
            "rank_ic_t": rank_ic.mean() / rank_ic.std(ddof=1) * np.sqrt(rank_ic.count())}


def oos_r2_by_period(preds: pd.DataFrame, target_col: str = "excess_ret") -> pd.Series:
    """Out-of-sample R² per calendar year.

    Inputs : preds — panel with "pred" and target_col; target_col — str
    Output : pd.Series, index = year (int), values = oos_r2 for that year

    Test   : >>> oos_r2_by_period(res["test_preds"])
    """
    years = preds.index.get_level_values("date").year
    return preds.groupby(years).apply(lambda g: oos_r2(g[target_col].values, g["pred"].values))


def plot_oos_r2(preds: pd.DataFrame, target_col: str = "excess_ret"):
    """Out-of-sample R² by year (left) and cumulative R² over time (right).

    The cumulative line rises while the model beats a zero forecast and falls while it loses.

    Inputs : preds — panel with "pred" and target_col; target_col — str
    Output : Figure, saved as figures_regression/oos_r2.png

    Test   : >>> plot_oos_r2(res["test_preds"])
    """
    by_year = oos_r2_by_period(preds, target_col)
    err = (preds[target_col] - preds["pred"]) ** 2
    sse = err.groupby(level="date").sum().cumsum()
    sst = (preds[target_col] ** 2).groupby(level="date").sum().cumsum()
    cum_r2 = 1 - sse / sst
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.bar(by_year.index.astype(str), by_year.values)
    a1.axhline(0, color="k", lw=0.8); a1.set(title="Out-of-sample R² by year", ylabel="R²")
    a2.plot(cum_r2); a2.axhline(0, color="k", lw=0.8)
    a2.set(title="Cumulative out-of-sample R²", ylabel="R²")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    FIG_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "oos_r2.png", dpi=150)
    return fig


# --------------------------------------------------------------------------------------
# One call to produce everything for the test set
# --------------------------------------------------------------------------------------
def evaluate_all(results: dict, spy_close: pd.Series, ppy: int = 252,
                 trends_results: dict | None = None, target_col: str = "excess_ret") -> dict:
    """Compute all test-set metrics and save all figures for the regression strategy.

    Inputs
    ------
    results        : dict returned by train_regression.run_pipeline
    spy_close      : pd.Series — SPY adjusted close, DatetimeIndex covering the test period
    ppy            : periods per year (252 daily, 52 weekly)
    trends_results : optional run_pipeline dict with Google Trends features added
    target_col     : label column the model was trained on (train_regression.TARGET_COL)

    Output
    ------
    dict with keys:
        "comparison"  : pd.DataFrame — model vs SPY vs equal-weight (+ trends)
        "strategy"    : pd.Series    — gross vs net stats, turnover, cost drag
        "forecast"    : dict         — forecast_metrics (oos_r2, rmse, rank IC, ...)
        "r2_by_year"  : pd.Series    — oos_r2_by_period
        "capm"        : dict         — alpha_ann, beta, alpha_t vs SPY
        "by_year"     : dict[str, pd.DataFrame] — period_stats per strategy
    Side effect: PNGs in ./figures_regression/ (cumulative_returns, drawdown, sharpe_by_model,
                 gross_vs_net, sharpe_by_period, coefficients, oos_r2, [tech_vs_trends])

    Test
    ----
        >>> rep = evaluate_all(res, spy_close)
        >>> rep["comparison"][["ann_return", "sharpe", "max_drawdown"]]
    """
    port, preds = results["test_portfolio"], results["test_preds"]
    spy = spy_buy_and_hold(spy_close, port.index)
    ew = equal_weight_universe(preds).reindex(port.index)
    name = f"{results['best_spec'].name} (net)"
    series = {name: port["net_ret"], "SPY B&H": spy, "Equal-weight": ew}
    if trends_results is not None:
        series["+ Google Trends (net)"] = trends_results["test_portfolio"]["net_ret"]

    out = {
        "comparison": comparison_table(series, ppy),
        "strategy": strategy_report(port, ppy),
        "forecast": forecast_metrics(preds, target_col),
        "r2_by_year": oos_r2_by_period(preds, target_col),
        "capm": capm_alpha(port["net_ret"], spy, ppy),
        "by_year": {k: period_stats(v, ppy) for k, v in series.items()},
    }
    plot_cumulative(series)
    plot_drawdown(series)
    plot_sharpe_by_model(results["val_table"])
    plot_gross_vs_net(port)
    plot_period_sharpe(out["by_year"])
    plot_coefficients(results["coefficients"])
    plot_oos_r2(preds, target_col)
    if trends_results is not None:
        plot_tech_vs_trends(port, trends_results["test_portfolio"])
    plt.close("all")
    return out
