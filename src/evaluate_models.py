"""
evaluate.py — DSA5205 Project 2: evaluation metrics, benchmarks and plots.

Purpose
-------
Takes the outputs of train.py and produces the numbers and figures for §4.9 of the report:
post-cost return / volatility / Sharpe, drawdown, turnover and costs, forecast accuracy / AUC,
CAPM alpha, per-period results, and comparisons with SPY and an equal-weight benchmark.

Data formats used here
----------------------
returns series   : pd.Series, index = DatetimeIndex, values = periodic simple returns (0.01 = 1%)
portfolio frame  : output of train.build_portfolio — index = date,
                   columns gross_ret, turnover, cost, net_ret, n_holdings, holdings
predictions panel: output of train.walk_forward_predict — MultiIndex (date, ticker),
                   columns include prob, target, fwd_ret
ppy              : periods per year — 252 for daily data, 52 for weekly

All plot functions save a PNG to ./figures/<name>.png and return the matplotlib Figure.

Quick test
----------
    >>> import numpy as np, pandas as pd
    >>> r = pd.Series(np.random.normal(5e-4, 0.01, 500), index=pd.bdate_range("2023-01-02", periods=500))
    >>> perf_stats(r)              # dict of metrics
    >>> period_stats(r)            # one row per year
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

FIG_DIR = Path("figures")


# --------------------------------------------------------------------------------------
# 1. Benchmarks
# --------------------------------------------------------------------------------------
def spy_buy_and_hold(spy_close: pd.Series, dates: pd.Index) -> pd.Series:
    """SPY buy-and-hold returns, aligned to the strategy's dates and timing.

    Uses the same convention as the strategy: the value on date t is SPY's return from the
    close of t to the close of t+1.

    Inputs
    ------
    spy_close : pd.Series — SPY adjusted close, index = DatetimeIndex (must cover `dates` + 1 day)
    dates     : pd.Index  — the dates to return, e.g. test_portfolio.index

    Output
    ------
    pd.Series named "SPY B&H", index = dates, values = returns (missing filled with 0)

    Test
    ----
        >>> c = pd.Series([100, 101, 99.99], index=pd.bdate_range("2023-01-02", periods=3))
        >>> spy_buy_and_hold(c, c.index[:2])        # [0.01, -0.01]
    """
    return spy_close.pct_change().shift(-1).reindex(dates).fillna(0.0).rename("SPY B&H")


def equal_weight_universe(preds: pd.DataFrame) -> pd.Series:
    """Equal-weight benchmark: hold every ticker at 1/N, rebalanced every period, no costs.

    Input  : preds — panel with column "fwd_ret"
    Output : pd.Series named "Equal-weight", index = date, value = mean fwd_ret across tickers

    Test
    ----
        >>> equal_weight_universe(res["test_preds"]).head()
    """
    return preds["fwd_ret"].groupby(level="date").mean().rename("Equal-weight")


# --------------------------------------------------------------------------------------
# 2. Performance metrics
# --------------------------------------------------------------------------------------
def max_drawdown(r: pd.Series) -> float:
    """Largest peak-to-trough fall of the cumulative wealth curve.

    Input  : r — returns series
    Output : float <= 0, e.g. -0.18 means the worst loss from a previous high was 18%

    Test
    ----
        >>> max_drawdown(pd.Series([0.10, -0.50, 0.20]))   # -0.5
    """
    wealth = (1 + r).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def perf_stats(r: pd.Series, ppy: int = 252) -> dict:
    """Standard performance summary for one returns series.

    Inputs
    ------
    r   : returns series (NaNs are dropped)
    ppy : periods per year (252 daily, 52 weekly)

    Output
    ------
    dict with float values:
        ann_return   — compound annual growth rate: (prod(1+r))^(ppy/n) - 1
        ann_vol      — std(r) * sqrt(ppy)
        sharpe       — sqrt(ppy) * mean(r) / std(r), risk-free rate taken as 0
        max_drawdown — see max_drawdown()
        total_return — prod(1+r) - 1
        hit_rate     — share of periods with r > 0

    Test
    ----
        >>> perf_stats(pd.Series([0.01] * 252 + [-0.01] * 10))["hit_rate"]   # ~0.962
    """
    r = r.dropna()
    ann_ret = (1 + r).prod() ** (ppy / len(r)) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(ppy)
    sharpe = np.sqrt(ppy) * r.mean() / r.std(ddof=1)
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_drawdown": max_drawdown(r), "total_return": (1 + r).prod() - 1,
            "hit_rate": (r > 0).mean()}


def strategy_report(port: pd.DataFrame, ppy: int = 252) -> pd.Series:
    """Gross vs net performance plus turnover and cost figures for one strategy.

    Inputs
    ------
    port : portfolio frame (build_portfolio output)
    ppy  : periods per year

    Output
    ------
    pd.Series with keys
        net_<metric>, gross_<metric> for every metric in perf_stats
        avg_turnover  — mean turnover per period
        ann_turnover  — avg_turnover * ppy (e.g. 70 = portfolio traded 70x per year)
        ann_cost_drag — mean cost * ppy (annual return lost to costs, e.g. 0.035 = 3.5%)

    Test
    ----
        >>> s = strategy_report(res["test_portfolio"])
        >>> s["gross_ann_return"] - s["net_ann_return"]     # roughly ann_cost_drag
    """
    net, gross = perf_stats(port["net_ret"], ppy), perf_stats(port["gross_ret"], ppy)
    out = {f"net_{k}": v for k, v in net.items()}
    out.update({f"gross_{k}": v for k, v in gross.items()})
    out.update({"avg_turnover": port["turnover"].mean(),
                "ann_turnover": port["turnover"].mean() * ppy,
                "ann_cost_drag": port["cost"].mean() * ppy})
    return pd.Series(out)


def comparison_table(series: dict[str, pd.Series], ppy: int = 252) -> pd.DataFrame:
    """Side-by-side perf_stats for several strategies / benchmarks.

    Inputs
    ------
    series : dict[str, returns series], e.g. {"Model (net)": r1, "SPY B&H": r2, "Equal-weight": r3}
             (all should cover the same dates for a fair comparison)
    ppy    : periods per year

    Output
    ------
    pd.DataFrame — rows = series names, columns = perf_stats keys

    Test
    ----
        >>> comparison_table({"A": r, "B": r * 0.5})
    """
    return pd.DataFrame({k: perf_stats(v, ppy) for k, v in series.items()}).T


def capm_alpha(r: pd.Series, mkt: pd.Series, ppy: int = 252) -> dict:
    """CAPM regression r_t = alpha + beta * mkt_t + e_t by OLS (plain standard errors, not HAC).

    Inputs
    ------
    r   : strategy returns series (e.g. test_portfolio["net_ret"])
    mkt : market returns series on the same dates (e.g. SPY buy-and-hold)
    ppy : periods per year, used to annualise alpha

    Output
    ------
    dict: alpha_ann (float, annualised alpha), beta (float), alpha_t (float, t-stat of alpha)

    Test
    ----
        >>> m = pd.Series(np.random.normal(0, 0.01, 1000))
        >>> capm_alpha(0.0001 + 0.5 * m, m)["beta"]     # 0.5
    """
    df = pd.concat([r, mkt], axis=1).dropna()
    y, X = df.iloc[:, 0].values, np.column_stack([np.ones(len(df)), df.iloc[:, 1].values])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(np.diag(resid @ resid / (len(y) - 2) * np.linalg.inv(X.T @ X)))
    return {"alpha_ann": beta[0] * ppy, "beta": beta[1], "alpha_t": beta[0] / se[0]}


def period_stats(r: pd.Series, ppy: int = 252, periods: dict | None = None) -> pd.DataFrame:
    """perf_stats split by calendar year, or by named market periods.

    Inputs
    ------
    r       : returns series
    ppy     : periods per year
    periods : None -> split by calendar year;
              or dict[str, (start, end)] with date strings, e.g.
              {"2023 rally": ("2023-01-01", "2023-12-31"), "2025 tariff shock": ("2025-03-01", "2025-05-31")}

    Output
    ------
    pd.DataFrame — rows = year (str) or period name, columns = perf_stats keys
                   (periods with fewer than 2 observations are skipped)

    Test
    ----
        >>> period_stats(r).index.tolist()             # ["2023", "2024", ...]
    """
    if periods is None:
        groups = {str(y): g for y, g in r.groupby(r.index.year)}
    else:
        groups = {k: r.loc[a:b] for k, (a, b) in periods.items()}
    return pd.DataFrame({k: perf_stats(g, ppy) for k, g in groups.items() if len(g) > 1}).T


# --------------------------------------------------------------------------------------
# 3. Forecast metrics
# --------------------------------------------------------------------------------------
def forecast_metrics(preds: pd.DataFrame) -> dict:
    """How good are the probability forecasts themselves (independent of trading)?

    Input
    -----
    preds : predictions panel with columns "prob" (float 0-1), "target" (0/1), "fwd_ret" (float)

    Output
    ------
    dict with float values:
        accuracy       — share of rows where (prob > 0.5) == target
        auc_pooled     — ROC AUC over all rows (0.5 = no skill)
        auc_daily_mean — AUC computed within each date, then averaged
                         (measures ranking across tickers, which is what the strategy uses)
        log_loss       — cross-entropy (lower is better; 0.693 = coin flip)
        rank_ic_mean   — average per-date Spearman correlation of prob vs fwd_ret
        rank_ic_t      — t-stat of rank_ic_mean (|t| > 2 suggests real signal)

    Test
    ----
        >>> fm = forecast_metrics(res["test_preds"])
        >>> 0 <= fm["auc_pooled"] <= 1                  # True
    """
    y, p = preds["target"].values, preds["prob"].values
    daily_auc = preds.groupby(level="date").apply(
        lambda g: roc_auc_score(g["target"], g["prob"]) if g["target"].nunique() == 2 else np.nan)
    rank_ic = preds.groupby(level="date").apply(
        lambda g: g["prob"].corr(g["fwd_ret"], method="spearman"))
    return {"accuracy": accuracy_score(y, p > 0.5), "auc_pooled": roc_auc_score(y, p),
            "auc_daily_mean": daily_auc.mean(), "log_loss": log_loss(y, p),
            "rank_ic_mean": rank_ic.mean(),
            "rank_ic_t": rank_ic.mean() / rank_ic.std(ddof=1) * np.sqrt(rank_ic.count())}


# --------------------------------------------------------------------------------------
# 4. Plots (each returns the Figure and saves ./figures/<name>.png)
# --------------------------------------------------------------------------------------
def set_figure_dir(path: str | Path) -> Path:
    """Point all figure output at a run-specific folder, e.g. figures/<run_tag>/, and create it.

    Input  : path — folder to write PNGs into
    Output : the same Path

    Test
    ----
        >>> set_figure_dir("figures/weekly_roll2y_g0.00_2025"); sort(glob("figures/*/*.png"))
    """
    global FIG_DIR
    FIG_DIR = Path(path)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR


def _save(fig, name):
    """Apply tight layout and save fig to FIG_DIR/<name>.png (creates the folder).

    Input  : fig — matplotlib Figure; name — str file stem, e.g. "drawdown"
    Output : the same Figure
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    return fig


def save_figure(fig, name: str):
    """Save a figure into the current figure folder set by set_figure_dir."""
    return _save(fig, name)


def plot_cumulative(series: dict[str, pd.Series], title="Cumulative net return (test)",
                    name: str = "cumulative_returns"):
    """Growth of $1 for each strategy / benchmark on one chart.

    Inputs : series — dict[str, returns series]; title — str; name — output file stem
    Output : Figure, saved as FIG_DIR/<name>.png

    Test   : >>> plot_cumulative({"A": r, "B": r * 0.5})
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, r in series.items():
        ax.plot((1 + r).cumprod(), label=label)
    ax.set(title=title, ylabel="Growth of $1")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, name)


def plot_sharpe_by_model(val_table: pd.DataFrame):
    """Horizontal bar chart of validation net Sharpe for every model, coloured by family.

    Input  : val_table — results["val_table"] (needs columns model, family, net_sharpe)
    Output : Figure, saved as figures/sharpe_by_model.png

    Test   : >>> plot_sharpe_by_model(res["val_table"])
    """
    t = val_table.sort_values("net_sharpe")
    colors = t["family"].map({"ridge": "C0", "lasso": "C1", "enet": "C2"})
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(t))))
    ax.barh(t["model"], t["net_sharpe"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(title="Validation net Sharpe by model", xlabel="Net Sharpe")
    return _save(fig, "sharpe_by_model")


def plot_gross_vs_net(port: pd.DataFrame):
    """Two panels: gross vs net growth of $1 (top) and cumulative costs in bps (bottom).

    Input  : port — portfolio frame (needs gross_ret, net_ret, cost)
    Output : Figure, saved as figures/gross_vs_net.png

    Test   : >>> plot_gross_vs_net(res["test_portfolio"])
    """
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    a1.plot((1 + port["gross_ret"]).cumprod(), label="Gross")
    a1.plot((1 + port["net_ret"]).cumprod(), label="Net")
    a1.set(title="Gross vs net cumulative return", ylabel="Growth of $1"); a1.legend()
    a2.plot(port["cost"].cumsum() * 1e4)
    a2.set(title="Cumulative transaction costs", ylabel="bps")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    return _save(fig, "gross_vs_net")


def plot_turnover_vs_gate(sweep: dict[float, pd.DataFrame], ppy: int = 252):
    """Annual turnover (left axis) and net Sharpe (right axis) against the gate threshold.

    Inputs : sweep — train.gate_sweep output, dict[gate -> portfolio frame]; ppy
    Output : Figure, saved as figures/turnover_vs_gate.png

    Test   : >>> plot_turnover_vs_gate(gate_sweep(res["val_preds"][res["best_spec"].name]))
    """
    gates = list(sweep)
    turn = [sweep[g]["turnover"].mean() * ppy for g in gates]
    sharpe = [perf_stats(sweep[g]["net_ret"], ppy)["sharpe"] for g in gates]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(gates, turn, "o-", color="C0"); ax1.set_ylabel("Annual turnover", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(gates, sharpe, "s--", color="C3"); ax2.set_ylabel("Net Sharpe", color="C3")
    ax1.set(title="Turnover and net Sharpe vs trading gate", xlabel="Gate (probability gap)")
    ax1.grid(alpha=0.3)
    return _save(fig, "turnover_vs_gate")


def plot_tech_vs_trends(tech: pd.DataFrame, trends: pd.DataFrame):
    """Net growth of $1: technical-only vs technical + Google Trends, on the test set.

    Inputs : tech, trends — portfolio frames (need net_ret) on the same dates
    Output : Figure, saved as figures/tech_vs_trends.png

    Test   : >>> plot_tech_vs_trends(res["test_portfolio"], res_gt["test_portfolio"])
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot((1 + tech["net_ret"]).cumprod(), label="Technical only")
    ax.plot((1 + trends["net_ret"]).cumprod(), label="Technical + Google Trends")
    ax.set(title="Technical vs Google Trends-enhanced (net, test)", ylabel="Growth of $1")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, "tech_vs_trends")


def plot_period_sharpe(stats_by_strategy: dict[str, pd.DataFrame]):
    """Grouped bar chart of Sharpe ratio per period for each strategy.

    Input  : stats_by_strategy — dict[str, period_stats output], e.g.
             {"Model": period_stats(r_model), "SPY B&H": period_stats(r_spy)}
    Output : Figure, saved as figures/sharpe_by_period.png

    Test   : >>> plot_period_sharpe({"A": period_stats(r)})
    """
    df = pd.DataFrame({k: v["sharpe"] for k, v in stats_by_strategy.items()})
    fig, ax = plt.subplots(figsize=(9, 5))
    df.plot.bar(ax=ax)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="Sharpe ratio by period", ylabel="Sharpe")
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, "sharpe_by_period")


def plot_drawdown(series: dict[str, pd.Series]):
    """Drawdown curve (fall from running peak) for each strategy / benchmark.

    Input  : series — dict[str, returns series]
    Output : Figure, saved as figures/drawdown.png

    Test   : >>> plot_drawdown({"A": r})
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, r in series.items():
        w = (1 + r).cumprod()
        ax.plot(w / w.cummax() - 1, label=name)
    ax.set(title="Drawdown", ylabel="Drawdown"); ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, "drawdown")


def plot_coefficients(coefs: pd.Series):
    """Bar chart of the final model's standardised coefficients (zeros = dropped by LASSO).

    Input  : coefs — results["coefficients"], index = feature names, values = float
    Output : Figure, saved as figures/coefficients.png

    Test   : >>> plot_coefficients(pd.Series({"mom_20": 0.3, "vol_20": -0.1}))
    """
    c = coefs.sort_values()
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(c))))
    ax.barh(c.index, c.values)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(title="Standardised coefficients (final model)")
    return _save(fig, "coefficients")


# --------------------------------------------------------------------------------------
# 5. One call to produce everything for the test set
# --------------------------------------------------------------------------------------
def evaluate_all(results: dict, spy_close: pd.Series, ppy: int = 252,
                 trends_results: dict | None = None,
                 include_spy_benchmark: bool = True) -> dict:
    """Compute all test-set metrics and save all standard figures in one call.

    Inputs
    ------
    results        : dict returned by train.run_pipeline (technical-only model)
    spy_close      : pd.Series — SPY adjusted close, DatetimeIndex covering the test period
    ppy            : periods per year (252 daily, 52 weekly)
    trends_results : optional dict from run_pipeline with Google Trends features; if given,
                     it is added to every comparison and plot_tech_vs_trends is drawn
    include_spy_benchmark : bool — if False, SPY buy-and-hold is excluded from the comparison
                     table and return plots; it is still used as the CAPM market factor

    Output
    ------
    dict with keys:
        "comparison" : pd.DataFrame — comparison_table of model vs SPY vs equal-weight (+ trends)
        "strategy"   : pd.Series    — strategy_report of the chosen model
        "forecast"   : dict         — forecast_metrics on test predictions
        "capm"       : dict         — capm_alpha of net returns on SPY
        "by_year"    : dict[str, pd.DataFrame] — period_stats per strategy
    Side effect: PNGs in ./figures/ (cumulative_returns, drawdown, sharpe_by_model,
                 gross_vs_net, sharpe_by_period, coefficients, [tech_vs_trends])

    Test
    ----
        >>> rep = evaluate_all(res, spy_close)
        >>> rep["comparison"][["ann_return", "sharpe", "max_drawdown"]]
    """
    port, preds = results["test_portfolio"], results["test_preds"]
    spy = spy_buy_and_hold(spy_close, port.index)
    ew = equal_weight_universe(preds).reindex(port.index)
    name = f"{results['best_spec'].name} (net)"
    series = {name: port["net_ret"], "Equal-weight": ew}
    if include_spy_benchmark:
        series["SPY B&H"] = spy
    if trends_results is not None:
        series["+ Google Trends (net)"] = trends_results["test_portfolio"]["net_ret"]

    out = {
        "comparison": comparison_table(series, ppy),
        "strategy": strategy_report(port, ppy),
        "forecast": forecast_metrics(preds),
        "capm": capm_alpha(port["net_ret"], spy, ppy),
        "by_year": {k: period_stats(v, ppy) for k, v in series.items()},
    }
    plot_cumulative(series)
    plot_drawdown(series)
    plot_sharpe_by_model(results["val_table"])
    plot_gross_vs_net(port)
    plot_period_sharpe(out["by_year"])
    plot_coefficients(results["coefficients"])
    if trends_results is not None:
        plot_tech_vs_trends(port, trends_results["test_portfolio"])
    plt.close("all")
    return out
