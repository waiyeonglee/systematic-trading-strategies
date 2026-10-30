"""Every figure produced by the pipeline. Each function saves one PNG and returns it."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from labels import TARGET_COL
from metrics import oos_r2_by_period, sharpe
from utils.plotting import save_figure


def plot_cumulative(series: dict[str, pd.Series], results_dir: Path,
                    title: str = "Cumulative net return (test)",
                    name: str = "cumulative_returns") -> plt.Figure:
    """Growth of $1 for each strategy and benchmark on one chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, returns in series.items():
        ax.plot((1 + returns).cumprod(), label=label)
    ax.set(title=title, ylabel="Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    return save_figure(fig, results_dir, name)


def plot_drawdown(series: dict[str, pd.Series], results_dir: Path,
                  name: str = "drawdown") -> plt.Figure:
    """Drawdown curve, the fall from each running peak."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, returns in series.items():
        wealth = (1 + returns).cumprod()
        ax.plot(wealth / wealth.cummax() - 1, label=label)
    ax.set(title="Drawdown", ylabel="Drawdown")
    ax.legend()
    ax.grid(alpha=0.3)
    return save_figure(fig, results_dir, name)


def plot_sharpe_by_model(val_table: pd.DataFrame, results_dir: Path) -> plt.Figure:
    """Horizontal bar chart of validation net Sharpe, coloured by model family."""
    table = val_table.sort_values("net_sharpe")
    palette = {"ridge": "C0", "lasso": "C1", "enet": "C2"}
    colors = [palette.get(family, "C3") for family in table["family"]]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(table))))
    ax.barh(table["model"], table["net_sharpe"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(title="Validation net Sharpe by model", xlabel="Net Sharpe")
    return save_figure(fig, results_dir, "sharpe_by_model")


def plot_cumulative_by_gate(sweep: dict[float, pd.DataFrame], equal_weight: pd.Series,
                            results_dir: Path, title: str,
                            name: str = "cumulative_returns_by_gate") -> plt.Figure:
    """Growth of $1 for every tested gate, with equal-weight as a reference line.

    Args:
        sweep: Gate value -> portfolio frame, as returned by ``portfolio.gate_sweep``.
        equal_weight: Equal-weight benchmark returns on the same dates.
        results_dir: Run folder for the PNG.
        title: Figure title, e.g. ``"Cumulative net return by gate (validation)"``.
        name: File stem.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for gate, portfolio in sweep.items():
        ax.plot((1 + portfolio["net_ret"]).cumprod(), label=f"gate={gate:g}")
    ax.plot((1 + equal_weight).cumprod(), ":", color="0.4", lw=1.6, label="Equal-weight")
    ax.set(title=title, ylabel="Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    return save_figure(fig, results_dir, name)


def plot_gross_vs_net(port: pd.DataFrame, results_dir: Path) -> plt.Figure:
    """Gross versus net growth of $1, plus cumulative costs in basis points."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    top.plot((1 + port["gross_ret"]).cumprod(), label="Gross")
    top.plot((1 + port["net_ret"]).cumprod(), label="Net")
    top.set(title="Gross vs net cumulative return", ylabel="Growth of $1")
    top.legend()
    bottom.plot(port["cost"].cumsum() * 1e4)
    bottom.set(title="Cumulative transaction costs", ylabel="bps")
    for ax in (top, bottom):
        ax.grid(alpha=0.3)
    return save_figure(fig, results_dir, "gross_vs_net")


def plot_turnover_vs_gate(sweep: dict[float, pd.DataFrame], results_dir: Path,
                          ppy: int = 252) -> plt.Figure:
    """Annual turnover and net Sharpe against the gate threshold."""
    gates = list(sweep)
    turnover = [sweep[gate]["turnover"].mean() * ppy for gate in gates]
    sharpes = [sharpe(sweep[gate]["net_ret"], ppy) for gate in gates]

    fig, left = plt.subplots(figsize=(8, 5))
    left.plot(gates, turnover, "o-", color="C0")
    left.set_ylabel("Annual turnover", color="C0")
    right = left.twinx()
    right.plot(gates, sharpes, "s--", color="C3")
    right.set_ylabel("Net Sharpe", color="C3")
    left.set(title="Turnover and net Sharpe vs trading gate",
             xlabel="Gate (predicted-return gap)")
    left.grid(alpha=0.3)
    return save_figure(fig, results_dir, "turnover_vs_gate")


def plot_tech_vs_trends(tech: pd.DataFrame, trends: pd.DataFrame, results_dir: Path) -> plt.Figure:
    """Net growth of $1: technical-only versus technical plus Google Trends."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot((1 + tech["net_ret"]).cumprod(), label="Technical only")
    ax.plot((1 + trends["net_ret"]).cumprod(), label="Technical + Google Trends")
    ax.set(title="Technical vs Google Trends-enhanced (net, test)", ylabel="Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    return save_figure(fig, results_dir, "tech_vs_trends")


def plot_period_sharpe(stats_by_strategy: dict[str, pd.DataFrame], results_dir: Path,
                       name: str = "sharpe_by_period") -> plt.Figure:
    """Grouped bar chart of Sharpe ratio per period for each strategy."""
    table = pd.DataFrame({label: stats["sharpe"] for label, stats in stats_by_strategy.items()})
    fig, ax = plt.subplots(figsize=(9, 5))
    table.plot.bar(ax=ax)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="Sharpe ratio by period", ylabel="Sharpe")
    ax.grid(alpha=0.3, axis="y")
    return save_figure(fig, results_dir, name)


def plot_coefficients(coefficients: pd.Series, results_dir: Path) -> plt.Figure:
    """Bar chart of the final model's standardised coefficients."""
    ordered = coefficients.sort_values()
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(ordered))))
    ax.barh(ordered.index, ordered.values)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(title="Standardised coefficients (final model)")
    return save_figure(fig, results_dir, "coefficients")


def plot_oos_r2(preds: pd.DataFrame, results_dir: Path,
                target_col: str = TARGET_COL) -> plt.Figure:
    """Out-of-sample RÃ‚Â² by year, with the cumulative RÃ‚Â² path alongside."""
    by_year = oos_r2_by_period(preds, target_col)
    error = (preds[target_col] - preds["pred"]) ** 2
    sse = error.groupby(level="date").sum().cumsum()
    sst = (preds[target_col] ** 2).groupby(level="date").sum().cumsum()
    cumulative = 1 - sse / sst

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    left.bar(by_year.index.astype(str), by_year.values)
    left.axhline(0, color="k", lw=0.8)
    left.set(title="Out-of-sample RÃ‚Â² by year", ylabel="RÃ‚Â²")
    right.plot(cumulative)
    right.axhline(0, color="k", lw=0.8)
    right.set(title="Cumulative out-of-sample RÃ‚Â²", ylabel="RÃ‚Â²")
    for ax in (left, right):
        ax.grid(alpha=0.3)
    return save_figure(fig, results_dir, "oos_r2")

