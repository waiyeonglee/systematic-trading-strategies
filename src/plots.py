"""Report figures for one backtest run, generated inside the pipeline.

Every run folder now ships a ``plots/`` directory with the time-series figures a
report needs, for **both** windows of the run (the test window the book is
measured on, and the validation window the model was selected on):

    01_cumulative_returns   net wealth vs Equal-weight, SPY B&H and the AR(1) baseline
    02_gross_vs_net         gross vs net wealth, cost/sleeve drag shaded
    03_drawdown             underwater curves (strategy + benchmarks + AR(1) baseline)
    04_costs_and_trades     cumulative transaction cost + turnover, with
                            net-buy / net-sell markers and long/short shading
    05_exposure             gross / net exposure, vol-target scale, leverage blocks
    06_weights_held         per-ticker weights held (longs up, shorts down)
    07_weights_heatmap      date x ticker weight heatmap
    08_holdings_counts      n holdings / n long / n short over time
    09_annual_returns       calendar-year net returns vs benchmarks + AR(1) baseline
    10_oos_r2_by_year       out-of-sample R2 per calendar year

The AR(1) baseline is drawn in every figure that shows a benchmark: a per-ticker
one-lag regression of the forward return on the trailing return, traded with the same
book and costs (``baselines.ar1_baseline``). It is a no-feature reference so a panel
answers "do the fitted features add anything over the trailing return?" at a glance.
It reaches the figures through ``plot_benchmarks`` and is never part of ``benchmarks``,
which feed the success gate and ``comparison.csv``.

The figures are drawn from the objects the pipeline already computes (the book
frame with its per-date ``weights_wide`` / ``orders_wide`` attachments, the
benchmark return series and the predictions panel), so no extra simulation is
needed. Output is deterministic: the Agg backend, a fixed style, no randomness.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import max_drawdown, oos_r2_by_period, perf_stats

# ---------------------------------------------------------------------------
# style + colours (fixed, report-friendly)
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
})

C_STRATEGY = "#1f4e79"   # dark blue: the strategy book
C_GROSS = "#8c8c8c"      # gray: gross book
C_COST = "#c0392b"       # red: costs / drawdown
C_EW = "#27ae60"         # green: Equal-weight-16 benchmark
C_SPY = "#d35400"        # orange: SPY buy-and-hold benchmark
C_AR1 = "#7f7f7f"        # gray, dashed: AR(1) no-feature baseline
C_VOLSCALE = "#8e44ad"   # purple: vol-target scale
C_LONG = "#1f4e79"
C_SHORT = "#c0392b"

AR1_LABEL = "AR(1) baseline"


def _benchmark_style(label: str) -> dict:
    """Line style of one benchmark: colour, width and dash pattern."""
    if "SPY" in label:
        return {"color": C_SPY, "lw": 1.4}
    if label == AR1_LABEL:
        return {"color": C_AR1, "lw": 1.2, "ls": "--"}
    return {"color": C_EW, "lw": 1.4}

_TICKER_COLORS = [plt.cm.tab20(i) for i in range(20)]  # 16 tradables + slack


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _wealth(returns: pd.Series) -> pd.Series:
    """Growth-of-1 curve from periodic returns."""
    return (1 + returns.fillna(0.0)).cumprod()


def _underwater(returns: pd.Series) -> pd.Series:
    """Drawdown series (<= 0) from periodic returns."""
    wealth = _wealth(returns)
    return wealth / wealth.cummax() - 1.0


def _annual_returns(series: pd.Series) -> pd.Series:
    """Compound return per calendar year."""
    return series.groupby(series.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)


def _finish(fig, path: Path, subtitle: str | None = None) -> None:
    """Lay out (reserving room for the subtitle), save and close the figure."""
    if subtitle:
        fig.suptitle(subtitle, fontsize=7.5, color="#555555", y=0.96)
        rect = (0, 0, 1, 0.90)
    else:
        rect = (0, 0, 1, 1)
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _date_axis(ax) -> None:
    """Year ticks on a date axis."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _stats_line(book: pd.DataFrame, ppy: int, overlay: str) -> str:
    """Subtitle with the headline stats of one window's book."""
    stats = perf_stats(book["net_ret"], ppy)
    return (f"net Sharpe {stats['sharpe']:.3f} · maxDD {stats['max_drawdown']:.2%} · "
            f"total {stats['total_return']:.2%} · ann ret {stats['ann_return']:.2%} · "
            f"ann vol {stats['ann_vol']:.2%} · {overlay}")


def _overlay_label(book: pd.DataFrame, params) -> str:
    """Describe the volatility overlay and how often it actually scaled the book."""
    if params.vol_target is None:
        return "overlay off"
    bound = int((book["vol_scale"] != 1.0).sum())
    return (f"vol target {params.vol_target:.0%} cap {params.vol_target_cap:g} "
            f"(scaled on {bound}/{len(book)} bars)")


# ---------------------------------------------------------------------------
# individual figures
# ---------------------------------------------------------------------------

def _plot_cumulative_returns(book: pd.DataFrame, benchmarks: dict[str, pd.Series],
                             path: Path, header: str, subtitle: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    net = _wealth(book["net_ret"])
    ax.plot(net.index, net.values, label="Strategy (net)", color=C_STRATEGY, lw=2.0)
    for label, rets in benchmarks.items():
        cum = _wealth(rets.reindex(net.index))
        ax.plot(cum.index, cum.values, label=label, **_benchmark_style(label))
    for i, line in enumerate(ax.get_lines()):
        # stagger label offsets so coincident terminal values don't overlap
        off = (0, 6) if i % 2 == 0 else (0, -14)
        ax.annotate(f"{line.get_ydata()[-1]:.2f}x", xy=(line.get_xdata()[-1], line.get_ydata()[-1]),
                    xytext=off, textcoords="offset points",
                    ha="center", fontsize=7, color=line.get_color())
    ax.axhline(1.0, color="#999999", lw=0.8, ls=":")
    ax.set_title(header)
    ax.set_ylabel("Growth of 1 (cumulative)")
    ax.legend(loc="upper left")
    _date_axis(ax)
    _finish(fig, path, subtitle)


def _plot_gross_vs_net(book: pd.DataFrame, benchmarks: dict[str, pd.Series],
                       path: Path, header: str, subtitle: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    gross, net = _wealth(book["gross_ret"]), _wealth(book["net_ret"])
    ax.plot(gross.index, gross.values, label="Gross", color=C_GROSS, lw=1.4)
    ax.plot(net.index, net.values, label="Net (after cost + sleeve)", color=C_STRATEGY, lw=2.0)
    ax.fill_between(gross.index, gross.values, net.values,
                    color=C_COST, alpha=0.15, label="drag (cost + rf/lev sleeve)")
    drag = float(gross.iloc[-1] - net.iloc[-1])
    ax.annotate(f"total drag {drag:.2%}", xy=(net.index[-1], net.iloc[-1]),
                xytext=(0, 6), textcoords="offset points", ha="center",
                fontsize=7.5, color=C_COST)
    ax.set_title(header)
    ax.set_ylabel("Growth of 1 (cumulative)")
    ax.legend(loc="upper left")
    _date_axis(ax)
    _finish(fig, path, subtitle)


def _plot_drawdown(book: pd.DataFrame, benchmarks: dict[str, pd.Series],
                   path: Path, header: str, subtitle: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    under = _underwater(book["net_ret"])
    ax.fill_between(under.index, under.values, 0.0, color=C_COST, alpha=0.30,
                    label=f"Strategy (maxDD {max_drawdown(book['net_ret']):.2%})")
    for label, rets in benchmarks.items():
        u = _underwater(rets.reindex(under.index))
        ax.plot(u.index, u.values, label=f"{label} (maxDD {u.min():.2%})",
                **_benchmark_style(label))
    ax.axhline(0.0, color="#555555", lw=0.8)
    ax.set_title(header)
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=7.5)
    _date_axis(ax)
    _finish(fig, path, subtitle)


def _plot_costs_and_trades(book: pd.DataFrame, benchmarks: dict[str, pd.Series], path: Path, header: str, subtitle: str) -> None:
    """Cumulative transaction cost + turnover, with buy/sell markers and the
    long/short regime shaded (the 'pointer where the action was' view)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True,
                                   height_ratios=(0.9, 1.0))
    cum_cost = book["cost"].cumsum() * 100.0
    ax1.step(cum_cost.index, cum_cost.values, where="post", color=C_COST, lw=1.6)
    ax1.fill_between(cum_cost.index, cum_cost.values, 0.0, step="post",
                     color=C_COST, alpha=0.12)
    ax1.set_ylabel("Cumulative cost (% of equity)")
    ax1.set_title(header)

    orders = getattr(book, "orders_wide", None)
    net_order = orders.sum(axis=1) if orders is not None else pd.Series(0.0, index=book.index)
    rebal = book["rebalanced"].astype(bool)
    turnover = book["turnover"] * 100.0

    # long / short regime shading (background pointer)
    exposure = book["net_exposure"]
    runs, cur_sign, start = [], None, None
    for date, value in exposure.items():
        sign = 1 if value > 0 else (-1 if value < 0 else 0)
        if sign != cur_sign:
            if cur_sign is not None:
                runs.append((start, date, cur_sign))
            cur_sign, start = sign, date
    runs.append((start, exposure.index[-1], cur_sign))
    for a, b, sign in runs:
        if sign > 0:
            ax2.axvspan(a, b, color=C_LONG, alpha=0.05)
        elif sign < 0:
            ax2.axvspan(a, b, color=C_SHORT, alpha=0.05)

    ax2.bar(turnover.index, turnover.values, width=3.0, color="#c9c9c9",
            label="turnover %")
    buy = rebal & (net_order > 0)
    sell = rebal & (net_order < 0)
    flat = rebal & (net_order == 0)
    ax2.scatter(turnover.index[buy], turnover[buy], marker="^", s=22,
                color=C_COST, label="net buy", zorder=5)
    ax2.scatter(turnover.index[sell], turnover[sell], marker="v", s=22,
                color=C_EW, label="net sell", zorder=5)
    ax2.scatter(turnover.index[flat], turnover[flat], marker="o", s=8,
                color="#555555", label="rebalance (no trade)", zorder=5)
    ax2.set_ylabel("Turnover (%)")
    ax2.set_xlabel("")
    ax2.legend(loc="upper right", ncol=2, fontsize=7)
    _date_axis(ax2)
    _finish(fig, path, subtitle)


def _plot_exposure(book: pd.DataFrame, benchmarks: dict[str, pd.Series], path: Path, header: str, subtitle: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.plot(book.index, book["gross_exposure"] * 100.0, label="Gross exposure %",
            color=C_STRATEGY, lw=1.5)
    ax.plot(book.index, book["net_exposure"] * 100.0, label="Net exposure %",
            color=C_COST, lw=1.5)
    ax.axhline(100.0, color="#999999", lw=0.8, ls=":")
    ax.set_ylabel("Exposure (% of equity)")
    ax2 = ax.twinx()
    ax2.plot(book.index, book["vol_scale"], label="Vol-target scale",
             color=C_VOLSCALE, lw=1.2, alpha=0.85)
    ax2.set_ylim(0.0, max(1.05, float(book["vol_scale"].max()) * 1.05))
    ax2.set_ylabel("Vol-target scale", color=C_VOLSCALE)
    ax2.tick_params(axis="y", labelcolor=C_VOLSCALE)
    blocked = book["leverage_blocked"].astype(bool)
    if blocked.any():
        ax.scatter(book.index[blocked], book["gross_exposure"][blocked] * 100.0,
                   marker="x", s=20, color="#111111", label="leverage blocked", zorder=5)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=7.5)
    _date_axis(ax)
    _finish(fig, path, subtitle)


def _plot_weights_held(book: pd.DataFrame, benchmarks: dict[str, pd.Series], path: Path, header: str, subtitle: str) -> None:
    """Stacked area of per-ticker weights: longs stack up from 0, shorts down."""
    weights = getattr(book, "weights_wide", None)
    if weights is None:
        return
    order = weights.abs().mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    colours = {ticker: _TICKER_COLORS[i % len(_TICKER_COLORS)]
               for i, ticker in enumerate(order)}
    ax.stackplot(weights.index, weights[order].clip(lower=0.0).T.values,
                 labels=[f"{t}" for t in order], colors=[colours[t] for t in order], alpha=0.9)
    ax.stackplot(weights.index, weights[order].clip(upper=0.0).T.values,
                 colors=[colours[t] for t in order], alpha=0.9)
    ax.axhline(0.0, color="#555555", lw=0.8)
    ax.set_title(header)
    ax.set_ylabel("Portfolio weight (long up / short down)")
    ax.set_ylim(min(-0.05, float(weights.clip(upper=0.0).sum(axis=1).min()) * 1.08),
                max(0.05, float(weights.clip(lower=0.0).sum(axis=1).max()) * 1.08))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4,
              fontsize=6, columnspacing=0.8)
    _date_axis(ax)
    _finish(fig, path, subtitle)


def _plot_weights_heatmap(book: pd.DataFrame, benchmarks: dict[str, pd.Series], path: Path, header: str, subtitle: str) -> None:
    weights = getattr(book, "weights_wide", None)
    if weights is None:
        return
    tickers = list(weights.columns)
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    extent = [mdates.date2num(weights.index[0]), mdates.date2num(weights.index[-1]),
              len(tickers) - 0.5, -0.5]
    vmax = max(0.05, float(np.nanpercentile(np.abs(weights.values), 99)))
    image = ax.imshow(weights.T.values, aspect="auto", extent=extent,
                      cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_yticks(range(len(tickers)), tickers)
    ax.set_title(header)
    ax.set_xlabel("")
    fig.colorbar(image, ax=ax, shrink=0.85, label="weight")
    _finish(fig, path, subtitle)


def _plot_holdings_counts(book: pd.DataFrame, benchmarks: dict[str, pd.Series], path: Path, header: str, subtitle: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.step(book.index, book["n_holdings"], where="post", color=C_GROSS,
            label="n holdings", lw=1.3)
    ax.step(book.index, book["n_long"], where="post", color=C_EW, label="n long", lw=1.6)
    ax.step(book.index, book["n_short"], where="post", color=C_COST, label="n short", lw=1.6)
    ax.set_title(header)
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    _date_axis(ax)
    _finish(fig, path, subtitle)


def _plot_annual_returns(book: pd.DataFrame, benchmarks: dict[str, pd.Series],
                         path: Path, header: str, subtitle: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    years = sorted(set(book["net_ret"].index.year) |
                   {y for s in benchmarks.values() for y in s.index.year})
    positions = np.arange(len(years))
    bars = [("Strategy (net)", book["net_ret"], C_STRATEGY)]
    for label, rets in benchmarks.items():
        bars.append((label, rets, _benchmark_style(label)["color"]))
    width = 0.8 / len(bars)
    for offset, (label, rets, colour) in enumerate(bars):
        values = _annual_returns(rets).reindex(years).fillna(0.0)
        ax.bar(positions + (offset - (len(bars) - 1) / 2) * width, values.values,
               width, label=label, color=colour)
    ax.axhline(0.0, color="#555555", lw=0.8)
    ax.set_xticks(positions, [str(y) for y in years])
    ax.set_title(header)
    ax.set_ylabel("Annual net return")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=len(bars),
              fontsize=7.5)
    _finish(fig, path, subtitle)


def _plot_oos_r2_by_year(preds: pd.DataFrame, target_col: str,
                         path: Path, header: str, subtitle: str) -> None:
    by_year = oos_r2_by_period(preds, target_col)
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.bar([str(y) for y in by_year.index], by_year.values, color=C_STRATEGY,
           width=0.55)
    ax.axhline(0.0, color="#555555", lw=0.8)
    ax.set_title(header)
    ax.set_ylabel("OOS R² (vs zero forecast)")
    _finish(fig, path, subtitle)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

def _figure_plan(results: dict):
    """(window name, book, preds, benchmarks) for both windows of the run.

    ``plot_benchmarks`` / ``val_plot_benchmarks`` are the gate benchmarks plus the AR(1)
    baseline, so the published figures can be read against a no-feature rule.
    """
    config = results["config"]
    return [
        ("test", results["portfolio"], results["preds"],
         results["plot_benchmarks"], config.ppy),
        ("validation", results["val_portfolio"], results["val_preds"],
         results["val_plot_benchmarks"], config.ppy),
    ]


def plot_run(results: dict, out_dir: Path) -> Path:
    """Write the full figure set for both windows into ``<out_dir>/plots/``.

    Args:
        results: Output of :func:`pipeline.run_pipeline` (book frames with
            ``weights_wide`` / ``orders_wide`` attachments).
        out_dir: Run folder; figures land in ``<out_dir>/plots/``.

    Returns:
        The plots directory that was created.
    """
    out = Path(out_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    selected = results["selected"]
    run_tag = results["run_tag"]
    params = results["config"].forecast_vol
    steps = (
        ("01_cumulative_returns", _plot_cumulative_returns),
        ("02_gross_vs_net", _plot_gross_vs_net),
        ("03_drawdown", _plot_drawdown),
        ("04_costs_and_trades", _plot_costs_and_trades),
        ("05_exposure", _plot_exposure),
        ("06_weights_held", _plot_weights_held),
        ("07_weights_heatmap", _plot_weights_heatmap),
        ("08_holdings_counts", _plot_holdings_counts),
        ("09_annual_returns", _plot_annual_returns),
    )
    for window, book, preds, benchmarks, ppy in _figure_plan(results):
        header = (f"{selected} · {window} window "
                  f"{book.index.min().date()} → {book.index.max().date()}")
        # two-line subtitle: headline stats first, run tag second (a single long
        # line would clip at the figure edges)
        subtitle = (f"{_stats_line(book, ppy, _overlay_label(book, params))}"
                    f"\nrun {run_tag}")
        for stem, draw in steps:
            draw(book, benchmarks, out / f"{window}_{stem}.png", header, subtitle)
        _plot_oos_r2_by_year(preds, results["config"].target_col,
                             out / f"{window}_10_oos_r2_by_year.png", header, subtitle)
    return out
