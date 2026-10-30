"""
main.py — end-to-end run on the real data: features -> split -> train/select -> test -> evaluate.

Data
----
Reads the sector-ETF OHLCV CSVs produced by ``src/fetch_data.py`` (``data/stocks_<freq>/raw``)
plus the Fear & Greed index, builds the 19 features defined in ``src/feature_selection.py``,
joins the next-bar label from ``make_labels``, and runs the penalised-logistic-regression
pipeline in ``src/train_models.py``. All randomness is seeded with ``SEED`` (5205), so a run
is deterministic given the same input CSVs.

Model
-----
The model is fixed by GRID (one spec) and refitted on the --refit-every cadence over a rolling
--window-years window of bars (5y = 1260 daily or 260 weekly), dropping the most recent bar
before each fit (embargo) so no label looks into the period being predicted. The --val-months
window before the test period is used only for the gate sweep, never for choosing the model.

Figures
-------
Every run writes its PNGs to its own figures/<run_tag>/ folder, where run_tag encodes the
configuration, e.g. figures/daily_roll5y_val36m_mrefit_lassoC1.0_g0.00_2020-2025/.

Run
---
    uv run python src/fetch_data.py --frequency weekly   # once per frequency, needs network
    uv run python main.py --frequency weekly                            # test 2025
    uv run python main.py --frequency daily --test-start 2020-01-01 \
        --window-years 5 --val-months 36 --refit-every ME               # monthly refit
    uv run python main.py --frequency weekly --test-start 2020-01-01 --window-years 1 --val-months 6
    uv run python main.py --frequency daily --compare-refit             # refit cadence study
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))   # src modules import each other by bare module name

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluate_models import (comparison_table, equal_weight_universe, evaluate_all,
                             plot_cumulative, plot_turnover_vs_gate, save_figure,
                             set_figure_dir, spy_buy_and_hold)
from feature_selection import FEATURES, LABEL, build_feature_frame, load_fear_greed_index, load_ohlcv
from fetch_data import BENCHMARKS, CORE_ETFS
from train_models import (PERIODS_PER_YEAR, SEED, ModelSpec, PipelineConfig, build_portfolio,
                          gate_sweep, make_labels, run_pipeline, walk_forward_predict)

FEAR_GREED_PATH = PROJECT_ROOT / "data" / "fear_and_greed_index" / "fear-greed.csv"

# Fixed model for prototyping: no per-run selection, so GRID has a single spec.
GRID = [ModelSpec("lasso_C1.0", "l1", 1.0)]

TOP_K = 5
COST_BPS = 5.0
GATE = 0.0
GATE_CANDIDATES = (0.0, 0.02, 0.05)
REFIT_EVERY = "ME"
REFIT_CADENCE_CHOICES = ("D", "W-FRI", "ME", "QE")
REFIT_CADENCES = {"ME": "monthly refit", "W-FRI": "weekly refit", "D": "every-bar refit"}
REFIT_GATES = (0.0, 0.05)

WINDOW_YEARS = 2.0   # rolling training window length, in years of bars
VAL_MONTHS = 24      # validation window immediately before the test period
TEST_END = "2025-12-31"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train, backtest and evaluate on real sector ETF data.")
    parser.add_argument(
        "--frequency",
        choices=("daily", "weekly"),
        default="weekly",
        help="Bar frequency of the input data; must match a fetched data/stocks_<freq>/raw folder.",
    )
    parser.add_argument("--test-start", default="2025-01-01",
                        help="First date of the reported test period (default 2025-01-01).")
    parser.add_argument("--test-end", default=TEST_END,
                        help=f"Last date of the reported test period (default {TEST_END}).")
    parser.add_argument("--window-years", type=float, default=WINDOW_YEARS,
                        help=f"Rolling training window in years of bars (default {WINDOW_YEARS:g}).")
    parser.add_argument("--val-months", type=int, default=VAL_MONTHS,
                        help=f"Validation window in months before the test period (default {VAL_MONTHS}).")
    parser.add_argument("--refit-every", default=REFIT_EVERY, choices=REFIT_CADENCE_CHOICES,
                        help=f"Refit cadence as a pandas offset alias (default {REFIT_EVERY}).")
    parser.add_argument("--compare-refit", action="store_true",
                        help="Also compare every-bar refitting against weekly refitting on the test period.")
    return parser.parse_args()


def short_spec_name(spec: ModelSpec) -> str:
    """Compact spec label for figure folders: "lasso_C1.0" -> "lassoC1.0"."""
    return spec.name.replace("_C", "C")


def run_tag(frequency: str, test_start: str, test_end: str, gate: float,
            window_years: float, val_months: int,
            refit_every: str = REFIT_EVERY) -> str:
    """Human-readable short config label used as the figures/ subfolder name.

    Example
    -------
        >>> run_tag("weekly", "2020-01-01", "2025-12-31", 0.05, 1.0, 6, "W-FRI")
        'weekly_roll1y_val6m_wkrefit_lassoC1.0_g0.05_2020-2025'
        >>> run_tag("daily", "2020-01-01", "2025-12-31", 0.00, 5.0, 36, "ME")
        'daily_roll5y_val36m_mrefit_lassoC1.0_g0.00_2020-2025'
    """
    refit = {"D": "drefit", "W-FRI": "wkrefit", "ME": "mrefit", "QE": "qrefit"}.get(
        refit_every, refit_every)
    years = f"{pd.Timestamp(test_start):%Y}-{pd.Timestamp(test_end):%Y}"
    return (f"{frequency}_roll{window_years:g}y_val{val_months}m_{refit}_"
            f"{short_spec_name(GRID[0])}_g{gate:.2f}_{years}")


def load_panel(frequency: str) -> tuple[pd.DataFrame, pd.Series]:
    """Build the feature + label panel and the SPY benchmark from the saved raw CSVs.

    Input  : frequency — "daily" or "weekly"
    Output : tuple (panel, spy_close)
        panel     : pd.DataFrame, MultiIndex (date, ticker), FEATURES + ["fwd_ret", "target"],
                    no NaNs, dates ascending
        spy_close : pd.Series, DatetimeIndex, SPY adjusted close (full history)

    Test
    ----
        >>> panel, spy = load_panel("weekly")
        >>> panel[FEATURES + ["fwd_ret", "target"]].isna().sum().sum()     # 0
        >>> panel.index.names                                              # ["date", "ticker"]
    """
    raw_dir = PROJECT_ROOT / "data" / f"stocks_{frequency}" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Missing {raw_dir}; run `uv run python src/fetch_data.py --frequency {frequency}` first."
        )

    prices = load_ohlcv(raw_dir, tuple(CORE_ETFS))
    benchmark = load_ohlcv(raw_dir, tuple(BENCHMARKS))
    closes = pd.DataFrame({ticker: prices[ticker]["Close"] for ticker in CORE_ETFS}).sort_index()

    features = build_feature_frame(
        prices,
        frequency=frequency,
        sectors=tuple(CORE_ETFS),
        fear_greed_index=load_fear_greed_index(FEAR_GREED_PATH),
        benchmark_prices=benchmark["SPY"],
    )
    # LABEL is the same continuous forward return as "fwd_ret"; run_pipeline needs the
    # cross-sectional binary "target", so let make_labels supply both columns.
    panel = (
        features.drop(columns=[LABEL])
        .join(make_labels(closes), how="inner")
        .dropna()
        .sort_index()
    )
    if panel.empty:
        raise ValueError("Panel is empty after joining features and labels.")
    return panel, benchmark["SPY"]["Close"].rename("SPY")


def build_splits(panel: pd.DataFrame, test_start: str, test_end: str,
                 val_months: int = VAL_MONTHS) -> dict[str, tuple[str, str]]:
    """Derive train/val/test windows: val is the val_months before the test period, train is
    everything earlier (the rolling window caps what each refit actually uses).

    Inputs
    ------
    panel      : panel from load_panel (only its first date is used)
    test_start : first reported date, e.g. "2020-01-01"
    test_end   : last reported date, e.g. "2025-12-31"
    val_months : length of the validation window in months

    Output
    ------
    dict with "train"/"val"/"test" -> (start, end) ISO date strings

    Test
    ----
        >>> build_splits(panel, "2020-01-01", "2025-12-31", 24)["val"]
        ('2018-01-01', '2019-12-31')
        >>> build_splits(panel, "2020-01-01", "2025-12-31", 6)["val"]
        ('2019-07-01', '2019-12-31')
    """
    start = pd.Timestamp(test_start)
    val_start = start - pd.DateOffset(months=val_months)
    one_day = timedelta(days=1)
    first = panel.index.get_level_values("date").min()
    return {
        "train": (f"{first:%Y-%m-%d}", f"{val_start - one_day:%Y-%m-%d}"),
        "val": (f"{val_start:%Y-%m-%d}", f"{start - one_day:%Y-%m-%d}"),
        "test": (f"{start:%Y-%m-%d}", f"{pd.Timestamp(test_end):%Y-%m-%d}"),
    }


def split_panel(panel: pd.DataFrame,
                splits: dict[str, tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the panel by date using splits from build_splits (non-overlapping, in time order).

    Inputs
    ------
    panel  : pd.DataFrame, MultiIndex (date, ticker)
    splits : dict from build_splits

    Output
    ------
    tuple (train, val, test) — three panels with the same columns

    Test
    ----
        >>> tr, va, te = split_panel(panel, splits)
        >>> tr.index.get_level_values("date").max() < va.index.get_level_values("date").min()  # True
    """
    dates = panel.index.get_level_values("date")
    parts = []
    for name in ("train", "val", "test"):
        start, end = splits[name]
        part = panel[(dates >= start) & (dates <= end)]
        if part.empty:
            raise ValueError(f"No rows in the {name} window {start}..{end}; widen the data range.")
        parts.append(part)
    return tuple(parts)


def annualised_sharpe(returns: pd.Series, ppy: int) -> float:
    """Annualised Sharpe ratio (risk-free = 0) of a periodic return series."""
    sd = returns.std(ddof=1)
    return float(np.sqrt(ppy) * returns.mean() / sd) if sd > 0 else np.nan


def test_gate_comparison(results: dict, ppy: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the test portfolio at each gate in GATE_CANDIDATES and compare it with equal-weight.

    Predictions do not depend on the gate, so this reuses results["test_preds"] instead of
    refitting. The gate is a free parameter that was swept on validation; these test numbers
    are reported side by side for reference only, not used to pick anything.

    Inputs
    ------
    results : dict from run_pipeline
    ppy     : periods per year (252 daily, 52 weekly)

    Outputs
    -------
    tuple (performance, diagnostics)

    Test
    ----
        >>> perf, diag = test_gate_comparison(results, 252)
        >>> list(perf.index)      # ["gate=0.00", "gate=0.02", "gate=0.05", "Equal-weight"]
    """
    preds = results["test_preds"]
    ports = {f"gate={gate:.2f}": build_portfolio(preds, TOP_K, gate, COST_BPS)
             for gate in GATE_CANDIDATES}
    series = {name: port["net_ret"] for name, port in ports.items()}
    series["Equal-weight"] = equal_weight_universe(preds).reindex(results["test_portfolio"].index)

    performance = comparison_table(series, ppy)
    diagnostics = pd.DataFrame({
        "ann_turnover": {name: p["turnover"].mean() * ppy for name, p in ports.items()},
        "ann_cost_drag": {name: p["cost"].mean() * ppy for name, p in ports.items()},
        "total_turnover": {name: p["turnover"].sum() for name, p in ports.items()},
        "n_rebalance_days": {name: int((p["turnover"] > 1e-12).sum()) for name, p in ports.items()},
        "avg_holdings": {name: p["n_holdings"].mean() for name, p in ports.items()},
    })
    plot_cumulative(series)
    return performance, diagnostics


def refit_cadence_comparison(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                             ppy: int, window: int,
                             base_cadence: str = REFIT_EVERY) -> pd.DataFrame:
    """Refit the fixed model at each cadence in REFIT_CADENCES and compare them.

    Only the refit cadence changes: window, model spec, cost and universe are identical.
    Predicted rows are the same, so the gate interacts with cadence only through how often
    the model is allowed to change. Gates are the ones already explored on validation; they
    are not re-selected per cadence. Saves refit_cadence.png in the current figure folder.

    Inputs
    ------
    train, val, test : panels (see load_panel); history for the test walk-forward is train+val
    ppy              : periods per year (252 daily, 52 weekly)
    window           : rolling training window in bars
    base_cadence     : cadence used by the main run; always included, even if not in
                       REFIT_CADENCES

    Output
    ------
    pd.DataFrame, one row per cadence/gate plus Equal-weight

    Test
    ----
        >>> refit_cadence_comparison(train, val, test, 252, 1260).index
    """
    history = pd.concat([train, val]).sort_index()
    cadences = {base_cadence: f"{base_cadence} refit"}
    cadences.update({c: label for c, label in REFIT_CADENCES.items() if c != base_cadence})
    series, diagnostics = {}, {}
    preds = None
    for cadence, label in cadences.items():
        preds = walk_forward_predict(history, test, FEATURES, GRID[0],
                                     refit_every=cadence, window=window)
        for gate in REFIT_GATES:
            port = build_portfolio(preds, TOP_K, gate, COST_BPS)
            name = f"{label} gate={gate:.2f}"
            series[name] = port["net_ret"]
            diagnostics[name] = dict(
                ann_turnover=port["turnover"].mean() * ppy,
                ann_cost_drag=port["cost"].mean() * ppy,
                n_rebalance_days=int((port["turnover"] > 1e-12).sum()),
            )
    series["Equal-weight"] = equal_weight_universe(preds).reindex(
        test.index.get_level_values("date").unique())

    performance = comparison_table(series, ppy).join(pd.DataFrame(diagnostics).T)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = dict(zip(REFIT_GATES, ("tab:blue", "tab:olive")))
    line_styles = ("-", "--", ":")
    styles = {c: line_styles[i % len(line_styles)] for i, c in enumerate(cadences)}
    for cadence, label in cadences.items():
        for gate in REFIT_GATES:
            name = f"{label} gate={gate:.2f}"
            ax.plot((1 + series[name]).cumprod(), styles[cadence], color=colors[gate], label=name)
    ax.plot((1 + series["Equal-weight"]).cumprod(), ":", color="0.4", lw=1.6, label="Equal-weight")
    ax.set(title="Refit cadence comparison (test)", ylabel="Growth of $1")
    ax.legend(); ax.grid(alpha=0.3)
    save_figure(fig, "refit_cadence")
    plt.close(fig)
    return performance


def plot_vs_spy(results: dict, spy_close: pd.Series, ppy: int) -> pd.DataFrame:
    """Plot every gate, equal-weight and SPY buy-and-hold on one cumulative-return chart.

    SPY is a reference line only: it is tech-heavy relative to this sector universe, so it is
    kept out of the main comparison table but shown here for context. Reuses the existing test
    predictions, so no refitting happens. Saves cumulative_returns_vs_spy.png.

    Inputs
    ------
    results   : dict from run_pipeline
    spy_close : pd.Series — SPY adjusted close covering the test period
    ppy       : periods per year (252 daily, 52 weekly)

    Output
    ------
    pd.DataFrame — comparison_table of the plotted series (net returns)

    Test
    ----
        >>> plot_vs_spy(results, spy_close, 52).index   # includes "SPY B&H"
    """
    preds = results["test_preds"]
    index = results["test_portfolio"].index
    series = {f"gate={gate:.2f}": build_portfolio(preds, TOP_K, gate, COST_BPS)["net_ret"]
              for gate in GATE_CANDIDATES}
    series["Equal-weight"] = equal_weight_universe(preds).reindex(index)
    series["SPY B&H"] = spy_buy_and_hold(spy_close, index)
    plot_cumulative(series, title="Cumulative net return vs SPY (test)",
                    name="cumulative_returns_vs_spy")
    return comparison_table(series, ppy)


def main() -> None:
    """Run the full pipeline on real data and print the evaluation summary."""
    args = parse_args()
    np.random.seed(SEED)
    ppy = PERIODS_PER_YEAR[args.frequency]
    window = int(round(args.window_years * ppy))

    panel, spy_close = load_panel(args.frequency)
    splits = build_splits(panel, args.test_start, args.test_end, args.val_months)
    train, val, test = split_panel(panel, splits)
    n_dates = lambda df: df.index.get_level_values("date").nunique()
    tag = run_tag(args.frequency, args.test_start, args.test_end, GATE,
                  args.window_years, args.val_months, args.refit_every)
    fig_dir = set_figure_dir(PROJECT_ROOT / "figures" / tag)

    print(f"Panel: {args.frequency} bars, {panel.index.get_level_values('ticker').nunique()} tickers, "
          f"{panel.index.get_level_values('date').min().date()} -> "
          f"{panel.index.get_level_values('date').max().date()}")
    print(f"  train {n_dates(train)} bars ({splits['train'][0]}..{splits['train'][1]})"
          f" | val {n_dates(val)} bars ({splits['val'][0]}..{splits['val'][1]})"
          f" | test {n_dates(test)} bars ({splits['test'][0]}..{splits['test'][1]})")
    print(f"  rolling window: {args.window_years:g}y = {window} bars, refit {args.refit_every}, "
          f"gate {GATE:.2f}")
    if n_dates(train) < window:
        print(f"  WARNING: only {n_dates(train)} training bars precede the validation window but the "
              f"rolling window is {window}; the first refits use a shorter history. "
              f"Use a later --test-start.")
    print(f"  figures -> {fig_dir.relative_to(PROJECT_ROOT)}")
    print(f"Features ({len(FEATURES)}): {', '.join(FEATURES)}")

    cfg = PipelineConfig(freq=args.frequency, top_k=TOP_K, gate=GATE, cost_bps=COST_BPS,
                         refit_every=args.refit_every, window=window, grid=GRID)
    results = run_pipeline(train, val, test, FEATURES, cfg)
    print("\n== Validation ranking (top 5) ==\n",
          results["val_table"].head(5).to_string(index=False))

    sweep = gate_sweep(results["val_preds"][results["best_spec"].name], top_k=TOP_K, cost_bps=COST_BPS)
    plot_turnover_vs_gate(sweep)
    print("\n== Validation gate sweep (mean turnover / net Sharpe) ==")
    for gate, port in sweep.items():
        print(f"  gate={gate:<5} turnover={port['turnover'].mean():.3f}  "
              f"net Sharpe={annualised_sharpe(port['net_ret'], ppy):.3f}")

    report = evaluate_all(results, spy_close, ppy=ppy, include_spy_benchmark=False)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 200)
    print("\n== Test comparison ==\n", report["comparison"])

    gate_perf, gate_diag = test_gate_comparison(results, ppy)
    print("\n== Test gate comparison (predictions fixed, gate varied) ==\n", gate_perf)
    print("\n", gate_diag)

    print("\n== Test comparison vs SPY (reference only) ==\n", plot_vs_spy(results, spy_close, ppy))

    if args.compare_refit:
        print(f"\n== Refit cadence comparison (test, base {args.refit_every}) ==")
        print(refit_cadence_comparison(train, val, test, ppy, window,
                                       args.refit_every).to_string())

    print("\n== Strategy (gross vs net) ==\n", report["strategy"])
    print("\n== Forecast ==\n", pd.Series(report["forecast"]))
    print("\n== CAPM ==\n", pd.Series(report["capm"]))
    print(f"\nFigures saved to {fig_dir.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
