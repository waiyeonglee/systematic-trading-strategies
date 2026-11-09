#!/usr/bin/env python3
"""Measure the rule-based trend reference book (SMA40 / MOM52) out of sample.

The rule path has no fitted model: the signal is ``trend feature > 0`` at each
weekly bar, the book is ``trend_long_flat`` (inverse-vol weights on the on-signal
assets, cash otherwise, 4-week rebalance, moomoo commissions, no cost gate). This
script runs it on the same validation (2017-2022) and test (2023-2025) windows and
against the same benchmarks (SPY B&H, equal-weight 16) as the fitted-model study,
so the numbers are directly comparable as the naive rule-based baseline.

Usage:
    python scripts/run_trend_baseline.py                 # both rules, both windows
    python scripts/run_trend_baseline.py --rule sma40    # one rule
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd

from features import TREND_SIGNAL_COLUMNS, trend_rule_signals
from metrics import (comparison_table, equal_weight_universe, period_stats,
                     spy_buy_and_hold, strategy_report)
from portfolio import TrendParams, build_trend_portfolio
from run_backtest import build_panel
from utils.config import seed_everything
from utils.paths import RESULTS_DIR

WINDOWS = {"val": ("2017-01-01", "2022-12-31"), "test": ("2023-01-01", "2025-12-31")}
PERIODS_PER_YEAR = 52          # weekly bars
INITIAL_CAPITAL_USD = 100_000.0


def run_rule(signals: pd.DataFrame, benchmark_close: pd.Series, rule: str,
             window: str, params: TrendParams) -> dict:
    """Run one rule on one window; return the book plus its table set."""
    start, end = (pd.Timestamp(value) for value in WINDOWS[window])
    dates = signals.index.get_level_values("date")
    book_panel = signals[(dates >= start) & (dates <= end)]
    if book_panel.empty:
        raise ValueError(f"rule {rule!r} has no rows in window {window!r}")
    # Volatility is estimated from strictly earlier bars only: the walk-forward
    # equivalent of the study's shared 48-bar warmup, per window.
    history = signals[dates < start]

    book = build_trend_portfolio(book_panel, params, initial_capital_usd=INITIAL_CAPITAL_USD,
                                 vol_history=history if not history.empty else None)
    benchmarks = {
        "SPY B&H": spy_buy_and_hold(benchmark_close, book.index),
        "Equal-weight": equal_weight_universe(book_panel),
    }
    comparison = comparison_table({f"{rule} long/flat": book["net_ret"], **benchmarks},
                                  ppy=PERIODS_PER_YEAR)
    strategy = strategy_report(book, ppy=PERIODS_PER_YEAR)
    periods = period_stats(book["net_ret"], ppy=PERIODS_PER_YEAR)
    return dict(book=book, comparison=comparison, strategy=strategy, periods=periods)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule", default="all", choices=("all", *TREND_SIGNAL_COLUMNS),
                        help="trend rule to run (default: all)")
    parser.add_argument("--out-dir", default=None,
                        help="results folder (default: results/trend_baseline)")
    args = parser.parse_args()

    seed_everything()
    panel, benchmark_close, _ = build_panel()
    rules = list(TREND_SIGNAL_COLUMNS) if args.rule == "all" else [args.rule]
    params = TrendParams(vol_window=24, vol_min_periods=20, rebalance="4w",
                         broker="moomoo", cost_bps=0.0)
    out_root = Path(args.out_dir) if args.out_dir else (RESULTS_DIR / "trend_baseline")

    summary_rows = []
    metadata = {"portfolio": "trend_long_flat", "params": {
        "vol_window": params.vol_window, "vol_min_periods": params.vol_min_periods,
        "rebalance": params.rebalance, "broker": params.broker,
        "cost_bps": params.cost_bps, "initial_capital_usd": INITIAL_CAPITAL_USD,
        "periods_per_year": PERIODS_PER_YEAR},
        "rule_definition": "pred = +1 if trend feature > 0 else 0 (no fitted model)",
        "signal_columns": TREND_SIGNAL_COLUMNS, "windows": WINDOWS}
    for rule in rules:
        signals = trend_rule_signals(panel, rule)
        for window in WINDOWS:
            run = run_rule(signals, benchmark_close, rule, window, params)
            folder = out_root / rule / window
            folder.mkdir(parents=True, exist_ok=True)
            run["comparison"].to_csv(folder / "comparison.csv")
            run["strategy"].to_csv(folder / "strategy.csv")
            run["periods"].to_csv(folder / "periods.csv")
            pd.DataFrame({
                "net_ret": run["book"]["net_ret"],
                "gross_ret": run["book"]["gross_ret"],
                "cost": run["book"]["cost"],
                "turnover": run["book"]["turnover"],
                "equity_usd": run["book"]["equity_usd"],
                "n_holdings": run["book"]["n_holdings"],
            }).to_csv(folder / "series.csv")
            test = run["comparison"].loc[f"{rule} long/flat"]
            summary_rows.append(dict(rule=rule, window=window,
                                     net_sharpe=test["sharpe"],
                                     ann_return=test["ann_return"],
                                     ann_vol=test["ann_vol"],
                                     max_drawdown=test["max_drawdown"],
                                     total_return=test["total_return"],
                                     hit_rate=test["hit_rate"]))
            print(f"\n== {rule} / {window} ({start_end_text(window)}) ==")
            print(f"{run['comparison'].to_string()}\n")
            print(f"turnover {run['strategy']['ann_turnover']:.2f}/yr, "
                  f"cost {run['strategy']['ann_cost_drag'] * 100:.2f}%/yr")

    summary = pd.DataFrame(summary_rows).set_index(["rule", "window"])
    out_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_root / "summary.csv")
    (out_root / "run_info.json").write_text(json.dumps(metadata, indent=2))
    print(f"\n== summary ==\n{summary.to_string()}\nresults saved to {out_root}")


def start_end_text(window: str) -> str:
    """``"2017-01-01..2022-12-31"`` for a window key."""
    return f"{WINDOWS[window][0]}..{WINDOWS[window][1]}"


if __name__ == "__main__":
    main()
