#!/usr/bin/env python3
"""Run the AR(1) baseline: forecasts, traded book, and everything the report quotes.

The baseline predicts the H-bar forward return from the trailing H-bar return, one
lag, fitted per ticker on a rolling window and refit on the study's cadence. Two
outputs are produced for each evaluation window:

* forecast metrics (OOS R² and RMSE), and
* a traded book, so the baseline can be compared with the fitted models on Sharpe
  ratio and drawdown rather than on forecast accuracy alone.

The book inherits the cadence of the selected study winner (see ``STUDY_BOOK``) so the
comparison reflects the forecasts rather than a different portfolio rule. Both windows
are always traded; no extra flags are needed.

Usage:
    python scripts/run_forecast_baseline.py                    # H=4, the micro winner's horizon
    python scripts/run_forecast_baseline.py --horizon 1        # textbook AR(1) on 1-bar returns
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd

from baselines import ar1_book, ar1_forecast, ar1_warmup, r2_against, r2_against_by_year
from labels import target_horizon
from metrics import forecast_metrics, oos_r2_by_period, period_stats, strategy_report
from portfolio import ForecastVolParams
from run_backtest import build_panel
from utils.config import duration_to_bars, periods_per_year, seed_everything
from utils.paths import RESULTS_DIR

WINDOWS = {"val": ("2019-01-01", "2022-12-31"), "test": ("2023-01-01", "2025-12-31")}

# Book and refit cadence per horizon, taken from the selected validation winner of each
# study: ME refit, 10% vol target, 3% cash, cap 1.0, and that winner's own vol window.
STUDY_HORIZON = {
    4: dict(refit_every="ME", book=ForecastVolParams(
        vol_window=4, vol_min_periods=4, rebalance="4w",
        vol_target=0.10, vol_target_cap=1.0, vol_target_rf=0.03)),
    12: dict(refit_every="ME", book=ForecastVolParams(
        vol_window=24, vol_min_periods=20, rebalance="4w",
        vol_target=0.10, vol_target_cap=1.0, vol_target_rf=0.03)),
}
DEFAULT_BOOK = ForecastVolParams(vol_window=24, vol_min_periods=20, rebalance="4w",
                                 vol_target=0.10, vol_target_cap=1.0, vol_target_rf=0.03)
VOL_SEED_BARS = 48


def slice_window(frame: pd.DataFrame, window: str) -> pd.DataFrame:
    """Rows of ``frame`` (indexed by date) inside one named window."""
    start, end = (pd.Timestamp(value) for value in WINDOWS[window])
    dates = frame.index.get_level_values("date")
    return frame[(dates >= start) & (dates <= end)]


def window_metrics(preds: pd.DataFrame, target_col: str) -> dict:
    """All forecast metrics for one evaluated window."""
    metrics = forecast_metrics(preds, target_col)
    metrics["oos_r2_vs_mean"] = r2_against(preds, "pred_mean", target_col)
    metrics["n_rows"] = len(preds)
    metrics["n_dates"] = preds.index.get_level_values("date").nunique()
    return metrics


def trade_window(panel: pd.DataFrame, window: str, target_col: str, *,
                 refit_every: str, window_train: str, min_obs: int,
                 book: ForecastVolParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast and trade the AR(1) baseline inside one window.

    The fit uses only bars strictly before the window and the volatility estimate is
    seeded from earlier predictions, so nothing from the evaluated window leaks in.

    Returns:
        ``(preds, book_frame)`` for the evaluated window.
    """
    start, end = (pd.Timestamp(value) for value in WINDOWS[window])
    dates = panel.index.get_level_values("date")
    history = panel[dates < start]
    evaluated = panel[(dates >= start) & (dates <= end)]
    kwargs = dict(refit_every=refit_every,
                  window=duration_to_bars(window_train, "weekly"),
                  embargo=target_horizon(target_col), min_obs=min_obs)

    preds, _ = ar1_forecast(pd.concat([history, evaluated]), target_col, **kwargs)
    preds = slice_window(preds, window)
    if preds.empty:
        raise ValueError(f"AR(1) produced no tradable predictions in window {window!r}")

    warmup = ar1_warmup(history, target_col, VOL_SEED_BARS, **kwargs)
    frame = ar1_book(preds, book, ppy=periods_per_year("weekly"),
                     vol_history=None if warmup.empty else warmup)
    return preds, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=4, choices=(1, 4, 12),
                        help="target horizon in bars (default: 4, the ranking winner)")
    parser.add_argument("--refit-every", default=None,
                        help="refit cadence (default: the study's own, ME)")
    parser.add_argument("--window-train", default="8y",
                        help="rolling training window (default: 8y = 416 weekly bars)")
    parser.add_argument("--min-obs", type=int, default=40,
                        help="minimum training pairs for the OLS slope (default: 40)")
    parser.add_argument("--out-dir", default=None,
                        help="results folder (default: results/forecast_baseline)")
    args = parser.parse_args()

    study = STUDY_HORIZON.get(args.horizon, {})
    refit_every = args.refit_every or study.get("refit_every", "QE")
    book = replace(study.get("book", DEFAULT_BOOK))

    seed_everything()
    panel, _, _ = build_panel()
    panel = panel.sort_index()
    target_col = f"fwd_ret_{args.horizon}bar"
    preds, coefficients = ar1_forecast(
        panel, target_col, refit_every=refit_every,
        window=duration_to_bars(args.window_train, "weekly"),
        embargo=args.horizon, min_obs=args.min_obs)

    out_root = Path(args.out_dir) if args.out_dir else (RESULTS_DIR / "forecast_baseline")
    root = out_root / f"ar1_h{args.horizon}"

    rows, yearly, book_rows, book_yearly = [], [], [], []
    for window in WINDOWS:
        window_preds = slice_window(preds, window)
        if window_preds.empty:
            raise ValueError(f"AR(1) produced no predictions in window {window!r}")
        rows.append(dict(window=window, **window_metrics(window_preds, target_col)))
        yearly.append(pd.DataFrame({
            "oos_r2_zero": oos_r2_by_period(window_preds, target_col),
            "oos_r2_vs_mean": r2_against_by_year(window_preds, "pred_mean", target_col),
        }).rename_axis("year").assign(window=window))

        folder = root / window
        folder.mkdir(parents=True, exist_ok=True)
        window_preds[["pred", "pred_mean", target_col, "fwd_ret"]].to_csv(
            folder / "predictions.csv")

        _, book_frame = trade_window(
            panel, window, target_col, refit_every=refit_every,
            window_train=args.window_train, min_obs=args.min_obs, book=book)
        book_frame.to_csv(folder / "book.csv")
        book_rows.append(dict(window=window,
                              **strategy_report(book_frame,
                                                periods_per_year("weekly")).to_dict()))
        book_yearly.append(
            period_stats(book_frame["net_ret"], periods_per_year("weekly"))
            .rename_axis("year").assign(window=window))

    summary = pd.DataFrame(rows).set_index("window")
    yearly_all = pd.concat(yearly).reset_index().set_index(["window", "year"])
    book_summary = pd.DataFrame(book_rows).set_index("window")
    book_yearly_all = pd.concat(book_yearly).reset_index().set_index(["window", "year"])

    summary.to_csv(root / "metrics.csv")
    yearly_all.to_csv(root / "by_year.csv")
    coefficients.to_csv(root / "coefficients.csv")
    book_summary.to_csv(root / "book_metrics.csv")
    book_yearly_all.to_csv(root / "book_by_year.csv")
    book_summary[[c for c in book_summary.columns if c.startswith("net_")]].rename(
        columns=lambda c: c.removeprefix("net_")).to_csv(root / "comparison.csv")

    combined_path = out_root / "summary.csv"
    combined = summary.assign(horizon=args.horizon).reset_index()
    if combined_path.exists():
        old = pd.read_csv(combined_path)
        combined = pd.concat([old[old["horizon"] != args.horizon], combined])
    combined.sort_values(["horizon", "window"]).to_csv(combined_path, index=False)

    (root / "run_info.json").write_text(json.dumps({
        "model": "AR(1): y = a + b*return_Hbar, per ticker, OLS",
        "target": target_col,
        "refit_every": refit_every,
        "window_train": args.window_train,
        "embargo": args.horizon,
        "min_obs": args.min_obs,
        "vol_seed_bars": VOL_SEED_BARS,
        "periods_per_year": periods_per_year("weekly"),
        "windows": WINDOWS,
        "book": {
            "rebalance": book.rebalance,
            "vol_window": book.vol_window,
            "vol_min_periods": book.vol_min_periods,
            "vol_target": book.vol_target,
            "vol_target_cap": book.vol_target_cap,
            "vol_target_rf": book.vol_target_rf,
            "vol_target_lev_fee": book.vol_target_lev_fee,
            "cost_gate_multiple": book.cost_gate_multiple,
            "broker": book.broker,
        },
    }, indent=2))

    pd.set_option("display.float_format", "{:.4f}".format)
    print(f"\n== AR(1) H={args.horizon} ({target_col}) ==")
    print(summary[["n_rows", "n_dates", "oos_r2", "oos_r2_vs_mean", "rmse"]].to_string())
    print(f"\n== OOS R2 by year ==\n{yearly_all.to_string()}")

    slopes = coefficients.groupby("refit")["slope"].mean()
    print(f"\nslope across refits: mean {slopes.mean():.4f}, "
          f"median {slopes.median():.4f}, share negative {float((slopes < 0).mean()):.2f}")

    print("\n== traded book (net) ==")
    print(book_summary[["net_ann_return", "net_ann_vol", "net_sharpe",
                        "net_max_drawdown", "net_hit_rate",
                        "avg_turnover"]].to_string())
    print("\n== book by year (net) ==")
    print(book_yearly_all[["sharpe", "ann_return", "max_drawdown"]]
          .unstack("window").to_string())
    print(f"\nresults saved to {root}")


if __name__ == "__main__":
    main()
