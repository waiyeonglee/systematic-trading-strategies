#!/usr/bin/env python3
"""Run the AR(1) forecast baseline and score it against the historical mean.

The baseline predicts the H-bar forward return from the trailing H-bar return, one
lag, fitted per ticker on a rolling window and refit on the study's cadence. It is a
forecast-space baseline only: no positions, no costs — the metrics are OOS R²,
RMSE, direction accuracy and cross-sectional rank IC.

Usage:
    python scripts/run_forecast_baseline.py                    # H=4, the winner's horizon
    python scripts/run_forecast_baseline.py --horizon 1        # textbook AR(1) on 1-bar returns
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

from baselines import ar1_forecast, r2_against, r2_against_by_year
from metrics import forecast_metrics, oos_r2_by_period
from run_backtest import build_panel
from utils.config import duration_to_bars, periods_per_year, seed_everything
from utils.paths import RESULTS_DIR

WINDOWS = {"val": ("2017-01-01", "2022-12-31"), "test": ("2023-01-01", "2025-12-31")}


def slice_window(preds: pd.DataFrame, window: str) -> pd.DataFrame:
    """Panel rows inside one named window."""
    start, end = (pd.Timestamp(value) for value in WINDOWS[window])
    dates = preds.index.get_level_values("date")
    return preds[(dates >= start) & (dates <= end)]


def window_metrics(preds: pd.DataFrame, target_col: str) -> dict:
    """All forecast metrics for one evaluated window."""
    metrics = forecast_metrics(preds, target_col)
    metrics["oos_r2_vs_mean"] = r2_against(preds, "pred_mean", target_col)
    metrics["n_rows"] = len(preds)
    metrics["n_dates"] = preds.index.get_level_values("date").nunique()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=4, choices=(1, 4, 12),
                        help="target horizon in bars (default: 4, the ranking winner)")
    parser.add_argument("--refit-every", default="QE", help="refit cadence (default: QE)")
    parser.add_argument("--window-train", default="5y",
                        help="rolling training window (default: 5y = 260 weekly bars)")
    parser.add_argument("--min-obs", type=int, default=40,
                        help="minimum training pairs for the OLS slope (default: 40)")
    parser.add_argument("--out-dir", default=None,
                        help="results folder (default: results/forecast_baseline)")
    args = parser.parse_args()

    seed_everything()
    panel, _, _ = build_panel()
    target_col = f"fwd_ret_{args.horizon}bar"
    preds, coefficients = ar1_forecast(
        panel, target_col, refit_every=args.refit_every,
        window=duration_to_bars(args.window_train, "weekly"),
        embargo=args.horizon, min_obs=args.min_obs)

    out_root = Path(args.out_dir) if args.out_dir else (RESULTS_DIR / "forecast_baseline")
    root = out_root / f"ar1_h{args.horizon}"
    rows, yearly = [], []
    for window in WINDOWS:
        window_preds = slice_window(preds, window)
        if window_preds.empty:
            raise ValueError(f"AR(1) produced no predictions in window {window!r}")
        metrics = window_metrics(window_preds, target_col)
        rows.append(dict(window=window, **metrics))
        yearly.append(pd.DataFrame({
            "oos_r2_zero": oos_r2_by_period(window_preds, target_col),
            "oos_r2_vs_mean": r2_against_by_year(window_preds, "pred_mean", target_col),
        }).rename_axis("year").assign(window=window))
        folder = root / window
        folder.mkdir(parents=True, exist_ok=True)
        window_preds[["pred", "pred_mean", target_col, "fwd_ret"]].to_csv(
            folder / "predictions.csv")

    summary = pd.DataFrame(rows).set_index("window")
    yearly_all = pd.concat(yearly).reset_index().set_index(["window", "year"])
    summary.to_csv(root / "metrics.csv")
    yearly_all.to_csv(root / "by_year.csv")
    coefficients.to_csv(root / "coefficients.csv")
    combined_path = out_root / "summary.csv"
    combined = summary.assign(horizon=args.horizon).reset_index()
    if combined_path.exists():
        old = pd.read_csv(combined_path)
        combined = pd.concat([old[old["horizon"] != args.horizon], combined])
    combined.sort_values(["horizon", "window"]).to_csv(combined_path, index=False)
    (root / "run_info.json").write_text(json.dumps({
        "model": "AR(1): y = a + b*return_Hbar, per ticker, OLS",
        "target": target_col,
        "refit_every": args.refit_every,
        "window_train": args.window_train,
        "embargo": args.horizon,
        "min_obs": args.min_obs,
        "periods_per_year": periods_per_year("weekly"),
        "windows": WINDOWS,
    }, indent=2))

    pd.set_option("display.float_format", "{:.4f}".format)
    print(f"\n== AR(1) H={args.horizon} ({target_col}) ==")
    print(summary[["n_rows", "n_dates", "oos_r2", "oos_r2_vs_mean", "rmse",
                   "direction_acc", "corr", "rank_ic_mean", "rank_ic_t"]].to_string())
    print(f"\n== OOS R² by year ==\n{yearly_all.to_string()}")
    slopes = coefficients.groupby("refit")["slope"].mean()
    print(f"\nslope across refits: mean {slopes.mean():.4f}, "
          f"median {slopes.median():.4f}, share negative {float((slopes < 0).mean()):.2f}")
    print(f"results saved to {root}")


if __name__ == "__main__":
    main()
