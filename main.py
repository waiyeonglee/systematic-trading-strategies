"""Command-line entry point: fetch data, build feature reports, run the backtest."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # src modules import each other by bare module name

import pandas as pd

import data
from features import (build_feature_frame, feature_columns, feature_label_correlation,
                      save_correlation_heatmap)
from metrics import comparison_table, equal_weight_universe, sharpe
from models import specs_from_config
from pipeline import PipelineConfig, evaluate_all, run_pipeline, run_tag
from plots import plot_cumulative_by_gate, plot_turnover_vs_gate
from portfolio import gate_sweep
from splits import build_splits, split_panel
from utils.artifacts import save_parameters, save_table
from utils.cli import build_parser
from utils.config import (BENCHMARKS, duration_to_bars, load_settings,
                          periods_per_year)
from utils.paths import FEAR_GREED_PATH, FEATURES_DIR, RESULTS_DIR, raw_data_dir
from utils.plotting import close_all


def run_fetch(settings) -> None:
    """Download the raw OHLCV data at the configured frequency."""
    data.fetch_and_save(settings.frequency, settings.data_start, settings.data_end)


def run_features(settings) -> None:
    """Write the feature table and correlation report at the configured frequency."""
    frequency = settings.frequency
    columns = feature_columns(settings.use_fear_greed)
    fear_greed = data.load_fear_greed_index(FEAR_GREED_PATH) if settings.use_fear_greed else None

    folder = raw_data_dir(frequency)
    prices = data.load_ohlcv(folder)
    benchmark = data.load_ohlcv(folder, BENCHMARKS)[BENCHMARKS[0]]
    features = build_feature_frame(prices, frequency=frequency,
                                   fear_greed_index=fear_greed,
                                   benchmark_prices=benchmark)
    correlation, sample_end = feature_label_correlation(features, columns)

    output_dir = FEATURES_DIR / frequency
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "features.csv")
    correlation.to_csv(output_dir / "feature_label_correlation.csv")
    save_correlation_heatmap(
        correlation,
        output_dir / "feature_label_correlation.png",
        f"{frequency.title()} Feature/Label Correlation "
        f"(All Labeled Samples through {sample_end.date()})",
    )
    print(f"{frequency.title()}: {features.shape[0]:,} rows x {len(columns)} features; "
          f"labels through {sample_end.date()}; saved to {output_dir}")


def run_backtest(settings) -> None:
    """Train on the real data, trade the test period and print the evaluation summary."""
    ppy = periods_per_year(settings.frequency)
    feature_cols = feature_columns(settings.use_fear_greed)
    panel, benchmark_close = data.load_panel(settings.frequency,
                                             use_fear_greed=settings.use_fear_greed)
    train, val, test = split_panel(panel, build_splits(panel, settings.test_start,
                                                       settings.test_end, settings.window_val))

    window_train = duration_to_bars(settings.window_train, settings.frequency)
    grid = specs_from_config(settings.model_entries())
    tag = run_tag(settings, grid)
    results_dir = RESULTS_DIR / tag

    print(f"panel: {settings.frequency} bars, "
          f"{panel.index.get_level_values('ticker').nunique()} tickers, "
          f"{len(feature_cols)} features, "
          f"{panel.index.get_level_values('date').min().date()} -> "
          f"{panel.index.get_level_values('date').max().date()}")
    for name, frame in (("train", train), ("val", val), ("test", test)):
        dates = frame.index.get_level_values("date")
        print(f"  {name:<5} {dates.nunique():4d} bars ({dates.min().date()}..{dates.max().date()})")

    print(f"  window_train {settings.window_train} = {window_train} bars, "
          f"window_val {settings.window_val}, refit {settings.refit_every}")
    print(f"  model [{settings.model_mode}]: {grid[0].name}"
          + (f" + {len(grid) - 1} more" if len(grid) > 1 else ""))
    print(f"  portfolio [{settings.strategy}]: top_k={settings.top_k}, gate={settings.gate:.2f}, "
          f"cost_bps={settings.cost_bps:g}")
    print(f"  results -> {results_dir.relative_to(PROJECT_ROOT)}")

    save_parameters({
        "run": tag,
        "seed": settings.seed,
        "data": {"start_date": settings.data_start, "end_date": settings.data_end},
        "features": {"frequency": settings.frequency,
                     "use_fear_greed": settings.use_fear_greed,
                     "columns": feature_cols},
        "validation": {"test_start": settings.test_start, "test_end": settings.test_end,
                       "window_train": settings.window_train,
                       "window_train_bars": window_train,
                       "window_val": settings.window_val,
                       "refit_every": settings.refit_every},
        "model": {"mode": settings.model_mode,
                  "specs": [{"name": spec.name, "family": spec.name.split("_")[0],
                             "alpha": spec.alpha, "l1_ratio": spec.l1_ratio}
                            for spec in grid]},
        "portfolio": {"strategy": settings.strategy, "k": settings.top_k,
                      "gate": settings.gate, "cost_bps": settings.cost_bps,
                      "gate_sweep": list(settings.gates)},
    }, results_dir)

    cfg = PipelineConfig(freq=settings.frequency, strategy=settings.strategy, top_k=settings.top_k,
                         gate=settings.gate, cost_bps=settings.cost_bps,
                         refit_every=settings.refit_every, window_train=window_train, grid=grid,
                         seed=settings.seed)
    results = run_pipeline(train, val, test, feature_cols, cfg)

    if settings.strategy == "top_k":
        validation_preds = results["val_preds"][results["best_spec"].name]
        sweep = gate_sweep(validation_preds, settings.gates, settings.top_k, settings.cost_bps)
        plot_turnover_vs_gate(sweep, results_dir, ppy)
        plot_cumulative_by_gate(
            sweep, equal_weight_universe(validation_preds).reindex(next(iter(sweep.values())).index),
            results_dir, "Cumulative net return by gate (validation)")
        save_table(pd.DataFrame({
            "ann_turnover": {gate: port["turnover"].mean() * ppy
                             for gate, port in sweep.items()},
            "sharpe": {gate: sharpe(port["net_ret"], ppy) for gate, port in sweep.items()},
        }).rename_axis("gate"), results_dir, "gate_sweep")
        close_all()

    report = evaluate_all(results, benchmark_close, ppy, results_dir)

    if settings.strategy == "top_k":
        test_sweep = gate_sweep(results["test_preds"], settings.gates,
                                settings.top_k, settings.cost_bps)
        plot_cumulative_by_gate(
            test_sweep,
            equal_weight_universe(results["test_preds"]).reindex(results["test_portfolio"].index),
            results_dir, "Cumulative net return by gate (test)",
            name="cumulative_returns_by_gate_test")
        save_table(comparison_table({f"gate={gate:g}": port["net_ret"]
                                     for gate, port in test_sweep.items()}, ppy),
                   results_dir, "gate_comparison_test")
        close_all()

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 200)
    print("\n== Validation ranking (top 5) ==\n", results["val_table"].head(5).to_string(index=False))
    print("\n== Test comparison (vs SPY) ==\n", report["comparison"])
    print("\n== Test comparison (ex-SPY) ==\n", report["comparison_no_spy"])
    print("\n== Strategy (gross vs net) ==\n", report["strategy"])
    print("\n== Forecast ==\n", pd.Series(report["forecast"]))
    print("\n== OOS R2 by year ==\n", report["r2_by_year"])
    print("\n== CAPM ==\n", pd.Series(report["capm"]))

    print(f"\nresults saved to {results_dir}")


def main() -> None:
    """Load the settings, parse the command and dispatch to it."""
    settings = load_settings()
    args = build_parser().parse_args()
    commands = {"fetch": run_fetch, "features": run_features, "backtest": run_backtest}
    commands[args.command](settings)


if __name__ == "__main__":
    main()
