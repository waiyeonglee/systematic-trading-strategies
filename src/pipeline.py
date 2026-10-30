"""End-to-end pipeline: select a model on validation, trade and evaluate on test."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from labels import TARGET_COL
from metrics import (capm_alpha, comparison_table, equal_weight_universe, forecast_metrics,
                     oos_r2_by_period, period_stats, sharpe, spy_buy_and_hold, strategy_report)
from models import ModelSpec, default_grid, model_family, walk_forward_predict
from plots import (plot_coefficients, plot_cumulative, plot_drawdown, plot_gross_vs_net,
                   plot_oos_r2, plot_period_sharpe, plot_sharpe_by_model,
                   plot_tech_vs_trends)
from portfolio import build_positions
from utils.artifacts import save_nested, save_table
from utils.checks import require_columns
from utils.config import DEFAULT_SEED, duration_slug, periods_per_year
from utils.plotting import close_all

REFIT_SLUGS = {"D": "drefit", "W-FRI": "wkrefit", "ME": "mrefit", "QE": "qrefit"}


def run_tag(settings, grid: list[ModelSpec]) -> str:
    """Build the results-folder label from the settings that change the results.

    Args:
        settings: Loaded ``Settings``.
        grid: Model specifications the run will score.

    Returns:
        A filesystem-safe tag such as
        ``daily_fg_roll5y_val3y_mrefit_lasso_a0.0001_topk5_g0_c5_2024-2025``.
    """
    model = grid[0].name if len(grid) == 1 else f"grid{len(grid)}"
    strategy = f"{settings.strategy.replace('_', '')}{settings.top_k}"
    return (
        f"{settings.frequency}_"
        f"{'fg' if settings.use_fear_greed else 'nofg'}_"
        f"roll{duration_slug(settings.window_train)}_"
        f"val{duration_slug(settings.window_val)}_"
        f"{REFIT_SLUGS.get(settings.refit_every, settings.refit_every)}_"
        f"{model}_{strategy}_g{settings.gate:g}_c{settings.cost_bps:g}_"
        f"{pd.Timestamp(settings.test_start):%Y}-{pd.Timestamp(settings.test_end):%Y}"
    )


@dataclass
class PipelineConfig:
    """Settings for one pipeline run.

    Attributes:
        freq: ``"daily"`` or ``"weekly"``; sets periods per year.
        strategy: Portfolio strategy name; must exist in ``portfolio.STRATEGIES``.
        top_k: Number of assets held (used by the ``top_k`` strategy).
        gate: Predicted-return gap needed to swap, in return units.
        cost_bps: One-way transaction cost in basis points.
        refit_every: Refit cadence as a pandas offset alias.
        window_train: Rolling training window in dates; ``None`` uses all past data.
        grid: Model specifications to compare on validation.
        seed: Random state for the global numpy seed and every model refit.
    """

    freq: str = "daily"
    strategy: str = "top_k"
    top_k: int = 5
    gate: float = 0.0
    cost_bps: float = 5.0
    refit_every: str = "QE"
    window_train: int | None = None
    grid: list[ModelSpec] = field(default_factory=default_grid)
    seed: int = DEFAULT_SEED


def select_model(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str],
                 cfg: PipelineConfig) -> tuple[ModelSpec, pd.DataFrame, dict, dict]:
    """Rank every model in the grid by validation net Sharpe.

    Models whose predictions are identical across tickers on every date (all coefficients
    shrunk to zero) are flagged as degenerate and excluded, because their picks are arbitrary.

    Args:
        train: Panel before the validation window.
        val: Validation panel.
        feature_cols: Model inputs.
        cfg: Pipeline settings.

    Returns:
        ``(best, table, val_preds, val_portfolios)`` where ``table`` has one row per model
        sorted best first.

    Raises:
        ValueError: if every configured model is degenerate (all coefficients shrunk to
            zero, so every asset gets the same prediction).
    """
    ppy = periods_per_year(cfg.freq)
    rows, val_preds, val_portfolios = [], {}, {}
    for spec in cfg.grid:
        predictions = walk_forward_predict(train, val, feature_cols, spec,
                                           cfg.refit_every, cfg.window_train,
                                           seed=cfg.seed)
        portfolio = build_positions(predictions, cfg)
        val_preds[spec.name], val_portfolios[spec.name] = predictions, portfolio

        degenerate = predictions.groupby(level="date")["pred"].std().mean() < 1e-12
        rows.append({
            "model": spec.name,
            "family": model_family(spec.name),
            "degenerate": degenerate,
            "net_sharpe": np.nan if degenerate else sharpe(portfolio["net_ret"], ppy),
            "gross_sharpe": sharpe(portfolio["gross_ret"], ppy),
            "avg_turnover": portfolio["turnover"].mean(),
        })
        print(f"  [val] {spec.name:<24} net Sharpe = {rows[-1]['net_sharpe']:.3f}")

    table = (pd.DataFrame(rows)
             .sort_values("net_sharpe", ascending=False, na_position="last")
             .reset_index(drop=True))

    if table["net_sharpe"].isna().all():
        names = ", ".join(f"{spec.name} (alpha={spec.alpha:g})" for spec in cfg.grid)
        raise ValueError(
            f"every configured model is degenerate: {names}. "
            "Every coefficient shrank to zero, so all assets get the same prediction. "
            f"On {cfg.freq} bars the excess return is ~0.01, so LASSO/Elastic Net need a "
            "much smaller alpha (try 1e-5 to 1e-3) unless the data is weekly and the "
            "signal large; Ridge tolerates larger alpha."
        )
    best = next(spec for spec in cfg.grid if spec.name == table.loc[0, "model"])
    return best, table, val_preds, val_portfolios


def run_pipeline(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                 feature_cols: list[str], cfg: PipelineConfig | None = None) -> dict:
    """Select a model on validation, then run it once on the test period.

    Args:
        train: Training panel.
        val: Validation panel, dated after ``train``.
        test: Test panel, dated after ``val``.
        feature_cols: Model inputs.
        cfg: Pipeline settings; defaults to :class:`PipelineConfig`.

    Returns:
        Dict with ``config``, ``best_spec``, ``val_table``, ``val_preds``,
        ``val_portfolios``, ``test_preds``, ``test_portfolio`` and ``coefficients``.

    Raises:
        AssertionError: if a panel lacks required columns or the panels overlap in time.
    """
    cfg = cfg or PipelineConfig()
    np.random.seed(cfg.seed)

    for name, frame in (("train", train), ("val", val), ("test", test)):
        require_columns(frame, feature_cols + [TARGET_COL, "fwd_ret"], f"{name} panel")
    assert train.index.get_level_values("date").max() < val.index.get_level_values("date").min()
    assert val.index.get_level_values("date").max() < test.index.get_level_values("date").min()

    label = "model" if len(cfg.grid) == 1 else f"{len(cfg.grid)} model specs"
    print(f"scoring {label} on validation...")
    best, val_table, val_preds, val_portfolios = select_model(train, val, feature_cols, cfg)
    print(f"selected: {best.name}")

    print("out-of-sample test with the selected model...")
    history = pd.concat([train, val])
    test_preds = walk_forward_predict(history, test, feature_cols, best,
                                      cfg.refit_every, cfg.window_train, seed=cfg.seed)
    test_portfolio = build_positions(test_preds, cfg)

    final_model = best.build(cfg.seed).fit(history[feature_cols].values, history[TARGET_COL].values)
    coefficients = pd.Series(final_model.named_steps["reg"].coef_.ravel(), index=feature_cols)

    return {
        "config": cfg,
        "best_spec": best,
        "val_table": val_table,
        "val_preds": val_preds,
        "val_portfolios": val_portfolios,
        "test_preds": test_preds,
        "test_portfolio": test_portfolio,
        "coefficients": coefficients,
    }


def evaluate_all(results: dict, benchmark_close: pd.Series, ppy: int, results_dir: Path,
                 trends_results: dict | None = None,
                 target_col: str = TARGET_COL) -> dict:
    """Compute the test-set metrics, then write the figures and tables for the run.

    Every comparison is written twice: once including SPY buy-and-hold and once with only
    the strategy and the equal-weight benchmark.

    Args:
        results: Output of :func:`run_pipeline`.
        benchmark_close: Benchmark adjusted close covering the test period.
        ppy: Periods per year.
        results_dir: Run folder for the PNGs and CSVs.
        trends_results: Optional second run with extra features, added to the comparison.
        target_col: Label the model was trained on.

    Returns:
        Dict with ``comparison``, ``comparison_no_spy``, ``strategy``, ``forecast``,
        ``r2_by_year``, ``capm`` and ``by_year``.
    """
    portfolio, preds = results["test_portfolio"], results["test_preds"]
    model_name = f"{results['best_spec'].name} (net)"
    benchmark = spy_buy_and_hold(benchmark_close, portfolio.index)
    equal_weight = equal_weight_universe(preds).reindex(portfolio.index)

    without_spy = {model_name: portfolio["net_ret"], "Equal-weight": equal_weight}
    with_spy = {model_name: portfolio["net_ret"], "SPY B&H": benchmark,
                "Equal-weight": equal_weight}
    if trends_results is not None:
        trends = trends_results["test_portfolio"]["net_ret"]
        without_spy["+ Google Trends (net)"] = trends
        with_spy["+ Google Trends (net)"] = trends

    report = {
        "comparison": comparison_table(with_spy, ppy),
        "comparison_no_spy": comparison_table(without_spy, ppy),
        "strategy": strategy_report(portfolio, ppy),
        "forecast": forecast_metrics(preds, target_col),
        "r2_by_year": oos_r2_by_period(preds, target_col),
        "capm": capm_alpha(portfolio["net_ret"], benchmark, ppy),
        "by_year": {name: period_stats(values, ppy) for name, values in with_spy.items()},
        "by_year_no_spy": {name: period_stats(values, ppy) for name, values in without_spy.items()},
    }

    plot_cumulative(with_spy, results_dir)
    plot_cumulative(without_spy, results_dir, title="Cumulative net return (test, ex-SPY)",
                    name="cumulative_returns_no_spy")
    plot_drawdown(with_spy, results_dir)
    plot_drawdown(without_spy, results_dir, name="drawdown_no_spy")
    plot_sharpe_by_model(results["val_table"], results_dir)
    plot_gross_vs_net(portfolio, results_dir)
    plot_period_sharpe(report["by_year"], results_dir)
    plot_period_sharpe(report["by_year_no_spy"], results_dir, name="sharpe_by_period_no_spy")
    plot_coefficients(results["coefficients"], results_dir)
    plot_oos_r2(preds, results_dir, target_col)
    if trends_results is not None:
        plot_tech_vs_trends(portfolio, trends_results["test_portfolio"], results_dir)

    save_table(results["val_table"], results_dir, "validation_ranking")
    save_table(results["coefficients"], results_dir, "coefficients")
    save_table(report["comparison"], results_dir, "comparison")
    save_table(report["comparison_no_spy"], results_dir, "comparison_no_spy")
    save_table(report["strategy"], results_dir, "strategy")
    save_table(report["forecast"], results_dir, "forecast")
    save_table(report["r2_by_year"], results_dir, "oos_r2_by_year")
    save_table(report["capm"], results_dir, "capm")
    save_nested(report["by_year"], results_dir, "performance_by_period")
    save_nested(report["by_year_no_spy"], results_dir, "performance_by_period_no_spy")

    close_all()
    return report

