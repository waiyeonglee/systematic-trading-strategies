"""Model selection and the walk-forward backtest pipeline.

Design (agents/model_design.py): select the grid point with the best net Sharpe on the
validation window (degenerate models excluded), then measure it once on the test window
against SPY buy-and-hold and an equal-weight-16 benchmark, with the success gate
``net Sharpe >= weaker benchmark`` and ``maxDD <= better benchmark``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from labels import TARGET_COL
from metrics import (capm_alpha, comparison_table, equal_weight_universe,
                     forecast_metrics, max_drawdown, oos_r2_by_period,
                     oos_r2_vs_baseline, perf_stats, period_stats, sharpe,
                     spy_buy_and_hold, strategy_report)
from models import ModelSpec, ar1_walk_forward, walk_forward_predict
from portfolio import ForecastVolParams, TrendParams, build_all_positions
from splits import build_splits, split_panel
from utils.artifacts import save_nested, save_parameters, save_table
from utils.config import DEFAULT_SEED, duration_slug, duration_to_bars, periods_per_year

REFIT_SLUGS = {"D": "drefit", "W-FRI": "wkrefit", "ME": "mrefit", "QE": "qrefit"}


def _vol_target_segment(params: ForecastVolParams) -> str:
    """Filename segment for the vol-target overlay, e.g. ``"vt0.1c1rf3"``."""
    if params.vol_target is None:
        return ""
    segment = f"vt{params.vol_target:g}c{params.vol_target_cap:g}"
    if params.vol_target_rf:
        segment += f"rf{params.vol_target_rf * 100:g}"
    if params.vol_target_lev_fee:
        segment += f"lv{params.vol_target_lev_fee * 100:g}"
    return segment


@dataclass(frozen=True)
class PipelineConfig:
    """Settings of one walk-forward run.

    Attributes:
        freq: Bar frequency of the panel, ``"daily"`` or ``"weekly"``.
        strategy: Registered portfolio rule, e.g. ``"forecast_vol_scaled"``.
        initial_capital_usd: Starting equity of the simulated book.
        refit_every: Refit block as a pandas offset alias, e.g. ``"QE"``.
        window_train: Rolling training window as text (e.g. ``"5y"``), or ``None`` to
            use all past data at every refit.
        embargo: Most recent dates dropped before each refit (label horizon).
        grid: Candidate :class:`ModelSpec` points; empty defaults to the design grid.
        seed: Random state shared by every refit and selection step.
        target_col: Label the models are trained on.
        forecast_vol: Settings of the ``forecast_vol_scaled`` book.
        trend: Settings of the ``trend_long_flat`` book (documented reference).
    """

    freq: str = "weekly"
    strategy: str = "forecast_vol_scaled"
    mode: str = "selection"     # "selection" (grid ranking) or "single" (pinned spec)
    initial_capital_usd: float = 100_000.0
    refit_every: str = "QE"
    window_train: str | None = "5y"
    embargo: int = 2
    grid: tuple[ModelSpec, ...] = ()
    seed: int = DEFAULT_SEED
    target_col: str = TARGET_COL
    forecast_vol: ForecastVolParams = ForecastVolParams()
    trend: TrendParams = TrendParams()

    @property
    def ppy(self) -> int:
        """Periods per year of ``freq`` (52 weekly, 252 daily)."""
        return periods_per_year(self.freq)

    @property
    def window_train_bars(self) -> int | None:
        """``window_train`` converted to bars, or ``None`` for all past data."""
        return duration_to_bars(self.window_train, self.freq) if self.window_train else None


def run_tag(config: PipelineConfig, test_start: str, test_end: str,
            window_val: str = "3y") -> str:
    """Filename of the run folder, e.g.
    ``weekly_fg_roll5y_val3y_qrefit_grid6_vt0.1c1rf3_2023-2025``.
    """
    parts = [config.freq, "fg"]
    if config.strategy.startswith("trend"):
        parts.append("trend")
    else:
        parts.append(f"roll{duration_slug(config.window_train or 'all')}")
        parts.append(f"val{duration_slug(window_val)}")
        parts.append(REFIT_SLUGS.get(config.refit_every, config.refit_every.lower()))
        if config.mode == "single" and len(config.grid) == 1:
            parts.append(config.grid[0].name)
        else:
            parts.append(f"grid{len(config.grid)}")
        segment = _vol_target_segment(config.forecast_vol)
        if segment:
            parts.append(segment)
        if config.forecast_vol.rebalance == "weekly":
            parts.append("rweekly")
        elif config.forecast_vol.rebalance == "12w":
            parts.append("r12w")
    parts.append(f"{str(test_start)[:4]}-{str(test_end)[:4]}")
    return "_".join(parts)


def select_model(history: pd.DataFrame, val_df: pd.DataFrame,
                 feature_cols: list[str], config: PipelineConfig) -> tuple[str, pd.DataFrame]:
    """Pick the grid point with the best validation net Sharpe.

    Every spec is walked forward over the validation window and its book is built with
    ``apply_vol_target=False`` (so ranking matches the plain ADR-07 book); a spec whose
    predictions are constant on the window is degenerate and excluded. The ranking
    favours the highest net Sharpe.

    Args:
        history: Training panel (rows strictly before the validation window).
        val_df: Validation panel.
        feature_cols: Model inputs.
        config: Pipeline settings (grid, window, embargo, seed, target).

    Returns:
        ``(selected_name, ranking)`` with ranking sorted by ``net_sharpe`` descending.

    Raises:
        ValueError: if every grid point is degenerate on the validation window.
    """
    if not config.grid:
        raise ValueError("select_model needs at least one model spec in config.grid")
    rows = []
    for spec in config.grid:
        preds = walk_forward_predict(history, val_df, feature_cols, spec,
                                     refit_every=config.refit_every,
                                     window=config.window_train_bars,
                                     embargo=config.embargo,
                                     seed=config.seed,
                                     target_col=config.target_col)
        degenerate = bool(float(preds.groupby(level="date")["pred"].std().mean()) < 1e-12)
        if not degenerate:
            book = build_all_positions(preds, config, apply_vol_target=False)[config.strategy]
            net_sharpe = sharpe(book["net_ret"], config.ppy)
            gross_sharpe = sharpe(book["gross_ret"], config.ppy)
            avg_turnover = float(book["turnover"].mean())
        else:
            net_sharpe = gross_sharpe = np.nan
            avg_turnover = np.nan
        rows.append(dict(model=spec.name, family=spec.name.split("_")[0],
                         degenerate=degenerate, net_sharpe=net_sharpe,
                         gross_sharpe=gross_sharpe, avg_turnover=avg_turnover))

    ranking = (pd.DataFrame(rows)
               .sort_values("net_sharpe", ascending=False, na_position="last")
               .reset_index(drop=True))
    viable = ranking[~ranking["degenerate"]]
    if viable.empty:
        raise ValueError("every model in the grid is degenerate on the validation window")
    return str(viable.iloc[0]["model"]), ranking


def trend_success_verdict(book_net_ret: pd.Series,
                          benchmarks: dict[str, pd.Series], ppy: int) -> dict:
    """Design §3.2 gate: beat the weaker benchmark on Sharpe and the better on drawdown.

    ``sharpe_bar`` is the lower of the benchmark Sharpes (weakest rival on return per
    risk); ``maxdd_bar`` is the *larger* (least negative) benchmark drawdown, because a
    smaller absolute drawdown is the harder target. Both conditions must hold.

    Args:
        book_net_ret: Net returns of the strategy book on the test window.
        benchmarks: Label -> benchmark returns on the same dates.
        ppy: Periods per year.

    Returns:
        ``{"sharpe_bar", "maxdd_bar", "sharpe_ok", "max_drawdown_ok", "success"}``.
    """
    book_sharpe = sharpe(book_net_ret, ppy)
    book_maxdd = max_drawdown(book_net_ret)
    bench_sharpes = {name: sharpe(values, ppy) for name, values in benchmarks.items()}
    bench_drawdowns = {name: max_drawdown(values) for name, values in benchmarks.items()}
    sharpe_bar = min(bench_sharpes.values())
    maxdd_bar = max(bench_drawdowns.values())
    return {
        "sharpe_bar": sharpe_bar,
        "maxdd_bar": maxdd_bar,
        "sharpe_ok": bool(book_sharpe >= sharpe_bar),
        "max_drawdown_ok": bool(book_maxdd >= maxdd_bar),
        "success": bool(book_sharpe >= sharpe_bar and book_maxdd >= maxdd_bar),
    }


def run_pipeline(panel: pd.DataFrame, feature_cols: list[str], benchmark_close: pd.Series,
                 config: PipelineConfig, test_start: str = "2023-01-01",
                 test_end: str = "2025-12-31", window_val: str = "3y",
                 out_dir: Path | None = None) -> dict:
    """Select on the validation window and measure the winner once on the test window.

    Args:
        panel: Feature panel indexed by ``(date, ticker)``.
        feature_cols: Model inputs (21 features in the v2 panel).
        benchmark_close: Benchmark (SPY) adjusted close covering the panel, by date.
        config: Pipeline settings.
        test_start / test_end / window_val: Split bounds; validation is the
            ``window_val`` span before ``test_start``.
        out_dir: Where tables are saved; ``None`` skips persistence.

    Returns:
        Dict with ``windows``, ``train/val/test``, ``ranking``, ``selected``, ``preds``,
        ``naive``, ``portfolio``, ``benchmarks``, ``verdict`` and ``run_tag``.

    Raises:
        ValueError: if the split yields an empty window or the selection fails.
    """
    windows = build_splits(panel, test_start, test_end, window_val)
    train, val, test = split_panel(panel, windows)
    selected_name, ranking = select_model(train, val, feature_cols, config)
    selected = next(spec for spec in config.grid if spec.name == selected_name)

    history = pd.concat([train, val])
    preds = walk_forward_predict(history, test, feature_cols, selected,
                                 refit_every=config.refit_every,
                                 window=config.window_train_bars,
                                 embargo=config.embargo,
                                 seed=config.seed,
                                 target_col=config.target_col)
    naive = ar1_walk_forward(history, test, config.target_col,
                             refit_every=config.refit_every,
                             window=config.window_train_bars,
                             embargo=config.embargo)

    portfolios = build_all_positions(preds, config, apply_vol_target=True)
    book = portfolios[config.strategy]
    spy = spy_buy_and_hold(benchmark_close, book.index)
    benchmarks = {"Equal-weight": equal_weight_universe(preds), "SPY B&H": spy}
    comparison = {f"{selected_name} (net)": book["net_ret"], **benchmarks}
    verdict = trend_success_verdict(book["net_ret"], benchmarks, config.ppy)

    tag = run_tag(config, test_start, test_end, window_val)
    results = dict(windows=windows, train=train, val=val, test=test,
                   ranking=ranking, selected=selected_name, preds=preds, naive=naive,
                   portfolio=book, benchmarks=benchmarks, verdict=verdict,
                   run_tag=tag, window_val=window_val, config=config)

    if out_dir is not None:
        evaluate_all(out_dir, results, feature_cols)
    return results


def evaluate_all(out_dir: Path, results: dict, feature_cols: list[str]) -> Path:
    """Write the run's tables (validation ranking, comparison, forecasts, coefficients).

    Args:
        out_dir: Destination folder for the CSV tables.
        results: Output of :func:`run_pipeline`.
        feature_cols: Model input columns (excludes labels and book helpers).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config: PipelineConfig = results["config"]
    book = results["portfolio"]
    preds, naive = results["preds"], results["naive"]
    benchmarks = results["benchmarks"]
    selected = results["selected"]

    save_table(results["ranking"], out_dir, "validation_ranking")

    comparison = {f"{selected} (net)": book["net_ret"], **benchmarks}
    save_table(comparison_table(comparison, config.ppy), out_dir, "comparison")
    save_table(comparison_table({k: v for k, v in comparison.items() if k != "SPY B&H"},
                                config.ppy), out_dir, "comparison_no_spy")

    save_table(strategy_report(book, config.ppy), out_dir, "strategy")

    model_metrics = forecast_metrics(preds, config.target_col)
    model_metrics["oos_r2_vs_ar1"] = oos_r2_vs_baseline(
        preds[config.target_col].values, preds["pred"].values, naive["pred"].values)
    save_table(model_metrics, out_dir, "forecast")
    save_table(forecast_metrics(naive, config.target_col), out_dir, "naive_forecast")

    save_table(oos_r2_by_period(preds, config.target_col), out_dir, "oos_r2_by_year")

    capm = {selected: capm_alpha(book["net_ret"], benchmarks["SPY B&H"], config.ppy)}
    save_table(pd.DataFrame(capm).T, out_dir, "capm")

    # final model: fit the selected spec on the whole history (train + val)
    history = pd.concat([results["train"], results["val"]])
    spec = selected_spec(config, selected)
    model = spec.build(config.seed)
    model.fit(history[feature_cols].values, history[config.target_col].values)
    if spec.penalty == "rff":
        # RFF coefficients live in the D-dim random-feature space, not the 21 input
        # features; record the spec plus summary stats instead of a per-feature row.
        coefs = np.asarray(model.named_steps["reg"].coef_)
        info = {"family": "rff", "alpha": spec.alpha, "rff_dim": spec.rff_dim,
                "gamma": spec.gamma, "coef_abs_mean": float(np.abs(coefs).mean()),
                "coef_norm": float(np.linalg.norm(coefs))}
        save_table(pd.DataFrame([info]), out_dir, "coefficients")
    else:
        save_table(pd.Series(model.named_steps["reg"].coef_, index=feature_cols),
                   out_dir, "coefficients")

    per_period = {name: period_stats(values, config.ppy)
                  for name, values in comparison.items()}
    save_nested(per_period, out_dir, "performance_by_period")
    no_spy = {name: period_stats(values, config.ppy)
              for name, values in comparison.items() if name != "SPY B&H"}
    save_nested(no_spy, out_dir, "performance_by_period_no_spy")

    save_parameters(run_parameters(results), out_dir)

    print_summary(results)
    return out_dir


def selected_spec(config: PipelineConfig, name: str) -> ModelSpec:
    """The grid point whose name is ``name`` (raises if absent)."""
    for spec in config.grid:
        if spec.name == name:
            return spec
    raise ValueError(f"selected model {name!r} is not in the configured grid")


def run_parameters(results: dict) -> dict:
    """Plain mapping of the effective run settings, for ``run_config.yaml``."""
    from features import feature_columns
    from utils.config import END_DATE_INCLUSIVE, START_DATE

    config: PipelineConfig = results["config"]
    windows = results["windows"]
    feature_cols = feature_columns(use_fear_greed=True)
    return {
        "run": results["run_tag"],
        "seed": config.seed,
        "data": {"start_date": START_DATE, "end_date": END_DATE_INCLUSIVE},
        "features": {"frequency": config.freq, "use_fear_greed": True,
                     "columns": feature_cols},
        "validation": {
            "test_start": windows["test"][0],
            "test_end": windows["test"][1],
            "window_train": config.window_train,
            "window_train_bars": config.window_train_bars,
            "window_val": results["window_val"],
            "refit_every": config.refit_every,
            "ic_t_min": None,
            "ic_gate_window": None,
        },
        "model": {
            "mode": config.mode,
            "target": next(name for name, col in
                           __import__("labels", fromlist=["MODEL_TARGETS"]).MODEL_TARGETS.items()
                           if col == config.target_col),
            "embargo_bars": config.embargo,
            "specs": [{"name": spec.name, "family": spec.name.split("_")[0],
                       "alpha": spec.alpha, "l1_ratio": spec.l1_ratio,
                       "rff_dim": spec.rff_dim, "gamma": spec.gamma}
                      for spec in config.grid],
        },
        "portfolio": {
            "strategy": config.strategy,
            "initial_capital_usd": config.initial_capital_usd,
            "forecast_vol_scaled": {
                "vol_window": config.forecast_vol.vol_window,
                "vol_min_periods": config.forecast_vol.vol_min_periods,
                "rebalance": config.forecast_vol.rebalance,
                "broker": config.forecast_vol.broker,
                "cost_bps": config.forecast_vol.cost_bps,
                "cost_gate_multiple": config.forecast_vol.cost_gate_multiple,
                "vol_target": config.forecast_vol.vol_target,
                "vol_target_cap": config.forecast_vol.vol_target_cap,
                "vol_target_rf": config.forecast_vol.vol_target_rf,
                "vol_target_lev_fee": config.forecast_vol.vol_target_lev_fee,
            },
        },
    }


def print_summary(results: dict) -> None:
    """Human-readable summary of the run (also written next to the tables)."""
    config: PipelineConfig = results["config"]
    book = results["portfolio"]
    verdict = results["verdict"]
    windows = results["windows"]
    net_stats = perf_stats(book["net_ret"], config.ppy)
    lines = [
        f"run: {results['run_tag']}",
        f"split: train {windows['train'][0]}..{windows['train'][1]} | "
        f"val {windows['val'][0]}..{windows['val'][1]} | "
        f"test {windows['test'][0]}..{windows['test'][1]}",
        f"selected model: {results['selected']} (best validation net Sharpe)",
        f"test net Sharpe {net_stats['sharpe']:.4f} | "
        f"maxDD {net_stats['max_drawdown']:.2%} | "
        f"total {net_stats['total_return']:.2%}",
        f"gate: sharpe_ok={verdict['sharpe_ok']} (bar {verdict['sharpe_bar']:.4f}), "
        f"max_drawdown_ok={verdict['max_drawdown_ok']} (bar {verdict['maxdd_bar']:.2%}), "
        f"SUCCESS={verdict['success']}",
    ]
    print("\n".join(lines))
