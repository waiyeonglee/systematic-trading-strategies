#!/usr/bin/env python3
"""Compare the two refit arms of one feature block: ME (left) vs QE (right).

For every feature block the joint ranking produces one best candidate per refit arm
(monthly, quarterly). This module draws those two arms side by side, marks which arm
won the ranking with a star, and reports their validation and test metrics.

Outputs, inside ``results/_study_<block>/comparison/``::

    figures/
      cumulative_test.png        ME (left) and QE (right) on the test window
      cumulative_val.png         the same on the validation window
      drawdown_test.png          underwater curves, same two-panel layout
    metrics_aggregate.txt        book metrics per arm x window
    metrics_forecast.txt         OOS R², rank IC, ...
    metrics_by_year.txt          the same metrics split by calendar year

The metric tables cover both arms on both windows, with the forecasts' OOS R²
reported next to the book metrics.

Usage:
    python scripts/compare_arms.py micro
    python scripts/compare_arms.py macro
    python scripts/compare_arms.py both

Each arm's candidate is re-run through the pipeline (the predictions come from the
cache, so this is the book simulation only), which is what produces both the figures
and the metric tables from a single source of truth.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import run_study
from macro_panel import FEATURE_SETS, build_panel
from metrics import forecast_metrics, oos_r2_by_period, period_stats, perf_stats
from study_worker import VALIDATION

TEST = run_study.TEST
VAL_WINDOW_SLUG = run_study.VAL_WINDOW_SLUG


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def arm_candidates(feature_set: str) -> list[dict]:
    """Best live candidate of each refit arm.

    Returns:
        One dict per arm (``label`` in ``{"ME", "QE"}``) in ranking order, each with
        ``row`` (the ranking row); which arm won the ranking is marked later, in
        :func:`arm_panels`, so no duplicate "net best" row is carried around.
    """
    path = run_study.STUDY_ROOTS[feature_set] / "validation_ranking.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run `python scripts/run_study.py rank "
                         f"--features {feature_set}` first")
    ranking = pd.read_csv(path)
    live = ranking[~ranking["degenerate"] & ranking["net_sharpe"].notna()]
    if live.empty:
        raise SystemExit(f"no non-degenerate candidate in {path}")

    best = {refit: group.iloc[0] for refit, group in live.groupby("refit")}
    missing = [label for label in ARM_ORDER if label not in best]
    if missing:
        raise SystemExit(f"no live {missing} row in {path}")
    # fixed order (ME then QE), independent of which arm the ranking put first
    return [{"label": label, "kind": "arm", "row": best[label]}
            for label in ARM_ORDER]


def run_arm(entry: dict, feature_set: str, settings: dict) -> dict:
    """Re-run one arm's candidate and capture its two windows' books and forecasts."""
    row = entry["row"]
    config = run_study._config(int(row["look_ahead"]), row["rebalance"], row["refit"],
                               row["model"],
                               {"vol_window": int(row["vol_window"]),
                                "vol_target": run_study._optional(row["vol_target"]),
                                "vol_rf": float(row["vol_rf"]),
                                "vol_cap": float(row["vol_cap"]),
                                "lev_fee": float(row["lev_fee"])},
                               feature_set)
    results = run_study.run_pipeline_no_outputs(settings, config)
    return {**entry, "results": results, "config": config, "row": row}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

ARM_ORDER = ("ME", "QE")
# One colour for the strategy in every panel, so the arms are told apart by position
# and title (ME left, QE right) rather than by colour; the benchmarks keep theirs.
STRATEGY_COLOUR = "#1f77b4"
PANEL_COLOURS = {"ME": STRATEGY_COLOUR, "QE": STRATEGY_COLOUR}


def _cumulative(returns: pd.Series) -> pd.Series:
    """Growth of 1 over the series, starting at 1.0 on the first bar."""
    return (1 + returns.fillna(0.0)).cumprod()


def _drawdown(returns: pd.Series) -> pd.Series:
    """Underwater curve: drawdown from the running peak, as a fraction."""
    growth = _cumulative(returns)
    return growth / growth.cummax() - 1.0


def arm_panels(arms: list[dict], window: str) -> list[dict]:
    """The two refit arms in fixed order (ME left, QE right).

    The ``net best`` entry is dropped: it duplicates one arm, and which arm wins is
    signalled with a star instead of a third panel.

    Raises:
        SystemExit: if an arm is missing from the comparison.
    """
    by_label = {arm["label"]: arm for arm in arms}
    missing = [label for label in ARM_ORDER if label not in by_label]
    if missing:
        raise SystemExit(f"no {missing} row in the ranking; cannot compare arms")
    best = max(arms, key=lambda arm: arm["row"]["net_sharpe"])
    winner = best["label"]
    panels = []
    for label in ARM_ORDER:
        arm = dict(by_label[label])
        book_key = "portfolio" if window == "test" else "val_portfolio"
        arm["is_winner"] = label == winner
        arm["sharpe"] = perf_stats(arm["results"][book_key]["net_ret"],
                                   arm["config"].ppy)["sharpe"]
        panels.append(arm)
    return panels


def _panel_grid(n: int):
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 3.8), dpi=140, squeeze=False)
    return fig, axes[0]


def _panel_title(ax, arm: dict, detail: str) -> None:
    """Panel title, with a red ★ when this arm is the validation-selected one.

    The star is the only marker: no "winner" word and no bold. Matplotlib titles are
    single-colour, so the title carries a trailing space and the ★ is drawn as its own
    text object positioned immediately after the centred title.
    """
    title = f"{arm['label']} · {detail}"
    if not arm["is_winner"]:
        ax.set_title(title, fontsize=9.5)
        return
    ax.set_title(title + "   ", fontsize=9.5)
    ax.annotate("★", xy=(0.5, 1.0), xycoords="axes fraction",
                xytext=(len(title) * 3.1, 3), textcoords="offset points",
                fontsize=12, color="#d62728", va="bottom", ha="left",
                annotation_clip=False)


def plot_cumulative(arms: list[dict], window: str, out: Path) -> Path:
    """ME (left) and QE (right) cumulative net return, red star in the winner's title.

    ``window`` is ``"test"`` or ``"val"``; each panel also shows the two benchmarks so
    an arm is readable against Equal-weight and SPY.
    """
    panels = arm_panels(arms, window)
    fig, axes = _panel_grid(len(panels))
    for ax, arm in zip(axes, panels):
        book = arm["results"]["portfolio" if window == "test" else "val_portfolio"]
        bench = arm["results"]["benchmarks" if window == "test"
                               else "val_benchmarks"]
        ax.plot(_cumulative(book["net_ret"]), lw=1.4,
                color=PANEL_COLOURS[arm["label"]], label=f"{arm['label']} (net)")
        ax.plot(_cumulative(bench["Equal-weight"]), lw=1.0, color="#2ca02c",
                alpha=0.85, label="Equal-weight")
        ax.plot(_cumulative(bench["SPY B&H"]), lw=1.0, color="#ff7f0e",
                alpha=0.85, label="SPY B&H")
        stats = perf_stats(book["net_ret"], arm["config"].ppy)
        _panel_title(ax, arm, f"Sharpe {stats['sharpe']:.2f} · "
                              f"maxDD {stats['max_drawdown']:.1%}")
        ax.set_ylabel("growth of 1", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, frameon=False)
    label = "test 2023-2025" if window == "test" else "validation 2019-2022"
    fig.suptitle(f"Cumulative net return — {label}", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_drawdown(arms: list[dict], out: Path) -> Path:
    """Underwater curve, ME (left) and QE (right), red star in the winner's title."""
    panels = arm_panels(arms, "test")
    fig, axes = _panel_grid(len(panels))
    for ax, arm in zip(axes, panels):
        book = arm["results"]["portfolio"]
        ax.fill_between(book["net_ret"].index, _drawdown(book["net_ret"]), 0,
                        color=PANEL_COLOURS[arm["label"]], alpha=0.35,
                        label=f"{arm['label']} (net)")
        stats = perf_stats(book["net_ret"], arm["config"].ppy)
        _panel_title(ax, arm, f"maxDD {stats['max_drawdown']:.2%}")
        ax.set_ylabel("drawdown", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, frameon=False)
    fig.suptitle("Drawdown — test 2023-2025", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# metric tables
# ---------------------------------------------------------------------------

METRIC_KEYS = ("sharpe", "ann_return", "ann_vol", "max_drawdown", "total_return",
               "hit_rate")
FORECAST_KEYS = ("oos_r2", "direction_acc", "rank_ic_mean", "rank_ic_t", "corr")


def _forecast_block(arm: dict, window: str) -> dict:
    """Forecast-quality metrics of an arm's predictions on one window."""
    results = arm["results"]
    preds = results["preds"] if window == "test" else results["val_preds"]
    target = arm["config"].target_col
    try:
        metrics = forecast_metrics(preds, target)
    except Exception:
        metrics = {}
    return {key: metrics.get(key) for key in FORECAST_KEYS}


def winner_label(arms: list[dict]) -> str:
    """Label of the arm the validation ranking selected (highest val net Sharpe)."""
    return max(arms, key=lambda arm: arm["row"]["net_sharpe"])["label"]


def aggregate_rows(arms: list[dict], window: str) -> list[dict]:
    """One row per arm: book metrics (net + gross) plus forecast quality.

    The winning arm's ``arm`` cell carries a trailing ``*`` so the text tables mark it
    without adding a column that would widen the table.
    """
    winner = winner_label(arms)
    rows = []
    for arm in arms:
        results = arm["results"]
        book = results["portfolio" if window == "test" else "val_portfolio"]
        ppy = arm["config"].ppy
        net = perf_stats(book["net_ret"], ppy)
        gross = perf_stats(book["gross_ret"], ppy)
        row = {"arm": arm["label"] + (" *" if arm["label"] == winner else ""),
               "kind": arm["kind"], "window": window,
               "model": arm["row"]["model"], "combo": arm["row"]["combo"],
               "book": arm["row"]["book"]}
        row.update({f"net_{k}": net[k] for k in METRIC_KEYS})
        row.update({f"gross_{k}": gross[k] for k in METRIC_KEYS})
        row.update(_forecast_block(arm, window))
        rows.append(row)
    return rows


def _oos_lookup(oos: pd.Series, year) -> float:
    """OOS R² for a calendar year, tolerating the ``MultiIndex`` the groupby produces."""
    for key in oos.index:
        last = key[-1] if isinstance(key, tuple) else key
        if str(last) == str(year):
            return float(oos.loc[key])
    return float("nan")


def yearly_rows(arm: dict, winner: str = "") -> list[dict]:
    """Per-calendar-year rows for one arm, val and test separately."""
    label = arm["label"] + (" *" if arm["label"] == winner else "")
    out = []
    for window, book_key, preds_key in (
            ("val", "val_portfolio", "val_preds"),
            ("test", "portfolio", "preds")):
        results = arm["results"]
        book = results[book_key]
        ppy = arm["config"].ppy
        stats = period_stats(book["net_ret"], ppy)
        oos = oos_r2_by_period(results[preds_key], arm["config"].target_col)
        for year, values in stats.iterrows():
            out.append({"arm": label, "kind": arm["kind"], "window": window,
                        "year": year,
                        **{k: values[k] for k in METRIC_KEYS},
                        "oos_r2": _oos_lookup(oos, year)})
    return out


def _format_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if pd.isna(value):
            return "-"
        return f"{value:.4f}"
    return str(value)


def write_table(rows: list[dict], columns: list[str], title: str, note: str,
                out: Path) -> Path:
    """Fixed-width text table (no third-party tabulate dependency)."""
    widths = {c: max(len(c), *(len(_format_value(r.get(c))) for r in rows))
              if rows else len(c) for c in columns}
    lines = [title, "=" * len(title), ""]
    lines.append("  ".join(c.ljust(widths[c]) for c in columns))
    lines.append("  ".join("-" * widths[c] for c in columns))
    for row in rows:
        lines.append("  ".join(_format_value(row.get(c)).ljust(widths[c])
                               for c in columns))
    lines += ["", note, ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


AGGREGATE_COLUMNS = ["arm", "window", "model", "combo", "book",
                     "net_sharpe", "net_ann_return", "net_ann_vol",
                     "net_max_drawdown", "net_total_return", "net_hit_rate",
                     "gross_sharpe"]
FORECAST_COLUMNS = ["arm", "window", "model", "combo",
                    "oos_r2", "direction_acc", "rank_ic_mean", "rank_ic_t", "corr"]
YEARLY_COLUMNS = ["arm", "window", "year", "sharpe", "ann_return", "ann_vol",
                  "max_drawdown", "total_return", "hit_rate", "oos_r2"]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def compare_block(feature_set: str, panel, benchmark) -> dict:
    """Run the arms of one block and write its figures and metric tables."""
    settings = run_study.study_settings(feature_set)
    root = settings["study_dir"] / "comparison"
    print(f"\n=== {feature_set}: comparing refit arms -> {run_study._rel(root)} ===",
          flush=True)

    entries = arm_candidates(feature_set)
    arms = [run_arm(entry, feature_set, settings) for entry in entries]
    for arm in arms:
        stats = perf_stats(arm["results"]["portfolio"]["net_ret"], arm["config"].ppy)
        print(f"  {arm['label']:8s} {arm['row']['model']:18s} "
              f"{arm['row']['combo']:22s} {arm['row']['book']:22s} "
              f"test Sharpe {stats['sharpe']:.3f}", flush=True)

    figures = root / "figures"
    written = [
        plot_cumulative(arms, "test", figures / "cumulative_test.png"),
        plot_cumulative(arms, "val", figures / "cumulative_val.png"),
        plot_drawdown(arms, figures / "drawdown_test.png"),
    ]

    aggregate = [row for window in ("val", "test")
                 for row in aggregate_rows(arms, window)]
    winner = winner_label(arms)
    yearly = [row for arm in arms for row in yearly_rows(arm, winner)]
    star_note = (f"* = the arm selected on the validation window ({winner}); the test\n"
                 "window is reported for both arms but selects nothing.")

    written.append(write_table(
        aggregate, AGGREGATE_COLUMNS,
        f"{feature_set} - book metrics per refit arm (validation and test)",
        "net_* = after costs; gross_sharpe = before costs.\n" + star_note,
        root / "metrics_aggregate.txt"))
    written.append(write_table(
        [{k: row[k] for k in FORECAST_COLUMNS} for row in aggregate],
        FORECAST_COLUMNS,
        f"{feature_set} - forecast quality per refit arm (validation and test)",
        "oos_r2 = 1 - SSE/sum(y^2) against a zero forecast on the same rows;\n"
        "rank_ic_* = cross-sectional Spearman IC of the prediction vs the label.\n"
        + star_note,
        root / "metrics_forecast.txt"))
    written.append(write_table(
        yearly, YEARLY_COLUMNS,
        f"{feature_set} - metrics by calendar year, per refit arm",
        "one row per (arm, window, year); 2023-2025 are the test window, 2019-2022 the\n"
        "validation window that selected the arm.\n" + star_note,
        root / "metrics_by_year.txt"))

    for path in written:
        print(f"  wrote {run_study._rel(path)}", flush=True)
    return {"feature_set": feature_set, "root": root,
            "aggregate": aggregate, "yearly": yearly}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("features", nargs="?", default="both",
                        choices=(*FEATURE_SETS, "both"),
                        help="feature block to compare: micro, macro, or both")
    args = parser.parse_args()

    blocks = list(FEATURE_SETS) if args.features == "both" else [args.features]

    panel, benchmark, _ = build_panel()
    run_study.init_run(panel, benchmark)

    failures = []
    started = time.perf_counter()
    for block in blocks:
        try:
            compare_block(block, panel, benchmark)
        except SystemExit as exc:
            failures.append((block, str(exc)))
            print(f"!! {block} failed: {exc}", flush=True)
        except Exception:
            failures.append((block, traceback.format_exc(limit=3)))
            print(f"!! {block} failed:\n{traceback.format_exc()}", flush=True)

    print(f"\nfinished in {time.perf_counter() - started:.0f}s", flush=True)
    if failures:
        print(f"{len(failures)} block(s) failed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
