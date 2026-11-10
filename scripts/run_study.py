#!/usr/bin/env python3
"""Validation study — ONE joint ranking over feature set x model x strategy parameters.

There is no separate model-selection phase and no separate book-ablation phase. The
walk-forward predictions for every (spec, feature block, look-ahead, refit cadence) are
stored under ``results/_study_<block>/predictions``; this script replays them through
every book configuration and ranks the **whole combined space** by net Sharpe on the
combined validation window (2019-01-01..2022-12-31).

The feature block is selected with ``--features`` (default: run both in turn). Each
block is a separate study with its own prediction cache, ranking and published figures,
so the two never share a folder or a cache entry:

    results/_study_micro/     the 19 firm features, no macro interaction
    results/_study_macro/     the 19 firm features x (const + 5 macro state variables)

The blocks differ only in which columns of the panel the model reads — the panel itself
always carries the full 19 x 6 Kronecker block (see ``scripts/macro_panel.py``):

    micro       19 columns   (the firm features: return_1bar, rsi_14bar, ...)
    macro      114 columns   (19 x (5 macro + const), e.g. return_1bar__x__tbl)

so a firm feature can change sign and magnitude with the macro regime in the macro
block and cannot in the micro block.

The whole space of one block:

    model spec   20   (4 alphas x 5 elastic-net l1_ratio values)
    combo        11   (rebalance, look-ahead, refit) under look-ahead >= cadence and
                      refit_interval >= cadence
    book         45   (vol window 4/8/12/24/48w x vol target 20%/10%/off x leverage
                      cap 1.0/2.0/3.0, rf sleeve fixed at 3%; caps 2.0/3.0 pay a
                      6.8% ann. borrow fee)
    ---------------
    candidates   20 x 11 x 45 = 9,900 per block

Only the book simulation runs — no refitting — so a full ranking is minutes, and the
predictions are reused byte-identically across every book cell of a candidate. Cells
that differ only in an inert knob (the leverage cap when the vol target is off) are
simulated once and share their metrics.

Split (calendar-anchored, fixed before the run)::

    train        2011-01-01 .. 2018-12-31   (413 weekly bars; the rolling 8y window)
    validation   2019-01-01 .. 2022-12-31   (209 weekly bars, the ranking window)
    test         2023-01-01 .. 2025-12-31   (single draw of the winner, measured once)

The interaction panel starts 2011-02-04 (its first lagged Fear & Greed month), so the
train window holds 413 bars, just short of the 416-bar rolling window: that window never
binds in 2011-2018, and starts binding only for refits inside validation, where 8y of
history has to be left behind.

Draws live under ``<study>/draws/<arm>/<run_tag>/`` with their figures under
``<study>/plots/<arm>/{test,val}/`` and the arm comparison in
``<study>/arm_comparison.txt``.

Determinism: ``SEED`` (== ``utils.config.DEFAULT_SEED``) is recorded in each cache
entry and in the ranking/winner files; a cache entry written under another seed is
recomputed. All book cells of a candidate share one 48-bar volatility warmup.

Usage:
    python scripts/run_study.py rank                   # both blocks, in turn
    python scripts/run_study.py rank --features macro  # one block only
    python scripts/run_study.py cache                  # fit only missing predictions
    python scripts/run_study.py cache --refits ME      # one refit arm of the cache
    python scripts/run_study.py draw                   # one test draw per refit arm
    python scripts/run_study.py rank --refits QE       # quarterly arm only
    python scripts/run_study.py rank --jobs 4
    python scripts/run_study.py test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd
from joblib import Parallel, delayed

import study_worker
from macro import MACRO_COLUMNS
from macro_panel import FIRM_COLUMNS, FEATURE_SETS, build_panel, feature_columns
from metrics import perf_stats
from models import ModelSpec, specs_from_config
from pipeline import (CUMULATIVE_HEADER, cumulative_by_window,
                      run_pipeline, run_tag)
from run_backtest import (FREQ_LA_PAIRS, FREQ_TO_BARS, GRID_ENTRIES,
                          REFIT_OPTIONS, REFIT_TO_BARS)
from study_worker import BOOKS, VALIDATION, book_config, combo_slug
from utils.artifacts import save_arm_summary, save_table
from utils.config import DEFAULT_SEED, seed_everything
from utils.paths import RESULTS_DIR

SEED = DEFAULT_SEED
TRAIN = ("2011-01-01", "2018-12-31")
TEST = ("2023-01-01", "2025-12-31")          # single pre-registered draw
WINDOW_TRAIN = "8y"                          # rolling training window per refit
VAL_WINDOW_SLUG = "4y"                       # window_val that reproduces 2019-2022 before 2023
BATCH = 5                                    # candidates per worker task

STUDY_ROOTS = {name: RESULTS_DIR / f"_study_{name}" for name in FEATURE_SETS}
# Set in ``main`` for the selected block (``choose_study``)
STUDY_DIR: Path = STUDY_ROOTS["macro"]
CACHE_ROOT: Path = STUDY_DIR / "predictions"

# (rebalance, look-ahead, refit) combos under the two design rules.
COMBOS = [(rebalance, horizon, refit)
          for refit in REFIT_OPTIONS
          for rebalance, horizon in FREQ_LA_PAIRS
          if REFIT_TO_BARS[refit] >= FREQ_TO_BARS[rebalance]]

# Set in ``main`` (the panel is built once and handed to the workers).
PANEL: pd.DataFrame | None = None
BENCHMARK_CLOSE: pd.Series | None = None
# feature block -> the panel columns a model reads
FEATURE_COLS: dict[str, list[str]] = {}
# the feature block the current command runs on
FEATURE_SET: str = "macro"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sanitize(value):
    """Convert numpy/pandas scalars to plain JSON types."""
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


def _save(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(obj), indent=2, default=str))
    return path


def _rel(path: Path) -> str:
    """Project-relative posix path, for portable records."""
    return Path(path).relative_to(PROJECT_ROOT).as_posix()


def _optional(value):
    """NaN -> None so records round-trip cleanly through JSON."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _feature_sets(text: str | None) -> list[str]:
    """Requested feature blocks, in canonical order (default: all)."""
    if not text:
        return list(FEATURE_SETS)
    wanted = {part.strip() for part in text.split(",") if part.strip()}
    unknown = wanted - set(FEATURE_SETS)
    if unknown:
        raise SystemExit(f"unknown feature set(s) {sorted(unknown)}; "
                         f"choose from {list(FEATURE_SETS)}")
    return [name for name in FEATURE_SETS if name in wanted]


def choose_study(feature_set: str, study_dir: str | Path | None = None) -> Path:
    """Point ``STUDY_DIR``/``CACHE_ROOT`` at the selected block's study folder.

    Each feature block is a separate study with its own cache and outputs:
    ``results/_study_micro`` and ``results/_study_macro``. ``study_dir`` overrides the
    default root (relative paths resolve against the project root).

    Returns:
        The study root in use.
    """
    global STUDY_DIR, CACHE_ROOT
    root = Path(study_dir) if study_dir else STUDY_ROOTS[feature_set]
    STUDY_DIR = root if root.is_absolute() else PROJECT_ROOT / root
    CACHE_ROOT = STUDY_DIR / "predictions"
    return STUDY_DIR


def study_settings(feature_set: str,
                   study_dir: str | Path | None = None) -> dict:
    """Everything one block's stages need, so a caller can run them out of order.

    ``init_run`` must have been called first (it builds the panel and the column
    lists). Returns a dict with the block name, its columns, and the resolved study
    root and cache root; passing it to :func:`run_cache`/:func:`run_rank`/
    :func:`run_test` makes those stages independent of the module globals.
    """
    root = Path(study_dir) if study_dir else STUDY_ROOTS[feature_set]
    root = root if root.is_absolute() else PROJECT_ROOT / root
    if PANEL is None:
        raise RuntimeError("call init_run(panel, benchmark_close) before study_settings")
    return {"feature_set": feature_set,
            "feature_cols": FEATURE_COLS[feature_set],
            "study_dir": root,
            "cache_root": root / "predictions"}


def init_run(panel: pd.DataFrame, benchmark_close: pd.Series) -> None:
    """Bind the panel, benchmark and per-block columns used by every stage."""
    global PANEL, BENCHMARK_CLOSE, FEATURE_COLS
    PANEL = panel
    BENCHMARK_CLOSE = benchmark_close
    FEATURE_COLS = {name: feature_columns(name) for name in FEATURE_SETS}


def plots_root_for(feature_set: str, study_dir: str | Path | None = None) -> Path:
    """Root of a block's published figures: ``results/_study_<block>/plots``."""
    return study_settings(feature_set, study_dir)["study_dir"] / "plots"


def run_pipeline_no_outputs(settings: dict, config) -> dict:
    """Run one candidate through the pipeline without writing tables or figures.

    Used where only the in-memory books and forecasts are wanted (e.g. an arm
    comparison that draws its own figures), so nothing is persisted from here.
    """
    return run_pipeline(PANEL, settings["feature_cols"], BENCHMARK_CLOSE, config,
                        test_start=TEST[0], test_end=TEST[1],
                        window_val=VAL_WINDOW_SLUG, out_dir=None,
                        publish_figures=False)


def _stamp_block(rows: list[dict], feature_set: str) -> None:
    """Label every ranking row with the block it was scored on."""
    for row in rows:
        row["features"] = feature_set


def _ranking_frame(rows: list[dict], tasks: list[tuple]) -> pd.DataFrame:
    """Rank the scored rows by net Sharpe, with a deterministic tie-break.

    ``score_batch`` is consumed from an *unordered* joblib generator, so two identical
    runs can collect tied rows in any order. Several books do tie exactly (the
    leverage cap never binds when the volatility overlay is off, so ``cap1``/``2``/``3``
    are the same book), and the winner is ``iloc[0]`` — so the rows are first restored
    to the candidate order (spec x combo, then book name) and then sorted with a stable
    sort. The winner is therefore a function of the scores alone, not of scheduling.
    """
    order = {(spec.name, rebalance, horizon, refit): index
             for index, (spec, rebalance, horizon, refit) in enumerate(tasks)}
    rows.sort(key=lambda row: (order[(row["model"], row["rebalance"],
                                     int(row["look_ahead"]), row["refit"])],
                               row["book"]))
    return (pd.DataFrame(rows)
            .sort_values("net_sharpe", ascending=False, na_position="last",
                         kind="stable")
            .reset_index(drop=True))


def _specs(text: str | None) -> list[ModelSpec]:
    specs = specs_from_config(GRID_ENTRIES)
    if not text:
        return specs
    wanted = {part.strip() for part in text.split(",") if part.strip()}
    unknown = wanted - {spec.name for spec in specs}
    if unknown:
        raise SystemExit(f"unknown model spec(s) {sorted(unknown)}")
    return [spec for spec in specs if spec.name in wanted]


def _parse_combos(text: str | None, combos: list[tuple[str, int, str]]) -> list[tuple[str, int, str]]:
    if not text:
        return combos
    known = {combo_slug(rebalance, horizon, refit): (rebalance, horizon, refit)
             for rebalance, horizon, refit in combos}
    wanted = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise SystemExit(f"unknown combo(s) {unknown}; choose from {sorted(known)}")
    return [known[name] for name in wanted]


def _combos(refits: str | None) -> list[tuple[str, int, str]]:
    """Combos restricted to the requested refit cadences (default: all)."""
    if not refits:
        return COMBOS
    wanted = {part.strip() for part in refits.split(",") if part.strip()}
    unknown = wanted - set(REFIT_OPTIONS)
    if unknown:
        raise SystemExit(f"unknown refit cadence(s) {sorted(unknown)}; "
                         f"choose from {sorted(REFIT_OPTIONS)}")
    return [combo for combo in COMBOS if combo[2] in wanted]


def _config(horizon: int, rebalance: str, refit: str, model: str, book: dict,
            feature_set: str | None = None):
    """Config of one cell: the candidate plus that cell's book knobs."""
    return book_config(horizon, rebalance=rebalance, refit=refit, model=model,
                       book=book, window_train=WINDOW_TRAIN,
                       grid_entries=GRID_ENTRIES,
                       features=feature_set or FEATURE_SET)


def _resolve(settings: dict | None) -> dict:
    """The stage's block/roots: explicit ``settings``, else the module state."""
    if settings is not None:
        return settings
    return {"feature_set": FEATURE_SET, "feature_cols": FEATURE_COLS[FEATURE_SET],
            "study_dir": STUDY_DIR, "cache_root": CACHE_ROOT}


def _initargs(block: dict) -> tuple:
    """Worker initialiser arguments for the given block."""
    return (PANEL, FEATURE_COLS, BENCHMARK_CLOSE, block["cache_root"], VALIDATION,
            WINDOW_TRAIN, block["feature_set"], GRID_ENTRIES)


def _batches(tasks: list[tuple]):
    return [tasks[i:i + BATCH] for i in range(0, len(tasks), BATCH)]


# ---------------------------------------------------------------------------
# cache — fill the walk-forward prediction cache, no book simulation
# ---------------------------------------------------------------------------

def run_cache(models: str | None, combos: str | None, refits: str | None,
              refresh: bool, jobs: int, settings: dict | None = None) -> None:
    """Fit every missing walk-forward prediction set of the requested space.

    The 11 combos map onto 6 ``(look-ahead, refit)`` forecast groups, so only those 6
    groups are filled: combos that differ only in the rebalance cadence share the same
    cached predictions. Nothing is ranked here; run ``rank`` afterwards.

    ``settings`` selects the feature block and its folders; ``None`` uses the module
    state set up by ``main``/``choose_study``.
    """
    block = _resolve(settings)
    feature_set = block["feature_set"]
    specs = _specs(models)
    groups = study_worker.forecast_groups(_parse_combos(combos, _combos(refits)))
    total = len(specs) * len(groups)
    print(f"prediction cache [{feature_set}, {len(block['feature_cols'])} columns]: "
          f"{len(specs)} specs x {len(groups)} forecast groups = {total} prediction sets "
          f"on {VALIDATION[0]}..{VALIDATION[1]} (window {WINDOW_TRAIN}, seed {SEED}, "
          f"jobs {jobs})", flush=True)

    tasks = [(spec, rebalance, horizon, refit)
             for rebalance, horizon, refit in groups for spec in specs]
    batches = _batches(tasks)
    progress_every = max(1, len(batches) // 10)

    # n_jobs=1 runs in-process and never calls the worker initializer, so seed the
    # state here as well (harmless when the pool does call it).
    study_worker.init(*_initargs(block))

    counts = {"cached": 0, "stored": 0, "degenerate": 0}
    started = time.perf_counter()
    for number, produced in enumerate(
            Parallel(n_jobs=jobs, return_as="generator_unordered",
                     initializer=study_worker.init, initargs=_initargs(block))(
                delayed(study_worker.cache_batch)(batch, refresh) for batch in batches),
            start=1):
        for record in produced:
            counts[record["status"]] += 1
            if record["status"] != "cached":
                print(f"  {record['candidate']}/{record['spec']}: "
                      f"{record['status']} ({record['seconds']:.1f}s)", flush=True)
        if number % progress_every == 0 or number == len(batches):
            done = min(number * BATCH, len(tasks))
            print(f"  {done}/{total} prediction sets checked", flush=True)

    print(f"cache: {counts['stored']} stored, {counts['degenerate']} degenerate, "
          f"{counts['cached']} already cached "
          f"({time.perf_counter() - started:.0f}s) -> {_rel(block['cache_root'])}")


# ---------------------------------------------------------------------------
# rank — the single validation stage
# ---------------------------------------------------------------------------

def run_rank(models: str | None, combos: str | None, refits: str | None,
             refresh: bool, jobs: int, settings: dict | None = None) -> Path:
    """Replay the cached predictions through the whole model x combo x book space.

    The winner is the highest net Sharpe on the validation window.

    ``refits`` restricts the refit cadences (``"QE"`` = the quarterly arm only).
    ``settings`` selects the feature block and its folders (``None`` = module state).
    """
    block = _resolve(settings)
    feature_set = block["feature_set"]
    specs = _specs(models)
    pairs = _parse_combos(combos, _combos(refits))
    total = len(specs) * len(pairs) * len(BOOKS)
    print(f"joint ranking [{feature_set}, {len(block['feature_cols'])} columns]: "
          f"{len(specs)} specs x {len(pairs)} combos x {len(BOOKS)} books = {total} "
          f"candidates on {VALIDATION[0]}..{VALIDATION[1]} (train {TRAIN[0]}.."
          f"{TRAIN[1]}, window {WINDOW_TRAIN}, seed {SEED}, jobs {jobs})", flush=True)

    tasks = [(spec, rebalance, horizon, refit)
             for rebalance, horizon, refit in pairs for spec in specs]
    batches = _batches(tasks)
    progress_every = max(1, len(batches) // 20)

    # n_jobs=1 runs in-process and never calls the worker initializer, so seed the
    # state here as well (harmless when the pool does call it).
    study_worker.init(*_initargs(block))

    rows: list[dict] = []
    for number, produced in enumerate(
            Parallel(n_jobs=jobs, return_as="generator_unordered",
                     initializer=study_worker.init, initargs=_initargs(block))(
                delayed(study_worker.score_batch)(batch, refresh) for batch in batches),
            start=1):
        rows.extend(produced)
        if number % progress_every == 0 or number == len(batches):
            done = min(number * BATCH, len(tasks))
            print(f"  {done}/{len(tasks)} candidates scored ({len(rows)} rows)",
                  flush=True)

    _stamp_block(rows, feature_set)
    ranking = _ranking_frame(rows, tasks)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    save_table(ranking, block["study_dir"], "validation_ranking", index=False)
    _save({"scheme": "joint model x strategy ranking",
           "features": _feature_record(feature_set, name_key=True),
           "split": {"train": list(TRAIN), "validation": list(VALIDATION),
                     "test": list(TEST)},
           "window_train": WINDOW_TRAIN,
           "window": list(VALIDATION), "seed": SEED,
           "specs": [spec.name for spec in specs],
           "combos": [combo_slug(*pair) for pair in pairs],
           "refits": sorted({pair[2] for pair in pairs}),
           "books": [{"name": name, "note": note, "knobs": knobs}
                     for name, knobs, note in BOOKS],
           "candidates": total,
           "cache": _rel(block["cache_root"])},
          block["study_dir"] / "run_parameters.json")

    live = ranking[~ranking["degenerate"] & ranking["net_sharpe"].notna()]
    if live.empty:
        raise SystemExit("no non-degenerate candidate; nothing to rank")
    best = live.iloc[0]

    winner_config = _config(int(best["look_ahead"]), best["rebalance"], best["refit"],
                            best["model"],
                            {"vol_window": int(best["vol_window"]),
                             "vol_target": _optional(best["vol_target"]),
                             "vol_rf": float(best["vol_rf"]),
                             "vol_cap": float(best["vol_cap"]),
                             "lev_fee": float(best["lev_fee"])},
                            feature_set)
    payload = {
        "phase": "joint-ranking",
        "features": _feature_record(feature_set, name_key=True),
        "split": {"train": list(TRAIN), "validation": list(VALIDATION),
                  "test": list(TEST)},
        "window": list(VALIDATION), "seed": SEED,
        "window_train": WINDOW_TRAIN,
        "candidates": {"specs": len(specs), "combos": len(pairs),
                       "books": len(BOOKS), "scored": int(len(ranking))},
        "model": {"name": best["model"], "family": best["family"],
                  "alpha": float(best["alpha"]),
                  "l1_ratio": _optional(best["l1_ratio"])},
        "combo": {"target": best["target"], "look_ahead": int(best["look_ahead"]),
                  "rebalance": best["rebalance"], "refit": best["refit"]},
        "book": {"name": best["book"], "vol_window": int(best["vol_window"]),
                 "vol_target": _optional(best["vol_target"]),
                 "vol_rf": float(best["vol_rf"]), "vol_cap": float(best["vol_cap"]),
                 "lev_fee": float(best["lev_fee"])},
        "metrics": {"net_sharpe": float(best["net_sharpe"]),
                    "gross_sharpe": float(best["gross_sharpe"]),
                    "max_drawdown": float(best["max_drawdown"]),
                    "total_return": float(best["total_return"]),
                    "ann_return": float(best["ann_return"]),
                    "ann_vol": float(best["ann_vol"]),
                    "hit_rate": float(best["hit_rate"]),
                    "avg_turnover": float(best["avg_turnover"])},
        "cache": _rel(block["cache_root"] / f"{VALIDATION[0]}_{VALIDATION[1]}"
                      / f"weekly_la{int(best['look_ahead'])}_refit{best['refit']}"
                      / f"{best['model']}.npz"),
        "window_train_bars": winner_config.window_train_bars,
        "ranking": _rel(block["study_dir"] / "validation_ranking.csv"),
        "note": "promotion still requires the pre-registered ADR and the single test "
                "draw",
    }
    winner_path = _save(payload, block["study_dir"] / "validation_winner.json")
    print(f"winner: {feature_set} / {best['model']} / {best['target']} / "
          f"{best['rebalance']} / {best['refit']} / {best['book']} -> net Sharpe "
          f"{best['net_sharpe']:.4f}")
    print(f"written: {_rel(winner_path)}")
    return winner_path


def _feature_record(name: str, name_key: bool = False) -> dict:
    """Provenance of one feature block: dimension and, for ``macro``, its state variables."""
    record = {"firm": len(FIRM_COLUMNS), "const": 1,
              "macro": len(MACRO_COLUMNS) if name == "macro" else 0,
              "model_columns": len(FEATURE_COLS[name])}
    if name_key:
        record = {"name": name, **record}
        record["macro_columns"] = list(MACRO_COLUMNS) if name == "macro" else []
    return record


# ---------------------------------------------------------------------------
# draws — one test draw per refit arm, each with its own figures
# ---------------------------------------------------------------------------

def draws_dir() -> Path:
    """Root of the study's run folders (``results/_study_<block>/draws``)."""
    return STUDY_DIR / "draws"


def plots_dir() -> Path:
    """Root of the study's published figures (``results/_study_<block>/plots``)."""
    return STUDY_DIR / "plots"


def arm_slug(refit: str) -> str:
    """Folder name of a refit arm, e.g. ``"QE"`` -> ``"qe"``."""
    return refit.lower()


def arm_plots_root(refit: str) -> Path:
    """Where an arm's figures are published: ``plots/<arm>/{test,val}/``."""
    return plots_dir() / arm_slug(refit)


def _arm_winner_rows(refits: str | None,
                     study_dir: Path | None = None) -> list[pd.Series]:
    """Best live ranking row of each refit arm (the ranking is Sharpe-sorted)."""
    path = (study_dir or STUDY_DIR) / "validation_ranking.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run `python scripts/run_study.py rank` first")
    ranking = pd.read_csv(path)
    live = ranking[~ranking["degenerate"] & ranking["net_sharpe"].notna()]
    keep = {part.strip() for part in refits.split(",")} if refits else None
    if keep:
        unknown = keep - set(live["refit"])
        if unknown:
            raise SystemExit(f"no live row for refit arm(s) {sorted(unknown)} in {path}")
        live = live[live["refit"].isin(keep)]
    if live.empty:
        raise SystemExit(f"no non-degenerate candidate in {path}")
    return [row for _, row in live.groupby("refit", sort=False).head(1).iterrows()]


def _row_config(row):
    """Config of a ranking row's candidate plus its book cell."""
    return _config(int(row["look_ahead"]), row["rebalance"], row["refit"],
                   row["model"],
                   {"vol_window": int(row["vol_window"]),
                    "vol_target": _optional(row["vol_target"]),
                    "vol_rf": float(row["vol_rf"]),
                    "vol_cap": float(row["vol_cap"]),
                    "lev_fee": float(row["lev_fee"])})


def run_draw(refits: str | None = None, settings: dict | None = None) -> Path:
    """Measure each refit arm's best candidate once on 2023-2025 and compare them.

    One run folder per arm under ``draws/<arm>/<run_tag>/`` and the arm's figures under
    ``plots/<arm>/{test,val}/``; the comparison lands in ``arm_comparison.txt`` at the
    study root. ``settings`` selects the feature block and its folders.
    """
    block = _resolve(settings)
    study_dir = block["study_dir"]
    rows = _arm_winner_rows(refits, study_dir)
    arms: list[dict] = []
    cumulative: list[dict] = []
    for row in rows:
        refit = row["refit"]
        arm = arm_slug(refit)
        config = _row_config(row)
        tag = run_tag(config, *TEST, window_val=VAL_WINDOW_SLUG)
        out_dir = study_dir / "draws" / arm / tag
        print(f"\n[{arm}] {row['features']} / {row['model']} / {row['target']} / "
              f"{row['rebalance']} / {refit} / {row['book']} "
              f"(validation net Sharpe {row['net_sharpe']:.4f})")
        results = run_pipeline(PANEL, block["feature_cols"], BENCHMARK_CLOSE,
                               config, test_start=TEST[0], test_end=TEST[1],
                               window_val=VAL_WINDOW_SLUG, out_dir=out_dir,
                               plots_root=study_dir / "plots" / arm)
        stats = perf_stats(results["portfolio"]["net_ret"], config.ppy)
        verdict = results["verdict"]
        arms.append({"arm": arm, "features": row["features"], "model": row["model"],
                     "combo": row["combo"], "book": row["book"],
                     "val_sharpe": float(row["net_sharpe"]),
                     "test_sharpe": float(stats["sharpe"]),
                     "max_drawdown": float(stats["max_drawdown"]),
                     "total_return": float(stats["total_return"]),
                     "ann_return": float(stats["ann_return"]),
                     "ann_vol": float(stats["ann_vol"]),
                     "gate": "pass" if verdict["success"] else "fail"})
        cumulative.append({"arm": arm, "test_sharpe": float(stats["sharpe"]),
                           **cumulative_by_window(results)})

    cumulative.sort(key=lambda row: row.pop("test_sharpe"), reverse=True)
    path = save_arm_summary(
        arms, study_dir,
        header=["arm", "features", "model", "combo", "book", "val_sharpe",
                "test_sharpe", "max_drawdown", "total_return", "ann_return", "ann_vol",
                "gate"],
        best_key="test_sharpe",
        title=f"Refit-arm test draws ({len(arms)} arms)",
        note=("note: the arm is chosen on the validation window, so ranking the arms by\n"
              "test-window net Sharpe here is descriptive, not a selection rule."),
        tables=((f"cumulative return {VALIDATION[0]}..{VALIDATION[1]} (val) and "
                 f"{TEST[0]}..{TEST[1]} (test); decimals, 0.4126 = +41.26%",
                 CUMULATIVE_HEADER, cumulative),))
    print(f"\nwritten: {_rel(path)}")
    return path


# ---------------------------------------------------------------------------
# test — the single draw of the ranking winner
# ---------------------------------------------------------------------------

def run_test(settings: dict | None = None) -> None:
    """Measure the ranking winner once on 2023-2025 and publish its figures.

    ``settings`` selects the feature block and its folders (``None`` = module state).
    """
    block = _resolve(settings)
    feature_set = block["feature_set"]
    study_dir = block["study_dir"]
    winner_path = study_dir / "validation_winner.json"
    if not winner_path.exists():
        raise SystemExit(f"missing {winner_path}; run `python scripts/run_study.py "
                         f"rank` first")
    winner = json.loads(winner_path.read_text())
    if (int(winner.get("seed", -1)) != SEED
            or list(winner.get("window", [])) != list(VALIDATION)):
        raise SystemExit("winner file was written under another seed/window; rerun rank")
    model, combo = winner["model"], winner["combo"]
    book = {key: value for key, value in winner["book"].items() if key != "name"}
    config = _config(int(combo["look_ahead"]), combo["rebalance"], combo["refit"],
                     model["name"], book, feature_set)
    figures = study_dir / "plots" / arm_slug(combo["refit"])
    out_dir = (study_dir / "draws" / arm_slug(combo["refit"])
               / run_tag(config, *TEST, window_val=VAL_WINDOW_SLUG))
    print(f"TEST draw [{feature_set}]: {model['name']} / {combo['target']} / "
          f"{combo['rebalance']} / {combo['refit']} with the {winner['book']['name']} "
          f"book on {TEST[0]}..{TEST[1]} (train 2011-2018, val 2019-2022)")
    run_pipeline(PANEL, block["feature_cols"], BENCHMARK_CLOSE, config,
                 test_start=TEST[0], test_end=TEST[1], window_val=VAL_WINDOW_SLUG,
                 out_dir=out_dir, plots_root=figures, publish_figures=True)
    print(f"results: {_rel(out_dir)}")
    print(f"figures: {_rel(figures)}/{{test,val}}/")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global FEATURE_SET
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="rank",
                        choices=("cache", "draw", "rank", "test"))
    parser.add_argument("--features", default=None,
                        help="feature block to study: micro or macro "
                             "(default: run both in turn, each into its own folder)")
    parser.add_argument("--models", default=None,
                        help="comma-separated spec names (subset runs)")
    parser.add_argument("--combos", default=None,
                        help="comma-separated la<h>_<cadence>_refit<X> (subset runs)")
    parser.add_argument("--refits", default=None,
                        help="comma-separated refit cadences to keep, e.g. QE "
                             "(default: all of ME,QE). Also selects which arms "
                             "`draw` measures")
    parser.add_argument("--study-dir", default=None,
                        help="study output root override for a single --features "
                             "value (default: results/_study_<block>)")
    parser.add_argument("--jobs", type=int, default=0,
                        help="worker processes (0 = min(8, cpu count))")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="recompute cached predictions instead of reusing them")
    args = parser.parse_args()

    feature_sets = _feature_sets(args.features)
    if args.study_dir and len(feature_sets) > 1:
        raise SystemExit("--study-dir needs a single --features value; each feature "
                         "block writes to its own results/_study_<block> folder")

    seed_everything(SEED)
    panel, benchmark, _ = build_panel()
    init_run(panel, benchmark)
    dates = PANEL.index.get_level_values("date")
    print(f"panel: {len(PANEL)} rows, {dates.min():%Y-%m-%d}..{dates.max():%Y-%m-%d}, "
          f"{len(FIRM_COLUMNS)} firm features, seed {SEED}")
    print(f"split: train {TRAIN[0]}..{TRAIN[1]} | validation {VALIDATION[0]}..{VALIDATION[1]} "
          f"| test {TEST[0]}..{TEST[1]} | rolling window {WINDOW_TRAIN} "
          f"({len(specs_from_config(GRID_ENTRIES))} specs)")

    jobs = args.jobs if args.jobs > 0 else min(8, os.cpu_count() or 1)
    for name in feature_sets:
        FEATURE_SET = name
        study = choose_study(name, args.study_dir)
        print(f"\n=== {name}: {len(FEATURE_COLS[name])} model columns -> {_rel(study)} ===",
              flush=True)
        settings = study_settings(name, args.study_dir)
        if args.command == "cache":
            run_cache(args.models, args.combos, args.refits, args.refresh_cache, jobs,
                      settings=settings)
        elif args.command == "draw":
            run_draw(args.refits, settings=settings)
        elif args.command == "rank":
            run_rank(args.models, args.combos, args.refits, args.refresh_cache, jobs,
                     settings=settings)
        else:
            run_test(settings=settings)


if __name__ == "__main__":
    main()
