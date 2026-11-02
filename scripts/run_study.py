#!/usr/bin/env python3
"""Execute the two-stage validation study defined in agents/validation_and_ablation.md.

2026-11-02 design revision (expanded grid): V1 is now a **joint selection** over
model x look-ahead x rebalance frequency x refit cadence on 2017-2019 — 144 specs
(ridge/lasso/enet on per-family alpha grids, enet x l1_ratio, plus the RFF ridge
family: alpha x rff_dim x gamma, rff_dim incl. 1000) x 14 valid (frequency,
look-ahead, refit) combos
under the rules **look-ahead >= frequency** and **refit_interval >=
prediction_interval** -> 2016 candidates, scored on the plain overlay-free
book; the winner locks (model, look-ahead, frequency, refit) together. V2 ablates
the remaining book knobs (vol window / vol target / rf / leverage); the cadence
row is gone because frequency is now locked in V1.

Phase ``v1`` — joint selection (§3): run each of the 14 valid (rebalance, horizon,
refit) combos with the full 144-spec grid, merge the per-combo rankings into one
2016-row ranking, lock the global best by validation net Sharpe. The pipeline maps
each combo to ``test_start=2020-01-01, window_val=3y``; the tiny January-2020 test
window is a byproduct of the pipeline shape and is never used for any decision
(selection reads only ``validation_ranking.csv``).

Phase ``v2`` — ablation (§4): with the V1-locked (model, look-ahead, frequency),
run the book rows on 2020-2022: base, vol window {4, 8, 12, 24, 48}w, vol-target
{None, 20%}, row3 rf 0%, row4 leverage cap {2.0, 3.0} + 6.8% fee.

Phase ``v2b`` — LEGACY (superseded 2026-11-02): the deferred look-ahead tie-break
from the old 2-look-ahead design. Kept only as history; V1 now selects look-ahead
jointly, so this phase needs re-specification before any reuse.

Result folders are named ``<tag>__<run-name>`` so the run tag does not collide across
rows. Extraction summaries land in ``results/_study/`` as
``validation_part1_*`` (per (freq, la) pair, rankings, winner),
``validation_part2_*.json`` (base + rows) and ``validation_part2b_*.json`` (legacy).

Usage:
    python scripts/run_study.py v1   # 2016-candidate joint selection
                                    #  -> results/_study/validation_part1_*.json
    python scripts/run_study.py v2   # V2 base + rows with the locked winner
                                    #  -> results/_study/validation_part2_*.json
    python scripts/run_study.py v2b  # LEGACY deferred look-ahead (re-specify first)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd

from pipeline import REFIT_SLUGS, run_pipeline, run_tag
from run_backtest import (FREQ_LA_PAIRS, FREQ_TO_BARS, REFIT_OPTIONS,
                          REFIT_TO_BARS, build_panel, default_config)
from utils.paths import RESULTS_DIR

STUDY_DIR = RESULTS_DIR / "_study"
V1_TEST = ("2020-01-01", "2020-01-31")   # pipeline byproduct only; selection is on val 2017-2019
V2_TEST = ("2020-01-01", "2022-12-31")   # Validation-2 ablation window

# V1 combos: (rebalance, horizon, refit) under the two design rules —
# look-ahead >= frequency AND refit_interval >= prediction_interval (refit no more
# frequent than the acting cadence; W-FRI=1, ME=4, QE=12 bars vs cadence 1/4/12).
# 3 refits x 6 pairs, filtered -> 14 combos (W-FRI: weekly pairs only; ME: 1w/4w
# cadences; QE: all 6).
V1_COMBOS = [(rebalance, horizon, refit)
             for refit in REFIT_OPTIONS
             for rebalance, horizon in FREQ_LA_PAIRS
             if REFIT_TO_BARS[refit] >= FREQ_TO_BARS[rebalance]]

# The V2 book rows (validation_and_ablation.md §4): cadence is no longer a row —
# frequency is locked with the model in V1 (2026-11-02 design revision). Vol-knob
# rows expanded 2026-11-02 (Step 6 follow-up 6): vol window {4, 8, 12, 24, 48} and
# vol target {None, 10% (base), 20%} as one-at-a-time deviations from the base book.
# Base book defaults re-based 2026-11-02 (follow-up 7): vol_window 24w (was 60w),
# vol_target 10%. volwin_24w duplicates base by construction (consistency check).
STUDY_ROWS = [
    ("base", dict(), "base book (V1-locked cadence, 24w, vt 10%, rf 3%, cap 1.0)"),
    ("volwin_4w", dict(vol_window=4), "vol window 4w"),
    ("volwin_8w", dict(vol_window=8), "vol window 8w"),
    ("volwin_12w", dict(vol_window=12), "vol window 12w"),
    ("volwin_24w", dict(vol_window=24), "vol window 24w (== base; consistency check)"),
    ("volwin_48w", dict(vol_window=48), "vol window 48w"),
    ("vt_none", dict(vol_target=False), "vol-target OFF"),
    ("vt_20", dict(vol_target=0.20), "vol-target 20%"),
    ("row3_rf0", dict(vol_rf=0.0), "row 3 variant: rf 0%"),
    ("row4_lev2", dict(vol_cap=2.0, lev_fee=0.068),
     "row 4 variant: leverage cap 2.0, fee 6.8%"),
    ("lev3", dict(vol_cap=3.0, lev_fee=0.068),
     "row 4 variant: leverage cap 3.0, fee 6.8%"),
]
V2B_MARGIN = 0.20   # legacy V2b promotion margin (superseded phase; kept for history)


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


def _save(obj, name: str) -> Path:
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    path = STUDY_DIR / name
    path.write_text(json.dumps(_sanitize(obj), indent=2, default=str))
    return path


def run(name: str, config, test_start: str, test_end: str) -> dict:
    """Run one backtest under a name, keep ``<tag>__<name>`` and extract the tables."""
    tag = run_tag(config, test_start, test_end, window_val="3y")
    dst = RESULTS_DIR / f"{tag}__{name}"
    print(f"\n######## {name}  tag={tag} ########")

    results = run_pipeline(PANEL, FEATURE_COLS, BENCHMARK_CLOSE, config,
                           test_start=test_start, test_end=test_end,
                           window_val="3y", out_dir=dst)

    summary = {"name": name, "tag": tag, "folder": str(dst.relative_to(PROJECT_ROOT)),
               "window": {"test_start": test_start, "test_end": test_end,
                          "window_val": "3y"},
               "target": config.target_col,
               "rebalance": config.forecast_vol.rebalance, "mode": config.mode}
    val = dst / "validation_ranking.csv"
    table = pd.read_csv(val)
    summary["val"] = table.to_dict("records")
    if len(table) and not pd.isna(table.iloc[0]["net_sharpe"]):
        summary["val_winner"] = {"model": str(table.iloc[0]["model"]),
                                 "net_sharpe": float(table.iloc[0]["net_sharpe"]),
                                 "gross_sharpe": float(table.iloc[0]["gross_sharpe"]),
                                 "degenerate": bool(table.iloc[0]["degenerate"]),
                                 "avg_turnover": float(table.iloc[0]["avg_turnover"])}
    for stem in ("strategy", "comparison", "comparison_no_spy"):
        frame = pd.read_csv(dst / f"{stem}.csv", index_col=0)
        summary[stem] = frame.to_dict("index")
    return summary


def spec_dict(name: str) -> dict:
    """Reconstruct a spec config dict from a spec name, e.g. ``ridge_a10`` ->
    ``{"family": "ridge", "alpha": 10.0}`` or ``rff_a0.01_d100_g0.1`` ->
    ``{"family": "rff", "alpha": 0.01, "rff_dim": 100, "gamma": 0.1}``."""
    family, rest = name.split("_", 1)
    if family == "rff":
        parts = rest.split("_")               # "a<alpha>_d<dim>_g<gamma>"
        return {"family": "rff", "alpha": float(parts[0][1:]),
                "rff_dim": int(parts[1][1:]), "gamma": float(parts[2][1:])}
    alpha = float(rest.split("_r")[0][1:])
    entry = {"family": family, "alpha": alpha}
    if "_r" in rest:
        entry["l1_ratio"] = float(rest.split("_r")[1])
    return entry


def _combo_slug(rebalance: str, horizon: int, refit: str) -> str:
    """Verbose per-combo file slug, e.g. ``la4_freq1w_wkrefit``
    (look-ahead 4w, weekly freq, weekly refits)."""
    return f"la{horizon}_freq{FREQ_TO_BARS[rebalance]}w_{REFIT_SLUGS[refit]}"


def phase_v1() -> None:
    """Joint selection: 144 specs x 14 valid (frequency, look-ahead, refit) combos
    = 2016 candidates on 2017-2019 (overlay-free); lock the global best by val
    net Sharpe.
    """
    rows = {}
    for rebalance, horizon, refit in V1_COMBOS:
        slug = _combo_slug(rebalance, horizon, refit)
        # overlay-free plain book for selection (validation_and_ablation.md §3)
        config = default_config(horizon, vol_target=False, vol_rf=0.0,
                                rebalance=rebalance, refit_every=refit)
        rows[slug] = run(f"v1_{slug}", config, *V1_TEST)
        _save(rows[slug], f"validation_part1_{slug}.json")

    ranked = []
    for rebalance, horizon, refit in V1_COMBOS:
        slug = _combo_slug(rebalance, horizon, refit)
        for row in rows[slug].get("val", []):
            net = row["net_sharpe"]
            if net is None or (isinstance(net, float) and math.isnan(net)):
                net = None   # NaN Sharpe = degenerate manifestation (0/0); unscoreable
            ranked.append({"target": f"absolute{horizon}",
                           "rebalance": rebalance,
                           "freq_bars": FREQ_TO_BARS[rebalance],
                           "refit": refit,
                           "model": row["model"],
                           "net_sharpe": net,
                           "gross_sharpe": row["gross_sharpe"],
                           "degenerate": row["degenerate"],
                           "avg_turnover": row["avg_turnover"]})
    ranked.sort(key=lambda r: (r["net_sharpe"] is not None, r["net_sharpe"]),
                reverse=True)
    _save(ranked, "validation_part1_rankings.json")

    live = [r for r in ranked
            if not r["degenerate"] and r["net_sharpe"] is not None]
    if not live:
        raise SystemExit("no non-degenerate candidate; nothing to lock")
    winner = {**live[0], "spec": spec_dict(live[0]["model"]),
              "locked_from":
                  f"results/_study/validation_part1_{_combo_slug(live[0]['rebalance'], int(live[0]['target'].replace('absolute', '')), live[0]['refit'])}.json"}
    _save(winner, "validation_part1_winner.json")
    print("\n======== V1 LOCKED WINNER ========")
    print(json.dumps(_sanitize(winner), indent=2, default=str))


def phase_v2() -> None:
    winner = json.loads((STUDY_DIR / "validation_part1_winner.json").read_text())
    locked_model = winner["model"]
    horizon = int(winner["target"].replace("absolute", ""))
    locked_rebalance = winner["rebalance"]
    locked_refit = winner["refit"]

    for book, kwargs, note in STUDY_ROWS:
        print(f"\n---- {note} ----")
        vol_target = kwargs.pop("vol_target", True)
        config = default_config(horizon, vol_target=vol_target, model=locked_model,
                                rebalance=locked_rebalance, refit_every=locked_refit,
                                **kwargs)
        summary = run(f"v2_{book}", config, *V2_TEST)
        _save(summary, f"validation_part2_{book}.json")
        print(f"  [extract] {summary.get('strategy', {})}")


def _net_sharpe(summary: dict) -> float:
    """Test-window net Sharpe from a summary's strategy table."""
    return float(summary["strategy"]["net_sharpe"]["value"])


def phase_v2b() -> None:
    """LEGACY V2b: deferred look-ahead tie-break — SUPERSEDED 2026-11-02.

    Written for the old 2-look-ahead design (absolute4/absolute12 x 7 books on
    2020-2022). V1 now selects (model, look-ahead, frequency) jointly, so this
    phase is kept only as history; re-specify before any reuse (its executed v2b
    record is archived in agents/validation_and_ablation.md §5.2; the old
    validation_and_ablation_extension.md was removed 2026-11-02).
    """
    winner = json.loads((STUDY_DIR / "validation_part1_winner.json").read_text())
    locked_model = winner["model"]
    print(f"======== V2b — locked spec {locked_model}, look-ahead deferred ========")

    cells = {}
    for book, kwargs, note in STUDY_ROWS:
        vol_target = kwargs.pop("vol_target", True)
        config = default_config(12, vol_target=vol_target, model=locked_model, **kwargs)
        name = f"v2b_a12_{book}"
        print(f"\n---- v2b absolute12 x {book} ({note}) ----")
        summary = run(name, config, *V2_TEST)
        summary = {**summary, "lookahead": "absolute12",
                   "source": "v2b phase (measured)"}
        cells[f"a12_{book}"] = summary
        _save(summary, f"validation_part2b_a12_{book}.json")
        print(f"  [extract] {summary.get('strategy', {})}")

    for book, kwargs, note in STUDY_ROWS:
        prev = json.loads((STUDY_DIR / f"validation_part2_{book}.json").read_text())
        summary = {**prev, "name": f"v2b_a4_{book}", "lookahead": "absolute4",
                   "source": "v2 phase (reused, byte-identical)"}
        cells[f"a4_{book}"] = summary
        _save(summary, f"validation_part2b_a4_{book}.json")

    incumbent = _net_sharpe(cells["a4_base"])
    threshold = incumbent + V2B_MARGIN
    best_key = max(cells, key=lambda k: _net_sharpe(cells[k]))
    best_ns = _net_sharpe(cells[best_key])
    col = best_key.split("_", 1)[0]                       # a4 | a12
    col_above = [k for k in cells
                 if k.startswith(col) and _net_sharpe(cells[k]) >= incumbent]
    margin_ok = best_ns >= threshold
    mechanism_ok = margin_ok and len(col_above) >= 2
    promoted = margin_ok and mechanism_ok
    verdict = {
        "phase": "v2b",
        "locked_model": locked_model,
        "incumbent": "a4_base",
        "incumbent_net_sharpe": incumbent,
        "margin_rule": {"M": V2B_MARGIN, "threshold": threshold},
        "best_cell": best_key,
        "best_net_sharpe": best_ns,
        "cells_above_incumbent_in_winner_column": col_above,
        "margin_ok": margin_ok,
        "mechanism_ok": mechanism_ok,
        "promoted": promoted,
        "reason": ("promote" if promoted else
                   ("margin not cleared" if not margin_ok else
                    "margin cleared but lone-cell edge (mechanism check failed)")),
        "next": ("pre-registered ADR + single 2023-2025 measurement" if promoted
                 else "keep incumbent lock; all cells logged as breadcrumbs"),
    }
    _save(verdict, "validation_part2b_verdict.json")
    if promoted:
        _save(cells[best_key], "validation_part2b_winner.json")
    print("\n======== V2b VERDICT ========")
    print(json.dumps(_sanitize(verdict), indent=2, default=str))


PANEL = None
FEATURE_COLS = None
BENCHMARK_CLOSE = None


def main() -> None:
    global PANEL, FEATURE_COLS, BENCHMARK_CLOSE
    phase = sys.argv[1] if len(sys.argv) > 1 else "v1"
    PANEL, BENCHMARK_CLOSE, FEATURE_COLS = build_panel()
    if phase == "v1":
        phase_v1()
    elif phase == "v2":
        phase_v2()
    elif phase == "v2b":
        phase_v2b()
    else:
        raise SystemExit(f"unknown phase {phase!r}; use v1 | v2 | v2b")


if __name__ == "__main__":
    main()
