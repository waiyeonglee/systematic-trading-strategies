#!/usr/bin/env python3
"""Plot selected micro/macro validation books with matching AR(1) baselines.

The figure compares the validation-selected micro and macro books on the 2019-2022
validation window. Each AR(1) baseline is traded through that block's selected book,
so the baseline distinction comes from the chosen target, refit cadence and portfolio
knobs rather than from different AR(1) features.

Usage:
    python scripts/plot_validation_comparison.py
"""

from __future__ import annotations

import json
import sys
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
from baselines import ar1_baseline
from macro_panel import FEATURE_SETS, build_panel
from study_worker import VOL_WARMUP_BARS


OUT = PROJECT_ROOT / "results" / "validation_comparison" / "cumulative_val_micro_macro_ar1.png"


def _cumulative(returns: pd.Series) -> pd.Series:
    """Growth of 1 over a return series."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def _winner(feature_set: str) -> dict:
    path = run_study.STUDY_ROOTS[feature_set] / "validation_winner.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run `python scripts/run_study.py rank --features {feature_set}`")
    return json.loads(path.read_text())


def _config(feature_set: str, winner: dict):
    combo = winner["combo"]
    book = winner["book"]
    return run_study._config(
        int(combo["look_ahead"]),
        combo["rebalance"],
        combo["refit"],
        winner["model"]["name"],
        {"vol_window": int(book["vol_window"]),
         "vol_target": book["vol_target"],
         "vol_rf": float(book["vol_rf"]),
         "vol_cap": float(book["vol_cap"]),
         "lev_fee": float(book["lev_fee"])},
        feature_set,
    )


def selected_block(feature_set: str) -> dict:
    """Return selected model validation book and matching AR(1) validation book."""
    winner = _winner(feature_set)
    settings = run_study.study_settings(feature_set)
    config = _config(feature_set, winner)
    results = run_study.run_pipeline_no_outputs(settings, config)
    _, ar1_book = ar1_baseline(results["train"], results["val"], config, VOL_WARMUP_BARS)
    return {"feature_set": feature_set,
            "winner": winner,
            "config": config,
            "results": results,
            "ar1_book": ar1_book}


def plot(blocks: dict[str, dict], out: Path = OUT) -> Path:
    """Write the six-line cumulative validation plot."""
    micro = blocks["micro"]
    macro = blocks["macro"]
    benchmarks = macro["results"]["val_benchmarks"]

    lines = [
        ("Selected micro book (net)",
         micro["results"]["val_portfolio"]["net_ret"], "#1f77b4", "-", 1.5),
        ("Selected macro book (net)",
         macro["results"]["val_portfolio"]["net_ret"], "#9467bd", "-", 1.5),
        ("Equal-weight 16 ETFs",
         benchmarks["Equal-weight"], "#2ca02c", "-", 1.1),
        ("SPY buy-and-hold",
         benchmarks["SPY B&H"], "#ff7f0e", "-", 1.1),
        ("AR(1) baseline, micro book",
         micro["ar1_book"]["net_ret"], "#7f7f7f", "--", 1.1),
        ("AR(1) baseline, macro book",
         macro["ar1_book"]["net_ret"], "#111111", ":", 1.3),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
    for label, returns, color, linestyle, width in lines:
        ax.plot(_cumulative(returns), label=label, color=color, ls=linestyle, lw=width)

    ax.set_title("Validation Cumulative Net Return — Selected Micro vs Macro", fontsize=11)
    ax.set_ylabel("growth of 1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    panel, benchmark, _ = build_panel()
    run_study.init_run(panel, benchmark)
    blocks = {feature_set: selected_block(feature_set) for feature_set in FEATURE_SETS}
    out = plot(blocks)
    print(f"wrote {out.relative_to(PROJECT_ROOT)}")
    for feature_set, block in blocks.items():
        winner = block["winner"]
        print(f"{feature_set}: {winner['model']['name']} / "
              f"{winner['combo']['target']} / {winner['combo']['rebalance']} / "
              f"{winner['combo']['refit']} / {winner['book']['name']} "
              f"(val net Sharpe {winner['metrics']['net_sharpe']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
