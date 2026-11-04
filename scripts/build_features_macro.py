#!/usr/bin/env python3
"""Write the macro-interaction feature panel to CSV (inspection artifact).

The macro study builds the same frame in memory (:mod:`macro_panel`); this CLI only
persists it so the design can be eyeballed: 19 firm features x (5 macro + const) = 114
interaction columns, the five raw macro levels, ``fwd_ret`` and the 1/4/12-bar labels.
The output is git-ignored.

Usage:
    python scripts/build_features_macro.py
    python scripts/build_features_macro.py --out data/features/weekly/features_macro.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd

from macro_panel import build_panel, macro_levels
from utils.paths import FEATURES_DIR

DEFAULT_OUT = FEATURES_DIR / "weekly" / "features_macro.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"destination CSV (default: {DEFAULT_OUT.relative_to(PROJECT_ROOT)})")
    args = parser.parse_args()

    panel, _, feature_cols = build_panel()
    frame = pd.concat([panel, macro_levels(panel)], axis=1).reset_index()
    frame = frame[["date", "ticker", *feature_cols,
                   *[column for column in frame.columns if column.startswith("macro_")],
                   "fwd_ret", "fwd_ret_1bar", "fwd_ret_4bar", "fwd_ret_12bar"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    dates = panel.index.get_level_values("date")
    print(f"wrote {args.out}")
    print(f"rows {len(panel)} | dates {dates.min():%Y-%m-%d}..{dates.max():%Y-%m-%d} "
          f"({dates.nunique()}) | tickers {panel.index.get_level_values('ticker').nunique()} "
          f"| interaction columns {len(feature_cols)}")
    print(f"columns: date, ticker, {len(feature_cols)} interactions, 5 macro levels, "
          f"fwd_ret + 3 labels")


if __name__ == "__main__":
    main()
