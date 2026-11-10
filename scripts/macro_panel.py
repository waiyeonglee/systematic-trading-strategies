"""Runnable panel for the joint feature-set study.

Firm block: the 19 asset-level features (the 21-column v2 set minus the two
market-wide columns, ``fear_greed_index`` and ``spy_return_1bar``). Macro block: the
five :data:`macro.MACRO_COLUMNS` plus a constant. The panel always carries the
Kronecker product of the two — 19 x (5 + 1) = 114 columns — and :func:`feature_columns`
selects the block a model reads:

    micro   19 columns    firm features x const (no macro interaction)
    macro  114 columns    firm features x (const + 5 macro state variables)

so a feature can carry a regime-dependent sign and magnitude in the macro block.

The panel is built in memory from the raw weekly CSVs and the Welch-Goyal workbook;
:mod:`build_features` writes the same frame to CSV for inspection (``features_macro.csv``).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd

from build_features import load_ohlcv
from features import FEATURES, TREND_FEATURES, build_feature_frame
from labels import make_labels
from macro import MACRO_COLUMNS, interaction_columns, kron_features, macro_state
from utils.config import BENCHMARKS, TRADABLE_ETFS
from utils.paths import raw_data_dir

# Market-wide columns of the v2 set; they become macro state instead of firm features.
MARKET_WIDE = ("fear_greed_index", "spy_return_1bar")

FIRM_COLUMNS = ([name for name in FEATURES if name not in MARKET_WIDE]
                + list(TREND_FEATURES))
assert len(FIRM_COLUMNS) == 19, f"expected 19 firm features, got {len(FIRM_COLUMNS)}"

FEATURE_SETS = ("micro", "macro")


def feature_columns(feature_set: str) -> list[str]:
    """Panel columns the model reads for one feature block.

    ``micro`` is the 19 firm features themselves, with no macro interaction.
    ``macro`` is the full 19 x 6 Kronecker block (firm x (const + 5 macro state)),
    which is the firm block plus one ``__x__`` column per macro variable.

    Raises:
        ValueError: if ``feature_set`` is not one of :data:`FEATURE_SETS`.
    """
    if feature_set == "micro":
        return list(FIRM_COLUMNS)
    if feature_set == "macro":
        return interaction_columns(FIRM_COLUMNS)
    raise ValueError(f"unknown feature set {feature_set!r}; use one of {list(FEATURE_SETS)}")


def build_panel() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build the macro-interaction panel from the raw weekly CSVs and macro inputs.

    Returns:
        ``(panel, benchmark_close, feature_cols)`` where the panel is indexed by
        ``(date, ticker)`` with the 114 interaction columns, ``fwd_ret`` and the
        1/4/12-bar training labels; ``feature_cols`` lists the 114 model inputs in
        build order (the ``macro`` block; see :func:`feature_columns`). Rows without a
        realised label or a macro value are dropped.
    """
    prices = load_ohlcv(raw_data_dir("weekly"), TRADABLE_ETFS)
    benchmark = load_ohlcv(raw_data_dir("weekly"), BENCHMARKS)[BENCHMARKS[0]]

    # No fear-greed / benchmark inputs here: both are macro state in this study.
    features = build_feature_frame(prices, frequency="weekly", sectors=TRADABLE_ETFS)
    close_wide = pd.DataFrame({ticker: prices[ticker]["Close"] for ticker in TRADABLE_ETFS}
                              ).sort_index()
    labels = make_labels(close_wide)

    base = (features[FIRM_COLUMNS]
            .join(labels, how="inner")
            .dropna()
            .sort_index())
    state = macro_state(base.index.get_level_values("date"))
    interactions = kron_features(base, FIRM_COLUMNS, state)
    panel = pd.concat([base, interactions], axis=1).dropna().sort_index()

    benchmark_close = benchmark["Close"].astype(float).sort_index()
    benchmark_close.index = pd.to_datetime(benchmark_close.index)
    return panel, benchmark_close, interaction_columns(FIRM_COLUMNS)


def macro_levels(panel: pd.DataFrame) -> pd.DataFrame:
    """The five raw macro values per panel date, for the inspection CSV."""
    state = macro_state(panel.index.get_level_values("date"))
    state = state[list(MACRO_COLUMNS)].reindex(panel.index.get_level_values("date"))
    state.index = panel.index
    return state.add_prefix("macro_")
