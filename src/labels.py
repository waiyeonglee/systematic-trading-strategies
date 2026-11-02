"""Labels: forward-return training targets at the momentum horizons.

Timing convention (ADR 01 §2.3 / ADR 07): features are computed at the close of
bar ``t``, the trade is executed at the close of bar ``t+1`` and held for ``H``
bars. The model therefore predicts ``price[t+H+1] / price[t+1] - 1`` for an
``H``-bar holding horizon. The horizon is a hyperparameter: ``absolute1`` (1 week),
``absolute4`` (4 weeks) or ``absolute12`` (12 weeks), with the walk-forward embargo
set to the horizon.

The 1-bar label is ``price[t+2] / price[t+1] - 1``: the return over the single
bar after entry. The look-ahead >= frequency rule (validation_and_ablation.md §3)
guarantees a 1-bar target only ever pairs with a 1-week (weekly) cadence.
"""

from __future__ import annotations

import re

import pandas as pd

DEFAULT_HORIZON = 12  # default look-ahead: 12 weekly bars

# Feature-table label at the default horizon.
LABEL = f"label_forward_return_{DEFAULT_HORIZON}bar"

# The fitted-model path trains on any of these horizons:
#   "absolute1"  -> fwd_ret_1bar   (1-week forward return, 2026-11-02 addition)
#   "absolute4"  -> fwd_ret_4bar   (4-week forward return)
#   "absolute12" -> fwd_ret_12bar  (12-week forward return, default)
MODEL_TARGETS = {"absolute1": "fwd_ret_1bar",
                 "absolute4": "fwd_ret_4bar",
                 "absolute12": "fwd_ret_12bar"}
TARGET_COL = MODEL_TARGETS["absolute12"]

# Holding horizon in bars, per target. The embargo excludes the latest H training
# dates, so their labels have fully realized by the next prediction date.
MODEL_EMBARGO = {"absolute1": 1, "absolute4": 4, "absolute12": 12}


def label_column(horizon: int) -> str:
    """Name of the feature-table label for a look-ahead horizon in bars."""
    return f"label_forward_return_{horizon}bar"


def target_horizon(target_col: str) -> int:
    """Label horizon in bars, parsed from a forward-return column name
    (1, 4, or 12 for the corresponding ``fwd_ret_*bar`` target)."""
    match = re.search(r"(\d+)bar", target_col)
    return int(match.group(1)) if match else DEFAULT_HORIZON


def make_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the label columns from a wide table of adjusted closes.

    For each date ``t`` and ticker:

    * ``fwd_ret``       = ``price[t+1] / price[t] - 1``, the close-to-close return the
      portfolio actually earns on the bar dated ``t`` (used for mark-to-market and
      realised performance).
    * ``fwd_ret_1bar``  = ``price[t+2] / price[t+1] - 1``, the 1-week forward return
      realised over the bar after the trade (the strict H=1 form is identically
      zero; see the module note).
    * ``fwd_ret_4bar``  = ``price[t+5] / price[t+1] - 1``, the return over four
      weekly bars after entry at the close of ``t+1``.
    * ``fwd_ret_12bar`` = ``price[t+13] / price[t+1] - 1``, the return over twelve
      weekly bars after entry at the close of ``t+1``.

    Rows are retained even when a forward target is not yet realized; those values
    are NaN. Callers can drop rows according to the targets they actually use.

    Args:
        prices: Wide frame, ``DatetimeIndex`` of trading dates, columns are tickers.

    Returns:
        DataFrame indexed by ``(date, ticker)`` with columns ``fwd_ret``,
        ``fwd_ret_1bar``, ``fwd_ret_4bar`` and ``fwd_ret_12bar``.
    """
    fwd_ret = prices.pct_change(fill_method=None).shift(-1)
    fwd_ret_1bar = prices.shift(-2) / prices.shift(-1) - 1
    fwd_ret_4bar = prices.shift(-5) / prices.shift(-1) - 1
    fwd_ret_12bar = prices.shift(-13) / prices.shift(-1) - 1
    labels = pd.concat(
        {
            "fwd_ret": fwd_ret.stack(future_stack=True),
            "fwd_ret_1bar": fwd_ret_1bar.stack(future_stack=True),
            "fwd_ret_4bar": fwd_ret_4bar.stack(future_stack=True),
            "fwd_ret_12bar": fwd_ret_12bar.stack(future_stack=True),
        },
        axis=1,
    )
    labels.index.names = ["date", "ticker"]
    return labels
