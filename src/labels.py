"""Labels: the forward-return column and the cross-sectional excess-return target."""

from __future__ import annotations

import pandas as pd

LABEL = "label_forward_return_1bar"
TARGET_COL = "excess_ret"


def make_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the label columns from a wide table of adjusted closes.

    For each date ``t`` and ticker:

    * ``fwd_ret``    = ``price[t+1] / price[t] - 1``
    * ``excess_ret`` = ``fwd_ret - median(fwd_ret across tickers at t)``

    The last date has no ``t+1`` price, so it is dropped.

    Args:
        prices: Wide frame, ``DatetimeIndex`` of trading dates, columns are tickers.

    Returns:
        DataFrame indexed by ``(date, ticker)`` with columns ``fwd_ret`` and ``excess_ret``.
    """
    fwd_ret = prices.pct_change(fill_method=None).shift(-1)
    excess = fwd_ret.sub(fwd_ret.median(axis=1), axis=0)
    labels = pd.concat(
        {
            "fwd_ret": fwd_ret.stack(future_stack=True),
            TARGET_COL: excess.stack(future_stack=True),
        },
        axis=1,
    )
    labels.index.names = ["date", "ticker"]
    return labels.dropna()
