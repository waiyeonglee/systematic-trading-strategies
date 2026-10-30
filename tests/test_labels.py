"""Checks for the label builder and the pipeline's default embargo."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from labels import TARGET_COL, make_labels
from models import walk_forward_predict


def test_make_labels_uses_next_bar_return_and_cross_sectional_median():
    dates = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 101.0, 103.0], "B": [100.0, 99.0, 102.0]}, index=dates)

    labels = make_labels(prices)

    first = dates[0]
    assert np.isclose(labels.loc[(first, "A"), "fwd_ret"], 101 / 100 - 1)
    assert np.isclose(labels.loc[(first, "B"), "fwd_ret"], 99 / 100 - 1)
    median = np.median([101 / 100 - 1, 99 / 100 - 1])
    assert np.isclose(labels.loc[(first, "A"), TARGET_COL], 101 / 100 - 1 - median)
    assert np.isclose(labels.loc[(first, "B"), TARGET_COL], 99 / 100 - 1 - median)


def test_make_labels_drops_the_final_date_without_a_next_bar():
    dates = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0]}, index=dates)

    labels = make_labels(prices)

    assert dates[-1] not in labels.index.get_level_values("date")
    assert labels.index.names == ["date", "ticker"]
    assert list(labels.columns) == ["fwd_ret", TARGET_COL]


def test_walk_forward_uses_a_one_bar_embargo_by_default():
    assert inspect.signature(walk_forward_predict).parameters["embargo"].default == 1
