import inspect

import numpy as np
import pandas as pd

from train_models import make_labels, walk_forward_predict


def test_make_labels_uses_close_t_plus_2_over_close_t_plus_1():
    dates = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0],
            "B": [100.0, 99.0, 103.0, 104.0],
        },
        index=dates,
    )

    labels = make_labels(prices)

    assert np.isclose(labels.loc[(dates[0], "A"), "fwd_ret"], 102 / 101 - 1)
    assert np.isclose(labels.loc[(dates[0], "B"), "fwd_ret"], 103 / 99 - 1)
    assert labels.loc[(dates[0], "A"), "target"] == 0
    assert labels.loc[(dates[0], "B"), "target"] == 1
    assert dates[-2] not in labels.index.get_level_values("date")
    assert dates[-1] not in labels.index.get_level_values("date")


def test_walk_forward_uses_two_bar_embargo_by_default():
    assert inspect.signature(walk_forward_predict).parameters["embargo"].default == 2
