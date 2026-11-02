import numpy as np
import pandas as pd
import pytest

from features import build_feature_frame, feature_columns
from labels import make_labels


def test_forward_labels_match_holding_horizon_and_keep_unrealized_rows():
    closes = pd.DataFrame({"ETF": np.arange(1.0, 20.0)},
                          index=pd.date_range("2025-01-03", periods=19, freq="W-FRI"))

    labels = make_labels(closes)
    date = closes.index[0]

    assert labels.loc[(date, "ETF"), "fwd_ret_1bar"] == pytest.approx(
        closes.iloc[2, 0] / closes.iloc[1, 0] - 1
    )
    assert labels.loc[(date, "ETF"), "fwd_ret_4bar"] == pytest.approx(
        closes.iloc[5, 0] / closes.iloc[1, 0] - 1
    )
    assert labels.loc[(date, "ETF"), "fwd_ret_12bar"] == pytest.approx(
        closes.iloc[13, 0] / closes.iloc[1, 0] - 1
    )
    assert pd.isna(labels.loc[(closes.index[-1], "ETF"), "fwd_ret_12bar"])


def test_feature_frame_does_not_create_labels():
    dates = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    values = np.linspace(100, 160, len(dates))
    bars = pd.DataFrame({
        "Open": values,
        "High": values + 1,
        "Low": values - 1,
        "Close": values,
        "Volume": np.arange(len(dates)) + 100,
    }, index=dates)

    frame = build_feature_frame({"ETF": bars}, sectors=("ETF",))

    assert frame.columns.tolist() == feature_columns()
    assert not any(column.startswith("label_") for column in frame.columns)
