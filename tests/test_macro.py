"""Macro state alignment and Kronecker interaction features."""

from __future__ import annotations

import pandas as pd
import pytest

from macro import (MACRO_COLUMNS, PREDICTOR_PATH, interaction_columns, kron_features,
                   load_fear_greed_monthly, load_macro_monthly, macro_state)


def _monthly() -> pd.DataFrame:
    """Three months of synthetic macro state, indexed by month end."""
    index = pd.to_datetime(["2019-01-31", "2019-02-28", "2019-03-31"])
    values = {column: [10.0, 20.0, 30.0] for column in MACRO_COLUMNS}
    values["tbl"] = [0.01, 0.02, 0.03]
    return pd.DataFrame(values, index=index)


def test_fear_greed_monthly_uses_average(tmp_path):
    path = tmp_path / "fear-greed.csv"
    path.write_text(
        "Date,Fear Greed,Rating\n"
        "2020-01-01,10,Fear\n"
        "2020-01-31,70,Greed\n"
        "2020-02-03,20,Fear\n"
        "2020-02-28,40,Neutral\n"
    )

    monthly = load_fear_greed_monthly(path)

    assert monthly.loc["2020-01-31"] == pytest.approx(40.0)
    assert monthly.loc["2020-02-29"] == pytest.approx(30.0)


def test_macro_state_uses_previous_month_end():
    dates = ["2019-02-01", "2019-02-28", "2019-03-01", "2019-03-31", "2019-04-05"]
    state = macro_state(dates, _monthly())

    assert list(state.columns) == ["const", *MACRO_COLUMNS]
    assert (state["const"] == 1.0).all()
    # February bars see January; a month-end bar never sees its own month.
    assert state.loc["2019-02-01", "tbl"] == 0.01
    assert state.loc["2019-02-28", "tbl"] == 0.01
    # March bars see February; April sees March.
    assert state.loc["2019-03-01", "tbl"] == 0.02
    assert state.loc["2019-03-31", "tbl"] == 0.02
    assert state.loc["2019-04-05", "tbl"] == 0.03


def test_macro_state_zero_lag_allows_same_month_end():
    dates = ["2019-01-30", "2019-01-31", "2019-02-05"]
    state = macro_state(dates, _monthly(), lag_months=0)
    assert pd.isna(state.loc["2019-01-30", "tbl"])
    assert state.loc["2019-01-31", "tbl"] == 0.01
    assert state.loc["2019-02-05", "tbl"] == 0.01


def test_macro_state_rejects_negative_lag():
    with pytest.raises(ValueError):
        macro_state(["2019-02-01"], _monthly(), lag_months=-1)


def test_kron_features_shape_and_values():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2019-02-01", "2019-03-01"]), ["AAA", "BBB"]],
        names=["date", "ticker"])
    panel = pd.DataFrame({"mom": [1.0, 2.0, 3.0, 4.0],
                          "vol": [0.5, 0.5, 0.5, 0.5]}, index=index)
    state = macro_state(panel.index.get_level_values("date"), _monthly())

    block = kron_features(panel, ["mom", "vol"], state)

    assert list(block.columns) == interaction_columns(["mom", "vol"])
    assert block.shape == (4, 2 * (1 + len(MACRO_COLUMNS)))
    assert block.loc[(pd.Timestamp("2019-02-01"), "AAA"), "mom__x__const"] == 1.0
    assert block.loc[(pd.Timestamp("2019-02-01"), "AAA"), "mom__x__tbl"] == pytest.approx(0.01)
    assert block.loc[(pd.Timestamp("2019-03-01"), "BBB"), "mom__x__tbl"] == pytest.approx(0.08)


def test_interaction_columns_count_for_nineteen_firm_features():
    firm = [f"f{i}" for i in range(19)]
    columns = interaction_columns(firm)
    assert len(columns) == 114
    assert columns[0] == "f0__x__const"
    assert columns[-1] == "f18__x__dfr"


@pytest.mark.skipif(not PREDICTOR_PATH.exists(),
                    reason="Welch-Goyal workbook not present")
def test_workbook_covers_the_test_window():
    monthly = load_macro_monthly()
    assert list(monthly.columns) == list(MACRO_COLUMNS)
    assert monthly.index.max() >= pd.Timestamp("2025-09-26")
    recent = monthly.loc["2011":]
    assert not recent.isna().any().any()
