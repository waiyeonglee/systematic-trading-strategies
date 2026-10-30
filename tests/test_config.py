"""Checks for window parsing and the validation-window split."""

from __future__ import annotations

import pandas as pd
import pytest

from splits import build_splits
from utils.config import (duration_slug, duration_to_bars, duration_to_offset, load_settings,
                          parse_duration)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("5y", (5, 0, 0)), ("18m", (0, 18, 0)), ("4w", (0, 0, 4)),
     ("2y 3m 4w", (2, 3, 4)), ("2y3m4w", (2, 3, 4)), ("1Y 2M", (1, 2, 0))],
)
def test_parse_duration_reads_years_months_and_weeks(text, expected):
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text", ["", "3d", "y", "0y", "-1y", "2m 3y"])
def test_parse_duration_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_duration(text)


def test_duration_to_bars_uses_periods_per_year():
    assert duration_to_bars("1y", "daily") == 252
    assert duration_to_bars("1y", "weekly") == 52
    assert duration_to_bars("6m", "weekly") == 26
    assert duration_to_bars("4w", "daily") == 19


def test_duration_slug_is_filename_safe():
    assert duration_slug("2y 3m 4w") == "2y3m4w"


def test_duration_to_offset_adds_all_three_units():
    assert pd.Timestamp("2025-01-01") - duration_to_offset("2y 3m 4w") == pd.Timestamp("2022-09-03")


def test_build_splits_places_the_validation_window_before_the_test_period():
    panel = pd.DataFrame(
        {"x": [1.0]},
        index=pd.MultiIndex.from_tuples([(pd.Timestamp("2010-01-01"), "A")], names=["date", "ticker"]),
    )

    splits = build_splits(panel, "2025-01-01", "2026-09-30", "3y")

    assert splits["val"] == ("2022-01-01", "2024-12-31")
    assert splits["test"] == ("2025-01-01", "2026-09-30")
    assert splits["train"] == ("2010-01-01", "2021-12-31")


def test_build_splits_accepts_month_and_week_windows():
    panel = pd.DataFrame(
        {"x": [1.0]},
        index=pd.MultiIndex.from_tuples([(pd.Timestamp("2010-01-01"), "A")], names=["date", "ticker"]),
    )

    assert build_splits(panel, "2025-01-01", "2026-09-30", "9m")["val"][0] == "2024-04-01"
    assert build_splits(panel, "2025-01-01", "2026-09-30", "1y 6w")["val"][0] == "2023-11-20"




def write_config(tmp_path, extra: str = ""):
    """Write a minimal config.yaml and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(
        "seed: 7\n"
        "data:\n"
        "  start_date: '2000-01-01'\n"
        "  end_date: '2020-12-31'\n"
        "features:\n"
        "  frequency: 'weekly'\n"
        "  alternative:\n"
        "    fear_greed: false\n"
        "validation:\n"
        "  test_start: '2021-01-01'\n"
        "  test_end: '2022-12-31'\n"
        "  window_train: '2y 3m 4w'\n"
        "  window_val: '9m'\n"
        "  refit_every: 'QE'\n"
        "model:\n"
        "  mode: 'single'\n"
        "  single: {family: 'lasso', alpha: 0.9}\n"
        "  selection:\n"
        "    - {family: 'ridge', alpha: [1, 10]}\n"
        "portfolio:\n"
        "  strategy: 'top_k'\n"
        "  top_k:\n"
        "    k: 4\n"
        "    gate: 0.001\n"
        "    cost_bps: 7.5\n"
        "    gate_sweep: [0.0, 0.002]\n" + extra
    )
    return path


def test_load_settings_reads_every_section(tmp_path):
    settings = load_settings(write_config(tmp_path))

    assert (settings.data_start, settings.data_end) == ("2000-01-01", "2020-12-31")
    assert settings.seed == 7
    assert (settings.frequency, settings.use_fear_greed) == ("weekly", False)
    assert (settings.test_start, settings.test_end) == ("2021-01-01", "2022-12-31")
    assert (settings.window_train, settings.window_val) == ("2y 3m 4w", "9m")
    assert settings.refit_every == "QE"
    assert settings.model_mode == "single"
    assert settings.model_entries() == ({"family": "lasso", "alpha": 0.9},)
    assert settings.strategy == "top_k"
    assert (settings.top_k, settings.gate, settings.cost_bps) == (4, 0.001, 7.5)
    assert settings.gates == (0.0, 0.002)


def test_load_settings_selection_mode_uses_the_grid(tmp_path):
    path = write_config(tmp_path)
    path.write_text(path.read_text().replace("mode: 'single'", "mode: 'selection'"))

    settings = load_settings(path)

    assert settings.model_mode == "selection"
    assert settings.model_entries() == ({"family": "ridge", "alpha": [1, 10]},)


def test_load_settings_rejects_an_unknown_model_mode(tmp_path):
    path = write_config(tmp_path)
    path.write_text(path.read_text().replace("mode: 'single'", "mode: 'auto'"))

    with pytest.raises(ValueError):
        load_settings(path)


def test_load_settings_reads_the_forecast_vol_scaled_block(tmp_path):
    path = write_config(tmp_path)
    path.write_text(
        path.read_text()
        .replace("  strategy: 'top_k'", "  strategy: 'forecast_vol_scaled'\n"
                                       "  initial_capital_usd: 50000.0")
        + "  forecast_vol_scaled:\n"
        "    vol_window: 30\n"
        "    vol_min_periods: 10\n"
        "    broker: 'ibkr'\n"
        "    cost_bps: 8.0\n"
    )

    settings = load_settings(path)

    assert settings.strategy == "forecast_vol_scaled"
    assert settings.initial_capital_usd == 50_000.0
    assert settings.forecast_vol.vol_window == 30
    assert settings.forecast_vol.vol_min_periods == 10
    assert settings.forecast_vol.broker == "ibkr"
    assert settings.forecast_vol.cost_bps == 8.0
    assert settings.portfolio_slug() == "fvs30_ibkr_c8_monthly"


def test_load_settings_rejects_a_non_positive_initial_capital(tmp_path):
    path = write_config(tmp_path)
    path.write_text(path.read_text().replace("  strategy: 'top_k'",
                                             "  strategy: 'top_k'\n"
                                             "  initial_capital_usd: 0"))

    with pytest.raises(ValueError):
        load_settings(path)


def test_load_settings_rejects_unknown_strategy_settings(tmp_path):
    path = write_config(tmp_path)
    path.write_text(
        path.read_text()
        .replace("strategy: 'top_k'", "strategy: 'forecast_vol_scaled'")
        + "  forecast_vol_scaled:\n"
        "    vol_windows: 30\n"
    )

    with pytest.raises(ValueError):
        load_settings(path)


def test_load_settings_rejects_an_unknown_strategy(tmp_path):
    path = write_config(tmp_path)
    path.write_text(path.read_text().replace("strategy: 'top_k'", "strategy: 'momentum'"))

    with pytest.raises(ValueError):
        load_settings(path)


def test_top_k_settings_keep_their_folder_slug(tmp_path):
    settings = load_settings(write_config(tmp_path))

    assert settings.portfolio_slug() == "topk4_g0.001_c7.5"
