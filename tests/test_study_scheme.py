from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from models import ModelSpec
from pipeline import PipelineConfig
from prediction_cache import (DegenerateCandidate, cache_path, cached_predictions,
                              horizon_slug, is_degenerate, matches, provenance)
from splits import fixed_splits
from utils.config import DEFAULT_SEED

FEATURES = ["feature_a", "feature_b"]
VALIDATION = ("2020-01-03", "2021-09-24")


def _panel(periods: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2019-01-04", periods=periods, freq="W-FRI")
    tickers = ["AAA", "BBB"]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.default_rng(DEFAULT_SEED)
    rows = len(index)
    frame = pd.DataFrame({
        "feature_a": rng.normal(size=rows),
        "feature_b": rng.normal(size=rows),
        "fwd_ret": rng.normal(0.0, 0.01, size=rows),
        "fwd_ret_1bar": rng.normal(0.0, 0.02, size=rows),
        "fwd_ret_4bar": rng.normal(0.0, 0.04, size=rows),
        "fwd_ret_12bar": rng.normal(0.0, 0.07, size=rows),
    }, index=index)
    return frame


def _config() -> PipelineConfig:
    return PipelineConfig(target_col="fwd_ret_4bar", embargo=4, refit_every="ME",
                          window_train="1y", seed=DEFAULT_SEED)


def test_fixed_splits_ends_train_the_day_before_validation():
    panel = _panel()
    windows = fixed_splits(panel, ("2020-01-03", "2021-09-24"))

    assert windows["val"] == ("2020-01-03", "2021-09-24")
    assert windows["train"][0] == "2019-01-04"
    assert pd.Timestamp(windows["train"][1]) == (pd.Timestamp("2020-01-03")
                                                 - pd.Timedelta(1, unit="D"))


def test_fixed_splits_rejects_reversed_window():
    with pytest.raises(ValueError):
        fixed_splits(_panel(), ("2021-09-24", "2020-01-03"))


def test_horizon_slug_maps_each_target():
    assert horizon_slug("fwd_ret_1bar") == "la1"
    assert horizon_slug("fwd_ret_12bar") == "la12"
    with pytest.raises(ValueError):
        horizon_slug("fwd_ret")


def test_cached_predictions_round_trip_is_float32_exact(tmp_path):
    panel = _panel()
    config = _config()
    spec = ModelSpec.for_family("ridge", 1.0)
    start, end = VALIDATION

    first = cached_predictions(panel, FEATURES, config, spec, start, end,
                               cache_root=tmp_path, verbose=False)
    cached = cached_predictions(panel, FEATURES, config, spec, start, end,
                                cache_root=tmp_path, verbose=False)
    recomputed = cached_predictions(panel, FEATURES, config, spec, start, end,
                                    cache_root=tmp_path, refresh=True, verbose=False)

    path = cache_path(config, spec.name, start, end, cache_root=tmp_path)
    assert path.exists() and path.suffix == ".npz"
    assert path.with_suffix(".json").exists()
    # the cache stores float32, so a reload is exact against the stored values and
    # within float32 precision of the fresh computation
    pd.testing.assert_series_equal(cached["pred"], first["pred"], rtol=1e-7)
    pd.testing.assert_series_equal(recomputed["pred"], first["pred"], rtol=1e-7)
    assert list(first.columns) == list(panel.columns) + ["pred"]
    assert path.stat().st_size < 0.25 * 1024 * 1024


def test_degenerate_candidate_is_marked_and_not_stored(tmp_path):
    panel = _panel()
    config = _config()
    # a lasso so weak that every prediction collapses to a constant (the intercept)
    spec = ModelSpec.for_family("lasso", 1e6)
    start, end = VALIDATION

    with pytest.raises(DegenerateCandidate):
        cached_predictions(panel, FEATURES, config, spec, start, end,
                           cache_root=tmp_path, verbose=False)

    path = cache_path(config, spec.name, start, end, cache_root=tmp_path)
    assert not path.exists()                       # nothing stored for it
    marker = (path.parent.parent / "_degenerate"
              / f"{path.parent.name}__{path.stem}.json")
    assert marker.exists()
    # a second call raises immediately from the marker (no refit)
    with pytest.raises(DegenerateCandidate):
        cached_predictions(panel, FEATURES, config, spec, start, end,
                           cache_root=tmp_path, verbose=False)


def test_is_degenerate_detects_constant_predictions():
    dates = pd.date_range("2020-01-03", periods=3, freq="W-FRI")
    index = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "ticker"])
    flat = pd.DataFrame({"pred": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}, index=index)
    varying = pd.DataFrame({"pred": [0.1, -0.1] * 3}, index=index)

    assert is_degenerate(flat)["reason"] == "constant_predictions"
    assert is_degenerate(varying) is None


def test_cache_key_ignores_rebalance_cadence(tmp_path):
    panel = _panel()
    config = _config()
    spec = ModelSpec.for_family("ridge", 1.0)
    other = replace(config, forecast_vol=replace(config.forecast_vol, rebalance="12w"))

    assert (cache_path(config, spec.name, *VALIDATION, cache_root=tmp_path)
            == cache_path(other, spec.name, *VALIDATION, cache_root=tmp_path))


def test_cache_entry_is_invalidated_by_a_different_seed():
    panel = _panel()
    config = _config()
    spec = ModelSpec.for_family("ridge", 1.0)

    stored = provenance(config, spec, *VALIDATION, panel)
    assert stored["seed"] == DEFAULT_SEED
    assert matches(stored, provenance(config, spec, *VALIDATION, panel))
    assert not matches(stored, provenance(replace(config, seed=1), spec,
                                           *VALIDATION, panel))
    assert not matches(stored, provenance(replace(config, window_train="2y"), spec,
                                           *VALIDATION, panel))


def test_cache_entry_is_invalidated_by_a_different_panel():
    panel = _panel()
    config = _config()
    spec = ModelSpec.for_family("ridge", 1.0)
    stored = provenance(config, spec, *VALIDATION, panel)
    other_panel = panel.iloc[:-2]

    assert not matches(stored, provenance(config, spec, *VALIDATION, other_panel))


def test_book_space_is_the_declared_product():
    import study_worker as worker

    names = [name for name, _, _ in worker.BOOKS]
    assert len(names) == 45 and len(set(names)) == 45
    assert {knobs["vol_window"] for _, knobs, _ in worker.BOOKS} == {4, 8, 12, 24, 48}
    assert {knobs["vol_target"] for _, knobs, _ in worker.BOOKS} == {0.20, 0.10, None}
    assert {knobs["vol_rf"] for _, knobs, _ in worker.BOOKS} == {0.03}
    assert {knobs["vol_cap"] for _, knobs, _ in worker.BOOKS} == {1.0, 2.0, 3.0}
    # the levered caps pay the borrow fee, the unlevered cap does not
    assert {knobs["vol_cap"]: knobs["lev_fee"]
            for _, knobs, _ in worker.BOOKS} == {1.0: 0.0, 2.0: 0.068, 3.0: 0.068}
    # without a volatility target the cap cannot bind, so 5 of the 45 cells repeat
    # an otherwise identical book and are simulated once
    assert len({worker.effective_key(knobs) for _, knobs, _ in worker.BOOKS}) == 40
