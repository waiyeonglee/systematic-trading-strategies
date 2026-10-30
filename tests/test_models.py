"""Checks for turning config entries into model specifications."""

from __future__ import annotations

import pytest

from models import default_grid, specs_from_config


def test_specs_from_config_derives_names_penalties_and_ratios():
    specs = specs_from_config([
        {"family": "lasso", "alpha": 0.9},
        {"family": "enet", "alpha": 0.0001, "l1_ratio": 0.5},
    ])

    assert [spec.name for spec in specs] == ["lasso_a0.9", "enet_a0.0001_r0.5"]
    assert [spec.penalty for spec in specs] == ["l1", "elasticnet"]
    assert specs[0].l1_ratio is None
    assert specs[1].l1_ratio == 0.5


def test_specs_from_config_expands_alpha_ranges_per_family():
    specs = specs_from_config([
        {"family": "ridge", "alpha": [1, 10]},
        {"family": "enet", "alpha": [1e-4], "l1_ratio": [0.25, 0.5]},
    ])

    assert [spec.name for spec in specs] == [
        "ridge_a1", "ridge_a10", "enet_a0.0001_r0.25", "enet_a0.0001_r0.5",
    ]
    assert [spec.penalty for spec in specs] == ["l2", "l2", "elasticnet", "elasticnet"]


def test_specs_from_config_ignores_l1_ratio_outside_elastic_net():
    specs = specs_from_config([{"family": "lasso", "alpha": 0.9, "l1_ratio": 0.5}])

    assert [spec.name for spec in specs] == ["lasso_a0.9"]
    assert specs[0].l1_ratio is None


def test_specs_from_config_rejects_an_unknown_family():
    with pytest.raises(ValueError):
        specs_from_config([{"family": "svm", "alpha": 1.0}])


def test_specs_from_config_rejects_an_empty_grid():
    with pytest.raises(ValueError):
        specs_from_config([])


def test_specs_from_config_rejects_a_missing_key():
    with pytest.raises(ValueError):
        specs_from_config([{"family": "lasso"}])


def test_default_grid_has_twenty_models_with_family_penalties():
    grid = default_grid()

    assert len(grid) == 20
    assert grid[0].name == "ridge_a1"
    assert grid[-1].name == "enet_a0.001_r0.75"
    expected = {"ridge": "l2", "lasso": "l1", "enet": "elasticnet"}
    assert all(spec.penalty == expected[spec.name.split("_")[0]] for spec in grid)