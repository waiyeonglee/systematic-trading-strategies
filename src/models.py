"""Model specifications and walk-forward prediction."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from labels import TARGET_COL
from utils.checks import require_columns, require_panel
from utils.config import DEFAULT_SEED

FAMILY_PENALTIES = {"ridge": "l2", "lasso": "l1", "enet": "elasticnet"}


@dataclass(frozen=True)
class ModelSpec:
    """One point in the hyperparameter grid.

    Attributes:
        name: Unique label ``"<family>_a<alpha>[_r<l1_ratio>]"``; the prefix must be
            ``ridge``, ``lasso`` or ``enet``.
        penalty: ``"l2"`` (Ridge), ``"l1"`` (LASSO) or ``"elasticnet"``; use
            :meth:`for_family` to derive it from the family instead of passing it in.
        alpha: Regularisation strength; larger means more shrinkage.
        l1_ratio: Only for ``"elasticnet"``: 0 is pure Ridge, 1 is pure LASSO.
    """

    name: str
    penalty: str
    alpha: float
    l1_ratio: float | None = None

    @classmethod
    def for_family(cls, family: str, alpha: float, l1_ratio: float | None = None) -> ModelSpec:
        """Build a spec whose penalty follows from ``family``.

        Args:
            family: ``"ridge"``, ``"lasso"`` or ``"enet"``.
            alpha: Regularisation strength.
            l1_ratio: Elastic Net mixing weight; ignored for the other families.

        Raises:
            ValueError: if ``family`` is unknown.
        """
        if family not in FAMILY_PENALTIES:
            raise ValueError(f"unknown model family {family!r}; use one of {sorted(FAMILY_PENALTIES)}")
        penalty = FAMILY_PENALTIES[family]
        if penalty != "elasticnet":
            l1_ratio = None
        suffix = f"_r{l1_ratio:g}" if l1_ratio is not None else ""
        return cls(f"{family}_a{alpha:g}{suffix}", penalty, alpha, l1_ratio)

    def build(self, seed: int = DEFAULT_SEED) -> Pipeline:
        """Return a fresh, unfitted scaler-plus-regressor pipeline.

        Args:
            seed: Random state for the regressor; keeps runs reproducible.
        """
        if self.penalty == "l2":
            regressor = Ridge(alpha=self.alpha, random_state=seed)
        elif self.penalty == "l1":
            regressor = Lasso(alpha=self.alpha, max_iter=10000, random_state=seed)
        else:
            regressor = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                                   max_iter=10000, random_state=seed)
        return Pipeline([("scaler", StandardScaler()), ("reg", regressor)])


def default_grid() -> list[ModelSpec]:
    """Return the pre-declared grid: 4 Ridge, 4 LASSO and 12 Elastic Net models."""
    return specs_from_config([
        {"family": "ridge", "alpha": [1, 10, 100, 1000]},
        {"family": "lasso", "alpha": [1e-6, 1e-5, 1e-4, 1e-3]},
        {"family": "enet", "alpha": [1e-6, 1e-5, 1e-4, 1e-3], "l1_ratio": [0.25, 0.5, 0.75]},
    ])


def model_family(spec_name: str) -> str:
    """Return the family prefix of a spec name, e.g. ``"enet_a0.0001_r0.5"`` -> ``"enet"``."""
    return spec_name.split("_")[0]


def _as_values(value: Any) -> list[float]:
    """Expand a scalar or a YAML list of numbers into a list of floats."""
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def specs_from_config(entries: Iterable[Mapping[str, Any]]) -> list[ModelSpec]:
    """Build specs from config entries, expanding alpha (and l1_ratio) ranges.

    Each entry describes one model family by ``family`` alone; the penalty (``l2``,
    ``l1`` or ``elasticnet``) follows from it. ``alpha`` may be a single value or a list,
    and ``l1_ratio`` (Elastic Net only) may also be a list. The cross product of the alpha
    and l1_ratio lists becomes one spec per combination.

    Args:
        entries: One mapping per family, e.g.
            ``{"family": "lasso", "alpha": [1e-4, 1e-3]}`` or
            ``{"family": "enet", "alpha": [1e-4], "l1_ratio": [0.25, 0.5]}``.

    Returns:
        One :class:`ModelSpec` per combination, named ``"<family>_a<alpha>[_r<l1_ratio>]"``.

    Raises:
        ValueError: if an entry lacks ``family`` or ``alpha``, names an unknown family,
            or the grid ends up empty.
    """
    specs = []
    for entry in entries:
        missing = {"family", "alpha"} - set(entry)
        if missing:
            raise ValueError(f"model entry {dict(entry)!r} is missing {sorted(missing)}")

        family = str(entry["family"])
        alphas = _as_values(entry["alpha"])
        if family == "enet" and entry.get("l1_ratio") is not None:
            ratios = _as_values(entry["l1_ratio"])
        else:
            ratios = [None]

        for alpha in alphas:
            for l1_ratio in ratios:
                specs.append(ModelSpec.for_family(family, alpha, l1_ratio))

    if not specs:
        raise ValueError("at least one model specification is required")
    return specs


def walk_forward_predict(
    history: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    spec: ModelSpec,
    refit_every: str = "QE",
    window: int | None = None,
    embargo: int = 1,
    *,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Predict ``eval_df`` out of sample by refitting block by block.

    For each block of ``refit_every`` calendar periods: train on the rows dated before the
    block, minus the last ``embargo`` dates, trimmed to the most recent ``window`` dates if
    given; fit a fresh model; predict every row in the block.

    Args:
        history: Panel with data before ``eval_df``.
        eval_df: Panel to predict, dated after ``history``.
        feature_cols: Model inputs.
        spec: Which model to fit.
        refit_every: Block length as a pandas offset alias.
        window: Rolling training window in dates; ``None`` uses all past data.
        embargo: Most recent dates dropped before each refit.
        seed: Random state passed to every refit.

    Returns:
        ``eval_df`` plus a ``pred`` column, indexed by ``(date, ticker)``.

    Raises:
        ValueError: if a required column is missing from either panel.
    """
    required = feature_cols + [TARGET_COL]
    require_columns(history, required, "history panel")
    require_columns(eval_df, required, "evaluation panel")
    require_panel(history)
    require_panel(eval_df)

    full = pd.concat([history, eval_df]).sort_index()
    all_dates = full.index.get_level_values("date").unique().sort_values()
    eval_dates = eval_df.index.get_level_values("date").unique().sort_values()

    predictions = []
    for _, block_dates in pd.Series(eval_dates, index=eval_dates).groupby(pd.Grouper(freq=refit_every)):
        if block_dates.empty:
            continue
        past = all_dates[all_dates < block_dates.iloc[0]]
        past = past[:-embargo] if embargo else past
        if window is not None:
            past = past[-window:]
        train_rows = full.loc[full.index.get_level_values("date").isin(past)]

        model = spec.build(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence warnings at extreme alphas
            model.fit(train_rows[feature_cols].values, train_rows[TARGET_COL].values)

        block = eval_df.loc[eval_df.index.get_level_values("date").isin(block_dates)].copy()
        block["pred"] = model.predict(block[feature_cols].values)
        predictions.append(block)

    return pd.concat(predictions).sort_index()
