"""Model specifications and walk-forward prediction.

Design (agents/model_design.py): ridge / lasso / elastic-net / RFF-ridge grid on a
walk-forward panel, embargo = target horizon, quarterly refits on a rolling history
window. The RFF ridge family (Random Fourier Features -> Ridge) approximates a
Gaussian-kernel ridge regression; the random projection is seeded with the shared
``DEFAULT_SEED`` so every refit samples the same features (deterministic, and the
kernel approximation is time-consistent across refits).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from labels import TARGET_COL, target_horizon
from utils.checks import require_columns, require_panel
from utils.config import DEFAULT_SEED

FAMILY_PENALTIES = {"ridge": "l2", "lasso": "l1", "enet": "elasticnet", "rff": "rff"}


@dataclass(frozen=True)
class ModelSpec:
    """One point in the hyperparameter grid.

    Attributes:
        name: Unique label ``"<family>_a<alpha>[_r<l1_ratio>]"`` for linear families
            and ``"rff_a<alpha>_d<rff_dim>_g<gamma>"`` for the RFF family; the
            prefix must be ``ridge``, ``lasso``, ``enet`` or ``rff``.
        penalty: ``"l2"`` (Ridge), ``"l1"`` (LASSO), ``"elasticnet"`` or ``"rff"``;
            use :meth:`for_family` to derive it from the family instead of passing
            it in.
        alpha: Regularisation strength; larger means more shrinkage.
        l1_ratio: Only for ``"elasticnet"``: 0 is pure Ridge, 1 is pure LASSO.
        rff_dim: Only for ``"rff"``: number of random Fourier features (D).
        gamma: Only for ``"rff"``: RBF kernel inverse bandwidth in
            ``k(x, y) = exp(-gamma * ||x - y||**2)``.
    """

    name: str
    penalty: str
    alpha: float
    l1_ratio: float | None = None
    rff_dim: int | None = None
    gamma: float | None = None

    @classmethod
    def for_family(cls, family: str, alpha: float, l1_ratio: float | None = None,
                   rff_dim: int | None = None, gamma: float | None = None) -> ModelSpec:
        """Build a spec whose penalty follows from ``family``.

        Raises:
            ValueError: if ``family`` is unknown, or if an ``rff`` spec lacks
                ``rff_dim`` / ``gamma``.
        """
        if family not in FAMILY_PENALTIES:
            raise ValueError(f"unknown model family {family!r}; use one of {sorted(FAMILY_PENALTIES)}")
        if family == "rff":
            if rff_dim is None or gamma is None:
                raise ValueError("rff specs need both rff_dim (int) and gamma (float)")
            return cls(f"rff_a{alpha:g}_d{rff_dim}_g{gamma:g}", "rff", alpha, None,
                       int(rff_dim), float(gamma))
        penalty = FAMILY_PENALTIES[family]
        if penalty != "elasticnet":
            l1_ratio = None
        suffix = f"_r{l1_ratio:g}" if l1_ratio is not None else ""
        return cls(f"{family}_a{alpha:g}{suffix}", penalty, alpha, l1_ratio)

    def build(self, seed: int = DEFAULT_SEED) -> Pipeline:
        """Return a fresh, unfitted scaler-plus-regressor pipeline."""
        if self.penalty == "rff":
            # Standardized features -> seeded RBFSampler (the same projection
            # across refits, since ``seed`` is fixed) -> Ridge on the
            # D-dimensional random feature space. sklearn >= 1.6 renamed
            # RandomFourierFeatures to RBFSampler; same RBF random-feature map.
            return Pipeline([
                ("scaler", StandardScaler()),
                ("rff", RBFSampler(n_components=self.rff_dim,
                                   gamma=self.gamma, random_state=seed)),
                ("reg", Ridge(alpha=self.alpha)),
            ])
        if self.penalty == "l2":
            regressor = Ridge(alpha=self.alpha, random_state=seed)
        elif self.penalty == "l1":
            regressor = Lasso(alpha=self.alpha, max_iter=10000, random_state=seed)
        else:
            regressor = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                                   max_iter=10000, random_state=seed)
        return Pipeline([("scaler", StandardScaler()), ("reg", regressor)])


def model_family(spec_name: str) -> str:
    """Return the family prefix of a spec name, e.g. ``"enet_a0.0001_r0.5"`` -> ``"enet"``,
    ``"rff_a0.01_d100_g0.1"`` -> ``"rff"``."""
    return spec_name.split("_")[0]


def _as_values(value: Any) -> list[float]:
    """Expand a scalar or a list of numbers into a list of floats."""
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def specs_from_config(entries: Iterable[Mapping[str, Any]]) -> list[ModelSpec]:
    """Build specs from config entries, expanding alpha (and l1_ratio) ranges.

    Each entry describes one model family by ``family`` alone; the penalty follows
    from it. ``alpha`` may be a single value or a list, and ``l1_ratio`` (Elastic Net
    only) may also be a list; the cross product becomes one spec per combination.
    An ``rff`` entry additionally expands ``rff_dim`` (list of ints) and ``gamma``
    (list of floats), one spec per (alpha, rff_dim, gamma) combination.

    Returns:
        One :class:`ModelSpec` per combination, named ``"<family>_a<alpha>[_r<l1_ratio>]"``
        or ``"rff_a<alpha>_d<rff_dim>_g<gamma>"``.

    Raises:
        ValueError: if an entry lacks ``family`` / ``alpha`` (or, for ``rff``,
            ``rff_dim`` / ``gamma``), names an unknown family, or the grid ends up
            empty.
    """
    specs = []
    for entry in entries:
        family = str(entry.get("family", ""))
        required = {"family", "alpha"} | ({"rff_dim", "gamma"} if family == "rff" else set())
        missing = required - set(entry)
        if missing:
            raise ValueError(f"model entry {dict(entry)!r} is missing {sorted(missing)}")

        alphas = _as_values(entry["alpha"])
        if family == "rff":
            raw_dims = entry["rff_dim"]
            dims = [int(v) for v in
                    (raw_dims if isinstance(raw_dims, (list, tuple)) else [raw_dims])]
            gammas = _as_values(entry["gamma"])
            for alpha in alphas:
                for dim in dims:
                    for gamma in gammas:
                        specs.append(ModelSpec.for_family(family, alpha,
                                                          rff_dim=dim, gamma=gamma))
            continue
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
    embargo: int = 2,
    *,
    seed: int = DEFAULT_SEED,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Predict ``eval_df`` out of sample by refitting block by block.

    For each block of ``refit_every`` calendar periods: train on the rows dated before
    the block, minus the last ``embargo`` dates, trimmed to the most recent ``window``
    dates if given; fit a fresh model; predict every row in the block.

    Args:
        history: Panel with data before ``eval_df``.
        eval_df: Panel to predict, dated after ``history``.
        feature_cols: Model inputs.
        spec: Which model to fit.
        refit_every: Block length as a pandas offset alias.
        window: Rolling training window in dates; ``None`` uses all past data.
        embargo: Most recent dates dropped before each refit; must be at least the label
            horizon so no training row's target is realised after the refit date.
        seed: Random state passed to every refit.
        target_col: Column the model is trained on.

    Returns:
        ``eval_df`` plus a ``pred`` column, indexed by ``(date, ticker)``.

    Raises:
        ValueError: if a required column is missing from either panel.
    """
    required = feature_cols + [target_col]
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
            model.fit(train_rows[feature_cols].values, train_rows[target_col].values)

        block = eval_df.loc[eval_df.index.get_level_values("date").isin(block_dates)].copy()
        block["pred"] = model.predict(block[feature_cols].values)
        predictions.append(block)

    return pd.concat(predictions).sort_index()


def ar1_walk_forward(
    history: pd.DataFrame,
    eval_df: pd.DataFrame,
    target_col: str = TARGET_COL,
    refit_every: str = "QE",
    window: int | None = None,
    embargo: int = 2,
    lag: int | None = None,
) -> pd.DataFrame:
    """Naive AR(1) baseline on the target series itself.

    Per ticker, fit ``y_s = a + phi * y_{s-lag}`` on the rows dated before each block
    (minus the last ``embargo`` dates, trimmed to the most recent ``window`` dates),
    then predict every row in the block from its lagged target value. ``lag`` defaults
    to the label horizon; rows with no lagged value get a zero forecast.

    Args:
        history: Panel with data before ``eval_df``.
        eval_df: Panel to predict, dated after ``history``.
        target_col: Column the AR(1) is fit on.
        refit_every: Block length as a pandas offset alias (same as the model).
        window: Rolling training window in dates; ``None`` uses all past data.
        embargo: Most recent dates dropped before each refit (same as the model).
        lag: AR lag in bars; defaults to the label horizon via :func:`target_horizon`.

    Returns:
        ``eval_df`` plus a ``pred`` column, indexed by ``(date, ticker)``.
    """
    require_columns(history, [target_col], "history panel")
    require_columns(eval_df, [target_col], "evaluation panel")
    require_panel(history)
    require_panel(eval_df)

    lag = lag or target_horizon(target_col)
    full = pd.concat([history, eval_df]).sort_index()
    full["y_lag"] = full.groupby(level="ticker")[target_col].shift(lag)
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
        train_rows = full.loc[full.index.get_level_values("date").isin(past)
                              & full["y_lag"].notna()]

        coeffs = {}
        for ticker, rows in train_rows.groupby(level="ticker"):
            design = np.column_stack([np.ones(len(rows)), rows["y_lag"].values])
            beta, *_ = np.linalg.lstsq(design, rows[target_col].values, rcond=None)
            coeffs[ticker] = beta

        block = eval_df.loc[eval_df.index.get_level_values("date").isin(block_dates)].copy()
        block["y_lag"] = full.loc[block.index, "y_lag"]
        pred = np.zeros(len(block))
        for ticker, beta in coeffs.items():
            mask = block.index.get_level_values("ticker") == ticker
            pred[mask] = beta[0] + beta[1] * block.loc[mask, "y_lag"].values
        block["pred"] = pred
        predictions.append(block)

    return pd.concat(predictions).sort_index()
