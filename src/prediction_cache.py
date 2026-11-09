"""Walk-forward prediction cache shared by the validation study.

Predictions are computed once per (model spec, look-ahead, refit cadence) and stored
under ``results/_study/predictions``; :func:`cached_predictions` recomputes only when
the stored provenance does not match the request (seed, target, refit cadence,
rolling training window, embargo, evaluation window, panel signature).

Predictions do not depend on the rebalance cadence or on any book knob, so the cache
key omits them: weekly / 4w / 12w candidates of the same (look-ahead, refit) reuse one
prediction set.

Storage format (2026-11-04): one ``.npz`` per prediction set holding the prediction
matrix as ``float32`` (``(dates, tickers)``, ~20 KB for six years of weekly bars,
versus ~185 KB for the old long CSV), plus a ``.json`` provenance sidecar. Degenerate
sets — predictions that are constant across tickers on the evaluation window — are not
stored at all: a marker file under ``_degenerate/`` records them, and later calls
raise :class:`DegenerateCandidate` instead of refitting or replaying useless forecasts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from labels import MODEL_TARGETS
from models import ModelSpec, walk_forward_predict
from utils.paths import PREDICTIONS_DIR

PROVENANCE_KEYS = ("freq", "spec", "target_col", "refit_every", "window_train_bars",
                   "embargo", "seed", "eval_start", "eval_end", "bars", "panel")

PRED_DTYPE = np.float32             # 4 bytes per prediction: |delta Sharpe| < 1e-7
DEGENERATE_STD = 1e-12              # mean cross-sectional std below this = degenerate
DEGENERATE_DIRNAME = "_degenerate"


class DegenerateCandidate(Exception):
    """The candidate's predictions are constant across tickers on the window.

    Nothing is stored for it (the forecasts carry no cross-sectional information),
    so callers should record the candidate as degenerate and skip its book rows.
    ``record`` defaults to ``{}`` so the exception survives pickling across a
    worker process (``BaseException.__reduce__`` replays only ``args``).
    """

    def __init__(self, key: str, record: dict | None = None):
        self.key = key
        self.record = record or {}
        super().__init__(f"degenerate candidate {key}: "
                         f"{self.record.get('reason', 'constant')}")


def horizon_slug(target_col: str) -> str:
    """``"fwd_ret_12bar"`` -> ``"la12"`` (look-ahead in bars)."""
    for name, column in MODEL_TARGETS.items():
        if column == target_col:
            return f"la{name.removeprefix('absolute')}"
    raise ValueError(f"unknown model target {target_col!r}; use one of "
                     f"{sorted(MODEL_TARGETS.values())}")


def panel_signature(panel: pd.DataFrame) -> str:
    """Compact fingerprint of the panel that predictions were computed from."""
    dates = panel.index.get_level_values("date")
    tickers = panel.index.get_level_values("ticker")
    return (f"{dates.min():%Y-%m-%d}..{dates.max():%Y-%m-%d}"
            f"|{len(panel)}|{tickers.nunique()}")


def _group_dir(config, cache_root: Path | None = None) -> Path:
    root = Path(cache_root or PREDICTIONS_DIR)
    return root / f"{config.freq}_{horizon_slug(config.target_col)}_refit{config.refit_every}"


def cache_path(config, spec_name: str, start: str, end: str, *,
               bars: int | None = None, cache_root: Path | None = None) -> Path:
    """Cache file (.npz) for one prediction set.

    ``bars`` marks the short volatility warmup window (``<spec>__warmup<bars>.npz``);
    omit it for the evaluation window.
    """
    root = Path(cache_root or PREDICTIONS_DIR)
    window = f"{pd.Timestamp(start):%Y-%m-%d}_{pd.Timestamp(end):%Y-%m-%d}"
    name = spec_name if bars is None else f"{spec_name}__warmup{bars}"
    return root / window / _group_dir(config, cache_root).name / f"{name}.npz"


def marker_path(path: Path) -> Path:
    """Degenerate marker matching a prediction payload ``path`` (``<group>__<name>.json``)."""
    return (path.parent.parent / DEGENERATE_DIRNAME
            / f"{path.parent.name}__{path.stem}.json")


def provenance(config, spec: ModelSpec, start: str, end: str, panel: pd.DataFrame,
               *, bars: int | None = None) -> dict:
    """Everything that must match for a cached prediction set to be reusable."""
    return {
        "freq": config.freq,
        "spec": spec.name,
        "family": spec.penalty,
        "alpha": spec.alpha,
        "l1_ratio": spec.l1_ratio,
        "target_col": config.target_col,
        "rebalance": config.forecast_vol.rebalance,
        "refit_every": config.refit_every,
        "window_train_bars": config.window_train_bars,
        "embargo": config.embargo,
        "seed": config.seed,
        "eval_start": f"{pd.Timestamp(start):%Y-%m-%d}",
        "eval_end": f"{pd.Timestamp(end):%Y-%m-%d}",
        "bars": bars,
        "panel": panel_signature(panel),
        "format": f"npz/{np.dtype(PRED_DTYPE).name}",
    }


def matches(stored: dict, request: dict) -> bool:
    """``stored`` covers the request on every provenance key except the format tag."""
    return all(stored.get(key) == request[key] for key in PROVENANCE_KEYS)


def is_degenerate(preds: pd.DataFrame) -> dict | None:
    """``None`` when the predictions vary across tickers, else a degenerate record."""
    if preds.empty:
        return {"reason": "empty", "std_mean": None, "rows": 0}
    std = preds.groupby(level="date")["pred"].std()
    std_mean = float(std.mean()) if std.notna().any() else 0.0
    if std_mean >= DEGENERATE_STD and np.isfinite(std_mean):
        return None
    return {"reason": "constant_predictions", "std_mean": std_mean,
            "rows": int(len(preds))}


def write_predictions(preds: pd.DataFrame, path: Path, meta: dict) -> None:
    """Store ``pred`` as a float32 ``(dates, tickers)`` matrix plus its axes."""
    wide = preds["pred"].unstack("ticker").sort_index()
    wide = wide[sorted(wide.columns)]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             pred=wide.to_numpy(dtype=PRED_DTYPE),
             dates=wide.index.to_numpy(dtype="datetime64[s]"),
             tickers=np.asarray(wide.columns, dtype=str))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))


def _read(path: Path) -> pd.Series:
    """Load the stored matrix back into a ``(date, ticker)`` series."""
    with np.load(path) as bundle:
        pred = bundle["pred"]
        dates = pd.DatetimeIndex(bundle["dates"])
        tickers = [str(value) for value in bundle["tickers"]]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    return pd.Series(pred.astype(float).ravel(), index=index).sort_index()


def _with_predictions(panel: pd.DataFrame, preds: pd.Series) -> pd.DataFrame:
    """Panel rows of the prediction dates, carrying the cached ``pred`` column."""
    missing = preds.index.difference(panel.index)
    if len(missing):
        raise ValueError(f"cached predictions cover {len(missing)} rows absent from the "
                         f"panel; delete the cache or rebuild the panel")
    out = panel.loc[preds.index].copy()
    out["pred"] = preds.astype(float)
    return out.sort_index()


def mark_degenerate(path: Path, request: dict, record: dict) -> Path:
    """Record a degenerate prediction set (and drop any stale payload)."""
    marker = marker_path(path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**request, **record,
                                  "detected": datetime.now(timezone.utc).isoformat()},
                                 indent=2, default=str))
    if path.exists():
        path.unlink()
    if path.with_suffix(".json").exists():
        path.with_suffix(".json").unlink()
    return marker


def _degenerate_record(path: Path) -> dict | None:
    marker = marker_path(path)
    if not marker.exists():
        return None
    return json.loads(marker.read_text())


def degenerate_markers(cache_root: Path | None = None) -> dict[str, dict]:
    """``{marker stem: record}`` for every degenerate prediction set in the cache."""
    root = Path(cache_root or PREDICTIONS_DIR)
    out = {}
    for marker in root.glob(f"*/{DEGENERATE_DIRNAME}/*.json"):
        out[marker.stem] = json.loads(marker.read_text())
    return out


def _cached_or_compute(path: Path, panel: pd.DataFrame, config, spec: ModelSpec,
                       start: str, end: str, refresh: bool, verbose: bool,
                       compute, *, check_degenerate: bool = True) -> pd.DataFrame:
    """Shared cache-hit / degenerate-marker / compute path."""
    request = provenance(config, spec, start, end, panel, bars=request_bars(path))
    if not refresh:
        if check_degenerate:
            record = _degenerate_record(path)
            if record is not None:
                raise DegenerateCandidate(path.stem, record)
        if path.exists():
            stored = json.loads(path.with_suffix(".json").read_text())
            if matches(stored, request):
                if verbose:
                    print(f"    [cache hit ] {path.parent.name}/{path.name}")
                return _with_predictions(panel, _read(path))

    if verbose:
        print(f"    [cache miss] {path.parent.name}/{path.name} — fitting")
    preds = compute()
    if check_degenerate:
        degenerate = is_degenerate(preds)
        if degenerate is not None:
            mark_degenerate(path, request, degenerate)
            raise DegenerateCandidate(path.stem, degenerate)
    write_predictions(preds, path, request)
    marker = marker_path(path)
    if marker.exists():          # a stale marker from an earlier degenerate run
        marker.unlink()
    return preds


def request_bars(path: Path) -> int | None:
    """Recover the warmup length encoded in a ``...__warmup<bars>.npz`` file name."""
    stem = path.stem
    if "__warmup" not in stem:
        return None
    return int(stem.rsplit("__warmup", 1)[1])


def cached_predictions(panel: pd.DataFrame, feature_cols: list[str], config,
                       spec: ModelSpec, start: str, end: str, *,
                       cache_root: Path | None = None, refresh: bool = False,
                       verbose: bool = True) -> pd.DataFrame:
    """Out-of-sample predictions on ``[start, end]``, computed once and reused.

    Args:
        panel: Feature panel indexed by ``(date, ticker)``.
        feature_cols: Model inputs.
        config: Pipeline settings (target, refit cadence, training window, embargo, seed).
        spec: The model spec to walk forward.
        start / end: Evaluation window bounds.
        cache_root: Cache root; defaults to ``results/_study/predictions``.
        refresh: Recompute even when a valid cache entry exists.
        verbose: Print a ``[cache] hit|miss`` line per prediction set.

    Returns:
        ``panel`` rows for the window plus a ``pred`` column.

    Raises:
        DegenerateCandidate: if the predictions are constant across tickers on the
            window (nothing is stored for them; the marker is written on first sight).
    """
    path = cache_path(config, spec.name, start, end, cache_root=cache_root)

    def compute() -> pd.DataFrame:
        dates = panel.index.get_level_values("date")
        history = panel[dates < pd.Timestamp(start)]
        eval_df = panel[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
        if history.empty or eval_df.empty:
            raise ValueError(f"empty history or evaluation window for {start}..{end}")
        return walk_forward_predict(history, eval_df, feature_cols, spec,
                                    refit_every=config.refit_every,
                                    window=config.window_train_bars,
                                    embargo=config.embargo,
                                    seed=config.seed,
                                    target_col=config.target_col)

    return _cached_or_compute(path, panel, config, spec, start, end,
                              refresh, verbose, compute)


def warmup_frame(panel: pd.DataFrame, feature_cols: list[str], config,
                 spec: ModelSpec, bars: int) -> pd.DataFrame:
    """Predictions on the last ``bars`` dates of ``panel`` (vol-estimate seed only).

    The panel must end where the evaluation window begins. Returns an empty frame
    when there is not enough history to fit and embargo the trailing bars.
    """
    dates = panel.index.get_level_values("date")
    unique_dates = dates.unique().sort_values()
    if len(unique_dates) <= bars + config.embargo:
        return panel.iloc[0:0].copy().assign(pred=pd.Series(dtype=float))
    warm_dates = unique_dates[-bars:]
    history = panel[dates < warm_dates[0]]
    eval_df = panel[dates.isin(warm_dates)]
    return walk_forward_predict(history, eval_df, feature_cols, spec,
                                refit_every=config.refit_every,
                                window=config.window_train_bars,
                                embargo=config.embargo,
                                seed=config.seed,
                                target_col=config.target_col)


def warmup_predictions(panel: pd.DataFrame, feature_cols: list[str], config,
                       spec: ModelSpec, eval_start: str, bars: int) -> pd.DataFrame:
    """Predictions on the ``bars`` dates immediately before ``eval_start``.

    Used only to seed the book's rolling forecast-volatility estimate, so the first
    evaluation bar already has a volatility history. Its returns are never traded.
    """
    unique_dates = panel.index.get_level_values("date").unique().sort_values()
    prior = unique_dates[unique_dates < pd.Timestamp(eval_start)]
    mask = panel.index.get_level_values("date").isin(prior)
    return warmup_frame(panel[mask], feature_cols, config, spec, bars)


def cached_warmup(panel: pd.DataFrame, feature_cols: list[str], config,
                  spec: ModelSpec, eval_start: str, *, bars: int,
                  cache_root: Path | None = None, refresh: bool = False,
                  verbose: bool = True) -> pd.DataFrame:
    """Cached version of :func:`warmup_predictions` (same provenance discipline)."""
    unique_dates = panel.index.get_level_values("date").unique().sort_values()
    prior = unique_dates[unique_dates < pd.Timestamp(eval_start)]
    if len(prior) <= bars + config.embargo:
        raise ValueError(f"fewer than {bars + config.embargo} bars before {eval_start}; "
                         f"cannot build a {bars}-bar volatility warmup")
    warm_dates = prior[-bars:]
    start, end = warm_dates[0], warm_dates[-1]
    path = cache_path(config, spec.name, start, end, bars=bars, cache_root=cache_root)

    def compute() -> pd.DataFrame:
        return warmup_predictions(panel, feature_cols, config, spec, eval_start, bars)

    # a flat warmup is harmless (it only seeds the volatility estimate), so it is
    # always stored and never recorded as a degenerate candidate
    return _cached_or_compute(path, panel, config, spec, start, end,
                              refresh, verbose, compute, check_degenerate=False)
