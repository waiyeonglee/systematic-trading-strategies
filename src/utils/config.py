"""Project constants and the settings loaded from ``config.yaml``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from portfolio import STRATEGIES, ForecastVolParams
from utils.paths import CONFIG_PATH

DEFAULT_SEED = 5205
FREQUENCIES = ("daily", "weekly")
PERIODS_PER_YEAR = {"daily": 252, "weekly": 52}

CORE_ETFS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
BENCHMARKS = ("SPY",)
TICKERS = CORE_ETFS + BENCHMARKS

_DURATION = re.compile(r"^\s*(?:(\d+)\s*y)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*w)?\s*$",
                       re.IGNORECASE)


def periods_per_year(frequency: str) -> int:
    """Return 252 for daily bars and 52 for weekly bars.

    Raises:
        ValueError: if ``frequency`` is not a supported bar frequency.
    """
    try:
        return PERIODS_PER_YEAR[frequency]
    except KeyError:
        raise ValueError(f"frequency must be one of {FREQUENCIES}") from None


def parse_duration(text: str) -> tuple[int, int, int]:
    """Parse a window such as ``"2y 3m 4w"`` into ``(years, months, weeks)``.

    Units are ``y`` (years), ``m`` (months) and ``w`` (weeks), in that order, with or
    without spaces; any subset may be given, e.g. ``"5y"``, ``"18m"``, ``"2y6w"``.

    Raises:
        ValueError: if the text is empty, uses an unknown unit, or adds up to zero.
    """
    match = _DURATION.match(str(text))
    if not match or not any(match.groups()):
        raise ValueError(
            f"invalid window {text!r}: use years, months and weeks in that order, "
            "e.g. '5y', '18m', '2y 3m 4w'"
        )
    years, months, weeks = (int(value or 0) for value in match.groups())
    if years + months + weeks == 0:
        raise ValueError(f"window {text!r} must be longer than zero")
    return years, months, weeks


def duration_to_offset(text: str) -> pd.DateOffset:
    """Convert a window such as ``"2y 3m 4w"`` into a calendar offset."""
    years, months, weeks = parse_duration(text)
    return pd.DateOffset(years=years, months=months, weeks=weeks)


def duration_to_bars(text: str, frequency: str) -> int:
    """Convert a window into a number of bars at ``frequency``, at least one."""
    years, months, weeks = parse_duration(text)
    span = years + months / 12 + weeks / 52
    return max(1, round(span * periods_per_year(frequency)))


def duration_slug(text: str) -> str:
    """Compact filename-safe form of a window, ``"2y 3m 4w"`` -> ``"2y3m4w"``."""
    return str(text).replace(" ", "").lower()


@dataclass(frozen=True)
class Settings:
    """Every setting read from ``config.yaml``; flags override individual values."""

    seed: int = DEFAULT_SEED
    # data
    data_start: str = "2004-01-01"
    data_end: str = "2026-09-30"
    # features
    frequency: str = "weekly"
    use_fear_greed: bool = True
    # validation
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"
    window_train: str = "5y"
    window_val: str = "3y"
    refit_every: str = "ME"
    # model
    model_mode: str = "single"
    model_single: dict = field(default_factory=lambda: {"family": "lasso", "alpha": 0.9})
    model_selection: tuple[dict, ...] = ()
    # portfolio
    strategy: str = "top_k"
    initial_capital_usd: float = 100_000.0
    top_k: int = 5
    gate: float = 0.0
    cost_bps: float = 5.0
    gates: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.002, 0.005)
    forecast_vol: ForecastVolParams = field(default_factory=ForecastVolParams)

    def model_entries(self) -> tuple[dict, ...]:
        """Return the model entries selected by :attr:`model_mode`.

        Raises:
            ValueError: if the mode is unknown or a selection grid is empty.
        """
        if self.model_mode == "single":
            return (self.model_single,)
        if self.model_mode == "selection" and self.model_selection:
            return self.model_selection
        raise ValueError(
            f"model.mode is {self.model_mode!r} but no model entries are configured; "
            "use 'single' with a `single:` block or 'selection' with a non-empty `selection:` list"
        )

    def portfolio_slug(self) -> str:
        """Compact, filename-safe label for the strategy and the settings it trades on."""
        if self.strategy == "forecast_vol_scaled":
            params = self.forecast_vol
            return f"fvs{params.vol_window}_{params.broker}_c{params.cost_bps:g}_{params.rebalance}"
        if self.strategy == "top_k":
            return f"topk{self.top_k}_g{self.gate:g}_c{self.cost_bps:g}"
        return str(self.strategy)


def _section(config: dict, name: str) -> dict:
    """Return one top-level mapping from the config, or an empty mapping."""
    section = config.get(name) or {}
    if not isinstance(section, dict):
        raise ValueError(f"config.yaml: '{name}' must be a mapping")
    return section


def _optional(value, cast):
    """Apply ``cast`` unless the value is missing."""
    return None if value is None else cast(value)


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    """Read ``config.yaml``, keeping :class:`Settings` defaults for missing keys.

    Sections: ``data``, ``features`` (with ``features.alternative``), ``validation``,
    ``model`` and ``portfolio`` (strategy name plus one block per strategy).

    Args:
        path: Config file to read; a missing file yields the defaults.

    Returns:
        The merged settings.

    Raises:
        ValueError: on a malformed window, an unknown ``model.mode``, ``selection`` mode with
            an empty grid, an unknown ``portfolio.strategy`` or an unknown strategy setting.
    """
    path = Path(path)
    if not path.exists():
        return Settings()

    with path.open() as file:
        config = yaml.safe_load(file) or {}

    data = _section(config, "data")
    features = _section(config, "features")
    alternative = _section(features, "alternative")
    validation = _section(config, "validation")
    model = _section(config, "model")
    portfolio = _section(config, "portfolio")

    strategy = portfolio.get("strategy") or Settings.strategy
    strategy_block = _section(portfolio, strategy)
    if strategy not in STRATEGIES:
        raise ValueError(
            f"portfolio.strategy must be one of {sorted(STRATEGIES)}, got {strategy!r}"
        )

    overrides = {
        "seed": _optional(config.get("seed"), int),
        "data_start": data.get("start_date"),
        "data_end": data.get("end_date"),
        "frequency": features.get("frequency"),
        "use_fear_greed": alternative.get("fear_greed"),
        "test_start": validation.get("test_start"),
        "test_end": validation.get("test_end"),
        "window_train": validation.get("window_train"),
        "window_val": validation.get("window_val"),
        "refit_every": validation.get("refit_every"),
        "model_mode": model.get("mode"),
        "model_single": model.get("single"),
        "model_selection": _optional(model.get("selection"), tuple),
        "strategy": strategy,
        "initial_capital_usd": _optional(portfolio.get("initial_capital_usd"), float),
    }
    if strategy == "forecast_vol_scaled":
        overrides["forecast_vol"] = ForecastVolParams.from_mapping(strategy_block)
    else:
        overrides.update({
            "top_k": _optional(strategy_block.get("k"), int),
            "gate": _optional(strategy_block.get("gate"), float),
            "cost_bps": _optional(strategy_block.get("cost_bps"), float),
            "gates": _optional(strategy_block.get("gate_sweep"), tuple),
        })
    settings = Settings(**{key: value for key, value in overrides.items() if value is not None})

    for name in ("window_train", "window_val"):
        parse_duration(getattr(settings, name))  # fail fast on a malformed window

    if settings.model_mode not in {"single", "selection"}:
        raise ValueError(
            f"model.mode must be 'single' or 'selection', got {settings.model_mode!r}"
        )
    settings.model_entries()  # fail fast on an empty or missing model block
    if settings.initial_capital_usd <= 0:
        raise ValueError(
            f"portfolio.initial_capital_usd must be positive, got {settings.initial_capital_usd}"
        )
    return settings
