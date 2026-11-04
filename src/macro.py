"""Macroeconomic state variables and their interaction (Kronecker) features.

Five monthly state variables feed the macro study:

    fear_greed  CNN Fear & Greed index (0-100), month-end value
    tbl         3-month T-bill rate (decimal)
    tms         term spread, ``lty - tbl`` (long govt yield - T-bill)
    dfy         default yield spread, ``BAA - AAA``
    dfr         default return spread, ``corpr - ltr``

``tbl`` / ``tms`` / ``dfy`` / ``dfr`` come from the Welch-Goyal predictor workbook
(``data/raw/Data2025.xlsx``, sheet ``Monthly``), downloaded from Amit Goyal's
predictor-data page <https://sites.google.com/view/agoyal145>; ``fear_greed`` comes
from the local daily CSV. The workbook's ``ReadMe`` sheet defines ``tms``, ``dfy`` and
``dfr`` as the derived columns used here, so the workbook values are taken as-is.

Timing (no lookahead): a bar dated in month ``M`` uses the month-end value of month
``M - 1``. The month-``M`` observation is only published after that month closes, so
using it on a bar inside month ``M`` would leak future information.
The interaction block is the Kronecker product of ``FIRM_COLUMNS`` with
``(const, fear_greed, tbl, tms, dfy, dfr)``: 19 firm features x 6 = 114 columns.
"""

from __future__ import annotations

import pandas as pd

from utils.paths import FEAR_GREED_PATH, RAW_DIR

# Monthly macro state variables, in the order the interaction block is built.
MACRO_COLUMNS = ("fear_greed", "tbl", "tms", "dfy", "dfr")

# Welch-Goyal "PredictorData" workbook, monthly sheet through 2025-12.
PREDICTOR_PATH = RAW_DIR / "Data2025.xlsx"

# Publication lag in months: bar in month M uses the month-end value of M - LAG_MONTHS.
LAG_MONTHS = 1


def _month_end(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Snap month-start stamps to their month end (idempotent on month-end stamps)."""
    return pd.DatetimeIndex(index) + pd.offsets.MonthEnd(0)


def load_fear_greed_monthly(path=FEAR_GREED_PATH) -> pd.Series:
    """Month-end Fear & Greed value (last daily observation of each month).

    Args:
        path: Daily Fear & Greed CSV (``Date,Fear Greed,Rating``).

    Returns:
        Series of the 0-100 index indexed by month end.
    """
    daily = (pd.read_csv(path, usecols=["Date", "Fear Greed"], parse_dates=["Date"])
             .set_index("Date")["Fear Greed"])
    monthly = daily.resample("ME").last()
    return monthly.rename("fear_greed")


def load_macro_monthly(predictor_path=PREDICTOR_PATH,
                       fear_greed_path=FEAR_GREED_PATH,
                       first: str = "1927-01-01") -> pd.DataFrame:
    """Monthly macro panel of :data:`MACRO_COLUMNS` indexed by month end.

    Args:
        predictor_path: Welch-Goyal workbook with a ``Monthly`` sheet.
        fear_greed_path: Daily Fear & Greed CSV.
        first: Earliest month kept (``dfr`` only exists from 1926-01).

    Returns:
        Frame with one row per month end and the five macro columns, complete rows.

    Raises:
        FileNotFoundError: if the workbook or the Fear & Greed CSV is missing.
    """
    if not predictor_path.exists():
        raise FileNotFoundError(
            f"missing Welch-Goyal workbook: {predictor_path} (see agents/macro_study.md)")
    monthly = pd.read_excel(predictor_path, sheet_name="Monthly",
                            usecols=["yyyymm", "tbl", "tms", "dfy", "dfr"])
    stamps = _month_end(pd.to_datetime(monthly.pop("yyyymm").astype("int64").astype(str),
                                       format="%Y%m"))
    frame = monthly.set_axis(stamps.rename("date"))
    frame["fear_greed"] = load_fear_greed_monthly(fear_greed_path)
    frame = frame[list(MACRO_COLUMNS)].sort_index()
    return frame.loc[first:].dropna()


def macro_state(dates, monthly: pd.DataFrame | None = None,
                lag_months: int = LAG_MONTHS) -> pd.DataFrame:
    """Macro state per bar date, lagged so nothing is used before it is published.

    The month-end value of month ``M`` becomes available the next day; with
    ``lag_months=1`` a bar in month ``M`` therefore gets the month ``M - 1`` value, and
    a bar dated on a month end gets that month's predecessor (never same-day data).
    ``lag_months=0`` allows a bar on the last day of a month to use that month's value.

    Args:
        dates: DatetimeIndex (or date-like sequence) of bar dates.
        monthly: Output of :func:`load_macro_monthly`; loaded on demand when ``None``.
        lag_months: Publication lag in months (>= 0).

    Returns:
        Frame indexed by the unique sorted ``dates`` with ``const`` (all ones) followed
        by :data:`MACRO_COLUMNS`; rows before the first available stamp are NaN.
    """
    if lag_months < 0:
        raise ValueError(f"lag_months must be >= 0, got {lag_months}")
    monthly = load_macro_monthly() if monthly is None else monthly
    available = monthly.copy()
    published = _month_end(available.index)
    if lag_months >= 1:
        # month M's value is known only after M closes -> available the next day,
        # then shifted by the remaining lag months
        published = published + pd.DateOffset(days=1, months=lag_months - 1)
    available.index = published
    right = available.sort_index().rename_axis("available").reset_index()
    left = pd.DataFrame({"date": pd.DatetimeIndex(dates).unique().sort_values()})
    merged = pd.merge_asof(left, right, left_on="date", right_on="available",
                           direction="backward")
    state = merged.drop(columns="available").set_index("date")
    state.insert(0, "const", 1.0)
    return state


def interaction_columns(firm_columns) -> list[str]:
    """Names of the interaction block, in build order (``const`` block first)."""
    return [f"{firm}__x__{macro}"
            for macro in ("const", *MACRO_COLUMNS)
            for firm in firm_columns]


def kron_features(panel: pd.DataFrame, firm_columns, state: pd.DataFrame) -> pd.DataFrame:
    """Kronecker (pairwise product) block of the firm features with the macro state.

    Args:
        panel: Panel indexed by ``(date, ticker)`` holding ``firm_columns``.
        firm_columns: The firm-level feature names to interact.
        state: Output of :func:`macro_state` covering the panel's dates.

    Returns:
        Frame indexed like ``panel`` with ``len(firm_columns) * (1 + len(MACRO_COLUMNS))``
        columns named ``<firm>__x__<macro>``; rows whose macro value is missing are NaN.
    """
    dates = panel.index.get_level_values("date")
    aligned = state.reindex(dates)
    blocks = {}
    for macro in ("const", *MACRO_COLUMNS):
        macro_values = aligned[macro].to_numpy(dtype=float)
        for firm in firm_columns:
            blocks[f"{firm}__x__{macro}"] = (panel[firm].to_numpy(dtype=float)
                                             * macro_values)
    return pd.DataFrame(blocks, index=panel.index)
