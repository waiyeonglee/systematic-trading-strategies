"""DESIGN reference for the trend-following-strategy-v2 project (standalone, latest).

STATUS: this project's design source of truth. The v2 pipeline (src/models.py,
src/splits.py, src/prediction_cache.py, src/portfolio.py, src/metrics.py,
src/pipeline.py, scripts/run_study.py, scripts/run_backtest.py) is implemented from
this file. The validation protocol and its measured record live in
agents/validation_and_ablation.md; results are logged there and under results/.

**Current scheme (2026-11-03): ONE joint ranking.** The walk-forward predictions for
every (model spec, look-ahead, refit cadence) are computed once and cached; the study
then replays them through every book configuration and ranks the whole combined space
of **model x (rebalance, look-ahead, refit) x book** on a single validation window
(2017-01-01..2022-12-31):

    93 specs x 14 combos x 45 book cells = 58,590 candidates, best net Sharpe wins.

There is no separate model-selection phase and no separate ablation phase. The cache
lives under results/_study/predictions. Seed `SEED` (§5) is fixed everywhere. The
2023-2025
window is measured once, afterwards, for the ranking winner only.

**Measured record:** the ranking CSV is `results/_study/validation_ranking.csv` and
the winner is `results/_study/validation_winner.json`; the numbers are summarised in
agents/validation_and_ablation.md §5. `HEADLINE_*` stays None until a winner passes
the test-window gate — the ranking itself promotes nothing.

Earlier protocols (2026-11-02 two-window V1/V2 cycles, the 114-/144-grid runs, the
pre-label-correction test draws) were removed on 2026-11-03 with their result
artifacts; only their summary numbers survive in agents/current_state.md Steps 1-16.

Design lineage (this project):
    agents/feature_engineering.md   — implementation reference: inputs, 21-feature
                                      panel, labels, embargo, effective window
    agents/validation_and_ablation.md — validation protocol (combined window, joint
                                      ranking space, cache, gate rules) + measured
                                      record
    agents/current_state.md         — project history and handover notes
    results/                        — measured tables per run tag (v2-owned)

Rationale in one line: the sister project measured zero cross-sectional signal on
9 highly-correlated sector ETFs because the median-subtracted label removes the shared
trend; this project harvests that absolute trend instead — with a fitted model
(ridge / elastic-net) on a walk-forward panel as the main implementation, and the
classic MA / momentum rules kept as the documented reference.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.config import DEFAULT_SEED  # noqa: E402  (the single source of truth)

# ---------------------------------------------------------------------------
# 1. Data (v2 fixed window: no 2026 data)
# ---------------------------------------------------------------------------
DATA_START = "2004-01-01"          # first weekly bar; pre-2011 rows are warmup only
DATA_END = "2026-01-01"            # v2 fixed cut, end INCLUSIVE (config
                                   # END_DATE_INCLUSIVE); the fetch bound is
                                   # exclusive 2026-01-02 (END_DATE_EXCLUSIVE);
                                   # the weekly panel runs to 2025-12-26
FREQUENCY = "weekly"               # Friday-ending bars, resampled from daily

CORE_ETFS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")   # 9 sectors
CROSS_ASSET_ETFS = ("TLT", "IEF", "GLD", "DBC", "UUP", "EFA", "EEM")          # 7 diversifiers
TRADABLES = CORE_ETFS + CROSS_ASSET_ETFS       # 16 tradables
BENCHMARK = "SPY"                  # benchmark #1 (B&H); benchmark #2 = equal-weight B&H
                                   # across all 16 tradables (EW-16)
# Per-ticker start (feature_engineering.md §1): GLD 2004-11, DBC 2006-02,
# UUP 2007-03, the others 2004-01. Fear & Greed (2011+) gates the runnable panel:
# with use_fear_greed: true the NaN warmup rows are dropped, so the panel starts
# 2011-01-07 (verified in the v2 backtest output).

# ---------------------------------------------------------------------------
# 2. Features & labels (implemented in src/features.py, src/labels.py;
#    full reference: agents/feature_engineering.md)
# ---------------------------------------------------------------------------
# Feature panel: 21 features per (date, ticker) with use_fear_greed: true (20
# without). Labels are generated separately by src/labels.py and joined by callers.
# The 12-bar holding target reaches t+13, so the runnable panel ends 2025-09-26.
FEATURE_COUNT = 21                 # 19 base + 2 trend; fear-greed on
TREND_SIGNAL_COLUMNS = {"sma40": "trend_sma40_ratio", "mom52": "trend_mom52"}
SIGNAL_SMA_WINDOW = 40             # weekly bars, ≈ 200-day MA (rule path)
SIGNAL_MOM_LB = 52                 # weekly bars, ≈ 12-month momentum (rule path)
# Rule path (trend book): configured trend feature value > 0 -> signal on
# (pred = +1 / 0 / -1). Kept as the documented reference; the fitted model is the
# main implementation.

# ML path (run_pipeline, mode: selection) — phase-1 hyperparameters.
# 2026-11-03 grid revision (user request): 4 families on reduced, PER-FAMILY alpha
# grids; enet adds the l1_ratio dimension; rff (Random Fourier Features -> Ridge)
# adds the rff_dim x gamma dimensions -> 4 + 5 + 12 + 72 = 93 specs, pre-declared
# before any validation.
# (2026-11-02 smoke finding: a shared {1e-3..100} grid sits above the L1 entry
# threshold on this data — max |corr|*std(target) = 0.00125/0.00315/0.00789 for the
# 1/4/12-week targets — so lasso/enet collapse to intercept-only; the per-family
# grids keep the L1 families in their active range.)
MODEL_FAMILIES = ("ridge", "lasso", "enet", "rff")     # 4 model families
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
LASSO_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1)
ENET_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1)
ENET_L1_RATIOS = (0.5, 0.25, 0.75)
# RFF ridge family (added 2026-11-02): ridge on seeded RandomFourierFeatures
# approximating a Gaussian-kernel ridge. alpha x rff_dim x gamma = 6 x 3 x 4 = 72.
#   rff_dim (D): {100, 200, 500} — D > p = 21; D = 10 (the first proposal) is a
#     bottleneck projection with ~1/sqrt(D) ~ 32% kernel-approximation error and
#     selection noise from the W draw, so the low anchor stays above the feature
#     count.
#   gamma: the RBF INVERSE bandwidth, k(x, y) = exp(-gamma * ||x-y||^2) (NOT the
#     lengthscale; sigma = 1/sqrt(2 gamma)). Features are standardized per fit
#     (p = 21), typical pair distance^2 ~ 2p = 42 -> median-heuristic gamma ~0.01.
#     The grid {1e-2, 1e-1, 1, 10} was set by the user on 2026-11-03; by the a-priori
#     argument gamma >= 1 already collapses the kernel towards a point mass (k(e^-42)
#     ~ 0 at the mean pair distance), so those cells are retained deliberately as a
#     measured check on that claim, not because signal is expected there.
RFF_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
RFF_DIMS = (100, 200, 500)
RFF_GAMMAS = (1e-2, 1e-1, 1.0, 10.0)
# Grid -> 93 specs: ridge {5 alphas} + lasso {4 alphas}
#                   + enet {4 alphas x 3 ratios} + rff {6 alphas x 3 dims x 4 gammas}
MODEL_TARGETS = {"absolute1": "fwd_ret_1bar",
                 "absolute4": "fwd_ret_4bar",
                 "absolute12": "fwd_ret_12bar"}
TARGET_COL = "fwd_ret_12bar"      # default until V1 locks the look-ahead
MODEL_EMBARGO = {"absolute1": 1, "absolute4": 4, "absolute12": 12}   # embargo = target horizon
# Rebalance cadence -> bars, and the valid (frequency, look-ahead) pairs under the
# design rule **look-ahead >= frequency** (2026-11-02). V1 selects over these jointly.
FREQ_TO_BARS = {"daily": 1, "weekly": 1, "4w": 4, "12w": 12}
FREQ_LA_PAIRS = [("weekly", 1), ("weekly", 4), ("weekly", 12),
                 ("4w", 4), ("4w", 12),
                 ("12w", 12)]
# Model refit cadence — V1 selection dimension (2026-11-02, Step 6 follow-up 2):
# refit weekly / monthly / quarterly. Design rule (an economy rule, NOT statistical
# validity like look-ahead >= frequency): refit is **no more frequent than the
# acting cadence** — REFIT_TO_BARS[refit] >= FREQ_TO_BARS[rebalance]. Refits finer
# than the cadence fit models whose intermediate predictions the book never uses;
# refit == cadence captures the freshness. Valid combos: W-FRI x {weekly pairs} (3),
# ME x {1w, 4w cadences} (5), QE x {all 6} -> 14 combos.
REFIT_OPTIONS = ("W-FRI", "ME", "QE")     # weekly / monthly / quarterly
REFIT_TO_BARS = {"W-FRI": 1, "ME": 4, "QE": 12}
# Label timing (feature_engineering.md §3): features at close t, enter at close t+1,
# hold H bars; label = price[t+H+1] / price[t+1] - 1. The 12-week target reaches
# t+13, so its final labelled date is 2025-09-26; benchmark bars run to 2025-12-31.

# ---------------------------------------------------------------------------
# 3. Position rules (per path)
# ---------------------------------------------------------------------------
#   Rule reference (trend book): LONG / FLAT — signal UP -> hold the asset
#     (weight per vol scaling), signal DOWN -> hold cash (weight 0, no financing
#     modelled). That is POSITION_RULE = "long_flat".
#   ML book (run_pipeline, mode: selection): the implemented strategy
#     "forecast_vol_scaled" — LONG + SHORT from the prediction sign, gross ~100%,
#     cost-gated orders (portfolio settings).
POSITION_RULE = "long_flat"        # rule path only; the ML book is long+short

# ---------------------------------------------------------------------------
# 4. Portfolio / execution
# ---------------------------------------------------------------------------
# Dataclass defaults = the reference book (vol target 10%, cap 1.0, rf 3%, 24w
# window). The joint ranking sweeps 45 book cells around it (§7) — there is no
# baseline row and no separate ablation stage. Inside run_pipeline the model ranking
# is scored OVERLAY-FREE (vol_target = None): the overlay changes position sizing,
# not forecasts, so it must not influence which model wins.
@dataclass(frozen=True)
class Portfolio:
    strategy: str = "forecast_vol_scaled"  # registered in src/portfolio.py STRATEGIES;
                                           # the ML book is the v2 main implementation
    weighting: str = "vol_scaled"       # inverse realized-vol weights
    vol_window: int = 24                # reference book; ranked over 4/8/12/24/48w
    vol_min_periods: int = 20           # history required before an ETF is tradable
    vol_floor: float = 1e-5             # same floor convention as the sister project
    rebalance: str = "4w"               # reference cadence; ranked with the model
                                        # (prediction frequency locked to cadence)
    gross_exposure: float = 1.0         # sum |weights| = 100% when any signal is on
    initial_capital_usd: float = 100_000.0
    broker: str = "moomoo"              # moomoo | ibkr (src/portfolio.py rates)
    cost_bps: float = 0.0               # moomoo commissions only (5 bps placeholder removed)
    # Overlay knobs — ranked as part of the 45-cell book space (§7); overlay-free
    # inside run_pipeline's own model ranking.
    vol_target: float | None = 0.10     # reference target; ranked 20% / 10% / off
    vol_target_rf: float = 0.03         # cash sleeve, fixed at 3% in every cell
    vol_target_cap: float = 1.0         # reference cap; ranked 1.0 / 2.0 / 3.0
    vol_target_lev_fee: float = 0.0     # moomoo holding fee on borrowed amount; 0.068 with caps 2.0/3.0
    # Run order: rows 1 -> 2 -> 3 -> 4 -> 5; leverage (row 5) after the
    # vol window (row 1) because leverage timing depends on the risk estimate.

# ---------------------------------------------------------------------------
# 5. Split & gates (protocol & discipline: agents/validation_and_ablation.md §1-§2)
# ---------------------------------------------------------------------------
WARMUP = ("2004-01-02", "2010-12-31")     # indicator history before training
TRAIN = ("2011-01-01", "2016-12-31")      # rolling 5y (~260 weeks) minus embargo
# ONE combined 6-year validation window: the joint model x book ranking runs on it,
# from the cached walk-forward predictions (results/_study/predictions).
VALIDATION = ("2017-01-01", "2022-12-31")
TEST = ("2023-01-01", "2025-12-31")       # holdout; effective end 2025-09-26
                                          # under absolute12 (≈ 2025-11-21 under absolute4)
SEED = DEFAULT_SEED                       # loaded from utils.config.DEFAULT_SEED —
                                          # never typed in the design reference; every
                                          # estimator and every cache entry uses it

# Walk-forward prediction cache (committed, npz): one matrix + JSON provenance per
# (spec, look-ahead, refit) on the validation window, plus the shared volatility
# warmup. Predictions do not depend on the rebalance cadence or on book knobs, so
# weekly / 4w / 12w candidates reuse one file.
PREDICTIONS_CACHE = "results/_study/predictions"
VOL_WARMUP_BARS = 48                      # covers every vol_window cell (4..48)

BENCHMARK_1 = "SPY"                # buy & hold
BENCHMARK_2 = "equal_weight"       # equal-weight B&H across all 16 tradables (EW-16)

# Success criterion (pre-registered protocol, validation_and_ablation.md §1.2): on the
# final test window, strategy net Sharpe >= the weaker benchmark AND maxDD <= the
# better benchmark. Gate bars (SPY / EW-16 on 2023-01-01..2025-12-31) are recomputed
# and pre-registered BEFORE any headline run. Measured once per promoted configuration;
# no tuning on the test window. Implemented as pipeline.trend_success_verdict.
SUCCESS_SHARPE_GE_BENCHMARK = True
SUCCESS_MAXDD_LE_BENCHMARK = True

# ---------------------------------------------------------------------------
# 6. Validation — ONE joint ranking over the combined model x strategy space
# ---------------------------------------------------------------------------
# There is no separate model-selection / ablation split (2026-11-03). The
# walk-forward predictions for every (spec, look-ahead, refit) are cached as CSVs;
# the study replays them through every book configuration and ranks the whole space
# by net Sharpe on the combined 2017-2022 window (validation_and_ablation.md §3):
#   93 specs x 14 valid (rebalance, look-ahead, refit) combos x 45 book cells
#   = 58,590 candidates, ONE ranking, degenerate cells excluded from the win.
# One selection criterion over that scored space (§3.1 of the protocol doc): the
# highest net Sharpe wins.
# Valid combos obey look-ahead >= rebalance cadence AND refit_interval >= rebalance
# cadence: W-FRI x {weekly pairs} (3), ME x {1w, 4w cadences} (5), QE x {all 6} (14).
# Walk-forward mechanics fixed (not tuned): rolling 5y, embargo = horizon, seed =
# SEED (§5, mirror of utils.config.DEFAULT_SEED).
# The winner is a candidate, not a headline; it still needs the pre-registered ADR
# and the single test draw.
# Cost: the book simulation is a numpy state machine (17-19 ms per book), and five
# of the forty-five cells are duplicates (inert lever cap with the overlay off), so a
# candidate runs 40 simulations; the whole ranking is ~5 min on 8 workers, all model
# fits cached.
RANK_CANDIDATES = 58_590           # 93 specs x 14 combos x 45 book cells
BOOK_SIMULATIONS = 40              # distinct books per candidate (see §7)

# Headline (None until a ranking winner passes the §1.2 gate on the test window).
HEADLINE_MODEL = None              # winning spec
HEADLINE_HORIZON = None            # winning look-ahead in bars
HEADLINE_REBALANCE = None          # winning cadence (weekly | 4w | 12w)
HEADLINE_FOLDER = None             # results folder of the single test measurement

# ---------------------------------------------------------------------------
# 7. Book cells inside the ranking
# ---------------------------------------------------------------------------
# The full product of three strategy knobs (no shared base row), 5 x 3 x 3:
#   1. Forecast-vol window:  4w / 8w / 12w / 24w / 48w
#   2. Vol target:           20% / 10% / off   (off = no overlay, gross stays 1.0)
#   3. Leverage cap:         1.0 / 2.0 / 3.0   (2.0 and 3.0 pay a 6.8% ann. borrow
#                                              fee on the levered fraction only)
# The risk-free rate is fixed at 3% ann. for the cash sleeve in every cell (the 0%
# arm was dropped: with a 3% cash return realisable in the sample, a 0% sleeve is a
# strictly worse book for the same signal, so it cannot inform the choice).
# The cap can only bind when the overlay is live (scale = min(cap, target/vol)), so
# the three "off x cap 2.0/3.0" cells reproduce the cap-1.0 book apart from the
# borrow fee charged on drift leverage: 45 cells, 40 distinct simulations.
# Every cell of a candidate shares one 48-bar volatility warmup and the candidate's
# cached predictions. Excluded by design: off-diagonal (cadence, look-ahead) pairs
# (rule: look-ahead >= cadence), real rf series (only the flat 3% rate).


# ---------------------------------------------------------------------------
# 8. Scope — deliberately NOT part of the design
# ---------------------------------------------------------------------------
#   - cross-sectional / rank features and median-subtracted labels: the target is
#     absolute (trend), not relative
#   - daily rebalance: weekly / 4w / 12w only
#   - off-diagonal (frequency, look-ahead) pairs where look-ahead < frequency
#   - real rf series (T-bill / SORA): only rf 3% vs 0% tested


def main() -> None:
    """Print the DESIGN summary."""
    print("=" * 74)
    print("trend-following-strategy-v2 — DESIGN reference (standalone, latest)")
    print("=" * 74)
    print(f"status      : design source of truth (copied & refreshed from the reference "
          f"project); first measured run validated byte-identical (ADR 09 stack, "
          f"results/weekly_fg_roll5y_val3y_qrefit_grid6_vt0.1c1rf3_2023-2025)")
    print(f"data        : {DATA_START}..{DATA_END} (end inclusive; fetch exclusive "
          f"2026-01-02), {FREQUENCY} bars, {len(TRADABLES)} tradables "
          f"({len(CORE_ETFS)} sectors + {len(CROSS_ASSET_ETFS)} cross-asset) + {BENCHMARK}")
    print(f"features    : {FEATURE_COUNT} columns (feature_engineering.md); labels "
          f"fwd_ret_1bar / fwd_ret_4bar / fwd_ret_12bar (embargo = horizon)")
    print(f"position    : rule long_flat | ML long+short ({Portfolio.weighting}, "
          f"rebalance {Portfolio.rebalance}, gross {Portfolio.gross_exposure})")
    print(f"portfolio   : vol_window {Portfolio.vol_window}w | vol_target "
          f"{Portfolio.vol_target} | rf {Portfolio.vol_target_rf} | cap "
          f"{Portfolio.vol_target_cap} | cost_bps {Portfolio.cost_bps} (moomoo)")
    print(f"split       : train {TRAIN[0]}..{TRAIN[1]} | validation (combined) "
          f"{VALIDATION[0]}..{VALIDATION[1]} | test {TEST[0]}..{TEST[1]}")
    print(f"phase       : ONE joint ranking — {RANK_CANDIDATES} candidates (93 specs x "
          f"14 (rebalance, look-ahead, refit) combos x 45 book cells), best validation "
          f"net Sharpe over the combined space")
    print(f"determinism : seed {SEED}; predictions cached under {PREDICTIONS_CACHE} "
          f"(vol warmup {VOL_WARMUP_BARS} bars shared by every row)")
    print(f"gates       : net Sharpe >= weaker benchmark AND maxDD <= better "
          f"benchmark (SPY / EW-16 recomputed on 2023-2025, pre-registered)")
    print(f"record      : agents/validation_and_ablation.md §5 (measured ranking) + "
          f"agents/current_state.md + results/_study/ (ranking CSV, winner JSON)")


if __name__ == "__main__":
    main()
