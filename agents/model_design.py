"""DESIGN reference for the trend-following-strategy-v2 project (standalone, latest).

STATUS: this project's copy of the design source of truth (copied from the reference
project's agents/model_design.py after a staleness check — the design constants match
the v2 implementation and the frozen v2 dataset, so it was refreshed only in window
wording). The v2 pipeline (src/models.py, src/splits.py, src/portfolio.py,
src/metrics.py, src/pipeline.py, scripts/run_backtest.py) is implemented from this
file. The validation protocol and its measured record live in
agents/validation_and_ablation.md (Validation 1 = selection on 2017-2019,
Validation 2 = ablation on 2020-2022, Test = final holdout 2023-2025); results are
logged there and under results/. This file is the single design source of truth: what
the pipeline is built from, how the layers split, and which knobs the study turns.

The 2026-11-02 experiment cycle's results are archived in
agents/validation_and_ablation.md, but are **superseded for current assessment**:
the 4-/12-bar target labels were corrected to match their holding periods, and
the feature builder now sources labels only from src/labels.py. Re-run V1 and V2
before drawing conclusions. The 2023–2025 test window was already measured under
the old selection; any corrected rerun on it is a remeasurement, not an untouched
holdout. No headline strategy is promoted; HEADLINE_* stays None. The 2026-11-02
design revision (4 model families
including the RFF ridge family + frequency x look-ahead grid {1,4,12} with
look-ahead >= frequency, + model refit cadence {W-FRI, ME, QE} with
refit_interval >= prediction_interval) superseded the previous selection, so
validation was re-run from scratch by the
expanded experiment before a single final test draw. See agents/current_state.md
(design revision, Step 6, Step 6 follow-up 2 entries) and
agents/validation_and_ablation.md §5/§6.
The Step-4 stack replication (ADR 09, enet_a0.001_r0.5 / absolute12) was validated
byte-identical to the reference and remains in results/ as the reference comparison
record.

Design lineage (this project):
    agents/feature_engineering.md   — implementation reference: inputs, 21-feature
                                      panel, labels, embargo, effective window
    agents/validation_and_ablation.md — validation & ablation protocol (V1 grid ->
                                      lock, V2 rows 1-5, gate rules) + measured record
    agents/current_state.md         — project history and handover notes (Step 1-5)
    results/                        — measured tables per run tag (v2-owned)

Rationale in one line: the sister project measured zero cross-sectional signal on
9 highly-correlated sector ETFs because the median-subtracted label removes the shared
trend; this project harvests that absolute trend instead — with a fitted model
(ridge / elastic-net) on a walk-forward panel as the main implementation, and the
classic MA / momentum rules kept as the documented reference.
"""

from __future__ import annotations

from dataclasses import dataclass

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

# ML path (run_pipeline, mode: selection) — Validation 1 hyperparameters.
# 2026-11-02 design revision: 4 families on PER-FAMILY alpha grids; enet adds the
# l1_ratio dimension; rff (Random Fourier Features -> Ridge) adds the rff_dim x
# gamma dimensions -> 6 + 6 + 12 + 120 = 144 specs, pre-declared before any validation.
# (2026-11-02 smoke finding: a shared {1e-3..100} grid sits above the L1 entry
# threshold on this data — max |corr|*std(target) = 0.00125/0.00315/0.00789 for the
# 1/4/12-week targets — so lasso/enet collapse to intercept-only; the per-family
# grids keep ridge {1e-3..100} and move the L1 families into their active range.)
MODEL_FAMILIES = ("ridge", "lasso", "enet", "rff")     # 4 model families
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
LASSO_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
ENET_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
ENET_L1_RATIOS = (0.5, 0.25)
# RFF ridge family (added 2026-11-02, Step 6 of agents/current_state.md): ridge on
# seeded RandomFourierFeatures approximating a Gaussian-kernel ridge. Alpha follows
# the ridge grid (same L2 range; alpha x rff_dim x gamma = 6 x 3 x 5 = 90 specs).
#   rff_dim (D): {50, 100, 500} — D > p = 21; D = 10 (the first proposal) is a
#     bottleneck projection with ~1/sqrt(D) ~ 32% kernel-approximation error and
#     selection noise from the W draw, so the low anchor was raised to 50.
#   gamma: the RBF INVERSE bandwidth, k(x, y) = exp(-gamma * ||x-y||^2) (NOT the
#     lengthscale; sigma = 1/sqrt(2 gamma)). Features are standardized per fit
#     (p = 21), typical pair distance^2 ~ 2p = 42 -> median-heuristic gamma ~0.01;
#     the grid {1e-3 .. 0.5} spans smooth -> local; gamma >= 1 collapses the kernel
#     to a point mass and aliases the RFF frequencies to noise, so the proposal's
#     {1, 10, 100} tail was dropped.
RFF_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
RFF_DIMS = (50, 100, 500, 1000)
RFF_GAMMAS = (1e-3, 1e-2, 5e-2, 1e-1, 5e-1)
# Grid -> 144 specs: ridge {6 alphas} + lasso {6 alphas} + enet {6 alphas x 2 ratios}
#                    + rff {6 alphas x 4 dims x 5 gammas}
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
# Dataclass defaults = the Validation 2 base book (ablation protocol §4). Validation 1
# runs OVERLAY-FREE (vol_target = None): the overlay must not influence model
# selection — it changes position sizing, not forecasts.
@dataclass(frozen=True)
class Portfolio:
    strategy: str = "forecast_vol_scaled"  # registered in src/portfolio.py STRATEGIES;
                                           # the ML book is the v2 main implementation
    weighting: str = "vol_scaled"       # inverse realized-vol weights
    vol_window: int = 24                # V2 row 1 base (4w / 8w / 12w / 48w variants; 24 = base)
    vol_min_periods: int = 20           # history required before an ETF is tradable
    vol_floor: float = 1e-5             # same floor convention as the sister project
    rebalance: str = "4w"               # default cadence; V1 locks it with the model
                                        # (prediction frequency locked to cadence)
    gross_exposure: float = 1.0         # sum |weights| = 100% when any signal is on
    initial_capital_usd: float = 100_000.0
    broker: str = "moomoo"              # moomoo | ibkr (src/portfolio.py rates)
    cost_bps: float = 0.0               # moomoo commissions only (5 bps placeholder removed)
    # Overlay knobs — V2 rows 3-5. OFF in V1 (overlay-free plain book).
    vol_target: float | None = 0.10     # V2 row 3 base; None = raw book
    vol_target_rf: float = 0.03         # V2 row 4 base; 0.0 = cash idle (only when vol target ON)
    vol_target_cap: float = 1.0         # V2 row 5 base; 2.0 + fee gate = variant
    vol_target_lev_fee: float = 0.0     # moomoo holding fee on borrowed amount; 0.068 with cap 2.0
    # Run order: rows 1 -> 2 -> 3 -> 4 -> 5; leverage (row 5) after the
    # vol window (row 1) because leverage timing depends on the risk estimate.

# ---------------------------------------------------------------------------
# 5. Split & gates (protocol & discipline: agents/validation_and_ablation.md §1-§2)
# ---------------------------------------------------------------------------
WARMUP = ("2004-01-02", "2010-12-31")     # indicator history before training
TRAIN = ("2011-01-01", "2016-12-31")      # rolling 5y (~260 weeks) minus embargo,
                                          # quarterly refit
VALIDATION_1 = ("2017-01-01", "2019-12-31")  # V1: hyperparameter / model selection
VALIDATION_2 = ("2020-01-01", "2022-12-31")  # V2: ablation rows
TEST = ("2023-01-01", "2025-12-31")       # holdout; effective end 2025-09-26
                                          # under absolute12 (≈ 2025-11-21 under absolute4)

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
# 6. Validation 1 — hyperparameter / model selection (2017-2019)
# ---------------------------------------------------------------------------
# Selection rule (validation_and_ablation.md §3, 2026-11-02 revision): **joint
# selection** — 144 specs x 14 valid (frequency, look-ahead, refit) combos = 2016
# candidates, ONE selection on 2017-2019, best validation net Sharpe on the plain
# (overlay-free) base book, degenerate excluded; AR(1) naive baseline reported for
# reference only. Valid combos obey look-ahead >= frequency AND refit_interval >=
# prediction_interval: W-FRI x {weekly pairs} (3), ME x {1w, 4w cadences} (5),
# QE x {all 6} (14 total). Walk-forward mechanics fixed (not tuned): rolling 5y,
# embargo = horizon. The winner — (model, look-ahead, frequency, refit) — is locked
# together and feeds every Validation 2 row. Measured record:
# validation_and_ablation.md §5.1/§5.3/§5.4 — V1 joint selection executed 2026-11-02
# on the 144-spec grid (2016 candidates): locked (enet_a0.001_r0.5, absolute1,
# weekly, ME), val net Sharpe 1.3370; V2 (11 rows, 2020-2022) winner vol_window 12w
# @ 0.6574 (base 24w @ 0.5180); single test draw 2023-2025 FAILED the §3.2 gate
# (null result) → nothing promoted. The earlier 114-grid run (lock 1.3674, V2 base
# 0.6274) and the 684/1596 runs are superseded breadcrumbs (§5.1/§5.3).
V1_CANDIDATES = 2016               # 144 specs x 14 (frequency, look-ahead, refit) combos

# Headline (CLOSED 2026-11-02) — the single test draw was measured and FAILED the
# §3.2 gate (validation_and_ablation.md §5.4), so nothing was promoted and these
# stay None. Any future iteration = new pre-registered cycle; the test window is
# consumed (one draw) and must not be re-measured.
HEADLINE_MODEL = None              # V1-locked spec (set after the expanded V1 run)
HEADLINE_HORIZON = None            # locked look-ahead in bars
HEADLINE_REBALANCE = None          # locked cadence (weekly | 4w | 12w)
HEADLINE_FOLDER = None             # results folder of the single test measurement

# ---------------------------------------------------------------------------
# 7. Validation 2 — ablation, portfolio-layer permutations (2020-2022)
# ---------------------------------------------------------------------------
# Base book for every row (validation_and_ablation.md §4): V1-locked (model, look-
# ahead, frequency), vol-target 10% / cap 1.0, rf 3%, moomoo, no leverage, 24-week
# vol window (re-based 2026-11-02 from 60w, Step 6 follow-up 7). Cadence is NO
# LONGER a row — frequency is locked with the model in V1 (2026-11-02 revision).
# One-variable rows, run in order 1 -> 4:
#   1. Vol estimation window: 24w -> 4w / 8w / 12w / 48w   (24 = base, in-sweep)
#   2. Vol target: 10% -> None / 20%
#   3. Risk-free sleeve: 3% -> 0%                (only when vol target ON)
#   4. Leverage: cap 1.0 -> cap 2.0 / 3.0 + fee gate   (only when vol target ON)
# Excluded by design: off-diagonal (frequency, look-ahead) pairs (rule: look-ahead
# >= frequency), real rf series (only 3% vs 0% tested). Measured record:
# validation_and_ablation.md §5.1/§5.3/§5.4 — closed cycle: lock @ 1.3370, V2
# winner volwin_12w @ 0.6574, test draw FAILED gate (null) → nothing promoted.

# ---------------------------------------------------------------------------
# 8. Scope — deliberately NOT part of the design
# ---------------------------------------------------------------------------
#   - cross-sectional / rank features and median-subtracted labels: the target is
#     absolute (trend), not relative
#   - daily rebalance: weekly / 4w / 12w only
#   - off-diagonal (frequency, look-ahead) pairs where look-ahead < frequency
#   - real rf series (T-bill / SORA): V2 tests rf 3% vs 0% only


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
    print(f"split       : val1 {VALIDATION_1[0]}..{VALIDATION_1[1]} | val2 "
          f"{VALIDATION_2[0]}..{VALIDATION_2[1]} | test {TEST[0]}..{TEST[1]}")
    print(f"v1          : {V1_CANDIDATES} candidates (144 specs x 14 (freq, la, "
          f"refit) combos, look-ahead >= frequency AND refit >= cadence), best val "
          f"net Sharpe on the plain overlay-free book")
    print(f"v2          : 4 one-variable rows (vol window / vol target / rf / "
          f"leverage); cadence locked in V1 (base book per the ablation protocol §4)")
    print(f"gates       : net Sharpe >= weaker benchmark AND maxDD <= better "
          f"benchmark (SPY / EW-16 recomputed on 2023-2025, pre-registered)")
    print(f"record      : agents/current_state.md (design revision 2026-11-02) + "
          f"agents/validation_and_ablation.md §5.1/§5.3/§5.4 (closed cycle) + results/ "
          f"(v2-owned tables)")


if __name__ == "__main__":
    main()
