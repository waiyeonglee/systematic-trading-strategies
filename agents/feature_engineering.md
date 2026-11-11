# Feature engineering — implementation reference

What the feature pipeline actually does, owned by this project. Staleness-checked
against the reference code and outputs, then re-implemented fresh in this project's
`src/`. Implementation lives in
`src/features.py`, `src/labels.py`, `src/utils/` and the `scripts/build_features.py`
command; the panel is consumed by both the rule path (trend signals) and the
fitted-model path (walk-forward ridge / elastic-net) in later steps.

## 1. Inputs

| Input | Location | Content |
| --- | --- | --- |
| Raw OHLCV (16 tradables + SPY) | `data/raw/weekly/<ticker>.csv` | yfinance auto-adjusted daily bars, resampled to **Friday week-ending bars** (`Open` first / `High` max / `Low` min / `Close` last / `Volume` sum); trailing partial week dropped |
| Fear & Greed index | `data/additional/fear-greed.csv` | daily 0–100 values from 2011-01-03, converted to monthly macro state in `src/macro.py` |
| Benchmark close | SPY column in `data/raw/weekly/` | used for the raw inspection panel's `spy_return_1bar`; excluded from the current study's 19-column firm block |

Universe (`src/utils/config.py`): 9 sector ETFs `CORE_ETFS` (XLB XLE XLF XLI XLK XLP
XLU XLV XLY) + 7 diversifiers `CROSS_ASSET_ETFS` (TLT IEF GLD DBC UUP EFA EEM) =
**16 tradables**; SPY is benchmark only. **Fixed dataset window:**
end inclusive `2026-01-01`, fetch end exclusive `2026-01-02` — data ends
2025-12-26 (weekly bars) / 2025-12-31 (fear-greed), nothing beyond. GLD starts
2004-11, DBC 2006-02, UUP 2007-03; the others start 2004-01.

## 2. Raw feature frame — 20 columns (weekly)

All lookbacks **count bars** (a 14-bar RSI is 14 weekly observations on weekly
data). Built per ticker from `Close` (and `Volume`), except the SPY market-wide
helper column. Names are exactly the panel column names. This is the raw inspection
frame, not the current study's `micro` model input set.

### Momentum returns (past close-to-close, `close.pct_change(n)`)
`return_1bar`, `return_2bar`, `return_3bar`, `return_4bar`, `return_12bar`,
`return_26bar`, `return_52bar`

### Moving-average ratios (talib SMA)
- `sma_ratio_4_26bar` = SMA(4) / SMA(26) − 1
- `sma_ratio_12_52bar` = SMA(12) / SMA(52) − 1

### Volatility & drawdown
- `volatility_12bar`, `volatility_26bar` = talib `STDDEV` of 1-bar returns
  (window 12 / 26) × √52 (annualised; `periods_per_year("weekly")` = 52)
- `drawdown_12bar` = close / MAX(close, 12) − 1

### Momentum / oscillator
- `rsi_14bar` = talib `RSI(14)` / 100
- `macd` = talib `MACD(12,26,9)` line / close; `macd_signal` = signal line / close
- `bollinger_position` = (close − SMA(20)) / (upper − lower) of talib
  `BBANDS(20, 2σ, SMA)`; NaN where the band width is zero
- `volume_change_4bar` = `Volume.pct_change(4)`

### Market-wide helper
- `spy_return_1bar` — SPY `Close.pct_change(1)`, forward-filled onto the bar date

### Trend-following inputs (always on; the rule's signals)
- `trend_sma40_ratio` = close / SMA(40) − 1 (≈ the classic 200-day MA rule)
- `trend_mom52` = `close.pct_change(52)` (trailing 52-bar ≈ 12-month return)

`TREND_SIGNAL_COLUMNS = {"sma40": "trend_sma40_ratio", "mom52": "trend_mom52"}`;
the rule thresholds the configured one at **value > 0 → signal on**
(pred = +1 / 0 / −1).

Current study feature blocks (`scripts/macro_panel.py`):

- `micro`: 19 firm columns = the 17 asset-level raw columns above, excluding
  `spy_return_1bar`, plus the two trend inputs.
- `macro`: 114 columns = the same 19 firm columns crossed with
  `(const + fear_greed + tbl + tms + dfy + dfr)`.

## 3. Labels (forward-return targets — `src/labels.py`)

Timing convention: features at the close of bar `t`, enter at the close of `t+1`,
then hold for `H` weekly bars. The label at row `t` is therefore
**`price[t+H+1] / price[t+1] − 1`**.

`make_labels(closes)` in `src/labels.py` is the single label implementation. It
produces the realized portfolio return and model targets on a `(date, ticker)` index:

| Column | Meaning |
| --- | --- |
| `fwd_ret` | `price[t+1] / price[t] − 1` — the close-to-close return the portfolio earns on bar `t` (mark-to-market / performance) |
| `fwd_ret_1bar` | 1-week holding-period return (t+1 → t+2) |
| `fwd_ret_4bar` | 4-week holding-period return (t+1 → t+5) |
| `fwd_ret_12bar` | 12-week holding-period return (t+1 → t+13) — **default target** |

Horizon is a hyperparameter: `MODEL_TARGETS = {"absolute1": "fwd_ret_1bar",
"absolute4": "fwd_ret_4bar", "absolute12": "fwd_ret_12bar"}`,
`TARGET_COL = "fwd_ret_12bar"` by default, and
the walk-forward embargo equals the holding horizon (`MODEL_EMBARGO = {"absolute1": 1,
"absolute4": 4, "absolute12": 12}`) so no unrealised labels leak into training.
Unrealized target values remain NaN in `make_labels`; consumers drop rows as needed.

## 4. Panel assembly (`build_feature_frame` → `scripts/build_features.py`)

1. `build_feature_frame` computes the 18 raw `FEATURES` columns plus 2 trend inputs
   per ticker; infinities → NaN. One of the 18 raw columns is the market-wide helper
   `spy_return_1bar`.
2. `scripts/build_features.py` loads the raw bars + SPY, builds features,
   calls `make_labels` from `src/labels.py`, selects the requested target, and joins
   it as `label_forward_return_{H}bar`. It writes `features.csv` (unrealized target
   values kept as NaN) and the correlation report, then (unless `--no-macro`) writes
   the 114-column macro-interaction panel `features_macro.csv` in the same run —
   crossing the current 19 firm features with `(const + 5 macro)`.

`data/features/weekly/features.csv` contains one selected label column in addition to
the 20 raw features. The 12-week label requires a close 13 bars after the feature date,
so its trailing 13 bars per ticker remain NaN (208 rows on the frozen weekly panel).
Rebuild outputs after changing label definitions; earlier numeric comparisons to the
reference panel no longer apply to corrected 4-/12-week labels.

**Consequence of the label lookahead:** the modelling panel ends 13 weeks before
the last raw bar (the last 13 feature dates have no 12-bar label), so the effective
final sample date is **2025-09-26**, not 2025-12-26. `features.csv` itself keeps those
trailing bars with NaN labels.

## 5. Consumers (planned for later steps)

- **Rule path**: reads the configured trend feature (`trend_sma40_ratio` default /
  `trend_mom52`) as a 0/1 signal; portfolio built on `pred`, `fwd_ret`,
  `return_1bar` (realized-vol sizing).
- **ML path** (mode `selection`): trains ridge / elastic-net on the feature
  columns against `TARGET_COL` (default `fwd_ret_12bar`); `fwd_ret` sizes and
  marks the forecast-vol book; `return_1bar` feeds realized volatility.
- **Diagnostics**: `feature_label_correlation` reports Pearson correlation of each
  feature vs the configured label across all labelled dates → saved as
  `feature_label_correlation.csv` + heatmap `.png`.

## 6. Reproduction

```bash
python scripts/fetch_data.py          # optional: re-fetch raw weekly CSVs (end exclusive 2026-01-02)
python scripts/build_features.py      # rebuild features.csv + features_macro.csv + correlation report
                                      # (weekly, H=12)
python scripts/build_features.py --horizon 4       # 4-bar label column
python scripts/build_features.py --no-macro        # skip the 114-column macro panel
```

Outputs land in `data/features/<frequency>/`. Settings that change the feature
set: `--frequency`, `--horizon` (1, 4, or 12-bar holding horizon H). The dataset
itself is frozen by the fixed end date 2026-01-01.
