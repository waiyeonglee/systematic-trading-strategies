# Feature engineering — v2-owned implementation reference

What the v2 feature pipeline actually does, owned by this project
(`trend-following-strategy-v2`). Copied from the reference project
(`../trend-following-strategy/agents/feature_engineering.md`) and refreshed for v2
conventions; staleness-checked against both reference code and outputs, then
re-implemented fresh in this project's `src/`. Implementation lives in
`src/features.py`, `src/labels.py`, `src/utils/` and the `scripts/build_features.py`
command; the panel is consumed by both the rule path (trend signals) and the
fitted-model path (walk-forward ridge / elastic-net) in later steps.

## 1. Inputs

| Input | Location | Content |
| --- | --- | --- |
| Raw OHLCV (16 tradables + SPY) | `data/raw/weekly/<ticker>.csv` | yfinance auto-adjusted daily bars, resampled to **Friday week-ending bars** (`Open` first / `High` max / `Low` min / `Close` last / `Volume` sum); trailing partial week dropped |
| Fear & Greed index | `data/additional/fear-greed.csv` | daily 0–100 values from 2011-01-03, forward-filled onto each ticker's bar dates (optional: `--no-fear-greed`) |
| Benchmark close | SPY column in `data/raw/weekly/` | used only for the `spy_return_1bar` market-return feature |

Universe (`src/utils/config.py`): 9 sector ETFs `CORE_ETFS` (XLB XLE XLF XLI XLK XLP
XLU XLV XLY) + 7 diversifiers `CROSS_ASSET_ETFS` (TLT IEF GLD DBC UUP EFA EEM) =
**16 tradables**; SPY is benchmark only. **Fixed dataset window (v2 convention):**
end inclusive `2026-01-01`, fetch end exclusive `2026-01-02` — data ends
2025-12-26 (weekly bars) / 2025-12-31 (fear-greed), nothing beyond. GLD starts
2004-11, DBC 2006-02, UUP 2007-03; the others start 2004-01.

## 2. Feature set — 21 columns (weekly, fear-greed on)

All lookbacks **count bars** (a 14-bar RSI is 14 weekly observations on weekly
data). Built per ticker from `Close` (and `Volume`), except the two market-wide
columns. Names are exactly the panel column names.

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

### Market-wide / alternative
- `fear_greed_index` — Fear & Greed value forward-filled onto the bar date
- `spy_return_1bar` — SPY `Close.pct_change(1)`, forward-filled onto the bar date

### Trend-following inputs (always on; the rule's signals)
- `trend_sma40_ratio` = close / SMA(40) − 1 (≈ the classic 200-day MA rule)
- `trend_mom52` = `close.pct_change(52)` (trailing 52-bar ≈ 12-month return)

`TREND_SIGNAL_COLUMNS = {"sma40": "trend_sma40_ratio", "mom52": "trend_mom52"}`;
the rule thresholds the configured one at **value > 0 → signal on**
(pred = +1 / 0 / −1). With fear-greed off the panel has 20 features (the
`fear_greed_index` column omitted); the two trend features are never removed.

## 3. Labels (forward-return targets — `src/labels.py`)

Timing convention: features at the close of bar `t`, trade at the close of `t+1`,
realised `H` bars later. The label at row `t` is therefore
**`price[t+H] / price[t+1] − 1`**.

`make_labels(closes)` produces three columns on a `(date, ticker)` index, from
`prices.shift(-H) / prices.shift(-1) − 1`:

| Column | Meaning |
| --- | --- |
| `fwd_ret` | `price[t+1] / price[t] − 1` — the close-to-close return the portfolio earns on bar `t` (mark-to-market / performance) |
| `fwd_ret_4bar` | 4-week forward return (t+1 → t+4) |
| `fwd_ret_12bar` | 12-week forward return (t+1 → t+12) — **default target** |

Horizon is a hyperparameter: `MODEL_TARGETS = {"absolute4": "fwd_ret_4bar",
"absolute12": "fwd_ret_12bar"}`, `TARGET_COL = "fwd_ret_12bar"` by default, and
the walk-forward embargo equals the horizon (`MODEL_EMBARGO = {"absolute4": 4,
"absolute12": 12}`) so no unrealised labels leak into training. Rows whose label
is NaN (last `H` bars per ticker) are dropped by the modelling panel.

## 4. Panel assembly (`build_feature_frame` → `scripts/build_features.py`)

1. `build_feature_frame` computes the 19 base + 2 trend features per ticker and
   appends the label column `label_forward_return_{H}bar` (H = `--horizon`,
   default 12); infinities → NaN.
2. `scripts/build_features.py` loads the raw weekly CSVs + fear-greed + SPY,
   builds the frame for the 16 tradables, writes `features.csv` (NaN labels kept)
   and the correlation report, and prints a summary.

**Verified result (v2, 2026-11-01):** `data/features/weekly/features.csv` =
**18,047 rows × 24 columns** (`date`, `ticker`, 21 features,
`label_forward_return_12bar`), dates 2004-01-02..2025-12-26, 16 tickers, 192
trailing rows with NaN label. **Numeric output is byte-identical to the reference
project's panel** (max abs diff 0.0 over all 22 numeric columns), confirming the
re-implementation matches the validated reference on the frozen dataset.

**Consequence of the label lookahead:** the modelling panel ends 12 weeks before
the last raw bar (last 12 weeks have no 12-bar label), so the effective final
sample date is **2025-10-03**, not 2025-12-26. `features.csv` itself keeps those
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
python scripts/build_features.py      # rebuild features.csv + correlation report (weekly, fear-greed on, H=12)
python scripts/build_features.py --no-fear-greed   # 20 features
python scripts/build_features.py --horizon 4       # 4-bar label column
```

Outputs land in `data/features/<frequency>/`. Settings that change the feature
set: `--frequency`, `--no-fear-greed`, `--horizon` (label horizon H). The dataset
itself is frozen by the fixed end date 2026-01-01.
