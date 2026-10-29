"""
train_regression.py — DSA5205 Project 2: REGRESSION model training, walk-forward prediction,
portfolio construction.  (Classification version: train.py)

Purpose
-------
Fits penalised linear regressions (Ridge / LASSO / Elastic Net) that predict each asset's
next-period EXCESS return (its return minus that day's cross-sectional median), selects the
best one on the validation period by net Sharpe, and runs it once on the test period.

How a return forecast becomes a trade
-------------------------------------
Each day the model outputs a predicted excess return per asset, e.g.
    XLK +0.08%, XLE +0.05%, XLV +0.01%, ..., XLU -0.06%
The strategy ranks assets by this number and holds the top k at equal weight. The gate
only swaps a holding when the newcomer's predicted return beats the weakest holding's by
more than `gate` (in return units), so a swap only happens when the expected gain plausibly
covers the trading cost (e.g. gate = 2 * cost_bps / 1e4 = one round trip).

=====================================================================================
WHERE THE OPTIONS ARE  (search the code for ">>> OPTION")
=====================================================================================
  OPTION 1  PipelineConfig.gate          min predicted-return gap to swap (0.001 = 0.1%)
  OPTION 2  PipelineConfig.cost_bps      one-way transaction cost in bps
  OPTION 3  PipelineConfig.top_k         number of assets held
  OPTION 4  PipelineConfig.refit_every   "ME" / "QE" / "YE" refit frequency
  OPTION 5  PipelineConfig.window        None = expanding window, int = rolling window
  OPTION 6  PipelineConfig.freq          "daily" (252/yr) | "weekly" (52/yr)
  OPTION 7  default_grid()               hyperparameter grid (alpha values, l1 ratios)
  OPTION 8  TARGET_COL                   "excess_ret" (default) or "fwd_ret" (raw return)
=====================================================================================

Standard data format ("panel") used by every function here
----------------------------------------------------------
pandas DataFrame, MultiIndex (date, ticker), one row per asset per date:

    index level "date"   : pd.Timestamp — decision date t (close of t)
    index level "ticker" : str          — e.g. "XLK"
    <feature columns>    : float        — known at the close of t (no future data)
    "fwd_ret"            : float        — asset's simple return t -> t+1 (e.g. 0.0042)
    "excess_ret"         : float        — fwd_ret minus the median fwd_ret across tickers at t

    Example:
                            mom_20   vol_20   fwd_ret  excess_ret
        date       ticker
        2015-01-02 XLB      0.0123   0.0110    0.0051      0.0032
                   XLE     -0.0310   0.0185   -0.0120     -0.0139

Timing convention: weights chosen at the close of t earn fwd_ret at t (trade at the same close).

Quick test
----------
    python run_example_regression.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 5205
PERIODS_PER_YEAR = {"daily": 252, "weekly": 52}

# >>> OPTION 8: label the model learns.
#   "excess_ret" — return minus the daily median: removes the market-wide move so the model
#                  focuses on WHICH asset does better (matches the ranking strategy).
#   "fwd_ret"    — raw next-period return.
TARGET_COL = "excess_ret"


# --------------------------------------------------------------------------------------
# 0. Labels (optional helper)
# --------------------------------------------------------------------------------------
def make_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the label columns ("fwd_ret", "excess_ret") from a table of prices.

    For each date t and ticker:
        fwd_ret    = price[t+1] / price[t] - 1
        excess_ret = fwd_ret - median fwd_ret across tickers on date t
    The last date has no t+1 price, so it is dropped.

    Input
    -----
    prices : pd.DataFrame, wide format
        index   = DatetimeIndex of trading dates (sorted ascending)
        columns = tickers, values = adjusted close prices (float)

    Output
    ------
    pd.DataFrame, MultiIndex (date, ticker), columns ["fwd_ret", "excess_ret"]

    Test
    ----
        >>> p = pd.DataFrame({"A": [100, 101, 102], "B": [100, 99, 103]},
        ...                  index=pd.bdate_range("2020-01-01", periods=3))
        >>> make_labels(p)
        # 2020-01-01: A fwd_ret 0.01, excess_ret 0.01;  B fwd_ret -0.01, excess_ret -0.01
    """
    fwd_ret = prices.pct_change().shift(-1)
    excess = fwd_ret.sub(fwd_ret.median(axis=1), axis=0)
    out = pd.concat({"fwd_ret": fwd_ret.stack(future_stack=True),
                     "excess_ret": excess.stack(future_stack=True)}, axis=1)
    out.index.names = ["date", "ticker"]
    return out.dropna()


# --------------------------------------------------------------------------------------
# 1. Model definitions + pre-declared grid
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """One point in the hyperparameter grid.

    Fields
    ------
    name     : str   — unique label "<family>_a<alpha>[_r<l1_ratio>]", e.g. "enet_a0.0001_r0.5".
                       The part before the first "_" must be "ridge", "lasso" or "enet".
    penalty  : str   — "l2" (Ridge), "l1" (LASSO) or "elasticnet".
    alpha    : float — regularisation strength; LARGER alpha = more shrinkage.
    l1_ratio : float | None — only for "elasticnet": 0 = pure Ridge, 1 = pure LASSO.

    Test
    ----
        >>> model = ModelSpec("ridge_a10", "l2", 10).build()     # unfitted Pipeline
    """
    name: str
    penalty: str
    alpha: float
    l1_ratio: float | None = None

    def build(self) -> Pipeline:
        """Create a fresh, unfitted model for this spec.

        The StandardScaler is inside the pipeline, so .fit() on training rows learns the
        mean/std from those rows only (no leakage into val/test).

        Input  : none (uses the spec's fields)
        Output : sklearn Pipeline ["scaler" -> StandardScaler, "reg" -> Ridge | Lasso | ElasticNet]
                 .fit(X, y): X np.ndarray (n_rows, n_features), y float array (n_rows,)
                 .predict(X) gives predicted returns (float array)

        Test
        ----
            >>> X = np.random.randn(200, 3); y = 0.01 * X[:, 0]
            >>> m = ModelSpec("ridge_a1", "l2", 1.0).build().fit(X, y)
            >>> m.predict(X[:5])        # close to y[:5]
        """
        if self.penalty == "l2":
            reg = Ridge(alpha=self.alpha, random_state=SEED)
        elif self.penalty == "l1":
            reg = Lasso(alpha=self.alpha, max_iter=10000, random_state=SEED)
        else:
            reg = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                             max_iter=10000, random_state=SEED)
        return Pipeline([("scaler", StandardScaler()), ("reg", reg)])


def default_grid() -> list[ModelSpec]:
    """Return the fixed hyperparameter grid, declared before looking at validation results.

    >>> OPTION 7: edit the alpha lists. Keep the grid small and report its size (§4.7).

    Daily returns are tiny (~0.01), so LASSO / ENet need tiny alphas or every coefficient
    becomes zero (such models are flagged "degenerate" and skipped in select_model).
        Ridge alpha        in {1, 10, 100, 1000}                      4 models
        LASSO alpha        in {1e-6, 1e-5, 1e-4, 1e-3}                4 models
        ENet  alpha (same) x l1_ratio {0.25, 0.5, 0.75}              12 models
                                                              total  20 models

    Input  : none
    Output : list[ModelSpec], length 20

    Test
    ----
        >>> len(default_grid()), default_grid()[0].name     # 20, "ridge_a1"
    """
    ridge_alphas = [1, 10, 100, 1000]
    sparse_alphas = [1e-6, 1e-5, 1e-4, 1e-3]
    ratios = (0.25, 0.5, 0.75)
    grid = [ModelSpec(f"ridge_a{a}", "l2", a) for a in ridge_alphas]
    grid += [ModelSpec(f"lasso_a{a}", "l1", a) for a in sparse_alphas]
    grid += [ModelSpec(f"enet_a{a}_r{r}", "elasticnet", a, r) for a in sparse_alphas for r in ratios]
    return grid


def model_family(spec_name: str) -> str:
    """Get the model family from a spec name.

    Input  : spec_name — str, e.g. "enet_a0.0001_r0.5"
    Output : str — "ridge", "lasso" or "enet"

    Test   : >>> model_family("lasso_a1e-05")       # "lasso"
    """
    return spec_name.split("_")[0]


# --------------------------------------------------------------------------------------
# 2. Walk-forward prediction
# --------------------------------------------------------------------------------------
def walk_forward_predict(
    history: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    spec: ModelSpec,
    refit_every: str = "QE",
    window: int | None = None,
    embargo: int = 1,
) -> pd.DataFrame:
    """Predict eval_df out-of-sample by refitting the model block by block.

    eval_df is cut into blocks (e.g. calendar quarters). For each block:
      1. training rows = rows dated BEFORE the block start (history + earlier eval rows),
         minus the last `embargo` dates (their labels use prices from inside the block);
         if `window` is set, keep only the most recent `window` dates;
      2. fit a fresh model from `spec` on TARGET_COL;
      3. predict the return for every row in the block.

    Inputs
    ------
    history      : panel — data before eval_df (e.g. train set)
    eval_df      : panel — the period to predict (e.g. val or test), dated after history
    feature_cols : list[str] — model inputs, e.g. ["mom_20", "vol_20"]
    spec         : ModelSpec — which model to fit
    refit_every  : str — block length: "ME" month, "QE" quarter, "YE" year
    window       : int | None — rolling training window in dates; None = all past data
    embargo      : int — most recent dates dropped before each refit (1 for next-day labels)

    Output
    ------
    pd.DataFrame — eval_df's rows and columns plus "pred" (float, predicted return,
                   e.g. 0.0008 = +0.08%), MultiIndex (date, ticker), sorted.

    Test
    ----
        >>> pr = walk_forward_predict(train, val, feats, ModelSpec("ridge_a10", "l2", 10))
        >>> len(pr) == len(val), pr["pred"].abs().max() < 0.1     # True, True
    """
    full = pd.concat([history, eval_df]).sort_index()
    all_dates = full.index.get_level_values("date").unique().sort_values()
    eval_dates = eval_df.index.get_level_values("date").unique().sort_values()

    blocks = pd.Series(eval_dates, index=eval_dates).groupby(pd.Grouper(freq=refit_every))
    preds = []
    for _, block_dates in blocks:
        if block_dates.empty:
            continue
        start = block_dates.iloc[0]
        past = all_dates[all_dates < start]
        past = past[:-embargo] if embargo else past
        if window is not None:
            past = past[-window:]
        train_rows = full.loc[full.index.get_level_values("date").isin(past)]

        model = spec.build()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # convergence warnings at extreme alphas
            model.fit(train_rows[feature_cols].values, train_rows[TARGET_COL].values)

        block = eval_df.loc[eval_df.index.get_level_values("date").isin(block_dates)].copy()
        block["pred"] = model.predict(block[feature_cols].values)
        preds.append(block)

    return pd.concat(preds).sort_index()


# --------------------------------------------------------------------------------------
# 3. Predicted returns -> portfolio (top-k, equal weight, with cost-aware action gate)
# --------------------------------------------------------------------------------------
def build_portfolio(
    preds: pd.DataFrame,
    top_k: int = 5,
    gate: float = 0.0,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Turn predicted returns into an equal-weight top-k portfolio with costs.

    Rule on each date t:
      1. Fill any empty slots with the highest-predicted-return names not already held.
      2. Action gate: while (best outsider pred) - (weakest holding pred) > gate, swap them.
         gate is in RETURN units: 0.001 means "only swap if the newcomer is expected to earn
         0.1% more". A natural choice is one round trip of costs: 2 * cost_bps / 1e4.
      3. Weight = 1/k for each holding, 0 otherwise.
      4. turnover = sum |w_t - w_{t-1}|; cost = turnover * cost_bps / 10,000.
      5. gross_ret = sum(w_t * fwd_ret_t); net_ret = gross_ret - cost.
    First date counts as full entry (turnover 1). Intra-period weight drift is ignored.

    Inputs
    ------
    preds    : panel with "pred" and "fwd_ret" (output of walk_forward_predict)
    top_k    : int — number of assets held (<= number of tickers)
    gate     : float — minimum predicted-return gap to swap, e.g. 0.001
    cost_bps : float — one-way cost per unit of turnover in bps (5 = 0.05%)

    Output
    ------
    pd.DataFrame indexed by date, columns:
        gross_ret (float), turnover (float, 0-2), cost (float), net_ret (float),
        n_holdings (int), holdings (str, e.g. "XLE,XLK,XLV")

    Test
    ----
        >>> port = build_portfolio(preds, top_k=5, gate=0.001)
        >>> (port["n_holdings"] == 5).all()                                  # True
        >>> np.allclose(port["net_ret"], port["gross_ret"] - port["cost"])   # True
    """
    pred = preds["pred"].unstack("ticker")
    rets = preds["fwd_ret"].unstack("ticker")

    holdings: set[str] = set()
    prev_w = pd.Series(0.0, index=pred.columns)
    rows = []
    for date, p in pred.iterrows():
        p = p.dropna()
        holdings &= set(p.index)
        if len(holdings) < top_k:
            for t in p.drop(list(holdings)).sort_values(ascending=False).index:
                if len(holdings) >= top_k:
                    break
                holdings.add(t)
        while True:
            outsiders = p.drop(list(holdings))
            if outsiders.empty:
                break
            best_out, weakest_in = outsiders.idxmax(), p[list(holdings)].idxmin()
            if p[best_out] - p[weakest_in] > gate:
                holdings.remove(weakest_in)
                holdings.add(best_out)
            else:
                break

        w = pd.Series(0.0, index=pred.columns)
        w[list(holdings)] = 1.0 / len(holdings)
        turnover = (w - prev_w).abs().sum()
        gross = float((w * rets.loc[date].fillna(0.0)).sum())
        cost = turnover * cost_bps / 1e4
        rows.append(dict(date=date, gross_ret=gross, turnover=turnover, cost=cost,
                         net_ret=gross - cost, n_holdings=len(holdings),
                         holdings=",".join(sorted(holdings))))
        prev_w = w

    return pd.DataFrame(rows).set_index("date")


# --------------------------------------------------------------------------------------
# 4. Selection on validation, final run on test
# --------------------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """All settings for one pipeline run (record these in the report).

    Fields (each is an >>> OPTION, see module docstring)
    ------
    freq        : str   — "daily" or "weekly"; sets periods per year (252 / 52)
    top_k       : int   — number of assets held
    gate        : float — predicted-return gap needed to swap (return units, 0.001 = 0.1%)
    cost_bps    : float — one-way transaction cost in bps
    refit_every : str   — "ME" / "QE" / "YE"
    window      : int | None — rolling training window in dates; None = expanding
    grid        : list[ModelSpec] — models to compare (default_grid() by default)

    Test
    ----
        >>> cfg = PipelineConfig(gate=0.001, grid=default_grid()[:2])
    """
    freq: str = "daily"                 # >>> OPTION 6
    top_k: int = 5                      # >>> OPTION 3
    gate: float = 0.0                   # >>> OPTION 1
    cost_bps: float = 5.0               # >>> OPTION 2
    refit_every: str = "QE"             # >>> OPTION 4
    window: int | None = None           # >>> OPTION 5
    grid: list[ModelSpec] = field(default_factory=default_grid)   # >>> OPTION 7


def _sharpe(r: pd.Series, ppy: int) -> float:
    """Annualised Sharpe ratio with risk-free rate 0.

    Input  : r — pd.Series of periodic returns; ppy — periods per year (252 or 52)
    Output : float — sqrt(ppy) * mean(r) / std(r); NaN if std is 0
    """
    return float(np.sqrt(ppy) * r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else np.nan


def select_model(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str],
                 cfg: PipelineConfig):
    """Run every model in cfg.grid walk-forward through val and rank them by net Sharpe.

    Models whose predictions are identical across tickers on every date (all coefficients
    shrunk to zero) are flagged as degenerate and excluded, because their picks are arbitrary.

    Inputs
    ------
    train, val   : panels (val dated after train)
    feature_cols : list[str]
    cfg          : PipelineConfig

    Outputs (tuple of 4)
    --------------------
    best      : ModelSpec with the highest validation net Sharpe
    table     : pd.DataFrame, one row per model, best first, columns:
                model, family, degenerate (bool), net_sharpe (NaN if degenerate),
                gross_sharpe, avg_turnover
    val_preds : dict[str, pd.DataFrame] — model name -> walk_forward_predict output
    val_ports : dict[str, pd.DataFrame] — model name -> build_portfolio output

    Test
    ----
        >>> best, table, vp, vport = select_model(train, val, feats,
        ...                                       PipelineConfig(grid=default_grid()[:3]))
        >>> table.loc[0, "model"] == best.name     # True
    """
    ppy = PERIODS_PER_YEAR[cfg.freq]
    rows, val_preds, val_ports = [], {}, {}
    for spec in cfg.grid:
        pr = walk_forward_predict(train, val, feature_cols, spec, cfg.refit_every, cfg.window)
        pf = build_portfolio(pr, cfg.top_k, cfg.gate, cfg.cost_bps)
        val_preds[spec.name], val_ports[spec.name] = pr, pf
        degenerate = pr.groupby(level="date")["pred"].std().mean() < 1e-12
        rows.append(dict(model=spec.name, family=model_family(spec.name),
                         degenerate=degenerate,
                         net_sharpe=np.nan if degenerate else _sharpe(pf["net_ret"], ppy),
                         gross_sharpe=_sharpe(pf["gross_ret"], ppy),
                         avg_turnover=pf["turnover"].mean()))
        print(f"  [val] {spec.name:<24} net Sharpe = {rows[-1]['net_sharpe']:.3f}")
    table = (pd.DataFrame(rows).sort_values("net_sharpe", ascending=False, na_position="last")
             .reset_index(drop=True))
    assert table["net_sharpe"].notna().any(), "Every model is degenerate: lower alpha"
    best = next(s for s in cfg.grid if s.name == table.loc[0, "model"])
    return best, table, val_preds, val_ports


def run_pipeline(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                 feature_cols: list[str], cfg: PipelineConfig | None = None) -> dict:
    """Main entry point: select a model on val, then test it once on test.

    Steps
      1. Check inputs (required columns present; train < val < test in time).
      2. select_model: walk forward from train through val for every grid point.
      3. Walk forward from train+val through test with the chosen model only.
      4. Fit the chosen model on all of train+val to report its coefficients.

    Inputs
    ------
    train, val, test : panels with feature_cols, TARGET_COL and "fwd_ret", in time order
    feature_cols     : list[str] — e.g. ["mom_20", "vol_20", "rsi_14"]
    cfg              : PipelineConfig | None — None uses defaults

    Output
    ------
    dict with keys:
        "config"         : PipelineConfig used
        "best_spec"      : ModelSpec chosen on validation
        "val_table"      : pd.DataFrame — ranking of all models
        "val_preds"      : dict[str, pd.DataFrame] — validation predictions per model
        "val_portfolios" : dict[str, pd.DataFrame] — validation portfolios per model
        "test_preds"     : pd.DataFrame — test panel + "pred" (best model)
        "test_portfolio" : pd.DataFrame — build_portfolio output on test
        "coefficients"   : pd.Series — feature -> coefficient (return per 1 std of feature)

    Test
    ----
        >>> res = run_pipeline(train, val, test, feats, PipelineConfig(grid=default_grid()[:2]))
        >>> res["test_portfolio"][["gross_ret", "net_ret"]].head()
    """
    cfg = cfg or PipelineConfig()
    np.random.seed(SEED)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        missing = set(feature_cols + [TARGET_COL, "fwd_ret"]) - set(df.columns)
        assert not missing, f"{name} is missing columns: {missing}"
    assert train.index.get_level_values("date").max() < val.index.get_level_values("date").min()
    assert val.index.get_level_values("date").max() < test.index.get_level_values("date").min()

    print(f"Model selection on validation ({len(cfg.grid)} models)...")
    best, val_table, val_preds, val_ports = select_model(train, val, feature_cols, cfg)
    print(f"Selected: {best.name}")

    print("Out-of-sample test with selected model...")
    history = pd.concat([train, val])
    test_preds = walk_forward_predict(history, test, feature_cols, best, cfg.refit_every, cfg.window)
    test_port = build_portfolio(test_preds, cfg.top_k, cfg.gate, cfg.cost_bps)

    final_model = best.build().fit(history[feature_cols].values, history[TARGET_COL].values)
    coefs = pd.Series(final_model.named_steps["reg"].coef_.ravel(), index=feature_cols)

    return dict(config=cfg, best_spec=best, val_table=val_table,
                val_preds=val_preds, val_portfolios=val_ports,
                test_preds=test_preds, test_portfolio=test_port, coefficients=coefs)


def gate_sweep(preds: pd.DataFrame, gates=(0.0, 0.0005, 0.001, 0.002, 0.005),
               top_k: int = 5, cost_bps: float = 5.0) -> dict[float, pd.DataFrame]:
    """Rebuild the portfolio for several gate values to see the turnover / cost trade-off.

    Use VALIDATION predictions to choose the gate; report test only for the chosen gate.

    Inputs
    ------
    preds    : panel with "pred" and "fwd_ret" (e.g. results["val_preds"][best_spec.name])
    gates    : iterable of float — predicted-return gaps to try (0.001 = 0.1%)
    top_k    : int
    cost_bps : float

    Output
    ------
    dict[float, pd.DataFrame] — gate value -> build_portfolio output

    Test
    ----
        >>> sweep = gate_sweep(res["val_preds"][res["best_spec"].name])
        >>> {g: p["turnover"].mean() for g, p in sweep.items()}   # falls as gate rises
    """
    return {g: build_portfolio(preds, top_k, g, cost_bps) for g in gates}
