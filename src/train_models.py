"""
train.py — DSA5205 Project 2: model training, walk-forward prediction, portfolio construction.

Purpose
-------
Fits penalised logistic regressions (Ridge / LASSO / Elastic Net) that predict whether each
asset beats the cross-sectional median return next period, selects the best one on the
validation period by net Sharpe, and runs it once on the untouched test period.

Standard data format ("panel") used by every function here
----------------------------------------------------------
pandas DataFrame, MultiIndex (date, ticker), one row per asset per date:

    index level "date"   : pd.Timestamp — decision date t (close of t)
    index level "ticker" : str          — e.g. "XLK"
    <feature columns>    : float        — known at the close of t (no future data)
    "target"             : int/float 0/1 — 1 if the asset's return t -> t+1 beats that
                                           day's cross-sectional median, else 0
    "fwd_ret"            : float        — the asset's simple return t -> t+1 (e.g. 0.0042)

    Example:
                            mom_20   vol_20  target   fwd_ret
        date       ticker
        2015-01-02 XLB      0.0123   0.0110     1.0    0.0051
                   XLE     -0.0310   0.0185     0.0   -0.0120

Timing convention: weights chosen at the close of t earn fwd_ret at t (trade at the same close).

Quick test
----------
    python run_example.py          # runs everything on synthetic data
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 5205
_SKLEARN_GE_18 = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
PERIODS_PER_YEAR = {"daily": 252, "weekly": 52}


# --------------------------------------------------------------------------------------
# 0. Labels (optional helper)
# --------------------------------------------------------------------------------------
def make_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the label columns ("fwd_ret", "target") from a table of prices.

    For each date t and ticker: fwd_ret = price[t+1] / price[t] - 1, and target = 1 if that
    return is above the median fwd_ret across all tickers on date t. The last date has no
    t+1 price, so it is dropped.

    Input
    -----
    prices : pd.DataFrame, wide format
        index   = DatetimeIndex of trading dates (sorted ascending)
        columns = tickers, values = adjusted close prices (float)
            Example:            XLB     XLE     XLF
                    2015-01-02  45.1    80.2    24.3
                    2015-01-05  44.6    77.9    23.9

    Output
    ------
    pd.DataFrame, MultiIndex (date, ticker), columns ["fwd_ret", "target"]
        len = (n_dates - 1) * n_tickers (rows with missing prices are dropped)

    Test
    ----
        >>> p = pd.DataFrame({"A": [100, 101, 102], "B": [100, 99, 103]},
        ...                  index=pd.bdate_range("2020-01-01", periods=3))
        >>> make_labels(p)
        # 2020-01-01: A fwd_ret 0.01, target 1;  B fwd_ret -0.01, target 0
        # (with 2 tickers the median is their mean, so exactly one is 1)
    """
    fwd_ret = prices.pct_change().shift(-1)                      # return from t to t+1
    beats_median = fwd_ret.gt(fwd_ret.median(axis=1), axis=0).astype(float)
    beats_median[fwd_ret.isna()] = np.nan                        # last row has no label
    out = pd.concat({"fwd_ret": fwd_ret.stack(future_stack=True),
                     "target": beats_median.stack(future_stack=True)}, axis=1)
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
    name     : str   — unique label, format "<family>_C<C>[_r<l1_ratio>]", e.g. "enet_C0.1_r0.5".
                       The part before the first "_" must be "ridge", "lasso" or "enet".
    penalty  : str   — "l2" (Ridge), "l1" (LASSO) or "elasticnet".
    C        : float — inverse regularisation strength; smaller C = more shrinkage.
    l1_ratio : float | None — only for "elasticnet": 0 = pure Ridge, 1 = pure LASSO.

    Test
    ----
        >>> spec = ModelSpec("ridge_C1.0", "l2", 1.0)
        >>> model = spec.build()            # sklearn Pipeline, not yet fitted
    """
    name: str
    penalty: str
    C: float
    l1_ratio: float | None = None

    def build(self) -> Pipeline:
        """Create a fresh, unfitted model for this spec.

        The StandardScaler is inside the pipeline, so calling .fit() on training rows
        learns the mean/std from those rows only (no leakage into val/test).

        Input  : none (uses the spec's fields)
        Output : sklearn.pipeline.Pipeline with steps
                     "scaler" -> StandardScaler
                     "clf"    -> LogisticRegression(solver="saga")
                 Use .fit(X, y) with X: np.ndarray (n_rows, n_features), y: 0/1 array (n_rows,)
                 and .predict_proba(X)[:, 1] for P(target = 1).

        Test
        ----
            >>> X = np.random.randn(200, 3); y = (X[:, 0] > 0).astype(int)
            >>> m = ModelSpec("lasso_C1.0", "l1", 1.0).build().fit(X, y)
            >>> m.predict_proba(X[:5])[:, 1]      # 5 probabilities in [0, 1]
        """
        ratio = {"l2": 0.0, "l1": 1.0}.get(self.penalty, self.l1_ratio)
        kw = dict(C=self.C, solver="saga", max_iter=5000, random_state=SEED)
        if _SKLEARN_GE_18:   # sklearn >= 1.8: penalty is set via l1_ratio only
            kw["l1_ratio"] = ratio
        else:
            kw.update(penalty=self.penalty,
                      l1_ratio=self.l1_ratio if self.penalty == "elasticnet" else None)
        clf = LogisticRegression(**kw)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def default_grid() -> list[ModelSpec]:
    """Return the fixed hyperparameter grid, declared before looking at validation results.

    Grid: C in {0.01, 0.1, 1, 10}
          Ridge  : 4 models
          LASSO  : 4 models
          ElasticNet : 4 C x 3 l1_ratio {0.25, 0.5, 0.75} = 12 models
          Total  : 20 models (report this number in §4.7)

    Input  : none
    Output : list[ModelSpec], length 20

    Test
    ----
        >>> g = default_grid(); len(g)          # 20
        >>> g[0].name                           # "ridge_C0.01"
    """
    Cs = [0.01, 0.1, 1.0, 10.0]
    grid = [ModelSpec(f"ridge_C{c}", "l2", c) for c in Cs]
    grid += [ModelSpec(f"lasso_C{c}", "l1", c) for c in Cs]
    grid += [ModelSpec(f"enet_C{c}_r{r}", "elasticnet", c, r) for c in Cs for r in (0.25, 0.5, 0.75)]
    return grid


def model_family(spec_name: str) -> str:
    """Get the model family from a spec name.

    Input  : spec_name — str, e.g. "enet_C0.1_r0.5"
    Output : str — "ridge", "lasso" or "enet"

    Test
    ----
        >>> model_family("lasso_C1.0")          # "lasso"
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
      1. training rows = all rows of history + earlier eval rows dated BEFORE the block start,
         minus the last `embargo` dates (their labels use prices from inside the block);
         if `window` is set, keep only the most recent `window` dates;
      2. fit a fresh model from `spec`;
      3. predict P(target = 1) for every row in the block.

    Inputs
    ------
    history      : panel (see module docstring) — data before eval_df (e.g. train set)
    eval_df      : panel — the period to predict (e.g. val or test); all its dates must be
                   later than history's dates
    feature_cols : list[str] — column names used as model inputs, e.g. ["mom_20", "vol_20"]
    spec         : ModelSpec — which model to fit
    refit_every  : str — pandas offset alias for block length: "ME" month, "QE" quarter, "YE" year
    window       : int | None — rolling training window in number of dates; None = use all past
    embargo      : int — number of most recent dates dropped before each refit (1 for next-day labels)

    Output
    ------
    pd.DataFrame — same rows/columns as eval_df plus a "prob" column (float in [0, 1]),
                   MultiIndex (date, ticker), sorted.

    Test
    ----
        >>> preds = walk_forward_predict(train, val, ["mom_20", "vol_20"],
        ...                              ModelSpec("ridge_C1.0", "l2", 1.0))
        >>> preds["prob"].between(0, 1).all()   # True
        >>> len(preds) == len(val)              # True
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
            warnings.simplefilter("ignore")   # saga convergence warnings on tiny C
            model.fit(train_rows[feature_cols].values, train_rows["target"].values)

        block = eval_df.loc[eval_df.index.get_level_values("date").isin(block_dates)].copy()
        block["prob"] = model.predict_proba(block[feature_cols].values)[:, 1]
        preds.append(block)

    return pd.concat(preds).sort_index()


# --------------------------------------------------------------------------------------
# 3. Predictions -> portfolio (top-k, equal weight, with cost-aware action gate)
# --------------------------------------------------------------------------------------
def build_portfolio(
    preds: pd.DataFrame,
    top_k: int = 5,
    gate: float = 0.0,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Turn predicted probabilities into a daily equal-weight top-k portfolio with costs.

    Rule on each date t:
      1. Fill any empty slots with the highest-probability names not already held.
      2. Action gate: while (best outsider prob) - (weakest holding prob) > gate,
         swap them. gate = 0 means plain "hold the top k".
      3. Weight = 1/k for each holding, 0 otherwise.
      4. turnover = sum |w_t - w_{t-1}|; cost = turnover * cost_bps / 10,000.
      5. gross_ret = sum(w_t * fwd_ret_t); net_ret = gross_ret - cost.
    The first date counts as a full entry (turnover = 1). Weights are reset to target each
    period (intra-period drift is ignored).

    Inputs
    ------
    preds    : panel with at least columns "prob" and "fwd_ret" (output of walk_forward_predict)
    top_k    : int — number of assets held (must be <= number of tickers)
    gate     : float — minimum probability gap to swap a holding, e.g. 0.02
    cost_bps : float — one-way cost per unit of turnover in basis points (5 = 0.05%)

    Output
    ------
    pd.DataFrame indexed by date (one row per date), columns:
        gross_ret  : float — portfolio return before costs
        turnover   : float — sum of absolute weight changes (0 to 2)
        cost       : float — cost as a return, e.g. 0.0002
        net_ret    : float — gross_ret - cost
        n_holdings : int   — number of names held (normally top_k)
        holdings   : str   — comma-separated tickers held, e.g. "XLE,XLK,XLV"

    Test
    ----
        >>> port = build_portfolio(preds, top_k=5, gate=0.0, cost_bps=5)
        >>> (port["n_holdings"] == 5).all()                       # True
        >>> np.allclose(port["net_ret"], port["gross_ret"] - port["cost"])   # True
        >>> build_portfolio(preds, gate=1.0)["turnover"].iloc[1:].sum()      # 0: gate=1 never swaps
    """
    probs = preds["prob"].unstack("ticker")
    rets = preds["fwd_ret"].unstack("ticker")

    holdings: set[str] = set()
    prev_w = pd.Series(0.0, index=probs.columns)
    rows = []
    for date, p in probs.iterrows():
        p = p.dropna()
        holdings &= set(p.index)                     # drop anything no longer tradable
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

        w = pd.Series(0.0, index=probs.columns)
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
    """All settings for one pipeline run (record these in the report for reproducibility).

    Fields
    ------
    freq        : str   — "daily" or "weekly"; sets periods per year (252 / 52) for Sharpe
    top_k       : int   — number of assets held
    gate        : float — action-gate probability gap (see build_portfolio)
    cost_bps    : float — one-way transaction cost in basis points
    refit_every : str   — refit block length, "ME" / "QE" / "YE"
    window      : int | None — rolling training window in dates; None = expanding
    grid        : list[ModelSpec] — models to compare (default_grid() by default)

    Test
    ----
        >>> cfg = PipelineConfig(top_k=3, cost_bps=10, grid=default_grid()[:2])
    """
    freq: str = "daily"
    top_k: int = 5
    gate: float = 0.0
    cost_bps: float = 5.0
    refit_every: str = "QE"
    window: int | None = None
    grid: list[ModelSpec] = field(default_factory=default_grid)


def _sharpe(r: pd.Series, ppy: int) -> float:
    """Annualised Sharpe ratio with risk-free rate 0.

    Input  : r — pd.Series of periodic returns (float); ppy — periods per year (252 or 52)
    Output : float — sqrt(ppy) * mean(r) / std(r); NaN if std is 0

    Test
    ----
        >>> _sharpe(pd.Series([0.01, -0.005, 0.002]), 252)     # a float
    """
    return float(np.sqrt(ppy) * r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else np.nan


def select_model(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str],
                 cfg: PipelineConfig):
    """Run every model in cfg.grid walk-forward through val and rank them by net Sharpe.

    Models whose predictions are identical across tickers on every date (all coefficients
    shrunk to zero) are flagged as degenerate and excluded, because their picks are arbitrary.

    Inputs
    ------
    train        : panel — training period (e.g. 2015-2020)
    val          : panel — validation period (e.g. 2021-2022), all dates after train
    feature_cols : list[str] — feature column names
    cfg          : PipelineConfig

    Outputs (tuple of 4)
    --------------------
    best      : ModelSpec — the spec with the highest validation net Sharpe
    table     : pd.DataFrame, one row per model, sorted best first, columns:
                    model (str), family (str), degenerate (bool),
                    net_sharpe (float, NaN if degenerate), gross_sharpe (float), avg_turnover (float)
    val_preds : dict[str, pd.DataFrame] — model name -> walk_forward_predict output on val
    val_ports : dict[str, pd.DataFrame] — model name -> build_portfolio output on val

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
        degenerate = pr.groupby(level="date")["prob"].std().mean() < 1e-6
        rows.append(dict(model=spec.name, family=model_family(spec.name),
                         degenerate=degenerate,
                         net_sharpe=np.nan if degenerate else _sharpe(pf["net_ret"], ppy),
                         gross_sharpe=_sharpe(pf["gross_ret"], ppy),
                         avg_turnover=pf["turnover"].mean()))
        print(f"  [val] {spec.name:<22} net Sharpe = {rows[-1]['net_sharpe']:.3f}")
    table = (pd.DataFrame(rows).sort_values("net_sharpe", ascending=False, na_position="last")
             .reset_index(drop=True))
    assert table["net_sharpe"].notna().any(), "Every model is degenerate: loosen shrinkage (raise C)"
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
    train, val, test : panels (see module docstring); must contain feature_cols,
                       "target" and "fwd_ret", and be in time order with no overlap
    feature_cols     : list[str] — e.g. ["mom_20", "vol_20", "rsi_14"]
    cfg              : PipelineConfig | None — None uses defaults

    Output
    ------
    dict with keys:
        "config"         : PipelineConfig used
        "best_spec"      : ModelSpec chosen on validation
        "val_table"      : pd.DataFrame — ranking of all models (see select_model)
        "val_preds"      : dict[str, pd.DataFrame] — validation predictions per model
        "val_portfolios" : dict[str, pd.DataFrame] — validation portfolios per model
        "test_preds"     : pd.DataFrame — panel + "prob" for the test period (best model)
        "test_portfolio" : pd.DataFrame — build_portfolio output on test (best model)
        "coefficients"   : pd.Series — index = feature_cols, values = standardised coefficients

    Test
    ----
        >>> res = run_pipeline(train, val, test, feats, PipelineConfig(grid=default_grid()[:2]))
        >>> res["test_portfolio"][["gross_ret", "net_ret"]].head()
        >>> res["coefficients"]
    """
    cfg = cfg or PipelineConfig()
    np.random.seed(SEED)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        missing = set(feature_cols + ["target", "fwd_ret"]) - set(df.columns)
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

    final_model = best.build().fit(history[feature_cols].values, history["target"].values)
    coefs = pd.Series(final_model.named_steps["clf"].coef_.ravel(), index=feature_cols)

    return dict(config=cfg, best_spec=best, val_table=val_table,
                val_preds=val_preds, val_portfolios=val_ports,
                test_preds=test_preds, test_portfolio=test_port, coefficients=coefs)


def gate_sweep(preds: pd.DataFrame, gates=(0.0, 0.01, 0.02, 0.05, 0.1),
               top_k: int = 5, cost_bps: float = 5.0) -> dict[float, pd.DataFrame]:
    """Rebuild the portfolio for several gate values to see the turnover / cost trade-off.

    Use VALIDATION predictions to choose the gate; report test only for the chosen gate.

    Inputs
    ------
    preds    : panel with "prob" and "fwd_ret" (e.g. results["val_preds"][best_spec.name])
    gates    : iterable of float — gate values to try
    top_k    : int
    cost_bps : float

    Output
    ------
    dict[float, pd.DataFrame] — gate value -> build_portfolio output

    Test
    ----
        >>> sweep = gate_sweep(res["val_preds"][res["best_spec"].name])
        >>> {g: p["turnover"].mean() for g, p in sweep.items()}   # should fall as gate rises
    """
    return {g: build_portfolio(preds, top_k, g, cost_bps) for g in gates}
