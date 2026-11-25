"""
Statistical evaluation of indicators.

* :func:`predictive_regression` – in-sample OLS of forward h-month excess log returns on a
  signal, Newey-West (HAC, h lags) t-stats.
* :func:`oos_r2` – Campbell & Thompson (2008) style out-of-sample R^2 from expanding-window
  regressions (training pairs only once their forward return is fully realised), with the
  Clark & West (2007) MSPE-adjusted test. Optional CT restrictions: wrong-signed slope ->
  use historical mean; negative premium forecast -> 0.
* :func:`drawdown_auc` – ability of a signal to flag a >= X% drawdown within the next h months.
* :func:`ath_bucket_table` – conditional forward returns by distance from the all-time high,
  optionally split by valuation tercile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Theory-implied sign of each predictor's slope on future returns (+1 / -1) and whether a
# HIGH value is "risk-off" for drawdown prediction (+1 = high value -> higher crash risk).
SIGNAL_SPECS: Dict[str, Dict[str, object]] = {
    # name                   ret sign, risk orientation, family
    "nearness_52w":        {"sign": +1, "risk": -1, "family": "ATH proximity"},
    "nearness_ath":        {"sign": -1, "risk": +1, "family": "ATH proximity"},
    "near_ath":            {"sign": -1, "risk": +1, "family": "ATH proximity"},
    "log_cape":            {"sign": -1, "risk": +1, "family": "Valuation"},
    "excess_cape_yield":   {"sign": +1, "risk": -1, "family": "Valuation"},
    "dividend_yield":      {"sign": +1, "risk": -1, "family": "Valuation"},
    "mcap_gdp":            {"sign": -1, "risk": +1, "family": "Valuation"},
    "aiae":                {"sign": -1, "risk": +1, "family": "Valuation"},
    "tsmom12":             {"sign": +1, "risk": -1, "family": "Trend"},
    "below_sma10":         {"sign": -1, "risk": +1, "family": "Trend"},
    "ma2_12_off":          {"sign": -1, "risk": +1, "family": "Trend"},
    "trend_off":           {"sign": -1, "risk": +1, "family": "Trend"},
    "runup_24m":           {"sign": -1, "risk": +1, "family": "Over-extension"},
    "dist_200d":           {"sign": -1, "risk": +1, "family": "Over-extension"},
    "bsadf":               {"sign": -1, "risk": +1, "family": "Over-extension"},
    "realized_vol":        {"sign": -1, "risk": +1, "family": "Risk"},
    "vix":                 {"sign": +1, "risk": +1, "family": "Risk"},
    "vrp":                 {"sign": +1, "risk": -1, "family": "Risk"},
    "term_spread":         {"sign": +1, "risk": -1, "family": "Macro"},
    "curve_inverted_12m":  {"sign": -1, "risk": +1, "family": "Macro"},
    "credit_spread":       {"sign": +1, "risk": +1, "family": "Macro"},
    "credit_widening":     {"sign": -1, "risk": +1, "family": "Macro"},
    "unemp_rising":        {"sign": -1, "risk": +1, "family": "Macro"},
    "sahm_on":             {"sign": -1, "risk": +1, "family": "Macro"},
    "nfci":                {"sign": -1, "risk": +1, "family": "Macro"},
    "macro_flag":          {"sign": -1, "risk": +1, "family": "Macro"},
}


# --------------------------------------------------------------------------------------------
# Forward outcomes
# --------------------------------------------------------------------------------------------

def excess_log_returns(df: pd.DataFrame) -> pd.Series:
    return (np.log1p(df["ret"]) - np.log1p(df["rf"])).rename("xr")


def forward_excess(df: pd.DataFrame, h: int) -> pd.Series:
    """Sum of excess log returns over months t+1..t+h (aligned at t)."""
    xr = excess_log_returns(df)
    fwd = xr[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)
    return fwd.rename(f"fwd_xr_{h}m")


def forward_total(df: pd.DataFrame, h: int) -> pd.Series:
    lr = np.log1p(df["ret"])
    return lr[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1).rename(f"fwd_tr_{h}m")


def forward_max_drawdown(df: pd.DataFrame, h: int = 12) -> pd.Series:
    """Worst peak-to-trough decline of the total-return index within months t+1..t+h,
    where the peak may be the value at t (i.e. includes losses from the entry point)."""
    tri = np.exp(np.log1p(df["ret"].fillna(0)).cumsum()).to_numpy()
    n = len(tri)
    out = np.full(n, np.nan)
    valid_ret = df["ret"].notna().to_numpy()
    for t in range(n - h):
        if not valid_ret[t + 1: t + h + 1].all():
            continue
        path = tri[t: t + h + 1]
        peak = np.maximum.accumulate(path)
        out[t] = (path / peak - 1.0).min()
    return pd.Series(out, index=df.index, name=f"fwd_mdd_{h}m")


# --------------------------------------------------------------------------------------------
# Regressions
# --------------------------------------------------------------------------------------------

@dataclass
class RegResult:
    beta: float
    tstat: float
    r2: float
    nobs: int


def predictive_regression(x: pd.Series, y: pd.Series, h: int) -> RegResult:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 60:
        return RegResult(np.nan, np.nan, np.nan, len(d))
    X = sm.add_constant(d.iloc[:, 0].astype(float))
    res = sm.OLS(d.iloc[:, 1], X).fit(cov_type="HAC", cov_kwds={"maxlags": max(h, 1)})
    return RegResult(float(res.params.iloc[1]), float(res.tvalues.iloc[1]), float(res.rsquared), len(d))


@dataclass
class OOSResult:
    r2_os: float
    cw_tstat: float
    cw_pvalue: float
    nobs: int
    forecasts: pd.Series
    benchmark: pd.Series


def oos_r2(
    x: pd.Series,
    y: pd.Series,
    h: int = 1,
    oos_start: str = "1950-01-31",
    min_train: int = 120,
    ct_restrict: bool = True,
    expected_sign: Optional[int] = None,
) -> OOSResult:
    """Expanding-window out-of-sample R^2 vs the historical-mean benchmark.

    At forecast origin t we only use training pairs (x_s, y_s) with s + h <= t, so the target
    of every training pair is fully observed at t."""
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1)
    d = d[d["x"].notna()]
    idx = d.index
    xs = d["x"].to_numpy(float)
    ys = d["y"].to_numpy(float)
    pos = {t: i for i, t in enumerate(idx)}
    f_list, b_list, y_list, t_list = [], [], [], []
    start = pd.Timestamp(oos_start)
    for i, t in enumerate(idx):
        if t < start or np.isnan(ys[i]):
            continue
        tr_end = i - h + 1  # pairs s <= i-h  ->  slice [:i-h+1]
        if tr_end < min_train:
            continue
        xt, yt = xs[:tr_end], ys[:tr_end]
        m = ~np.isnan(yt)
        xt, yt = xt[m], yt[m]
        if len(yt) < min_train:
            continue
        bench = yt.mean()
        A = np.column_stack([np.ones(len(xt)), xt])
        coef, *_ = np.linalg.lstsq(A, yt, rcond=None)
        f = coef[0] + coef[1] * xs[i]
        if ct_restrict:
            if expected_sign is not None and np.sign(coef[1]) != expected_sign:
                f = bench
            f = max(f, 0.0)
        f_list.append(f)
        b_list.append(bench)
        y_list.append(ys[i])
        t_list.append(t)
    if len(y_list) < 24:
        e = pd.Series(dtype=float)
        return OOSResult(np.nan, np.nan, np.nan, len(y_list), e, e)
    f_arr, b_arr, y_arr = map(np.asarray, (f_list, b_list, y_list))
    r2 = 1.0 - np.sum((y_arr - f_arr) ** 2) / np.sum((y_arr - b_arr) ** 2)
    adj = (y_arr - b_arr) ** 2 - ((y_arr - f_arr) ** 2 - (b_arr - f_arr) ** 2)
    res = sm.OLS(adj, np.ones(len(adj))).fit(cov_type="HAC", cov_kwds={"maxlags": max(h, 1)})
    tstat = float(res.tvalues[0])
    pval = float(1 - stats.norm.cdf(tstat))
    ti = pd.DatetimeIndex(t_list)
    return OOSResult(float(r2), tstat, pval, len(y_arr), pd.Series(f_arr, ti), pd.Series(b_arr, ti))


# --------------------------------------------------------------------------------------------
# Drawdown prediction
# --------------------------------------------------------------------------------------------

def drawdown_auc(signal: pd.Series, fwd_mdd: pd.Series, threshold: float = -0.20, risk: int = 1) -> Dict[str, float]:
    """AUC of a signal for predicting a >= |threshold| drawdown within the window.

    ``risk`` orients the signal so higher = riskier. Also reports the event rate when the
    signal is in its top quintile (or on, for binary signals) vs the unconditional base rate."""
    from sklearn.metrics import roc_auc_score

    d = pd.concat([signal.rename("s") * risk, fwd_mdd.rename("m")], axis=1).dropna()
    if len(d) < 60:
        return {"auc": np.nan, "base_rate": np.nan, "hit_rate_top": np.nan, "n": len(d)}
    event = (d["m"] <= threshold).astype(int)
    if event.nunique() < 2:
        return {"auc": np.nan, "base_rate": float(event.mean()), "hit_rate_top": np.nan, "n": len(d)}
    auc = roc_auc_score(event, d["s"])
    uniq = d["s"].nunique()
    top = d["s"] >= (d["s"].max() if uniq <= 2 else d["s"].quantile(0.8))
    return {
        "auc": float(auc),
        "base_rate": float(event.mean()),
        "hit_rate_top": float(event[top].mean()) if top.any() else np.nan,
        "n": int(len(d)),
    }


# --------------------------------------------------------------------------------------------
# Conditional returns near all-time highs
# --------------------------------------------------------------------------------------------

ATH_BUCKETS: List[Tuple[str, float, float]] = [
    ("At ATH (0%)", -1e-9, 1.0),
    ("0 to -2%", -0.02, -1e-9),
    ("-2 to -5%", -0.05, -0.02),
    ("-5 to -10%", -0.10, -0.05),
    ("-10 to -20%", -0.20, -0.10),
    ("Below -20%", -1.0, -0.20),
]


def ath_bucket_table(
    df: pd.DataFrame,
    drawdown: pd.Series,
    horizons: Sequence[int] = (1, 3, 12, 36, 60),
    split: Optional[pd.Series] = None,
    split_labels: Sequence[str] = ("Cheap", "Fair", "Expensive"),
    start: str = "1928-01-31",
) -> pd.DataFrame:
    """Mean forward excess log return (annualised for h >= 12) by distance-from-ATH bucket.

    Also: share of months in bucket, probability of a >= 20% drawdown over the next 12 months.
    ``split`` (e.g. valuation percentile) adds tercile sub-buckets."""
    fwd = {h: forward_excess(df, h) for h in horizons}
    mdd = forward_max_drawdown(df, 12)
    base = pd.DataFrame({"dd": drawdown, "mdd12": mdd, **{f"h{h}": fwd[h] for h in horizons}})
    if split is not None:
        base["grp"] = pd.cut(split, [-np.inf, 1 / 3, 2 / 3, np.inf], labels=list(split_labels))
    base = base[base.index >= pd.Timestamp(start)]
    rows = []
    groups = [("All", base)]
    if split is not None:
        groups += [(str(g), base[base["grp"] == g]) for g in split_labels]
    for gname, gdf in groups:
        for label, lo, hi in ATH_BUCKETS:
            sub = gdf[(gdf["dd"] > lo) & (gdf["dd"] <= hi)]
            row = {"group": gname, "bucket": label, "months": len(sub),
                   "share": len(sub) / max(len(gdf), 1)}
            for h in horizons:
                v = sub[f"h{h}"].dropna()
                scale = 12.0 / h if h >= 12 else 1.0
                row[f"fwd_{h}m"] = v.mean() * scale if len(v) else np.nan
                if h == 12:
                    row["pct_neg_12m"] = (v < 0).mean() if len(v) else np.nan
            m = sub["mdd12"].dropna()
            row["p_dd20_12m"] = (m <= -0.20).mean() if len(m) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# One-stop table
# --------------------------------------------------------------------------------------------

def evaluate_signals(
    df: pd.DataFrame,
    ind: pd.DataFrame,
    names: Optional[Iterable[str]] = None,
    start: str = "1928-01-31",
    oos_start: str = "1950-01-31",
    horizons: Sequence[int] = (1, 12),
) -> pd.DataFrame:
    """In-sample slope/t/R^2, OOS R^2 + Clark-West p-value, and 12m >=20% drawdown AUC."""
    ind = ind.copy()
    if "cape" in ind:
        ind["log_cape"] = np.log(ind["cape"])
    names = [n for n in (names or SIGNAL_SPECS.keys()) if n in ind]
    sl = df.index >= pd.Timestamp(start)
    fwd = {h: forward_excess(df, h)[sl] for h in horizons}
    mdd = forward_max_drawdown(df, 12)[sl]
    rows = []
    for n in names:
        spec = SIGNAL_SPECS[n]
        x = ind[n][sl]
        row = {"signal": n, "family": spec["family"], "first": x.first_valid_index()}
        for h in horizons:
            reg = predictive_regression(x, fwd[h], h)
            row[f"beta_{h}m"] = reg.beta
            row[f"t_{h}m"] = reg.tstat
            row[f"R2_{h}m"] = reg.r2
            oos = oos_r2(x, fwd[h], h=h, oos_start=max(pd.Timestamp(oos_start), (x.first_valid_index() or df.index[0]) + pd.DateOffset(years=12)).strftime("%Y-%m-%d"),
                         expected_sign=int(spec["sign"]))
            row[f"R2oos_{h}m"] = oos.r2_os
            row[f"CW_p_{h}m"] = oos.cw_pvalue
        auc = drawdown_auc(x, mdd, risk=int(spec["risk"]))
        row["AUC_dd20_12m"] = auc["auc"]
        row["hit_top"] = auc["hit_rate_top"]
        row["base_rate"] = auc["base_rate"]
        row["n"] = auc["n"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("signal")
