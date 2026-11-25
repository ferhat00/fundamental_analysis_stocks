"""
Monthly backtest engine and performance statistics.

Conventions
-----------
* A target cash weight decided at month end *t* (from information available at *t*) is
  applied to month *t+1* returns. Equity earns the total return, cash earns the T-bill.
* Weights drift with returns between decisions. A trade happens only when the target differs
  from the drifted weight by at least ``band`` (default 5pp no-trade band); then the portfolio
  is moved fully to target and pays ``cost_bps`` per unit of turnover (default 10bp).
* Benchmarks: buy-and-hold, and a *matched* static mix that holds the strategy's average cash
  weight every month (the fair comparison: any cash holding lowers risk mechanically).

Run ``python -m market_timing.backtest`` for the full report tables (saved to
``output/market_timing``).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from . import allocation as A
from . import data as D
from . import indicators as I

logger = logging.getLogger(__name__)
OUT_DIR = D.REPO_ROOT / "output" / "market_timing"


# --------------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------------

@dataclass
class BacktestResult:
    returns: pd.Series      # monthly portfolio returns
    cash_weight: pd.Series  # cash weight held during each month (after trading at prior month end)
    turnover: pd.Series     # one-way turnover traded at the start of each month


def run_backtest(
    equity_ret: pd.Series,
    cash_ret: pd.Series,
    target_cash: pd.Series,
    band: float = 0.05,
    cost_bps: float = 10.0,
) -> BacktestResult:
    """Simulate the strategy. ``target_cash`` at index t is applied to returns at t+1."""
    idx = equity_ret.index.intersection(cash_ret.index)
    tgt = target_cash.reindex(idx).shift(1)  # decision at t -> held over t+1
    re = equity_ret.reindex(idx).to_numpy(float)
    rc = cash_ret.reindex(idx).to_numpy(float)
    tg = tgt.to_numpy(float)
    n = len(idx)
    rets = np.full(n, np.nan)
    w_cash = np.full(n, np.nan)
    turn = np.zeros(n)
    c = None  # current (drifted) cash weight
    for i in range(n):
        if np.isnan(re[i]) or np.isnan(rc[i]) or np.isnan(tg[i]):
            continue
        if c is None:
            c = tg[i]
            traded = 0.0
        elif abs(tg[i] - c) >= band - 1e-12:
            traded = abs(tg[i] - c)
            c = tg[i]
        else:
            traded = 0.0
        cost = traded * cost_bps / 1e4
        r = (1 - c) * re[i] + c * rc[i] - cost
        rets[i] = r
        w_cash[i] = c
        turn[i] = traded
        # drift
        eq_val = (1 - c) * (1 + re[i])
        ca_val = c * (1 + rc[i])
        c = ca_val / (eq_val + ca_val) if (eq_val + ca_val) > 0 else c
    s = pd.Series(rets, idx).dropna()
    return BacktestResult(s, pd.Series(w_cash, idx).reindex(s.index), pd.Series(turn, idx).reindex(s.index))


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------

def max_drawdown(r: pd.Series) -> float:
    w = (1 + r).cumprod()
    return float((w / w.cummax() - 1).min())


def ulcer_index(r: pd.Series) -> float:
    w = (1 + r).cumprod()
    dd = (w / w.cummax() - 1) * 100
    return float(np.sqrt((dd ** 2).mean()))


def metrics(r: pd.Series, rf: pd.Series, bench: Optional[pd.Series] = None,
            cash_weight: Optional[pd.Series] = None, turnover: Optional[pd.Series] = None) -> Dict[str, float]:
    r = r.dropna()
    rf = rf.reindex(r.index)
    years = len(r) / 12
    cagr = (1 + r).prod() ** (1 / years) - 1
    ex = r - rf
    vol = r.std() * np.sqrt(12)
    sharpe = ex.mean() / ex.std() * np.sqrt(12)
    downside = ex[ex < 0].std() * np.sqrt(12)
    sortino = ex.mean() * 12 / downside if downside > 0 else np.nan
    mdd = max_drawdown(r)
    out = {
        "CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan, "Ulcer": ulcer_index(r),
        "Worst12m": float((1 + r).rolling(12).apply(np.prod, raw=True).min() - 1),
    }
    if bench is not None:
        b = bench.reindex(r.index)
        up, dn = b > 0, b < 0
        out["UpCapture"] = r[up].mean() / b[up].mean()
        out["DownCapture"] = r[dn].mean() / b[dn].mean()
    if cash_weight is not None:
        out["AvgCash"] = float(cash_weight.mean())
        out["TimeInCash>=20%"] = float((cash_weight >= 0.2 - 1e-9).mean())
    if turnover is not None:
        out["Turnover/yr"] = float(turnover.sum() / years)
        out["Trades/yr"] = float((turnover > 0).sum() / years)
    return out


def stationary_bootstrap_sharpe_diff(
    r1: pd.Series, r2: pd.Series, rf: pd.Series, n_boot: int = 2000, block: int = 12, seed: int = 1
) -> Dict[str, float]:
    """Politis-Romano stationary bootstrap of the Sharpe-ratio difference (r1 - r2)."""
    d = pd.concat([r1, r2, rf], axis=1).dropna().to_numpy()
    n = len(d)
    rng = np.random.default_rng(seed)
    p = 1.0 / block

    def sr(x, f):
        e = x - f
        return e.mean() / e.std() * np.sqrt(12)

    obs = sr(d[:, 0], d[:, 2]) - sr(d[:, 1], d[:, 2])
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        jumps = rng.random(n) < p
        starts = rng.integers(n, size=n)
        for t in range(1, n):
            idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % n
        s = d[idx]
        diffs[b] = sr(s[:, 0], s[:, 2]) - sr(s[:, 1], s[:, 2])
    return {"sharpe_diff": float(obs), "ci_lo": float(np.percentile(diffs, 2.5)),
            "ci_hi": float(np.percentile(diffs, 97.5)), "p_le_0": float((diffs <= 0).mean())}


def deflated_sharpe(excess: pd.Series, trial_sharpes_monthly: Sequence[float]) -> float:
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio.

    ``excess`` is the return series being judged (here: strategy minus its matched static mix),
    ``trial_sharpes_monthly`` the monthly Sharpe ratios of *all* strategies tried on the same
    question. Returns the probability that the true Sharpe exceeds the maximum expected from
    N unskilled trials, after adjusting for skewness and kurtosis."""
    e = excess.dropna()
    T = len(e)
    sr = e.mean() / e.std()
    g3 = stats.skew(e)
    g4 = stats.kurtosis(e, fisher=False)
    trials = np.asarray(trial_sharpes_monthly, float)
    N = max(len(trials), 2)
    var_sr = np.var(trials, ddof=1) if len(trials) > 1 else 0.0
    emc = 0.5772156649
    sr0 = np.sqrt(var_sr) * ((1 - emc) * stats.norm.ppf(1 - 1 / N) + emc * stats.norm.ppf(1 - 1 / (N * np.e)))
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf(z))


# --------------------------------------------------------------------------------------------
# Study driver
# --------------------------------------------------------------------------------------------

SUBPERIODS: List[Tuple[str, str, str]] = [
    ("1930-1949", "1930-01-31", "1949-12-31"),
    ("1950-1989", "1950-01-31", "1989-12-31"),
    ("1990-2009", "1990-01-31", "2009-12-31"),
    ("2010-now", "2010-01-31", "2100-01-01"),
]


def load(complete_only: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = D.us_monthly()
    if complete_only:
        df = df[df["complete"]]
    ind = I.compute_all(df)
    return df, ind


def compare(
    df: pd.DataFrame,
    strategies: Dict[str, pd.Series],
    start: str = "1930-01-31",
    end: Optional[str] = None,
    band: float = 0.05,
    cost_bps: float = 10.0,
    matched: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, BacktestResult]]:
    sl = slice(pd.Timestamp(start), pd.Timestamp(end) if end else None)
    eq, rf = df["ret"].loc[sl], df["rf"].loc[sl]
    results: Dict[str, BacktestResult] = {}
    rows = {}
    bh = run_backtest(eq, rf, pd.Series(0.0, index=eq.index), band, cost_bps).returns
    for name, tgt in strategies.items():
        res = run_backtest(eq, rf, tgt, band, cost_bps)
        results[name] = res
        rows[name] = metrics(res.returns, rf, bh, res.cash_weight, res.turnover)
        if matched and name != "buy_hold":
            avg = float(res.cash_weight.mean())
            st = run_backtest(eq, rf, pd.Series(avg, index=eq.index), band, cost_bps)
            m = metrics(st.returns, rf, bh, st.cash_weight, st.turnover)
            rows[name]["Matched_CAGR"] = m["CAGR"]
            rows[name]["Matched_Sharpe"] = m["Sharpe"]
            rows[name]["Matched_MaxDD"] = m["MaxDD"]
            rows[name]["dCAGR_vs_matched"] = rows[name]["CAGR"] - m["CAGR"]
            rows[name]["dSharpe_vs_matched"] = rows[name]["Sharpe"] - m["Sharpe"]
            results[f"{name}__matched"] = st
    return pd.DataFrame(rows).T, results


def walk_forward_composite(
    df: pd.DataFrame,
    ind: pd.DataFrame,
    first_oos: int = 1960,
    step_years: int = 10,
    start: str = "1930-01-31",
    objective: str = "Sharpe",
) -> Tuple[pd.Series, pd.DataFrame]:
    """Choose composite-rule parameters on an expanding window, apply to the next decade."""
    base = A.CashRule()
    grid = []
    for smax in (0.10, 0.20):
        for vlo in (0.5, 0.6, 0.7):
            for tb in (0.20, 0.30):
                for vnt in (True, False):
                    grid.append(A.CashRule(strategic_max=smax, val_lo=vlo, tactical_both=tb,
                                           val_needs_trend=vnt, name=f"s{smax}_v{vlo}_t{tb}_{'vt' if vnt else 'v'}"))
    cash_paths = {r.name: A.composite_cash(ind, r) for r in grid}
    chosen = []
    pieces = []
    last_year = df.index[-1].year
    for y0 in range(first_oos, last_year + 1, step_years):
        train_end = pd.Timestamp(f"{y0 - 1}-12-31")
        best, best_val = None, -np.inf
        for r in grid:
            res = run_backtest(df["ret"].loc[start:train_end], df["rf"].loc[start:train_end], cash_paths[r.name])
            val = metrics(res.returns, df["rf"])[objective]
            if val > best_val:
                best, best_val = r, val
        seg = cash_paths[best.name].loc[f"{y0 - 1}-12-31": f"{y0 + step_years - 1}-11-30"]
        pieces.append(seg)
        chosen.append({"from": y0, "params": best.name, f"train_{objective}": best_val})
    wf = pd.concat(pieces)
    wf = wf[~wf.index.duplicated(keep="last")].rename("composite_walkforward")
    return wf, pd.DataFrame(chosen)


def global_check(regions: Sequence[str] = ("UK", "Japan", "WORLD", "EUROPE"), start: str = "1985-01-31") -> pd.DataFrame:
    """Robustness: the same layered logic on French international data (price trend, dividend-
    yield valuation percentile, ATH proximity, term-spread inversion as the macro leg)."""
    rows = []
    for reg in regions:
        try:
            p = D.intl_monthly(reg)
        except Exception as err:  # noqa: BLE001
            logger.warning("Global check skipped for %s: %s", reg, err)
            continue
        ind = pd.DataFrame(index=p.index)
        ind["drawdown_from_ath"] = p["price"] / p["ath"] - 1
        ind["below_sma10"] = I.sma_signal(p, 10).astype(float)
        ind["trend_off"] = I.trend_off(p).astype(float)
        dy = p["div"] / p["price"]
        ind["valuation_pct"] = 1 - I.expanding_percentile(dy, min_periods=60)  # low yield = expensive
        ts = p["term_spread"]
        ind["macro_flag"] = ((ts < 0).astype(float).rolling(12, min_periods=1).max()).where(ts.notna())
        ind["runup_24m_pct"] = I.expanding_percentile(p["price"] / p["price"].shift(24) - 1, min_periods=60)
        ind["realized_vol"] = p["ret"].rolling(6).std() * np.sqrt(12)
        strategies = {
            "buy_hold": pd.Series(0.0, index=ind.index),
            "faber_sma10": A.faber_cash(ind),
            "valuation_only": A.valuation_only_cash(ind),
            "bank_gains": A.bank_gains_cash(ind),
            "composite": A.composite_cash(ind),
        }
        tab, _ = compare(p, strategies, start=start)
        tab.insert(0, "region", reg)
        rows.append(tab)
    return pd.concat(rows) if rows else pd.DataFrame()


def run_study(start: str = "1930-01-31", do_global: bool = True, n_boot: int = 2000, save: bool = True) -> Dict[str, pd.DataFrame]:
    df, ind = load()
    strategies = A.strategy_set(df, ind)
    wf, chosen = walk_forward_composite(df, ind)
    strategies["composite_walkforward"] = wf
    out: Dict[str, pd.DataFrame] = {}

    full, results = compare(df, strategies, start=start)
    out["full"] = full
    # post-1960 (walk-forward period) for a like-for-like comparison incl. walk-forward
    post, _ = compare(df, strategies, start="1960-01-31")
    out["post1960"] = post
    sub_rows = []
    for label, s, e in SUBPERIODS:
        t, _ = compare(df, {k: v for k, v in strategies.items() if k != "composite_walkforward"}, start=s, end=e)
        t.insert(0, "period", label)
        sub_rows.append(t)
    out["subperiods"] = pd.concat(sub_rows)
    out["walkforward_params"] = chosen

    # walk-forward only exists from 1960: keep it out of the full-period table
    out["full"] = full.drop(index="composite_walkforward")

    # bootstrap + deflated Sharpe of active returns (strategy - matched static mix)
    rf = df["rf"]
    active = {}
    for k in strategies:
        if k == "buy_hold":
            continue
        r = results[k].returns
        m = results[f"{k}__matched"].returns
        active[k] = (r - m).dropna()
    trial_srs = [a.mean() / a.std() for a in active.values()]
    boot_rows = {}
    for k, a in active.items():
        bs = stationary_bootstrap_sharpe_diff(results[k].returns, results[f"{k}__matched"].returns, rf, n_boot=n_boot)
        bs["active_IR_ann"] = float(a.mean() / a.std() * np.sqrt(12))
        bs["deflated_sharpe_prob"] = deflated_sharpe(a, trial_srs)
        boot_rows[k] = bs
    out["bootstrap"] = pd.DataFrame(boot_rows).T

    # cycle phase conditional returns
    from .evaluate import forward_excess, forward_max_drawdown
    ph = A.cycle_phase(ind)
    f12 = forward_excess(df, 12)
    f1 = forward_excess(df, 1)
    mdd = forward_max_drawdown(df, 12)
    phase_tab = pd.DataFrame({"phase": ph, "f1": f1, "f12": f12, "mdd": mdd}).loc[start:]
    out["phases"] = phase_tab.groupby("phase").agg(
        months=("f1", "size"),
        ann_excess_from_1m=("f1", lambda x: x.mean() * 12),
        fwd12_excess=("f12", "mean"),
        vol_1m=("f1", lambda x: x.std() * np.sqrt(12)),
        p_dd20_12m=("mdd", lambda x: (x <= -0.2).mean()),
    )
    if do_global:
        out["global"] = global_check()

    out["equity_curves"] = pd.DataFrame({k: (1 + v.returns).cumprod() for k, v in results.items() if "__" not in k})
    out["cash_paths"] = pd.DataFrame({k: v.cash_weight for k, v in results.items() if "__" not in k})
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for k, v in out.items():
            v.to_csv(OUT_DIR / f"{k}.csv")
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Backtest cash-allocation rules near all-time highs")
    ap.add_argument("--start", default="1930-01-31")
    ap.add_argument("--no-global", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = run_study(start=args.start, do_global=not args.no_global, n_boot=args.boot)
    cols = ["CAGR", "Vol", "Sharpe", "MaxDD", "Calmar", "Ulcer", "AvgCash", "Turnover/yr",
            "Matched_CAGR", "Matched_Sharpe", "Matched_MaxDD", "dCAGR_vs_matched", "dSharpe_vs_matched", "DownCapture"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        for k in ("full", "post1960"):
            print(f"\n=== {k} ===")
            print(out[k][[c for c in cols if c in out[k]]])
        print("\n=== subperiods (Sharpe / MaxDD / dSharpe vs matched) ===")
        print(out["subperiods"][["period", "CAGR", "Sharpe", "MaxDD", "AvgCash", "dSharpe_vs_matched"]])
        print("\n=== bootstrap ===")
        print(out["bootstrap"])
        print("\n=== walk-forward chosen params ===")
        print(out["walkforward_params"])
        print("\n=== cycle phases ===")
        print(out["phases"])
        if "global" in out and len(out["global"]):
            print("\n=== global check ===")
            print(out["global"][["region", "CAGR", "Sharpe", "MaxDD", "AvgCash", "dCAGR_vs_matched", "dSharpe_vs_matched", "Matched_MaxDD"]])
    print(f"\nTables saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
