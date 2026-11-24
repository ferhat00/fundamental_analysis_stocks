"""
Indicator library. Every function takes the point-in-time monthly panel from
:func:`market_timing.data.us_monthly` (or the international panel) and returns a Series
indexed by month end that uses only information available at that month end.

Families (and the literature each operationalises)
--------------------------------------------------
ATH proximity     Li & Yu (2012): nearness to 52-week high (+) vs historical high (-).
Valuation         Campbell & Shiller (1988, 1998) CAPE; Shiller's excess CAPE yield;
                  market cap / GDP; aggregate investor equity allocation.
Trend             Faber (2007) 10-month SMA; Moskowitz, Ooi & Pedersen (2012) 12m TSMOM;
                  Neely, Rapach, Tu & Zhou (2014) MA(2,12) rule; 200-day MA.
Over-extension    Greenwood, Shleifer & You (2019) run-ups; Phillips, Shi & Yu (2015) BSADF.
Risk              Realised volatility (Moreira & Muir 2017); VIX; variance risk premium
                  (Bollerslev, Tauchen & Zhou 2009).
Macro             Estrella & Mishkin (1998) term spread; unemployment trend / Sahm rule;
                  credit spreads (Gilchrist & Zakrajsek 2012); NFCI; CFNAI.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------------

def expanding_percentile(x: pd.Series, min_periods: int = 120) -> pd.Series:
    """Percentile rank (0-1) of x_t within {x_s : s <= t}. Strictly no look-ahead."""
    vals = x.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    hist: list = []
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        hist.append(v)
        if len(hist) >= min_periods:
            arr = np.asarray(hist)
            out[i] = (np.sum(arr < v) + 0.5 * np.sum(arr == v)) / len(arr)
    return pd.Series(out, index=x.index, name=f"{x.name}_pct" if x.name else None)


def zscore(x: pd.Series, window: int = 36) -> pd.Series:
    m = x.rolling(window, min_periods=window // 2).mean()
    s = x.rolling(window, min_periods=window // 2).std()
    return (x - m) / s


# --------------------------------------------------------------------------------------------
# 1. ATH proximity
# --------------------------------------------------------------------------------------------

def drawdown_from_ath(df: pd.DataFrame) -> pd.Series:
    return (df["price"] / df["ath"] - 1.0).rename("drawdown_from_ath")


def months_since_ath(df: pd.DataFrame, tol: float = 0.01) -> pd.Series:
    """Months since the month-end close was within ``tol`` of the running (daily) high."""
    at = (df["price"] >= df["ath"] * (1 - tol)).to_numpy()
    out = np.zeros(len(at))
    c = 0
    for i, a in enumerate(at):
        c = 0 if a else c + 1
        out[i] = c
    return pd.Series(out, index=df.index, name="months_since_ath")


def nearness_52w(df: pd.DataFrame) -> pd.Series:
    """Li & Yu x52: price / 52-week high (1 = at the 52-week high)."""
    return (df["price"] / df["high_52w"]).rename("nearness_52w")


def nearness_ath(df: pd.DataFrame) -> pd.Series:
    """Li & Yu xmax: price / historical high."""
    return (df["price"] / df["ath"]).rename("nearness_ath")


def near_ath_flag(df: pd.DataFrame, within: float = 0.02) -> pd.Series:
    return (drawdown_from_ath(df) >= -within).rename("near_ath")


# --------------------------------------------------------------------------------------------
# 2. Valuation
# --------------------------------------------------------------------------------------------

def cape(df: pd.DataFrame, years: int = 10) -> pd.Series:
    """Cyclically adjusted P/E using point-in-time CPI and 3-month-lagged earnings.

    Earnings at t are the trailing-12m figure as of t-3; deflate with CPI as of t-3 too
    (``cpi_pit`` is CPI(t-1), so CPI(t-3) = cpi_pit.shift(2))."""
    cpi_t = df["cpi_pit"]
    real_price = df["price"] / cpi_t
    real_earn = df["earn_pit"] / cpi_t.shift(2)
    avg = real_earn.rolling(12 * years, min_periods=12 * years).mean()
    return (real_price / avg).rename("cape")


def excess_cape_yield(df: pd.DataFrame) -> pd.Series:
    """Shiller's ECY: 1/CAPE minus the real 10y yield (10y nominal less trailing 10y CPI inflation).
    Higher = equities cheaper relative to bonds."""
    c = cape(df)
    infl10 = (df["cpi_pit"] / df["cpi_pit"].shift(120)) ** 0.1 - 1.0
    real10 = df["gs10"] / 100.0 - infl10
    return (1.0 / c - real10).rename("excess_cape_yield")


def dividend_yield(df: pd.DataFrame) -> pd.Series:
    return (df["div"] / df["price"]).rename("dividend_yield")


def market_cap_to_gdp(df: pd.DataFrame) -> pd.Series:
    return df["mcap_gdp"].rename("mcap_gdp")


def investor_equity_allocation(df: pd.DataFrame) -> pd.Series:
    return df["aiae"].rename("aiae")


# --------------------------------------------------------------------------------------------
# 3. Trend
# --------------------------------------------------------------------------------------------

def sma_signal(df: pd.DataFrame, months: int = 10) -> pd.Series:
    """True when the month-end price is BELOW its n-month simple moving average (risk-off)."""
    sma = df["price"].rolling(months, min_periods=months).mean()
    return (df["price"] < sma).where(sma.notna()).rename(f"below_sma{months}")


def tsmom(df: pd.DataFrame, months: int = 12) -> pd.Series:
    """Trailing n-month excess total return (Moskowitz-Ooi-Pedersen)."""
    ex = np.log1p(df["ret"]) - np.log1p(df["rf"])
    return ex.rolling(months, min_periods=months).sum().rename(f"tsmom{months}")


def ma_cross_signal(df: pd.DataFrame, short: int = 2, long: int = 12) -> pd.Series:
    """Neely et al. MA(s,l): True when MA(short) < MA(long) (risk-off)."""
    p = df["price"]
    s = p.rolling(short).mean()
    l = p.rolling(long).mean()
    return (s < l).where(l.notna()).rename(f"ma{short}_{long}_off")


def below_200d(df: pd.DataFrame) -> pd.Series:
    if "ma_200d" not in df:
        return pd.Series(np.nan, index=df.index, name="below_200d")
    return (df["price"] < df["ma_200d"]).where(df["ma_200d"].notna()).rename("below_200d")


def trend_off(df: pd.DataFrame) -> pd.Series:
    """Consensus trend break: at least 2 of {10m SMA, 12m TSMOM<0, MA(2,12)} risk-off."""
    votes = pd.concat([
        sma_signal(df, 10).astype(float),
        (tsmom(df, 12) < 0).astype(float).where(tsmom(df, 12).notna()),
        ma_cross_signal(df).astype(float),
    ], axis=1)
    return (votes.sum(axis=1, min_count=2) >= 2).where(votes.notna().sum(axis=1) >= 2).rename("trend_off")


# --------------------------------------------------------------------------------------------
# 4. Over-extension / bubble diagnostics
# --------------------------------------------------------------------------------------------

def runup(df: pd.DataFrame, months: int = 24) -> pd.Series:
    """Trailing n-month price run-up (Greenwood-Shleifer-You use 2 years)."""
    return (df["price"] / df["price"].shift(months) - 1.0).rename(f"runup_{months}m")


def acceleration(df: pd.DataFrame) -> pd.Series:
    """Recent 6m return minus a quarter of the 24m return: >0 when the run-up is speeding up."""
    r6 = df["price"] / df["price"].shift(6) - 1.0
    return (r6 - runup(df, 24) / 4.0).rename("acceleration")


def dist_from_200d(df: pd.DataFrame) -> pd.Series:
    if "ma_200d" not in df:
        ma = df["price"].rolling(10).mean()
    else:
        ma = df["ma_200d"].fillna(df["price"].rolling(10).mean())
    return (df["price"] / ma - 1.0).rename("dist_200d")


def bsadf(log_price: pd.Series, min_window: int = 36, max_window: int = 240, step: int = 3) -> pd.Series:
    """Backward sup-ADF statistic (Phillips, Shi & Yu 2015), ADF with no lags and a constant.

    For each end month t the statistic is the sup over start points s (t-max_window .. t-min_window,
    every ``step`` months) of the ADF t-stat on [s, t]. Computed in O(1) per window with
    cumulative sums. Positive large values signal explosive (bubble-like) behaviour."""
    y = log_price.to_numpy(dtype=float)
    n = len(y)
    out = np.full(n, np.nan)
    if n < min_window + 2:
        return pd.Series(out, index=log_price.index, name="bsadf")
    x = y[:-1]            # y_{t-1}
    z = np.diff(y)        # dy_t
    valid = ~(np.isnan(x) | np.isnan(z))
    x0 = np.where(valid, x, 0.0)
    z0 = np.where(valid, z, 0.0)
    c = lambda a: np.concatenate([[0.0], np.cumsum(a)])  # noqa: E731
    S1, Sx, Sz = c(valid.astype(float)), c(x0), c(z0)
    Sxx, Sxz, Szz = c(x0 * x0), c(x0 * z0), c(z0 * z0)
    for t in range(min_window, n - 1 + 1):
        j_end = t  # regression observations are indices [j0, j_end) of the diff arrays
        starts = np.arange(max(0, j_end - max_window), j_end - min_window + 1, step)
        if len(starts) == 0:
            continue
        m = S1[j_end] - S1[starts]
        sx = Sx[j_end] - Sx[starts]
        sz = Sz[j_end] - Sz[starts]
        sxx = Sxx[j_end] - Sxx[starts]
        sxz = Sxz[j_end] - Sxz[starts]
        szz = Szz[j_end] - Szz[starts]
        with np.errstate(invalid="ignore", divide="ignore"):
            vx = sxx - sx * sx / m
            b = (sxz - sx * sz / m) / vx
            a = (sz - b * sx) / m
            rss = szz - 2 * a * sz - 2 * b * sxz + m * a * a + 2 * a * b * sx + b * b * sxx
            s2 = rss / (m - 2)
            tstat = b / np.sqrt(s2 / vx)
        tstat = tstat[np.isfinite(tstat) & (m >= min_window - 1)]
        if len(tstat):
            out[t] = tstat.max()  # value for observation y[t]
    return pd.Series(out, index=log_price.index, name="bsadf")


def bsadf_critical_value(n_sims: int = 300, seed: int = 0, **kw) -> float:
    """95th percentile of the BSADF statistic under a driftless random walk (Monte Carlo)."""
    rng = np.random.default_rng(seed)
    max_window = kw.get("max_window", 240)
    stats = []
    for _ in range(n_sims):
        rw = pd.Series(np.cumsum(rng.standard_normal(max_window + 5)))
        stats.append(bsadf(rw, **kw).iloc[-1])
    return float(np.nanpercentile(stats, 95))


_BSADF_CV: Optional[float] = None


def bsadf_cv() -> float:
    """Cached 95% Monte Carlo critical value for the default BSADF configuration."""
    global _BSADF_CV
    if _BSADF_CV is None:
        _BSADF_CV = bsadf_critical_value()
    return _BSADF_CV


def real_log_price(df: pd.DataFrame) -> pd.Series:
    return np.log(df["price"] / df["cpi_pit"]).rename("real_log_price")


# --------------------------------------------------------------------------------------------
# 5. Risk
# --------------------------------------------------------------------------------------------

def realized_vol(df: pd.DataFrame) -> pd.Series:
    """Daily realised vol where available, else 6m rolling std of monthly returns (annualised)."""
    fallback = df["ret"].rolling(6, min_periods=6).std() * np.sqrt(12)
    if "rv_1m" in df:
        return df["rv_1m"].fillna(fallback).rename("realized_vol")
    return fallback.rename("realized_vol")


def variance_risk_premium(df: pd.DataFrame) -> pd.Series:
    """VIX^2 - realised variance (annualised, decimal^2). Available from 1990."""
    return ((df["vix"] / 100.0) ** 2 - realized_vol(df) ** 2).rename("vrp")


# --------------------------------------------------------------------------------------------
# 6. Macro / cycle
# --------------------------------------------------------------------------------------------

def term_spread(df: pd.DataFrame) -> pd.Series:
    return (df["gs10"] - df["tb3"]).rename("term_spread")


def curve_inverted_recently(df: pd.DataFrame, months: int = 12) -> pd.Series:
    """True if the 10y-3m spread has been negative at any point in the last n months
    (recessions typically follow inversions with a 6-18 month lag)."""
    inv = (term_spread(df) < 0).astype(float).where(term_spread(df).notna())
    return (inv.rolling(months, min_periods=1).max() > 0).where(term_spread(df).notna()).rename("curve_inverted_12m")


def credit_spread(df: pd.DataFrame) -> pd.Series:
    """Baa - Aaa corporate quality spread (since 1919)."""
    return (df["baa"] - df["aaa"]).rename("credit_spread")


def credit_widening(df: pd.DataFrame) -> pd.Series:
    """Spread rising: above its 12m average AND up over 6 months."""
    cs = credit_spread(df)
    flag = (cs > cs.rolling(12).mean()) & (cs.diff(6) > 0)
    return flag.where(cs.notna()).rename("credit_widening")


def unemployment_trend_bad(df: pd.DataFrame) -> pd.Series:
    """Unemployment (point-in-time) above its 12-month moving average."""
    u = df["unrate"]
    ma = u.rolling(12, min_periods=10).mean()
    return (u > ma).where(ma.notna() & u.notna()).rename("unemp_rising")


def sahm_on(df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    return (df["sahm"] >= threshold).where(df["sahm"].notna()).rename("sahm_on")


def cfnai_recession(df: pd.DataFrame) -> pd.Series:
    ma3 = df["cfnai"].rolling(3).mean()
    return (ma3 < -0.7).where(ma3.notna()).rename("cfnai_recession")


def nfci_tight(df: pd.DataFrame) -> pd.Series:
    return (df["nfci"] > 0).where(df["nfci"].notna()).rename("nfci_tight")


def macro_flag(df: pd.DataFrame) -> pd.Series:
    """Composite recession-risk flag, using whatever is available at each date.

    On if ANY of: unemployment above its 12m MA; Sahm rule triggered; CFNAI-MA3 < -0.7;
    or (curve inverted within 12m AND credit spreads widening). Before 1948 only the
    curve/credit leg exists."""
    parts = pd.concat([
        unemployment_trend_bad(df),
        sahm_on(df),
        cfnai_recession(df),
        (curve_inverted_recently(df).astype("boolean") & credit_widening(df).astype("boolean")),
    ], axis=1).astype(float)
    flag = parts.max(axis=1, skipna=True)
    return (flag > 0).where(parts.notna().any(axis=1)).rename("macro_flag")


# --------------------------------------------------------------------------------------------
# 7. Factor regime (diagnostic)
# --------------------------------------------------------------------------------------------

def factor_trailing(ff: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """Trailing n-month cumulative returns of SMB, HML and Mom (French data)."""
    cols = [c for c in ["SMB", "HML", "Mom"] if c in ff]
    return np.log1p(ff[cols]).rolling(months).sum().add_suffix(f"_{months}m")


# --------------------------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------------------------

def compute_all(df: pd.DataFrame, with_bsadf: bool = True, pct_min_periods: int = 240) -> pd.DataFrame:
    """All indicators in one frame plus expanding percentiles of the valuation measures."""
    out: Dict[str, pd.Series] = {
        "drawdown_from_ath": drawdown_from_ath(df),
        "months_since_ath": months_since_ath(df),
        "nearness_52w": nearness_52w(df),
        "nearness_ath": nearness_ath(df),
        "near_ath": near_ath_flag(df).astype(float),
        "cape": cape(df),
        "excess_cape_yield": excess_cape_yield(df),
        "dividend_yield": dividend_yield(df),
        "below_sma10": sma_signal(df, 10).astype(float),
        "tsmom12": tsmom(df, 12),
        "ma2_12_off": ma_cross_signal(df).astype(float),
        "trend_off": trend_off(df).astype(float),
        "runup_24m": runup(df, 24),
        "acceleration": acceleration(df),
        "dist_200d": dist_from_200d(df),
        "realized_vol": realized_vol(df),
        "term_spread": term_spread(df),
        "curve_inverted_12m": curve_inverted_recently(df).astype(float),
        "credit_spread": credit_spread(df),
        "credit_widening": credit_widening(df).astype(float),
        "macro_flag": macro_flag(df).astype(float),
    }
    for col, fn in [("mcap_gdp", market_cap_to_gdp), ("aiae", investor_equity_allocation)]:
        if col in df:
            out[col] = fn(df)
    if "unrate" in df:
        out["unemp_rising"] = unemployment_trend_bad(df).astype(float)
        out["sahm_on"] = sahm_on(df).astype(float)
    if "vix" in df:
        out["vix"] = df["vix"]
        out["vrp"] = variance_risk_premium(df)
    if "nfci" in df:
        out["nfci"] = df["nfci"]
    ind = pd.DataFrame(out)
    if with_bsadf:
        ind["bsadf"] = bsadf(real_log_price(df))
        ind["bsadf_flag"] = (ind["bsadf"] > bsadf_cv()).astype(float).where(ind["bsadf"].notna())
    for col in ["cape", "excess_cape_yield", "mcap_gdp", "aiae", "runup_24m", "dist_200d"]:
        if col in ind:
            ind[f"{col}_pct"] = expanding_percentile(ind[col], min_periods=pct_min_periods)
    # Valuation percentile: higher = more expensive. ECY is inverted (high ECY = cheap).
    val_parts = [ind.get("cape_pct")]
    if "excess_cape_yield_pct" in ind:
        val_parts.append(1.0 - ind["excess_cape_yield_pct"])
    ind["valuation_pct"] = pd.concat(val_parts, axis=1).mean(axis=1, skipna=True)
    return ind
