"""
Cash-allocation rules. Every function maps point-in-time indicators at month end *t* to a
*target cash weight* to hold during month t+1. Defaults respect the user's constraint band of
0-40% cash.

Rule families
-------------
* ``strategic_cash``    slow valuation layer (expanding-window valuation percentile).
* ``tactical_cash``     trend x macro ("growth-trend") layer plus an over-extension kicker.
* ``composite_cash``    strategic + tactical, clipped to [min_cash, max_cash].
* ``faber_cash``        10-month SMA rule mapped to {0, max_cash}.
* ``growth_trend_cash`` max_cash only when trend AND macro are both risk-off.
* ``bank_gains_cash``   state machine: trim to a cash target when near the ATH with expensive
                        valuations; redeploy after a drawdown or once valuations normalise.
* ``combination_forecast_cash`` Rapach-Strauss-Zhou mean of univariate OOS forecasts mapped to
                        a mean-variance equity weight (Campbell-Thompson restricted).
* ``horizon_adjustment`` / ``cycle_phase`` – life-stage and cycle overlays.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CashRule:
    min_cash: float = 0.0
    max_cash: float = 0.40
    # strategic (valuation) layer: linear ramp from val_lo to val_hi percentile
    strategic_max: float = 0.20
    val_lo: float = 0.60
    val_hi: float = 0.95
    # halve the valuation-driven cash while the trend is still up ("sin a little":
    # value timing works better when it does not fight momentum; Asness et al. 2017)
    val_needs_trend: bool = True
    val_trend_on_scale: float = 0.5
    # tactical layer
    tactical_both: float = 0.20     # trend off AND macro flag
    tactical_trend: float = 0.10    # trend off only
    tactical_macro: float = 0.05    # macro flag only
    # over-extension kicker when near ATH
    overext_cash: float = 0.05
    overext_runup_pct: float = 0.90
    near_ath_within: float = 0.05
    # optional volatility cap: extra cash when realised vol is far above normal
    vol_cap: Optional[float] = None   # e.g. 0.25 annualised
    vol_extra: float = 0.10
    name: str = "composite"


def strategic_cash(ind: pd.DataFrame, rule: CashRule = CashRule()) -> pd.Series:
    v = ind["valuation_pct"]
    ramp = ((v - rule.val_lo) / (rule.val_hi - rule.val_lo)).clip(0.0, 1.0)
    cash = rule.strategic_max * ramp
    if rule.val_needs_trend and "trend_off" in ind:
        scale = np.where(ind["trend_off"].fillna(0) > 0, 1.0, rule.val_trend_on_scale)
        cash = cash * scale
    return cash.fillna(0.0).rename("strategic_cash")


def tactical_cash(ind: pd.DataFrame, rule: CashRule = CashRule()) -> pd.Series:
    trend = ind["trend_off"].fillna(0) > 0
    macro = ind["macro_flag"].fillna(0) > 0
    cash = np.select([trend & macro, trend, macro], [rule.tactical_both, rule.tactical_trend, rule.tactical_macro], 0.0)
    cash = pd.Series(cash, index=ind.index)
    if rule.overext_cash > 0:
        near = ind["drawdown_from_ath"] >= -rule.near_ath_within
        hot = (ind.get("runup_24m_pct", pd.Series(0, index=ind.index)).fillna(0) >= rule.overext_runup_pct) | (
            ind.get("bsadf_flag", pd.Series(0, index=ind.index)).fillna(0) > 0)
        cash = cash + np.where(near & hot, rule.overext_cash, 0.0)
    if rule.vol_cap is not None:
        cash = cash + np.where(ind["realized_vol"].fillna(0) > rule.vol_cap, rule.vol_extra, 0.0)
    return cash.rename("tactical_cash")


def composite_cash(ind: pd.DataFrame, rule: CashRule = CashRule()) -> pd.Series:
    c = strategic_cash(ind, rule) + tactical_cash(ind, rule)
    return c.clip(rule.min_cash, rule.max_cash).rename(rule.name)


def explain(ind_row: pd.Series, rule: CashRule = CashRule()) -> Dict[str, float]:
    """Layer-by-layer breakdown for one date (used by the CLI)."""
    one = ind_row.to_frame().T
    s = float(strategic_cash(one, rule).iloc[0])
    t = float(tactical_cash(one, rule).iloc[0])
    total = float(np.clip(s + t, rule.min_cash, rule.max_cash))
    return {"strategic": s, "tactical": t, "total": total}


def faber_cash(ind: pd.DataFrame, cash_when_off: float = 0.40) -> pd.Series:
    return (ind["below_sma10"].fillna(0) * cash_when_off).rename("faber_sma10")


def trend_only_cash(ind: pd.DataFrame, cash_when_off: float = 0.40) -> pd.Series:
    return (ind["trend_off"].fillna(0) * cash_when_off).rename("trend_only")


def valuation_only_cash(ind: pd.DataFrame, max_cash: float = 0.40, lo: float = 0.5, hi: float = 0.95) -> pd.Series:
    ramp = ((ind["valuation_pct"] - lo) / (hi - lo)).clip(0, 1)
    return (max_cash * ramp).fillna(0.0).rename("valuation_only")


def growth_trend_cash(ind: pd.DataFrame, cash_when_off: float = 0.40) -> pd.Series:
    off = (ind["trend_off"].fillna(0) > 0) & (ind["macro_flag"].fillna(0) > 0)
    return (off.astype(float) * cash_when_off).rename("growth_trend")


def bank_gains_cash(
    ind: pd.DataFrame,
    trim_to: float = 0.20,
    near_within: float = 0.02,
    val_threshold: float = 0.80,
    redeploy_drawdown: float = 0.10,
    redeploy_val: float = 0.60,
) -> pd.Series:
    """Opportunistic 'bank some gains' rule.

    Enter (hold ``trim_to`` cash) when the market is within ``near_within`` of its ATH AND the
    valuation percentile is >= ``val_threshold``. Stay trimmed until EITHER the market has fallen
    ``redeploy_drawdown`` from its ATH (buy the dip) OR valuations drop below ``redeploy_val``."""
    dd = ind["drawdown_from_ath"].to_numpy()
    val = ind["valuation_pct"].to_numpy()
    out = np.zeros(len(ind))
    state = False
    for i in range(len(ind)):
        if np.isnan(val[i]):
            state = False
        elif not state:
            state = dd[i] >= -near_within and val[i] >= val_threshold
        else:
            if dd[i] <= -redeploy_drawdown or val[i] < redeploy_val:
                state = False
        out[i] = trim_to if state else 0.0
    return pd.Series(out, index=ind.index, name="bank_gains")


def combination_forecast_cash(
    df: pd.DataFrame,
    ind: pd.DataFrame,
    predictors: Sequence[Tuple[str, int]] = (
        ("excess_cape_yield", +1), ("term_spread", +1), ("trend_off", -1), ("curve_inverted_12m", -1),
        ("nearness_52w", +1), ("nearness_ath", -1), ("credit_spread", +1), ("dividend_yield", +1),
    ),
    risk_aversion: float = 3.0,
    min_equity: float = 0.60,
    max_equity: float = 1.00,
    oos_start: str = "1940-01-31",
    vol_window: int = 60,
) -> pd.Series:
    """Rapach, Strauss & Zhou (2010) combination (equal-weight mean) of univariate OOS
    forecasts of next-month excess return -> mean-variance weight f / (gamma * sigma^2)."""
    from .evaluate import forward_excess, oos_r2

    y = forward_excess(df, 1)
    fcs = []
    for name, sign in predictors:
        if name not in ind:
            continue
        res = oos_r2(ind[name], y, h=1, oos_start=oos_start, min_train=120, ct_restrict=True, expected_sign=sign)
        if len(res.forecasts):
            fcs.append(res.forecasts.rename(name))
    f = pd.concat(fcs, axis=1).mean(axis=1)
    xr = np.log1p(df["ret"]) - np.log1p(df["rf"])
    var = xr.rolling(vol_window, min_periods=24).var()
    w = (f / (risk_aversion * var.reindex(f.index))).clip(min_equity, max_equity)
    cash = (1.0 - w).reindex(ind.index)
    return cash.fillna(0.0).rename("combination_forecast")


# --------------------------------------------------------------------------------------------
# Cycle phase and horizon overlays
# --------------------------------------------------------------------------------------------

PHASES = ("early", "mid", "late", "contraction")


def cycle_phase(ind: pd.DataFrame) -> pd.Series:
    """Heuristic business/market-cycle phase from point-in-time indicators.

    contraction : trend off AND macro flag on (growth-trend 'risk-off')
    early       : within 24 months of a >= 20% drawdown trough, trend back on, macro flag off
    late        : trend on but (curve inverted within 12m OR valuation pct >= 0.85 with
                  unemployment rising / credit widening)
    mid         : everything else
    """
    trend = ind["trend_off"].fillna(0) > 0
    macro = ind["macro_flag"].fillna(0) > 0
    deep_dd_recent = (ind["drawdown_from_ath"] <= -0.20).astype(float).rolling(24, min_periods=1).max() > 0
    inverted = ind["curve_inverted_12m"].fillna(0) > 0
    stress = (ind.get("unemp_rising", pd.Series(0, index=ind.index)).fillna(0) > 0) | (ind["credit_widening"].fillna(0) > 0)
    rich = ind["valuation_pct"].fillna(0.5) >= 0.85
    phase = np.select(
        [trend & macro, deep_dd_recent & ~trend & ~macro, ~trend & (inverted | (rich & stress))],
        ["contraction", "early", "late"],
        "mid",
    )
    return pd.Series(phase, index=ind.index, name="phase")


# Suggested cash band by phase within the 0-40% envelope (see report, section 6).
PHASE_BANDS: Dict[str, Tuple[float, float]] = {
    "early": (0.00, 0.05),
    "mid": (0.00, 0.15),
    "late": (0.10, 0.30),
    "contraction": (0.25, 0.40),
}


def horizon_adjustment(years_to_need: float, phase: str = "mid", annual_withdrawal_pct: float = 0.0) -> Dict[str, object]:
    """Life-stage overlay. Returns the allowed cash ``envelope``, a ``scale`` applied to the
    rule's timing tilt, the phase band and a note.

    * < 2 years: money needed soon should not be in equities at all (liability matching).
    * 2-5 years from a planned draw: keep a withdrawal 'bucket' (2-3 years of draws) in cash /
      short gilts and apply the rule to the rest. Sequence-of-returns risk peaks around the start
      of withdrawals (Pfau & Kitces 2014), so the floor rises.
    * 5-20 years: full 0-40% envelope, tilt as the rule says.
    * > 20 years: halve the tilt (0-20% envelope). Cash drag compounds, lifecycle evidence favours
      high equity exposure for the young (Anarkulova, Cederburg & O'Doherty), and parameter
      uncertainty argues for modest timing (Barberis 2000).
    """
    lo, hi = PHASE_BANDS.get(phase, PHASE_BANDS["mid"])
    if years_to_need < 2:
        return {"envelope": (1.0, 1.0), "scale": 1.0, "phase_band": (1.0, 1.0),
                "note": "Funds needed within ~2 years: hold in cash / short-dated gilts, not equities."}
    if years_to_need < 5:
        bucket = min(0.40, max(0.10, 2.5 * annual_withdrawal_pct))
        return {"envelope": (bucket, 0.40), "scale": 1.0, "phase_band": (max(lo, bucket), max(hi, bucket)),
                "note": f"Withdrawal bucket: keep >= {bucket:.0%} (2-3 years of draws) in cash, then apply the rule."}
    if years_to_need <= 20:
        return {"envelope": (0.0, 0.40), "scale": 1.0, "phase_band": (lo, hi),
                "note": "Full 0-40% envelope; follow the rule."}
    return {"envelope": (0.0, 0.20), "scale": 0.5, "phase_band": (lo / 2, hi / 2),
            "note": "Long horizon (>20y): tilts halved; staying invested matters more than timing."}


# --------------------------------------------------------------------------------------------
# Strategy set
# --------------------------------------------------------------------------------------------

def strategy_set(df: pd.DataFrame, ind: pd.DataFrame, include_combo: bool = True) -> Dict[str, pd.Series]:
    """The pre-registered set of strategies compared in the backtest (target cash weights)."""
    base = CashRule()
    s = {
        "buy_hold": pd.Series(0.0, index=ind.index, name="buy_hold"),
        "valuation_only": valuation_only_cash(ind),
        "faber_sma10": faber_cash(ind),
        "trend_only": trend_only_cash(ind),
        "growth_trend": growth_trend_cash(ind),
        "bank_gains": bank_gains_cash(ind),
        "composite": composite_cash(ind, base),
        "composite_volcap": composite_cash(ind, replace(base, vol_cap=0.25, name="composite_volcap")),
    }
    if include_combo:
        s["combination_forecast"] = combination_forecast_cash(df, ind)
    return s
