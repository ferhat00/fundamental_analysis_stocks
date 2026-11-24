"""
Today's market-top dashboard and recommended cash weight.

    python -m market_timing.signal_today [--portfolio-value 250000] [--current-cash 0.05]
                                         [--years-to-need 20] [--annual-withdrawal 0.0]

Prints every indicator with its historical percentile and state, the layered cash
recommendation (strategic valuation + tactical trend/macro + over-extension), the cycle-phase
estimate and the horizon overlay. The current (partial) month is included, using the latest
published value of each input; the backtests use completed months only.

Not financial advice.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import allocation as A
from . import data as D
from . import indicators as I


def _pct(ind: pd.DataFrame, col: str) -> float:
    """Expanding percentile of the latest value of ``col`` (1928+ history for daily-era stats)."""
    s = ind[col].dropna()
    if s.empty:
        return float("nan")
    v = s.iloc[-1]
    return float((s < v).mean() + 0.5 * (s == v).mean())


def _fmt(v: float, kind: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if kind == "pct":
        return f"{v:+.1%}" if abs(v) < 10 else f"{v:.1f}"
    if kind == "flag":
        return "ON" if v > 0 else "off"
    if kind == "x":
        return f"{v:.2f}"
    return f"{v:.1f}"


ROWS = [
    # (label, column, kind, risk-off condition description, risk test)
    ("ATH proximity", None, None, None, None),
    ("Drawdown from ATH", "drawdown_from_ath", "pct", "near ATH = not a sell signal by itself", lambda v: v >= -0.02),
    ("Months since ATH", "months_since_ath", "num", "", None),
    ("Price / 52-week high", "nearness_52w", "x", "low -> trend damage", lambda v: v < 0.90),
    ("Valuation", None, None, None, None),
    ("CAPE (Shiller P/E10)", "cape", "num", "high pct -> lower 10y returns", lambda v: False),
    ("Excess CAPE yield", "excess_cape_yield", "pct", "low -> equities rich vs bonds", lambda v: v < 0.01),
    ("Market cap / GDP", "mcap_gdp", "x", "high pct -> drawdown risk", lambda v: False),
    ("Investor equity allocation", "aiae", "x", "high pct -> lower 10y returns", lambda v: False),
    ("Composite valuation pct", "valuation_pct", "x", ">= 0.60 starts the strategic layer", lambda v: v >= 0.6),
    ("Trend", None, None, None, None),
    ("Below 10-month SMA", "below_sma10", "flag", "Faber (2007)", lambda v: v > 0),
    ("12m TSMOM (excess)", "tsmom12", "pct", "<0 = risk-off", lambda v: v < 0),
    ("MA(2) < MA(12)", "ma2_12_off", "flag", "Neely et al. (2014)", lambda v: v > 0),
    ("Trend consensus (2 of 3)", "trend_off", "flag", "tactical layer input", lambda v: v > 0),
    ("Over-extension", None, None, None, None),
    ("24m run-up", "runup_24m", "pct", "top decile near ATH -> +5% cash", lambda v: False),
    ("Distance from 200-day MA", "dist_200d", "pct", "", None),
    ("BSADF bubble statistic", "bsadf", "x", "above MC 95% cv = explosive", lambda v: v > I.bsadf_cv()),
    ("Risk", None, None, None, None),
    ("Realised vol (1m, ann.)", "realized_vol", "pct", "> 25% = stress", lambda v: v > 0.25),
    ("VIX", "vix", "num", "> 30 = stress", lambda v: v > 30),
    ("Macro / cycle", None, None, None, None),
    ("10y - 3m term spread", "term_spread", "x", "inverted in last 12m -> late cycle", lambda v: v < 0),
    ("Curve inverted in last 12m", "curve_inverted_12m", "flag", "", lambda v: v > 0),
    ("Baa - Aaa credit spread", "credit_spread", "x", "", None),
    ("Credit spreads widening", "credit_widening", "flag", "", lambda v: v > 0),
    ("Unemployment > 12m avg", "unemp_rising", "flag", "growth-trend recession signal", lambda v: v > 0),
    ("Sahm rule triggered", "sahm_on", "flag", "", lambda v: v > 0),
    ("Chicago Fed NFCI", "nfci", "x", "> 0 = tighter than average", lambda v: v > 0),
    ("Macro recession flag", "macro_flag", "flag", "tactical layer input", lambda v: v > 0),
]


def build_dashboard(df: pd.DataFrame, ind: pd.DataFrame) -> pd.DataFrame:
    last = ind.iloc[-1]
    rows = []
    for label, col, kind, note, test in ROWS:
        if col is None:
            rows.append({"indicator": f"-- {label} --", "value": "", "hist_pct": "", "state": "", "note": ""})
            continue
        if col not in ind:
            continue
        v = last.get(col, np.nan)
        if (v is None or pd.isna(v)) and ind[col].notna().any():
            v = ind[col].dropna().iloc[-1]  # latest published reading
        pct = _pct(ind, col) if kind not in ("flag",) else np.nan
        state = ""
        if test is not None and not pd.isna(v):
            try:
                state = "RISK" if test(float(v)) else "ok"
            except Exception:  # noqa: BLE001
                state = ""
        if col in ("cape", "mcap_gdp", "aiae", "runup_24m") and not np.isnan(pct):
            state = "RISK" if pct >= 0.90 else ("elevated" if pct >= 0.75 else "ok")
        rows.append({"indicator": label, "value": _fmt(float(v), kind) if not pd.isna(v) else "n/a",
                     "hist_pct": "" if np.isnan(pct) else f"{pct:.0%}", "state": state, "note": note})
    return pd.DataFrame(rows)


def recommend(
    portfolio_value: Optional[float] = None,
    current_cash: Optional[float] = None,
    years_to_need: float = 10.0,
    annual_withdrawal: float = 0.0,
    rule: A.CashRule = A.CashRule(),
) -> dict:
    df = D.us_monthly()
    ind = I.compute_all(df)
    last = ind.iloc[-1].copy()
    # use latest published value for slow inputs that are blank in the partial month
    for c in ("valuation_pct", "macro_flag", "trend_off", "runup_24m_pct", "bsadf_flag", "curve_inverted_12m",
              "credit_widening", "unemp_rising"):
        if c in ind and pd.isna(last.get(c)) and ind[c].notna().any():
            last[c] = ind[c].dropna().iloc[-1]
    layers = A.explain(last, rule)
    phase = A.cycle_phase(ind.ffill(limit=2)).iloc[-1]
    hz = A.horizon_adjustment(years_to_need, phase, annual_withdrawal)
    lo, hi = hz["envelope"]
    rec = float(np.clip(layers["total"] * hz["scale"], lo, hi)) if years_to_need >= 2 else 1.0
    out = {
        "as_of": df.index[-1].date(),
        "partial_month": not bool(df["complete"].iloc[-1]),
        "last_price": float(df["price"].iloc[-1]),
        "dashboard": build_dashboard(df, ind),
        "layers": layers,
        "phase": phase,
        "phase_band": A.PHASE_BANDS.get(phase),
        "horizon": hz,
        "recommended_cash": rec,
        "growth_trend_cash": float(A.growth_trend_cash(ind.ffill(limit=2)).iloc[-1]),
        "faber_cash": float(A.faber_cash(ind).iloc[-1]),
    }
    if portfolio_value is not None:
        out["target_cash_value"] = rec * portfolio_value
        if current_cash is not None:
            out["trade_value"] = (rec - current_cash) * portfolio_value
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Market-top dashboard and recommended cash %")
    ap.add_argument("--portfolio-value", type=float, default=None)
    ap.add_argument("--current-cash", type=float, default=None, help="current cash weight, e.g. 0.05")
    ap.add_argument("--years-to-need", type=float, default=10.0, help="years until you need to draw on the money")
    ap.add_argument("--annual-withdrawal", type=float, default=0.0, help="planned annual withdrawal as % of portfolio")
    ap.add_argument("--currency", default="£")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    r = recommend(args.portfolio_value, args.current_cash, args.years_to_need, args.annual_withdrawal)
    print(f"\nS&P 500 market-top dashboard as of {r['as_of']}"
          f"{' (month in progress)' if r['partial_month'] else ''}  |  S&P 500 = {r['last_price']:,.0f}\n")
    with pd.option_context("display.width", 200, "display.max_colwidth", 48, "display.max_rows", 100):
        print(r["dashboard"].to_string(index=False))
    L = r["layers"]
    print("\nCash recommendation (0-40% envelope)")
    print(f"  Strategic (valuation) layer : {L['strategic']:.0%}")
    print(f"  Tactical (trend x macro +   : {L['tactical']:.0%}")
    print(f"            over-extension)")
    print(f"  Composite rule              : {L['total']:.0%}")
    print(f"  Cycle phase estimate        : {r['phase']}  (suggested band {r['phase_band'][0]:.0%}-{r['phase_band'][1]:.0%})")
    print(f"  Horizon overlay             : {r['horizon']['note']}")
    print(f"  ==> Recommended cash        : {r['recommended_cash']:.0%}")
    print(f"  (cross-checks: growth-trend rule {r['growth_trend_cash']:.0%}, Faber 10m SMA {r['faber_cash']:.0%})")
    cur = args.currency
    if "target_cash_value" in r:
        print(f"  Target cash value           : {cur}{r['target_cash_value']:,.0f}")
    if "trade_value" in r:
        tv = r["trade_value"]
        verb = "Move to cash" if tv > 0 else "Redeploy into equities"
        if abs(tv) < 0.05 * (args.portfolio_value or 0):
            print(f"  Trade                       : none (within the 5pp no-trade band; gap {cur}{tv:,.0f})")
        else:
            print(f"  Trade                       : {verb} {cur}{abs(tv):,.0f}")
    if r["partial_month"]:
        print("\n  NOTE: the month is still in progress. Trend and volatility readings use the latest close and are"
              "\n  provisional; the rules (and the backtests) act on month-end values only.")
    print("\nMonthly review; trades only when the target moves >= 5pp. Educational tool, not financial advice.\n")


if __name__ == "__main__":
    main()
