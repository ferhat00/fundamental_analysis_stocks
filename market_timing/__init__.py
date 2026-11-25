"""
market_timing — indicators and cash-allocation rules for equity markets near all-time highs.

Modules
-------
data        Cached loaders (Shiller, FRED, Ken French, Yahoo) assembled into one monthly,
            point-in-time (publication-lagged) panel.
indicators  One function per indicator family (ATH proximity, valuation, trend,
            over-extension, risk, macro, factor regime).
evaluate    Predictive regressions, out-of-sample R^2, Clark-West test, drawdown AUC,
            conditional returns by distance from the all-time high.
allocation  The layered cash rule (strategic valuation + tactical trend/macro + over-extension),
            combination-forecast alternative, horizon/cycle adjustment.
backtest    Monthly simulation, metrics, matched-cash benchmarks, bootstrap, deflated Sharpe.
signal_today  CLI that prints today's dashboard and recommended cash %.

Nothing here is financial advice; see research/market_highs_cash_allocation.md.
"""

__all__ = ["data", "indicators", "evaluate", "allocation", "backtest"]
