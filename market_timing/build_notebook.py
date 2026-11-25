"""
Generate ``market_top_analysis.ipynb`` (same nbformat pattern as
``.claude/commands/deep-stock-analyzer/scripts/generate_notebook.py``).

    python -m market_timing.build_notebook [--output market_top_analysis.ipynb]
    jupyter nbconvert --execute --to notebook --inplace market_top_analysis.ipynb
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from .data import REPO_ROOT


def build() -> nbformat.NotebookNode:
    today = datetime.now().strftime("%Y-%m-%d")
    cells = []
    cells.append(new_markdown_cell(f"""# Markets near all-time highs: indicators & cash allocation
**Generated:** {today} · companion to `research/market_highs_cash_allocation.md`

This notebook reproduces every number in the report from the `market_timing` package:
1. Data panel (S&P 500 1871–today, point-in-time).
2. Which indicators actually predict returns / drawdowns (in-sample, out-of-sample, AUC).
3. What happens after the market is at or near its all-time high, by valuation.
4. Backtests of cash rules within a 0–40% cash envelope, vs a static mix holding the *same average cash*.
5. Cycle phases, global robustness, and today's dashboard.

> Educational research, not financial advice. Past relationships can break.
"""))
    cells.append(new_code_cell("""import warnings, logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.ticker as mtick
from market_timing import data as D, indicators as I, evaluate as E, allocation as A, backtest as B
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 3)

# Reference palette (validated categorical slots 1-3; text/surface tokens)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
    "figure.dpi": 110,
})

df_all = D.us_monthly()
df = df_all[df_all["complete"]]
ind = I.compute_all(df)
print(f"Panel {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} months); latest partial month: {df_all.index[-1].date()}")
df.tail(3)"""))

    cells.append(new_markdown_cell("""## 1. The market relative to its all-time high
Shaded bands are NBER recessions (ex-post dates, shown for context only, never used as a signal)."""))
    cells.append(new_code_cell("""def shade_recessions(ax, s):
    rec = df["usrec"].reindex(s.index).fillna(0)
    on = rec.diff().fillna(rec.iloc[0]) == 1; off = rec.diff() == -1
    for a, b in zip(rec.index[on], list(rec.index[off]) + [rec.index[-1]] * (on.sum() - off.sum())):
        ax.axvspan(a, b, color="#d9d8d2", alpha=0.6, lw=0)

sl = slice("1928", None)
fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
real = (df["price"] / df["cpi_pit"])[sl]
shade_recessions(ax[0], real); ax[0].plot(real.index, real, color=C1)
ax[0].set_yscale("log"); ax[0].set_title("S&P 500, real (CPI-deflated) price, log scale"); ax[0].set_ylabel("Real index")
dd = ind["drawdown_from_ath"][sl]
shade_recessions(ax[1], dd); ax[1].fill_between(dd.index, dd, 0, color=C2, alpha=0.85, lw=0)
ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax[1].set_title("Drawdown from nominal all-time high")
plt.tight_layout(); plt.show()
print(f"Share of months since 1928 within 2% of the ATH: {(dd >= -0.02).mean():.0%}; within 5%: {(dd >= -0.05).mean():.0%}")"""))

    cells.append(new_markdown_cell("""## 2. Indicator scorecard
* `beta/t/R2` – in-sample predictive regression of forward excess log returns (Newey-West t).
* `R2oos` – Campbell-Thompson out-of-sample R² (expanding window from 1950, sign-restricted); `CW_p` – Clark-West p-value.
* `AUC_dd20_12m` – how well the signal (oriented so high = risky) flags a ≥20% drawdown within 12 months (0.5 = coin flip); `hit_top` – drawdown frequency when the signal is in its riskiest quintile / on, vs `base_rate`."""))
    cells.append(new_code_cell("""score = E.evaluate_signals(df, ind)
score.drop(columns=["first"]).sort_values(["family", "AUC_dd20_12m"], ascending=[True, False])"""))

    cells.append(new_markdown_cell("""## 3. What happens after all-time highs?
Mean forward excess return (12/36/60-month figures annualised) and the probability of a ≥20% drawdown within 12 months, by distance from the ATH at month end, split by the composite valuation percentile (terciles of CAPE & excess-CAPE-yield percentiles). Sample 1928–today; overlapping windows, so treat differences of a few % as noise."""))
    cells.append(new_code_cell("""ath = E.ath_bucket_table(df, ind["drawdown_from_ath"], split=ind["valuation_pct"])
display(ath[ath.group == "All"].drop(columns="group"))
piv = ath[ath.group != "All"].pivot(index="bucket", columns="group", values="fwd_12m").reindex([b[0] for b in E.ATH_BUCKETS])
pdd = ath[ath.group != "All"].pivot(index="bucket", columns="group", values="p_dd20_12m").reindex(piv.index)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, tab, title in [(axs[0], piv, "Next-12m excess return"), (axs[1], pdd, "P(>=20% drawdown within 12m)")]:
    x = np.arange(len(tab)); w = 0.26
    for j, (g, col) in enumerate(zip(["Cheap", "Fair", "Expensive"], [C3, C1, C2])):
        ax.bar(x + (j - 1) * w, tab[g], w - 0.02, color=col, label=g)
    ax.set_xticks(x); ax.set_xticklabels(tab.index, rotation=25, ha="right"); ax.set_title(title)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax.axhline(0, color=MUTED, lw=0.8)
axs[0].legend(title="Valuation tercile"); plt.tight_layout(); plt.show()
ath[ath.group != "All"]"""))

    cells.append(new_markdown_cell("""## 4. Valuation regime: CAPE and excess CAPE yield
Expanding-window percentiles (no look-ahead). The strategic cash layer ramps from 0% at the 60th to 20% at the 95th composite valuation percentile."""))
    cells.append(new_code_cell("""fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].plot(ind["cape"].index, ind["cape"], color=C1); ax[0].set_title("CAPE (point-in-time, extended past 2023 with NIPA profits)")
ax[0].axhline(ind["cape"].median(), color=MUTED, ls="--", lw=1); ax[0].annotate("median", (ind.index[40], ind["cape"].median()), color=INK2, va="bottom")
ax[1].plot(ind["excess_cape_yield"].index, ind["excess_cape_yield"], color=C2); ax[1].set_title("Excess CAPE yield (1/CAPE - real 10y yield)")
ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax[1].axhline(0, color=MUTED, lw=0.8)
plt.tight_layout(); plt.show()
ind[["cape", "cape_pct", "excess_cape_yield", "excess_cape_yield_pct", "mcap_gdp_pct", "aiae_pct", "valuation_pct"]].dropna(how="all").tail(3)"""))

    cells.append(new_markdown_cell("""## 5. Backtests (0–40% cash, monthly, 5pp no-trade band, 10bp costs, cash earns T-bills)
`Matched_*` = a static mix holding the strategy's *average* cash every month. Beating buy-and-hold on risk is trivial with cash; beating the matched mix is the real test. `composite_walkforward` re-selects the composite parameters each decade from 1960 using only prior data."""))
    cells.append(new_code_cell("""study = B.run_study(n_boot=1000, save=True)
cols = ["CAGR", "Vol", "Sharpe", "MaxDD", "Calmar", "Ulcer", "AvgCash", "Turnover/yr", "Matched_CAGR", "Matched_Sharpe", "Matched_MaxDD", "dCAGR_vs_matched", "dSharpe_vs_matched", "DownCapture"]
print("Full sample 1930-today"); display(study["full"][cols])
print("1960-today (incl. walk-forward)"); display(study["post1960"][cols])"""))
    cells.append(new_code_cell("""eqc = study["equity_curves"].loc["1960":]
eqc = eqc / eqc.iloc[0]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for name, col in [("buy_hold", MUTED), ("trend_only", C2), ("composite", C1)]:
    ax[0].plot(eqc.index, eqc[name], color=col, label=name, lw=2 if name == "composite" else 1.6)
    ddc = eqc[name] / eqc[name].cummax() - 1
    ax[1].plot(ddc.index, ddc, color=col, label=name, lw=1.4)
ax[0].set_yscale("log"); ax[0].set_title("Growth of 1 since 1960 (log)"); ax[0].legend()
ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax[1].set_title("Drawdown"); ax[1].legend(loc="lower left")
plt.tight_layout(); plt.show()
cp = study["cash_paths"]["composite"].loc["1960":]
fig, ax = plt.subplots(figsize=(12, 2.6)); ax.fill_between(cp.index, cp, 0, step="post", color=C1, alpha=0.8, lw=0)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax.set_ylim(0, 0.42); ax.set_title("Composite rule: cash weight held"); plt.tight_layout(); plt.show()"""))
    cells.append(new_code_cell("""print("Sub-periods"); display(study["subperiods"][["period", "CAGR", "Sharpe", "MaxDD", "AvgCash", "dCAGR_vs_matched", "dSharpe_vs_matched"]])
print("Bootstrap (stationary, 12m blocks) of Sharpe difference vs matched static; deflated Sharpe of active returns across all trials")
display(study["bootstrap"])
print("Walk-forward parameter choices"); display(study["walkforward_params"])"""))

    cells.append(new_markdown_cell("""## 6. Cycle phases and global robustness
Phase is classified point-in-time (see `allocation.cycle_phase`). The global check reuses the same logic on Ken French country/region indices (1985+, dividend-yield valuation, term-spread inversion as the macro leg)."""))
    cells.append(new_code_cell("""display(study["phases"])
display(study["global"][["region", "CAGR", "Sharpe", "MaxDD", "AvgCash", "Matched_MaxDD", "dCAGR_vs_matched", "dSharpe_vs_matched"]])"""))

    cells.append(new_markdown_cell("""## 7. Today's dashboard
Uses the latest (possibly partial) month and the latest *published* value of every input."""))
    cells.append(new_code_cell("""from market_timing import signal_today as S
r = S.recommend(portfolio_value=100_000, current_cash=0.0, years_to_need=10)
print(f"As of {r['as_of']} (partial month: {r['partial_month']}), S&P 500 {r['last_price']:,.0f}")
display(r["dashboard"])
print(r["layers"], "| phase:", r["phase"], "| recommended cash:", f"{r['recommended_cash']:.0%}")"""))

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(REPO_ROOT / "market_top_analysis.ipynb"))
    args = ap.parse_args()
    nb = build()
    Path(args.output).write_text(nbformat.writes(nb), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
