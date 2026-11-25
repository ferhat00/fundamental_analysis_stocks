# Banking gains near all-time highs: what the evidence says about indicators and cash allocation

*Research note · data to November 2026 · companion code in [`market_timing/`](../market_timing) and notebook [`market_top_analysis.ipynb`](../market_top_analysis.ipynb)*

> **Not financial advice.** This is an educational review of published research and historical
> backtests. Past relationships can and do break. Every number below was produced by the code
> in this repository (`python -m market_timing.backtest`); tables are saved to `output/market_timing/`.

---

## 1. Executive summary

**Question.** When the market is at or near an all-time high (ATH), which indicators tell you to
bank some gains into cash, and how much cash should you hold, given a 0–40% cash band, monthly
reviews, and an ISA (no tax on trimming)? How should that change with your horizon and with the
market cycle?

**Answer in seven points.**

1. **An all-time high is not, by itself, a sell signal.** Since 1928 the S&P 500 has closed within
   2% of its high in nearly a quarter of all months. Forward returns from those months are close to
   normal: about 5.6% a year excess return over the next 12 months, with a 13% chance of a ≥20%
   drawdown. That is *lower* than the 23% chance when the market is already more than 20% below its
   high. This matches Li & Yu (2012): being near the 52-week high predicts *higher* returns
   (anchoring and under-reaction). The negative effect of being near the *historical* high is
   weak and did not hold up out of sample in our tests. Selling purely because of a new high (our
   "bank gains" rule) slightly **under-performed** a static mix with the same average cash.
2. **ATHs matter only when combined with valuation.** In the most expensive third of history:
   - ATH months were followed by roughly **−2% a year** excess return over the next 3 years and
     about **+1% a year** over 5 years.
   - The same position in the cheapest or fair-value thirds gave about 6% a year over 3 years.
   - The worst case was expensive markets that had *already* broken down (>20% below the ATH): a
     42% chance of a further ≥20% fall within a year.
3. **Valuation sets the size of the long-run tilt; it does not time the exit.** The excess CAPE
   yield (Shiller's valuation of equities against real bond yields) was the best single
   predictor:
   - 12-month out-of-sample R² of +3.6% (Clark-West p = 0.003).
   - AUC of 0.68 for predicting a ≥20% drawdown.

   CAPE, market cap/GDP and the investors' equity-allocation ratio help mainly at multi-year
   horizons. This is the classic result (Campbell & Shiller 1998; Goyal & Welch 2008; Asness,
   Ilmanen & Maloney 2017): useful for setting expectations, weak for timing.
4. **Trend does the timing.** Three trend rules were combined, and the market counts as
   "trend-off" when at least two agree:
   - a 10-month moving average (Faber 2007);
   - 12-month time-series momentum (Moskowitz, Ooi & Pedersen 2012);
   - the MA(2,12) crossover (Neely, Rapach, Tu & Zhou 2014).

   This gave the largest risk-adjusted improvement. Against a static mix holding the *same
   average cash*, Sharpe was +0.09 (bootstrap 95% CI +0.04 to +0.14) and the maximum drawdown fell
   from −74% to −58% (1930–2026). Adding a **recession confirmation** helps further. The flag is on
   when any of these hold: unemployment above its 12-month average, the Sahm rule, CFNAI weakness,
   or a recently inverted yield curve combined with widening credit spreads. This "growth-trend"
   filter cuts the number of trades and the cash drag in bull markets.
5. **Recommended rule ("composite", 0–40% cash).** The rule has three layers:
   - **Strategic valuation layer, 0–20%.** Cash ramps from 0% at the 60th valuation percentile to
     20% at the 95th. Only half of it applies while the trend is still up, so value timing does not
     fight momentum (Asness et al. 2017).
   - **Tactical layer.** Add +20% when trend and macro are both risk-off, +10% for trend only, and
     +5% for macro only.
   - **Over-extension kicker.** Add +5% when the market is within 5% of its ATH *and* its 2-year
     run-up is in the top decile.

   The total is capped at 40%. From 1930 to today it averaged 10% cash. CAGR was 10.2% vs 10.1% for
   buy-and-hold, volatility 16.1% vs 18.3%, and maximum drawdown −69% vs −79%. Since 1960 the
   maximum drawdown was −41% vs −50%, and −36% for the walk-forward version. Its Sharpe beat the
   matched static mix by +0.04 (CI +0.01 to +0.07).
6. **Be modest.** After correcting for the nine rules tested, no rule's edge over the matched mix
   reaches conventional significance (deflated-Sharpe probability ≤ 0.64). Since 2010 most timing
   rules have *lagged* their matched static mix because of fast V-shaped recoveries (2020). The
   honest pitch is: **a similar long-run return with shallower, shorter drawdowns**, not higher
   returns. Parameter uncertainty argues for tilts, not all-in or all-out switches (Barberis 2000).
7. **Horizon matters more than the signal.**
   - Money you need within about 2 years should not be in equities at all.
   - From 2 to 5 years before you start drawing on the money, keep a cash bucket of 2–3 years of
     withdrawals. Sequence-of-returns risk peaks around the start of withdrawals (Pfau & Kitces
     2014).
   - With 5–20 years to go, use the full 0–40% rule.
   - With more than 20 years to go, halve the tilts (0–20%). Cash drag compounds, and lifecycle
     evidence favours high equity exposure for the young (Anarkulova, Cederburg & O'Doherty).

**Where we are now (24 Nov 2026, month in progress; S&P 500 ≈ 7,237, 7.4% below its October high).**
- **Valuation:** in the 95th percentile of its own history. CAPE is about 39, and market cap/GDP
  and investor equity allocation are at record highs.
- **Volatility:** high. VIX is 36 and one-month realised volatility is 35%.
- **Trend:** consensus still *on*. Only the 10-month SMA has tripped intra-month.
- **Macro:** the recession flag is *off*.

The composite rule says **≈9% cash**. Whether it moves depends on the **November month-end
close**:

| If at month end… | Composite cash |
|---|---|
| Trend consensus still on (today) | ~9% |
| Trend consensus flips off, macro still benign | ~28% |
| Trend off **and** macro flag turns on | ~38% (cap 40%) |

Run `python -m market_timing.signal_today --portfolio-value <£> --current-cash <w>` after the
month-end close.

---

## 2. Data and method (why the numbers can be trusted, and where they can't)

| Item | Detail |
|---|---|
| Market | S&P 500. Daily closes since 1928 (Yahoo `^GSPC`); before 1928, Shiller monthly averages. |
| Total return | Shiller (1871–1926) → CRSP value-weighted via Ken French (1926–Sep 2026) → `^SP500TR` |
| Cash | NBER commercial-paper rate before 1926 → 1-month T-bill (French) → 3-month T-bill (FRED) |
| Valuation inputs | Shiller earnings and CPI. **Earnings after mid-2023 are extended using NIPA after-tax corporate profits** (an approximation: Yale's public file ends 2023). Z.1 equity and debt levels and GDP (FRED). |
| Macro inputs | FRED: 10y and 3m Treasuries, Moody's Baa/Aaa, unemployment, real-time Sahm, CFNAI, NFCI, VIX |
| Point-in-time | Earnings lagged 3 months, CPI/unemployment/CFNAI/NFCI 1 month, Z.1 and GDP about 1 quarter after quarter end. Valuation percentiles use **expanding windows**. NBER recession dates are used only for shading charts, never as a signal. Unit tests check that perturbing future data leaves past signals unchanged. |
| Backtest | Decide at month end, hold through the next month. 5-percentage-point no-trade band, 10bp per unit of turnover, cash earns T-bills. Main sample 1930–Oct 2026; 1900 start as a robustness check. |
| Fair benchmark | For each rule, a **static mix holding the rule's average cash weight**. Any cash lowers volatility and drawdowns mechanically, so beating buy-and-hold on risk proves nothing. |
| Statistics | Newey-West t-stats; Campbell-Thompson out-of-sample R² (from 1950, sign-restricted); Clark-West test; AUC for ≥20% drawdowns within 12 months; stationary block bootstrap; Bailey & López de Prado deflated Sharpe; decade-by-decade walk-forward parameter choice. |
| Global check | Ken French country and region indices (UK, Japan, Europe, World), 1985–2025. Uses dividend-yield valuation, price trend and term-spread inversion. |

---

## 3. What the literature says, by indicator family

### 3.1 All-time-high proximity and anchoring
- **Li & Yu (2012, *JFE*)**:
  - Nearness of the Dow to its **52-week high positively** predicts aggregate returns.
  - Nearness to its **historical high negatively** predicts them.
  - They interpret both as anchoring and under/over-reaction, and report robustness across G7
    markets.
- **George & Hwang (2004, *JF*)**: 52-week-high anchoring explains much of cross-sectional
  momentum.
- *Our replication (1928–2026)*:
  - The 52-week-high effect is positive in sign but small (t = 0.4).
  - The historical-high effect is negative in sign (t = −1.3 at 12 months) but has negative
    out-of-sample R².
  - Being far *below* the 52-week high is a decent drawdown flag (AUC 0.65) because damage begets
    damage.
  - **Conclusion: proximity to an ATH is context, not a trigger.**

### 3.2 Valuation (fundamental)
- **Campbell & Shiller (1998, *JPM*)**: CAPE and dividend yield forecast **10-year** returns.
  High CAPE means lower subsequent real returns.
- **Goyal & Welch (2008, *RFS*)**: almost no valuation or macro predictor beats the historical
  mean **out of sample**. The relationships are unstable.
- **Campbell & Thompson (2008, *RFS*)**: weak sign restrictions restore small but economically
  meaningful out-of-sample R² values.
- **Asness, Ilmanen & Maloney (2017, *J. Investment Management*), "Market Timing: Sin a Little"**:
  - Value timing is directionally right but weak out of sample, because contrarian value fights
    momentum.
  - They recommend **combining value with momentum and making only modest tilts**.
- *Our replication*:
  - The excess CAPE yield is the strongest valuation signal (12-month out-of-sample R² +3.6%,
    p = 0.003; drawdown AUC 0.68).
  - Raw CAPE has t = −2.8 in sample but *negative* out-of-sample R².
  - Market cap/GDP and investor equity allocation flag drawdowns well (AUC 0.70 and 0.69), but
    their history only starts after WWII.
  - A valuation-only cash rule simply matched its static mix (+0.003 Sharpe).

### 3.3 Trend / technical
- **Faber (2007, *J. Wealth Management*)**: a 10-month SMA rule gives equity-like returns with
  much smaller drawdowns, using US data since 1900. Critics note data-snooping in the choice of
  10 months.
- **Moskowitz, Ooi & Pedersen (2012, *JFE*)**: time-series momentum appears in all 58 futures
  markets studied.
- **Hurst, Ooi & Pedersen (2017, *JPM*)**: the effect holds over a century and 67 markets.
- **Neely, Rapach, Tu & Zhou (2014, *Management Science*)**:
  - Technical indicators forecast the equity premium as well as or better than macro variables.
  - They detect the **decline in the premium near business-cycle peaks**, while macro variables
    catch the rise near troughs.
  - Combining the two works best.
- *Our replication*:
  - Trend rules are the only family with significant one-month out-of-sample gains (MA(2,12):
    out-of-sample R² +0.7%, Clark-West p = 0.008).
  - They give the largest backtest improvement, and also work in Japan, Europe and the World index
    (but barely in the UK).

### 3.4 Over-extension and bubbles
- **Greenwood, Shleifer & You (2019, *JFE*) "Bubbles for Fama"**:
  - Sharp run-ups do **not** on average predict low returns.
  - They do sharply raise **crash probability**, especially with high volatility, turnover,
    issuance and price acceleration.
- **Phillips, Shi & Yu (2015, *IER*)**: the GSADF/BSADF test date-stamps explosive episodes in the
  S&P 500 since 1871.
- *Our replication*:
  - The 24-month run-up has mildly negative 12-month predictive power (out-of-sample R² +1.6%,
    p = 0.05).
  - The BSADF statistic flags 1929 and 1999–2000 but adds little systematically (AUC 0.59).
  - We therefore use over-extension only as a small (+5%) kicker when the market is near its ATH.

### 3.5 Volatility and risk
- **Moreira & Muir (2017, *JF*)**: scaling exposure down when recent volatility is high raises
  Sharpe ratios.
- **Harvey et al. (2018, *JPM*)**: volatility targeting helps equities and credit and reduces tail
  events.
- **Cederburg, O'Doherty, Wang & Yan (2020, *JFE*)**: real-time, out-of-sample implementations
  generally do *not* beat the unmanaged portfolio.
- *Our replication*:
  - Realised volatility does not predict returns, but it flags drawdowns (AUC 0.69).
  - Adding a volatility cap to the composite changed little (Sharpe +0.008, drawdown slightly
    smaller, more trading).
  - We treat it as optional.

### 3.6 Macro and the business cycle
- **Estrella & Mishkin (1998, *REStat*)**: beyond one quarter, the **yield-curve slope** is the best
  single recession predictor.
- **Gilchrist & Zakrajšek (2012, *AER*)**: the credit-spread component that is not explained by
  default risk (the "excess bond premium") predicts activity and asset prices.
- The Sahm rule and real-time unemployment trends date recessions quickly.
- *Our replication*:
  - An inverted curve in the last 12 months predicts lower returns (12-month out-of-sample R²
    +2.8%, p = 0.04) and flags drawdowns (AUC 0.65).
  - The macro flag alone barely predicts returns. Its value is as **confirmation** of a trend
    break: "growth-trend" timing cut cash drag in bull markets and was the best rule in 1990–2009.

### 3.7 Sentiment
- **Baker & Wurgler (2006, *JF*)**: a sentiment index from six proxies (IPO volume and first-day
  returns, equity share of issuance, turnover, closed-end fund discount, dividend premium) predicts
  returns of hard-to-value stocks.
- **Huang, Jiang, Tu & Zhou (2015, *RFS*)**: an "aligned" sentiment index strongly predicts the
  *aggregate* market, in and out of sample.
- *Not replicated here*: the source data were not reachable from this environment. VIX and run-up
  measures act as partial proxies. This is a recommended extension.

### 3.8 Factors near market peaks
- Late bull markets tend to be led by high-beta, "junk" and momentum stocks.
- Defensive factors offer an *alternative to cash* for de-risking:
  - **betting-against-beta** (Frazzini & Pedersen 2014);
  - **quality-minus-junk** (Asness, Frazzini & Pedersen 2019).
- **Daniel & Moskowitz (2016)**: momentum crashes cluster in rebounds after bear markets.
- *In this study* factors are diagnostic only. Tilting the equity sleeve toward quality or
  low-volatility near tops is a reasonable complement to holding cash, not a substitute for it.

### 3.9 Portfolio choice, horizon and regimes
- **Barberis (2000, *JF*)**: once parameter uncertainty is acknowledged, predictability still
  matters, but **much less than fixed-parameter models imply**. The theoretical case is for
  modest tilts.
- **Kritzman, Page & Turkington (2012, *FAJ*)**: regime-switching (Markov) models add value mainly
  by **avoiding large losses**.
- **Pfau & Kitces (2014, *J. Financial Planning*)**: a *rising* equity glidepath through early
  retirement reduces failure risk. Low equity exposure (and more cash and bonds) matters most at
  the start of withdrawals.
- **Anarkulova, Cederburg & O'Doherty, "Beyond the Status Quo"**: for lifecycle savers, an
  all-equity, globally diversified mix beats conventional stock/bond glidepaths. This is a
  caution against large *permanent* cash holdings for long-horizon money.
- **Rapach, Strauss & Zhou (2010, *RFS*)**: equal-weight combinations of univariate forecasts beat
  the historical mean out of sample.
  - *Our replication*: a combination-forecast mean-variance rule **under-performed** (Sharpe −0.04
    vs its static mix). Its mean-variance mapping held too much cash during long bull markets.

---

## 4. Evidence table (S&P 500, 1928–2026)

Abbreviations: OOS R² = out-of-sample R²; CW p = Clark-West p-value; AUC = area under the curve
for predicting a ≥20% drawdown within 12 months.

| Indicator | Family | In-sample t (12m) | OOS R² 1m | OOS R² 12m (CW p) | AUC | Role in framework |
|---|---|---|---|---|---|---|
| Excess CAPE yield | Valuation | **4.34** | +0.8% | **+3.6% (0.003)** | 0.68 | Strategic layer (core) |
| CAPE (log) | Valuation | −2.82 | −1.0% | −11.7% (0.08) | 0.59 | Strategic layer (with ECY) |
| Market cap / GDP | Valuation | −1.10 | +0.1% | −2.1% (0.03) | **0.70** | Dashboard: drawdown risk |
| Investor equity allocation | Valuation | −2.33 | −0.3% | −3.6% (0.01) | 0.69 | Dashboard: drawdown risk |
| Trend consensus (2 of 3) | Trend | −0.39 | **+0.7% (p 0.009)** | −1.3% | 0.62 | Tactical layer (core) |
| MA(2,12) off | Trend | −0.36 | **+0.7% (p 0.008)** | −1.4% | 0.62 | Part of consensus |
| 10m SMA off | Trend | −0.95 | +0.2% (p 0.03) | −1.4% | 0.63 | Part of consensus |
| Curve inverted ≤12m | Macro | −1.71 | +0.4% (p 0.03) | +2.8% (0.04) | 0.65 | Macro flag |
| Term spread | Macro | 2.35 | +0.1% | −1.3% (0.04) | 0.60 | Phase classification |
| Macro recession flag | Macro | −0.11 | −0.2% | −3.2% | 0.54 | Confirms trend breaks |
| 24m run-up | Over-extension | −1.35 | 0.0% | +1.6% (0.05) | 0.48 | Kicker near ATH |
| BSADF bubble stat | Over-extension | −0.06 | −0.1% | −4.2% | 0.59 | Kicker near ATH |
| Realised vol | Risk | 0.0 | −0.1% | −0.9% | 0.69 | Optional volatility cap |
| VIX (1990+) | Risk | 1.07 | −0.5% | 0.0% | 0.66 | Dashboard |
| Price / 52-week high | ATH | 0.04 | −0.1% | −1.0% | 0.65 | Dashboard |
| Price / all-time high | ATH | −1.30 | −0.4% | −4.7% | 0.41 | Context only |

### Forward returns by distance from the ATH (1928–2026)

| Month-end distance from ATH | Share of months | Next-12m excess (ann.) | Next-36m excess (ann.) | P(≥20% drawdown in 12m) |
|---|---|---|---|---|
| At ATH | 6% | 3.8% | −0.5% | 15% |
| 0 to −2% | 17% | 5.6% | 5.4% | 13% |
| −2 to −5% | 13% | 4.9% | 4.6% | 13% |
| −5 to −10% | 12% | 3.2% | 3.6% | 14% |
| −10 to −20% | 16% | 7.1% | 6.2% | 13% |
| Below −20% | 36% | 8.1% | 9.0% | 23% |

The same buckets in the **expensive** valuation tercile only:

| Distance from ATH | Months | 12m excess | 36m excess (ann.) | 60m excess (ann.) | P(≥20% drawdown in 12m) |
|---|---|---|---|---|---|
| At ATH | 55 | 2.5% | **−2.1%** | 1.1% | 17% |
| 0 to −2% | 164 | 5.2% | 5.9% | 6.6% | 14% |
| −10 to −20% | 72 | 3.5% | 4.4% | 2.5% | 22% |
| Below −20% | 91 | **−7.4%** | −1.1% | −1.1% | **42%** |

Reading: being near an ATH in an expensive market lowers *multi-year* expected returns but does not
by itself raise near-term crash risk much. The danger zone is **expensive *and* already breaking
down**, which is exactly what the trend × valuation interaction in the composite rule targets.
(These windows overlap and sub-samples are small, so treat differences of a few percent as noise.)

---

## 5. Backtest results (0–40% cash, monthly, 10bp costs, 5pp no-trade band)

### 1930 – Oct 2026

Matched = a static mix holding the same average cash as the rule.

| Rule | CAGR | Vol | Sharpe | Max DD | Avg cash | Turnover/yr | Matched CAGR | Matched max DD | ΔSharpe vs matched (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| Buy & hold | 10.1% | 18.3% | 0.44 | −79% | 0% | 0 | – | – | – |
| Valuation only | 9.8% | 16.6% | 0.45 | −74% | 14% | 0.11 | 9.5% | −73% | +0.00 (−0.03, +0.04) |
| Faber 10m SMA | 10.2% | 14.3% | 0.53 | −58% | 13% | 0.60 | 9.5% | −74% | +0.09 (+0.03, +0.14) |
| Trend consensus | 10.4% | 14.5% | 0.53 | −58% | 13% | 0.38 | 9.5% | −74% | **+0.09 (+0.04, +0.14)** |
| Growth-trend (trend AND macro) | 10.6% | 16.4% | 0.50 | −68% | 6% | 0.23 | 10.0% | −76% | +0.05 (+0.01, +0.10) |
| Bank gains at ATH | 9.9% | 18.0% | 0.44 | −79% | 4% | 0.03 | 10.1% | −78% | −0.01 (−0.02, +0.01) |
| **Composite (recommended)** | **10.2%** | **16.1%** | **0.48** | **−69%** | **10%** | **0.21** | 9.7% | −75% | **+0.04 (+0.01, +0.07)** |
| Composite + volatility cap | 10.1% | 15.4% | 0.49 | −66% | 11% | 0.27 | 9.7% | −74% | +0.04 (+0.02, +0.08) |
| Combination forecast | 8.8% | 16.0% | 0.41 | −79% | 18% | 0.15 | 9.2% | −71% | −0.04 (−0.08, +0.02) |

### 1960 – Oct 2026 (includes the walk-forward version)

| Rule | CAGR | Sharpe | Max DD | Avg cash | Matched max DD | ΔSharpe vs matched |
|---|---|---|---|---|---|---|
| Buy & hold | 10.8% | 0.47 | −50% | 0% | – | – |
| Trend consensus | 10.9% | 0.54 | −34% | 11% | −46% | +0.07 |
| Growth-trend | 11.2% | 0.53 | −36% | 6% | −49% | +0.06 |
| Composite | 10.7% | 0.51 | −41% | 11% | −47% | +0.04 |
| Composite, walk-forward | 10.6% | 0.52 | **−36%** | 13% | −45% | +0.05 |
| Bank gains at ATH | 10.6% | 0.47 | −49% | 5% | −50% | −0.00 |

The walk-forward step re-selected the composite's parameters each decade from 1960, using only
prior data. It picked the same setting every time: strategic maximum 20%, ramp starting at the
50th percentile, 30% cash when trend and macro are both off, and the valuation layer halved while
the trend is up. That stability is reassuring.

### Sub-periods (ΔSharpe vs matched static mix)

| Rule | 1930–49 | 1950–89 | 1990–2009 | 2010–now |
|---|---|---|---|---|
| Valuation only | +0.03 | +0.03 | −0.02 | +0.03 |
| Trend consensus | +0.07 | +0.08 | +0.13 | −0.03 |
| Growth-trend | +0.04 | +0.04 | +0.19 | −0.03 |
| Composite | +0.04 | +0.04 | +0.09 | −0.01 |

### Robustness checks
- **Multiple testing.** Nine rules were tested. The deflated-Sharpe probability that the best
  rule's edge over its matched mix is genuine is 0.64 (composite). The trend rules score 0.24–0.37.
  None clears 0.95. The bootstrap intervals above do *not* account for having tried nine rules.
- **1900 start** (pre-1928 data are smoothed monthly averages):
  - Composite: CAGR 10.0% vs 9.9% buy-and-hold; max drawdown −75% vs −84%; ΔSharpe +0.04.
  - Trend consensus: ΔSharpe +0.09.
- **Global (1985–2025), ΔSharpe vs matched:**

  | Rule | Japan | World | Europe | UK |
  |---|---|---|---|---|
  | Faber 10m SMA | +0.08 | +0.06 | +0.06 | +0.01 |
  | Composite | +0.02 | +0.02 | +0.01 | +0.01 |
  | Valuation-only (dividend yield) | −0.02 | −0.02 | −0.07 | −0.02 |

  Composite maximum drawdown: UK −33% vs −36% for its matched mix; Japan −50% vs −55%.

---

## 6. The cash rule, precisely

Evaluate **once a month, after the month-end close**. Trade only if the target differs from your
current cash weight by **≥5 percentage points**.

```
valuation_pct  = mean( pct(CAPE), 1 - pct(excess CAPE yield) )      # expanding-window percentiles
trend_off      = at least 2 of { price < 10m SMA, 12m excess return < 0, MA(2) < MA(12) }
macro_flag     = unemployment > its 12m avg  OR  Sahm rule  OR  CFNAI-MA3 < -0.7
                 OR  (10y-3m inverted in last 12m AND Baa-Aaa spread widening)

strategic  = 20% x clip((valuation_pct - 0.60) / 0.35, 0, 1)  x (1.0 if trend_off else 0.5)
tactical   = 20% if trend_off and macro_flag
             10% if trend_off only
              5% if macro_flag only
              0% otherwise
kicker     = +5% if within 5% of ATH and 24m run-up in its top decile (or BSADF > 95% critical value)
cash       = clip(strategic + tactical + kicker, 0%, 40%)
```

Redeployment is automatic. When the trend turns back up, the tactical layer drops to zero and the
strategic layer halves. This avoids the classic failure of selling too early and never buying
back.

**ISA implementation notes.**
- Trims inside an ISA carry no capital gains tax, so a 5pp band and monthly checks are cheap.
- Hold the "cash" sleeve in something that earns roughly Bank Rate inside the ISA: a sterling
  money-market fund or ultra-short gilt ETF. Uninvested cash can earn little.
- A backtest using T-bills assumes you earn the short rate.
- If your equity holdings are global rather than S&P 500, the dashboard is still a reasonable
  global risk gauge. The global check shows trend rules travel well; valuation rules travel
  less well.

---

## 7. How the cash band should change over the market cycle

The phase is classified point-in-time (`allocation.cycle_phase`). Forward returns by phase, 1930–2026:

| Phase (definition) | Months | Excess return (ann.) | Vol | P(≥20% DD in 12m) | Suggested cash band |
|---|---|---|---|---|---|
| **Early** (≤24m after a ≥20% drawdown, trend back on, macro calm) | 308 | +8.8% | 15.6% | 13% | 0–5% (be invested) |
| **Mid** (trend on, no late-cycle stress) | 535 | +8.6% | 17.7% | 13% | 0–15% (valuation layer only) |
| **Late** (trend on but curve inverted ≤12m, or expensive with rising unemployment / widening spreads) | 154 | +6.8% | 15.0% | 11% | 10–30% |
| **Contraction** (trend off and macro flag on) | 165 | **−5.8%** | **25.7%** | **37%** | 25–40% |

Two observations matter for practice:
- **"Late cycle" still pays.** Returns stay positive until the trend actually breaks, which is why
  the rule only *halves* valuation-driven cash while the trend is up rather than leaving the
  market.
- **Nearly all the value of timing comes from the contraction state.** This is the state the
  trend × macro layer identifies. It is rare (about 14% of months), but it combines negative
  returns with high volatility.

---

## 8. How the cash percentage should change with your time horizon

| Years until you need the money | Envelope | How to apply the rule |
|---|---|---|
| < 2 | Not equities | Cash or short gilts (liability matching), whatever the signals say |
| 2 – 5 | Floor = 2–3 years of planned withdrawals (≥10%), cap 40% | Withdrawal bucket plus the rule on the rest. Sequence risk is highest here. |
| 5 – 20 | 0 – 40% | Follow the composite rule as specified |
| > 20 | 0 – 20% | Halve the tilts; cash drag compounds and the long horizon absorbs drawdowns |

In retirement, the Pfau & Kitces evidence supports starting withdrawals with *more* cash and bonds
(the bucket) and letting equity exposure rise over the first 10–15 years. Refill the bucket by
trimming equities in "mid" or "late" phases near highs, which is where banking gains *does* make
sense. Never refill it from equities during a contraction. Use
`python -m market_timing.signal_today --years-to-need N --annual-withdrawal 0.04`.

---

## 9. Monthly checklist

1. After the last trading day of the month, run `python -m market_timing.signal_today --portfolio-value <£> --current-cash <w>`.
2. Read three things:
   - the valuation percentile (strategic layer);
   - the trend consensus and macro flag (tactical layer);
   - the phase.
3. Trade only if the recommended cash differs from your current cash by ≥5pp. Move the full
   distance to the target in one go.
4. Do not override the rule because of headlines. The behavioural cost of discretionary
   in-and-out trading is well documented. The rule's edge is small, so it only works if followed
   mechanically.

---

## 10. Limitations

- **Small edges, many trials.** Improvements over a matched static mix are about +0.04 to +0.09
  Sharpe, and do not survive a strict deflated-Sharpe test. Expect long stretches of
  under-performance: 2010–2026 is one.
- **Regime change.** Valuation levels have trended up since the 1990s (globalisation, margins,
  index composition, lower real rates). Expanding-window percentiles adapt slowly, so the
  strategic layer has been "on" for much of the last 30 years.
- **Data splices.**
  - Post-2023 earnings use a corporate-profits proxy.
  - Pre-1928 prices are monthly averages, which smooth returns and flatter trend rules.
  - Total returns switch from S&P to CRSP-total-market in 1926 and back to S&P TR in Oct 2026.
- **Whipsaws.** Trend rules lose in V-shaped recoveries (1987, 2020). The 5pp band and the "2 of 3"
  consensus reduce, but do not remove, this.
- **Not replicated:** aggregate sentiment indices (Baker-Wurgler, Huang et al.), margin debt,
  insider-buying ratios and breadth. Their source data were not reachable here. These are
  candidates for future work.
- **US-centric evidence.** The global check uses a simpler signal set and shorter history.

---

## References

✓ = bibliographic details and core finding checked during this study.

- ✓ Anarkulova, A., Cederburg, S., & O'Doherty, M. *Beyond the Status Quo: A Critical Assessment of Lifecycle Investment Advice.* Working paper (SSRN; AEA 2025).
- ✓ Asness, C. (2003). Fight the Fed Model. *Journal of Portfolio Management*, Fall 2003.
- ✓ Asness, C., Ilmanen, A., & Maloney, T. (2017). Market Timing: Sin a Little. *Journal of Investment Management* 15(3), 23–40.
- Asness, C., Frazzini, A., & Pedersen, L. H. (2019). Quality minus junk. *Review of Accounting Studies* 24, 34–112.
- Baker, M., & Wurgler, J. (2006). Investor sentiment and the cross-section of stock returns. *Journal of Finance* 61(4).
- ✓ Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management* 40(5), 94–107.
- ✓ Barberis, N. (2000). Investing for the long run when returns are predictable. *Journal of Finance* 55(1), 225–264.
- ✓ Campbell, J. Y., & Shiller, R. J. (1998). Valuation ratios and the long-run stock market outlook. *Journal of Portfolio Management* 24(2), 11–26.
- ✓ Campbell, J. Y., & Thompson, S. B. (2008). Predicting excess stock returns out of sample: Can anything beat the historical average? *Review of Financial Studies* 21(4), 1509–1531.
- ✓ Cederburg, S., O'Doherty, M., Wang, F., & Yan, X. (2020). On the performance of volatility-managed portfolios. *Journal of Financial Economics* 138(1), 95–117.
- Clark, T., & West, K. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics* 138(1).
- Daniel, K., & Moskowitz, T. (2016). Momentum crashes. *Journal of Financial Economics* 122(2).
- ✓ Estrella, A., & Mishkin, F. (1998). Predicting U.S. recessions: Financial variables as leading indicators. *Review of Economics and Statistics* 80(1), 45–61.
- ✓ Faber, M. (2007). A quantitative approach to tactical asset allocation. *Journal of Wealth Management*, Spring 2007.
- Frazzini, A., & Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics* 111(1).
- George, T., & Hwang, C.-Y. (2004). The 52-week high and momentum investing. *Journal of Finance* 59(5).
- ✓ Gilchrist, S., & Zakrajšek, E. (2012). Credit spreads and business cycle fluctuations. *American Economic Review* 102(4), 1692–1720.
- ✓ Goyal, A., & Welch, I. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies* 21(4), 1455–1508.
- ✓ Greenwood, R., Shleifer, A., & You, Y. (2019). Bubbles for Fama. *Journal of Financial Economics* 131(1), 20–43.
- ✓ Harvey, C., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M., & Van Hemert, O. (2018). The impact of volatility targeting. *Journal of Portfolio Management* 45(1), 14–33.
- ✓ Huang, D., Jiang, F., Tu, J., & Zhou, G. (2015). Investor sentiment aligned: A powerful predictor of stock returns. *Review of Financial Studies* 28(3), 791–837.
- ✓ Hurst, B., Ooi, Y. H., & Pedersen, L. H. (2017). A century of evidence on trend-following investing. *Journal of Portfolio Management*, Fall 2017.
- ✓ Kritzman, M., Page, S., & Turkington, D. (2012). Regime shifts: Implications for dynamic strategies. *Financial Analysts Journal* 68(3), 22–39.
- ✓ Li, J., & Yu, J. (2012). Investor attention, psychological anchors, and stock return predictability. *Journal of Financial Economics* 104(2), 401–419.
- ✓ Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance* 72(4).
- ✓ Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics* 104(2), 228–250.
- ✓ Neely, C., Rapach, D., Tu, J., & Zhou, G. (2014). Forecasting the equity risk premium: The role of technical indicators. *Management Science* 60(7), 1772–1791.
- ✓ Pfau, W., & Kitces, M. (2014). Reducing retirement risk with a rising equity glide path. *Journal of Financial Planning* 27(1), 38–45.
- ✓ Phillips, P. C. B., Shi, S., & Yu, J. (2015). Testing for multiple bubbles: Historical episodes of exuberance and collapse in the S&P 500. *International Economic Review* 56(4), 1043–1078.
- ✓ Rapach, D., Strauss, J., & Zhou, G. (2010). Out-of-sample equity premium prediction: Combination forecasts and links to the real economy. *Review of Financial Studies* 23(2), 821–862.
- Sahm, C. (2019). Direct stimulus payments to individuals. Hamilton Project (origin of the Sahm recession rule).

Data: Robert Shiller (Yale) online data; Kenneth R. French Data Library; Federal Reserve Bank of St. Louis (FRED); Yahoo Finance.
