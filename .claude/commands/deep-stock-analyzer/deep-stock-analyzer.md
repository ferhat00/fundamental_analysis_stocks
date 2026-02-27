---
name: deep-stock-analyzer
description: Deep, institutional-grade stock analysis outputting a runnable Jupyter notebook. Use this skill whenever a user asks for deep, advanced, or comprehensive stock/equity analysis of any company by name or ticker — especially when they want Monte Carlo DCF simulation, Hidden Markov Model regime detection, advanced factor models (Fama-French + momentum/quality/macro), PCA dimensionality reduction, cointegration tests, XGBoost/LightGBM return prediction, SEC EDGAR filing scraping, portfolio optimization via cvxpy, or backtesting via vectorbt. Also triggers on: "analyze [company] stock", "deep dive on [stock]", "institutional analysis", "quant analysis", "factor model analysis", "regime detection", "Monte Carlo valuation". Always use this skill (not stock-analyzer) when the user wants the advanced quant/ML/probabilistic features.
---

# Deep Stock Analyzer Skill

Generates a **self-contained, runnable Jupyter notebook** with institutional-grade quantitative analysis for any publicly traded company. The user provides a company name; Claude resolves the yfinance-compatible ticker and generates the notebook.

## Workflow

1. **Resolve ticker** – Use yfinance + web search to confirm the correct ticker symbol for the company name provided
2. **Generate notebook** – Run `scripts/generate_notebook.py` to produce the full `.ipynb` file
3. **Present to user** – Use `present_files` to deliver the notebook

## Step 1: Ticker Resolution

When the user provides a company name (not a ticker), resolve it programmatically:

```python
import yfinance as yf

def resolve_ticker(company_name: str) -> str:
    """Search yfinance for ticker by company name."""
    # Try direct lookup first
    ticker = yf.Ticker(company_name)
    info = ticker.info
    if info.get('regularMarketPrice') or info.get('currentPrice'):
        return company_name.upper()
    # Use yfinance search
    results = yf.Search(company_name, max_results=5)
    quotes = results.quotes
    if quotes:
        return quotes[0]['symbol']
    raise ValueError(f"Could not resolve ticker for: {company_name}")
```

If resolution fails, use web search to find the correct ticker, then verify with yfinance.

## Step 2: Generate the Notebook

```bash
python /home/claude/deep-stock-analyzer/scripts/generate_notebook.py \
  --ticker AAPL \
  --company "Apple Inc" \
  --output /mnt/user-data/outputs/AAPL_deep_analysis.ipynb
```

The script generates a notebook with these sections (see `references/notebook_sections.md` for full code):

### Notebook Sections

| # | Section | Key Libraries |
|---|---------|---------------|
| 1 | Setup & Config | yfinance, pandas, numpy |
| 2 | Technical Analysis (1yr) | plotly, ta-lib / pandas_ta |
| 3 | Fundamentals & Financials | yfinance, beautifulsoup4 |
| 4 | SEC EDGAR Scraping | requests, beautifulsoup4 |
| 5 | Standard DCF | numpy, scipy |
| 6 | Monte Carlo DCF | numpy, scipy, plotly |
| 7 | HMM Regime Detection | hmmlearn |
| 8 | Advanced Factor Models | statsmodels, sklearn |
| 9 | PCA & Cointegration | sklearn, statsmodels |
| 10 | ML Return Prediction | xgboost, lightgbm |
| 11 | Portfolio Optimization | cvxpy |
| 12 | Backtesting | vectorbt |
| 13 | Summary & Investment Thesis | — |

## Step 3: Present the Notebook

```python
present_files(["/mnt/user-data/outputs/{TICKER}_deep_analysis.ipynb"])
```

## Important Notes

- **All code in the notebook is defensive**: try/except blocks around every data fetch, graceful degradation if a library is missing, clear install instructions at the top
- **Ticker resolution is the critical first step** — double-check with `yf.Ticker(ticker).info` before proceeding
- **SEC EDGAR**: only works for US-listed companies; skip gracefully for others
- **vectorbt**: may have install issues on some systems; notebook includes fallback manual backtest
- For non-US tickers, append exchange suffix if needed (e.g., `.L` for LSE, `.TO` for TSX)

## Reference Files

- `references/notebook_sections.md` — Full Python code for each notebook section
- `references/install_guide.md` — Library installation commands and known issues
- `scripts/generate_notebook.py` — The notebook generator script

## Required Libraries (notebook installs them automatically)

```
yfinance pandas numpy matplotlib plotly scipy statsmodels
scikit-learn xgboost lightgbm hmmlearn cvxpy vectorbt
pandas-ta requests beautifulsoup4 nbformat ipywidgets
```