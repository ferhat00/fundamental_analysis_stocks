# Deep Stock Analyzer — Library Installation Guide

## Core Installation (run all at once)

```bash
pip install yfinance pandas numpy matplotlib plotly scipy statsmodels \
    scikit-learn xgboost lightgbm hmmlearn cvxpy vectorbt \
    pandas-ta requests beautifulsoup4 nbformat ipywidgets seaborn lxml \
    --break-system-packages
```

## Known Issues & Fixes

### vectorbt
- May conflict with newer pandas. If install fails, the backtest section gracefully falls back to a manual pandas implementation.
- For M1/M2 Macs: `pip install vectorbt --no-binary vectorbt`

### hmmlearn
- Requires a C compiler. On Windows: install Visual C++ Build Tools first.
- On Ubuntu/Debian: `sudo apt-get install gcc`

### cvxpy
- If SCS solver fails, it auto-falls back to CLARABEL or ECOS.
- Full solver suite: `pip install cvxpy[CLARABEL]`

### ta-lib (optional — not required, using pandas_ta instead)
- Requires system library: on Ubuntu `sudo apt-get install ta-lib`
- We use pandas-ta as a pure-Python alternative (already included)

## Jupyter Setup

```bash
pip install jupyter notebook ipywidgets
jupyter nbextension enable --py widgetsnbextension
jupyter notebook your_analysis.ipynb
```

## Virtual Environment (recommended)

```bash
python -m venv deep_stock_env
source deep_stock_env/bin/activate  # Linux/Mac
deep_stock_env\Scripts\activate     # Windows
pip install [packages above]
```