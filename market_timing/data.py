"""
Data loaders for the market-timing study.

All series are cached under ``data/cache/raw`` and assembled into a single monthly
(month-end indexed) DataFrame by :func:`us_monthly`. Every predictor column in that frame is
**point-in-time**: the value stored at month-end *t* is something an investor could have known
at the close of month *t* (publication lags are applied here, once, so downstream code cannot
accidentally look ahead).

Sources
-------
* Robert Shiller, ``ie_data.xls`` (monthly S&P composite since 1871: price, dividends,
  earnings, CPI, long rate). The public Yale file ends 2023-09 (D/E end 2023-06).
* Yahoo Finance daily ``^GSPC`` (since 1927-12) and ``^SP500TR`` (since 1988) via yfinance.
* Ken French data library: CRSP value-weighted market and 1-month T-bill (1926-07 onward),
  international country/region market returns with and without dividends (1975 onward).
* FRED (no API key; ``fredgraph.csv`` endpoint): rates, credit spreads, unemployment, Sahm rule,
  NFCI, VIX, Z.1 equity and debt levels, GDP, corporate profits, CPI, NBER recession dates,
  NBER commercial-paper rate (pre-1934 cash proxy).

Splices (documented in the report)
----------------------------------
* Earnings after the last Shiller observation are extended with the growth of NIPA after-tax
  corporate profits (FRED ``CP``), scaled to match at the overlap. This is an approximation.
* CPI after Shiller's last month uses FRED ``CPIAUCSL`` ratio-spliced.
* Total returns: Shiller (1871-1926, built from *monthly-average* prices — smoothed), CRSP via
  French (1926-07 to the latest French month, month-end), then ``^SP500TR`` month-end.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "cache"
RAW_DIR = CACHE_DIR / "raw"

SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}.zip"

# FRED series used, with how they become point-in-time at month end.
#   freq: native frequency; agg: how to collapse to monthly; lag: months after the observation
#   month (or after quarter *start* for quarterly) before the value is usable.
FRED_SERIES: Dict[str, Dict[str, object]] = {
    "TB3MS": {"agg": "mean", "lag": 0},          # 3m T-bill, monthly avg of daily (known at m-end)
    "GS10": {"agg": "mean", "lag": 0},           # 10y Treasury, monthly avg
    "BAA": {"agg": "mean", "lag": 0},            # Moody's Baa yield
    "AAA": {"agg": "mean", "lag": 0},            # Moody's Aaa yield
    "UNRATE": {"agg": "mean", "lag": 1},         # released first Friday of next month
    "SAHMREALTIME": {"agg": "mean", "lag": 1},   # real-time Sahm rule, released with UNRATE
    "CFNAI": {"agg": "mean", "lag": 1},          # Chicago Fed National Activity Index (~3 weeks)
    "CPIAUCSL": {"agg": "mean", "lag": 1},       # CPI, mid-next-month
    "NFCI": {"agg": "last", "lag": 1},           # weekly; conservative 1-month lag
    "VIXCLS": {"agg": "last", "lag": 0},         # daily close, month-end value
    "BAMLH0A0HYM2": {"agg": "last", "lag": 0},   # HY OAS (FRED only serves recent 3y)
    "USREC": {"agg": "max", "lag": None},        # NBER recession dates: EX-POST, evaluation only
    "M13002US35620M156NNBR": {"agg": "mean", "lag": 0},  # NBER commercial paper rate 1857-1971
    # Quarterly (FRED dates quarters by their first day). Z.1 is published ~10 weeks after
    # quarter end, NIPA profits ~2 months: usable at quarter start + 5 months (month end).
    "NCBEILQ027S": {"agg": "q", "lag": 5},       # nonfin. corp. equities, market value
    "FBCELLQ027S": {"agg": "q", "lag": 5},       # financial corp. equities
    "TCMDO": {"agg": "q", "lag": 5},             # all-sector debt (credit market debt owed)
    "GDP": {"agg": "q", "lag": 5},
    "CP": {"agg": "q", "lag": 5},                # corporate profits after tax
    # UK, for the global check
    "IRLTLT01GBM156N": {"agg": "mean", "lag": 0},
    "IR3TIB01GBM156N": {"agg": "mean", "lag": 0},
}


# --------------------------------------------------------------------------------------------
# Download helpers
# --------------------------------------------------------------------------------------------

def _fresh(path: Path, max_age_days: float) -> bool:
    return path.exists() and path.stat().st_size > 0 and (
        time.time() - path.stat().st_mtime < max_age_days * 86400
    )


def _download(url: str, path: Path, max_age_days: float = 7.0, retries: int = 4) -> Path:
    """Download ``url`` to ``path`` unless a fresh cached copy exists. Retries with backoff;
    if every attempt fails but a stale copy exists, the stale copy is used (with a warning)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _fresh(path, max_age_days):
        return path
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=90, headers={"User-Agent": "Mozilla/5.0 (research script)"})
            r.raise_for_status()
            if b"<html" in r.content[:500].lower() and not url.endswith(".html"):
                raise ValueError(f"HTML returned instead of data for {url}")
            path.write_bytes(r.content)
            return path
        except Exception as err:  # noqa: BLE001 - network errors are heterogeneous
            last_err = err
            wait = 2 ** attempt
            logger.warning("Download failed (%s) attempt %d/%d; retrying in %ds", url, attempt + 1, retries, wait)
            time.sleep(wait)
    if path.exists() and path.stat().st_size > 0:
        logger.warning("Using stale cache for %s (%s)", path.name, last_err)
        return path
    raise RuntimeError(f"Could not download {url}: {last_err}")


def _month_end(idx: Iterable) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx)) + pd.offsets.MonthEnd(0)


# --------------------------------------------------------------------------------------------
# Individual sources
# --------------------------------------------------------------------------------------------

def fred_raw(series_id: str, max_age_days: float = 3.0) -> pd.Series:
    """Raw FRED series as published (no lag), indexed by observation date."""
    path = _download(FRED_URL.format(sid=series_id), RAW_DIR / f"fred_{series_id}.csv", max_age_days)
    df = pd.read_csv(path)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df["value"], errors="coerce")
    s.index = df["date"]
    s.name = series_id
    return s.dropna()


def fred_monthly(series_id: str, point_in_time: bool = True) -> pd.Series:
    """FRED series collapsed to month-end and (optionally) shifted by its publication lag."""
    spec = FRED_SERIES.get(series_id, {"agg": "mean", "lag": 0})
    raw = fred_raw(series_id)
    agg = spec["agg"]
    if agg == "q":
        s = raw.copy()
        lag = int(spec["lag"]) if point_in_time else 2  # 2 => quarter-end month
        s.index = _month_end(s.index + pd.DateOffset(months=lag))
        # forward-fill to monthly
        monthly_index = pd.date_range(s.index.min(), pd.Timestamp.today() + pd.offsets.MonthEnd(0), freq="ME")
        return s.reindex(monthly_index).ffill().rename(series_id)
    grouped = raw.groupby(_month_end(raw.index))
    s = getattr(grouped, agg)()
    s.index = pd.DatetimeIndex(s.index)
    lag = spec.get("lag")
    if point_in_time and lag:
        s = s.shift(int(lag), freq="ME")
    return s.rename(series_id)


def shiller() -> pd.DataFrame:
    """Shiller monthly data: P (monthly *average* price), D, E (trailing 12m), CPI, GS10."""
    path = _download(SHILLER_URL, RAW_DIR / "ie_data.xls", max_age_days=30)
    raw = pd.read_excel(path, sheet_name="Data", header=None, skiprows=8)
    raw = raw.iloc[:, [0, 1, 2, 3, 4, 6]]
    raw.columns = ["date", "P", "D", "E", "CPI", "GS10"]
    raw = raw[pd.to_numeric(raw["date"], errors="coerce").notna()].copy()
    d = raw["date"].astype(float)
    year = np.floor(d).astype(int)
    month = np.rint((d - year) * 100).astype(int)
    idx = _month_end(pd.to_datetime(dict(year=year, month=month, day=1)))
    out = raw[["P", "D", "E", "CPI", "GS10"]].apply(pd.to_numeric, errors="coerce")
    out.index = idx
    return out


def yahoo_daily(ticker: str, max_age_days: float = 1.0) -> pd.Series:
    """Daily close from Yahoo (cached CSV). Falls back to stale cache if Yahoo is unavailable."""
    path = RAW_DIR / f"yahoo_{ticker.replace('^', '')}.csv"
    if not _fresh(path, max_age_days):
        try:
            import yfinance as yf

            for attempt in range(4):
                df = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=False)
                if len(df):
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] for c in df.columns]
                    df.to_csv(path)
                    break
                time.sleep(3 * (attempt + 1))
        except Exception as err:  # noqa: BLE001
            logger.warning("Yahoo download failed for %s: %s", ticker, err)
    if not path.exists():
        raise RuntimeError(f"No Yahoo data for {ticker}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["Close"].astype(float).dropna().rename(ticker)


def _french_zip(name: str) -> zipfile.ZipFile:
    path = _download(FRENCH_URL.format(name=name), RAW_DIR / "ff" / f"{name}.zip", max_age_days=30)
    return zipfile.ZipFile(path)


def french_us_factors() -> pd.DataFrame:
    """Monthly Fama-French factors (Mkt-RF, SMB, HML, RF) in decimal, plus momentum (Mom)."""
    def _read(name: str) -> pd.DataFrame:
        z = _french_zip(name)
        text = z.read(z.namelist()[0]).decode("latin-1")
        lines = text.splitlines()
        start = next(i for i, l in enumerate(lines) if l.strip().startswith(",")) + 1
        rows = []
        for l in lines[start:]:
            parts = [p.strip() for p in l.split(",")]
            if len(parts) < 2 or not parts[0].isdigit() or len(parts[0]) != 6:
                break
            rows.append(parts)
        header = [h.strip() for h in lines[start - 1].split(",")][1:]
        df = pd.DataFrame([r[1:] for r in rows], columns=header, dtype=float) / 100.0
        df.index = _month_end(pd.to_datetime([r[0] for r in rows], format="%Y%m"))
        return df

    ff = _read("F-F_Research_Data_Factors_CSV")
    mom = _read("F-F_Momentum_Factor_CSV")
    mom.columns = ["Mom"]
    return ff.join(mom, how="left")


def french_international(region: str = "UK", local: bool = True) -> pd.DataFrame:
    """Monthly value-weighted market return with and without dividends for a country/region.

    region: a country file stem (e.g. ``UK``, ``Japan``) or ``WORLD`` / ``EUROPE`` for the
    international index files. Returns columns ``tr`` (with dividends) and ``pr`` (price only).
    Local-currency returns when ``local`` (country files only; index files are USD).
    """
    if region.upper() in {"WORLD", "EUROPE", "ASIA"}:
        fname = {"WORLD": "Ind_all.Dat", "EUROPE": "Ind_Eur_With_UK.Dat", "ASIA": "Ind_Asia_Pacific.Dat"}[region.upper()]
        zips = ("F-F_International_Indices", "F-F_International_Indices_Wout_Div")
        local = False
    else:
        fname = f"{region}.Dat"
        zips = ("F-F_International_Countries", "F-F_International_Countries_Wout_Div")

    def _section(zname: str) -> pd.Series:
        z = _french_zip(zname)
        member = next(n for n in z.namelist() if n.lower().endswith(fname.lower()))
        lines = z.read(member).decode("latin-1").splitlines()
        want = "Local" if local else "Dollar"
        start = next(i for i, l in enumerate(lines) if want in l and "Not Reqd" in l)
        vals = {}
        for l in lines[start + 3:]:
            parts = l.split()
            if not parts or not parts[0].isdigit() or len(parts[0]) != 6:
                break
            vals[parts[0]] = float(parts[1]) / 100.0
        s = pd.Series(vals)
        s.index = _month_end(pd.to_datetime(s.index, format="%Y%m"))
        return s

    tr = _section(zips[0])
    pr = _section(zips[1])
    return pd.DataFrame({"tr": tr, "pr": pr}).dropna()


# --------------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------------

def _splice_ratio(base: pd.Series, ext: pd.Series) -> pd.Series:
    """Extend ``base`` forward with ``ext`` rescaled to match at the last common date."""
    base = base.dropna()
    ext = ext.dropna()
    common = base.index.intersection(ext.index)
    if len(common) == 0:
        return base
    anchor = common[-1]
    scaled = ext * (base.loc[anchor] / ext.loc[anchor])
    return pd.concat([base, scaled[scaled.index > anchor]])


def _monthly_from_daily(px: pd.Series) -> pd.DataFrame:
    """Month-end close, running all-time high (daily closes), trailing 252-day high,
    distance from 200-day moving average and realised volatility of daily returns."""
    px = px.dropna()
    me = _month_end(px.index)
    ret = px.pct_change()
    out = pd.DataFrame({
        "close": px.groupby(me).last(),
        "ath_daily": px.cummax().groupby(me).last(),
        "high_252d": px.rolling(252, min_periods=200).max().groupby(me).last(),
        "ma_200d": px.rolling(200, min_periods=150).mean().groupby(me).last(),
        "rv_1m": ret.groupby(me).std() * np.sqrt(252),
        "rv_3m_daily_var": (ret ** 2).groupby(me).sum(),
        "n_days": ret.groupby(me).count(),
    })
    out.index = pd.DatetimeIndex(out.index)
    return out


def us_monthly(start: str = "1871-01-31") -> pd.DataFrame:
    """Assemble the point-in-time monthly US panel used throughout the study.

    Key columns
    -----------
    price        S&P 500 month-end close (Yahoo ^GSPC from 1928; Shiller monthly avg before)
    ret          monthly total return of US equities (Shiller -> CRSP -> ^SP500TR)
    rf           monthly cash (T-bill) return
    ath          running all-time high of the price index (daily closes where available)
    high_52w     trailing 52-week high
    ma_200d      200-day moving average at month end
    rv_1m        realised volatility (annualised) of daily returns in the month
    earn_pit     trailing-12m earnings available at t (3-month lag)
    div          trailing-12m dividends per index unit (as of t)
    cpi_pit      CPI available at t (1-month lag)
    gs10, tb3    10y and 3m yields (%), monthly average
    baa, aaa     Moody's corporate yields (%)
    unrate, sahm, cfnai, nfci, vix
    mcap_gdp     nonfinancial corporate equity market value / GDP (lagged)
    aiae         investor equity allocation proxy: equities / (equities + debt) (lagged)
    usrec        NBER recession dummy (ex-post: evaluation only, never a signal)
    """
    sh = shiller()

    # ---- price & daily-derived stats
    gspc = _monthly_from_daily(yahoo_daily("^GSPC"))
    price = pd.concat([sh["P"][sh.index < gspc.index[0]], gspc["close"]])
    price = price[~price.index.duplicated(keep="last")]

    # ---- total return
    sh_tr = (sh["P"] + sh["D"] / 12.0) / sh["P"].shift(1) - 1.0
    ff = french_us_factors()
    crsp = ff["Mkt-RF"] + ff["RF"]
    ret = pd.concat([sh_tr[sh_tr.index < crsp.index[0]], crsp])
    sptr_m: Optional[pd.Series] = None
    try:
        sptr = yahoo_daily("^SP500TR")
        sptr_m = sptr.groupby(_month_end(sptr.index)).last().pct_change()
        sptr_m.index = pd.DatetimeIndex(sptr_m.index)
        ret = pd.concat([ret, sptr_m[sptr_m.index > ret.index[-1]]])
    except RuntimeError:
        logger.warning("^SP500TR unavailable; total return ends at last French month")

    # ---- cash
    cp_rate = fred_monthly("M13002US35620M156NNBR") / 1200.0
    tb3 = fred_monthly("TB3MS")
    rf = pd.concat([cp_rate[cp_rate.index < ff.index[0]], ff["RF"]])
    rf = pd.concat([rf, (tb3 / 1200.0)[tb3.index > rf.index[-1]]])

    # ---- CPI (point in time: 1-month lag)
    cpi_ext = fred_monthly("CPIAUCSL", point_in_time=False)
    full_idx = pd.date_range(sh.index[0], price.index[-1], freq="ME")
    cpi = _splice_ratio(sh["CPI"], cpi_ext).reindex(full_idx).ffill()  # ffill: e.g. Oct-2025 gap
    cpi_pit = cpi.shift(1)

    # ---- earnings (3-month lag) with corporate-profit extension
    cp = fred_monthly("CP", point_in_time=False)  # aligned to quarter-end month, ffilled
    earn = _splice_ratio(sh["E"], cp).reindex(full_idx).ffill()
    earn_pit = earn.shift(3)

    # ---- dividends: Shiller, then implied from ^SP500TR vs ^GSPC
    div = sh["D"].dropna()
    if sptr_m is not None:
        # Same index family on both sides (S&P 500 TR vs S&P 500 price) so the gap is the
        # dividend return. Rolling 12m sum of monthly dividend cash per index point.
        pr_m = gspc["close"].pct_change()
        implied = ((sptr_m - pr_m) * gspc["close"].shift(1)).dropna().rolling(12).sum().dropna()
        div = _splice_ratio(div, implied)

    # ---- long rate
    gs10 = pd.concat([sh["GS10"][sh.index < "1953-04-30"], fred_monthly("GS10")])
    gs10 = gs10[~gs10.index.duplicated(keep="last")]
    tb3_full = pd.concat([cp_rate * 1200.0 - 0.0, tb3])  # % p.a. (CP rate pre-1934)
    tb3_full = tb3_full[~tb3_full.index.duplicated(keep="last")]

    # ---- Z.1-based valuation
    nfc_eq = fred_monthly("NCBEILQ027S")
    fin_eq = fred_monthly("FBCELLQ027S")
    debt = fred_monthly("TCMDO")
    gdp = fred_monthly("GDP")
    mcap_gdp = (nfc_eq / 1000.0) / gdp  # NCBEILQ027S in $m, GDP in $bn
    aiae = (nfc_eq + fin_eq) / (nfc_eq + fin_eq + debt)

    df = pd.DataFrame({
        "price": price,
        "ret": ret,
        "rf": rf,
        "cpi_pit": cpi_pit,
        "earn_pit": earn_pit,
        "div": div,
        "gs10": gs10,
        "tb3": tb3_full,
        "baa": fred_monthly("BAA"),
        "aaa": fred_monthly("AAA"),
        "unrate": fred_monthly("UNRATE"),
        "sahm": fred_monthly("SAHMREALTIME"),
        "cfnai": fred_monthly("CFNAI"),
        "nfci": fred_monthly("NFCI"),
        "vix": fred_monthly("VIXCLS"),
        "mcap_gdp": mcap_gdp,
        "aiae": aiae,
        "usrec": fred_monthly("USREC", point_in_time=False),
    })
    df = df.join(gspc[["ath_daily", "high_252d", "ma_200d", "rv_1m"]], how="left")

    # Running ATH: monthly prices before 1928, daily closes after (never below the monthly max).
    df["ath"] = df["price"].cummax()
    df["ath"] = np.fmax(df["ath"], df["ath_daily"].fillna(-np.inf))
    df["ath"] = df["ath"].cummax()
    df["high_52w"] = df["high_252d"].fillna(df["price"].rolling(12, min_periods=12).max())
    df = df.drop(columns=["ath_daily", "high_252d"])

    # Carry the latest *published* value of slow predictors forward up to 2 months so the
    # current (partial) month has a reading. This never uses future information.
    slow = ["gs10", "tb3", "baa", "aaa", "unrate", "sahm", "cfnai", "nfci", "vix", "cpi_pit"]
    df[slow] = df[slow].ffill(limit=2)
    df = df[df.index >= pd.Timestamp(start)]
    # Drop trailing months with no price (future index from quarterly ffill).
    last_price = df["price"].last_valid_index()
    df = df.loc[:last_price]
    # Flag a partial (in-progress) final month: usable for today's signal, not for backtests.
    df["complete"] = True
    if df.index[-1] > pd.Timestamp.today().normalize():
        df.loc[df.index[-1], "complete"] = False
    df.index.name = "date"
    return df


def intl_monthly(region: str) -> pd.DataFrame:
    """Monthly panel for the global robustness check (French data, 1975+).

    Columns: ``ret`` total return, ``price`` price index, ``div`` trailing-12m dividends,
    ``rf`` cash return. Local-currency for countries with a local short rate (UK); USD with the
    US T-bill otherwise.
    """
    region_u = region.upper()
    local = region_u == "UK"
    f = french_international(region if region_u not in {"WORLD", "EUROPE", "ASIA"} else region_u, local=local)
    price = (1.0 + f["pr"]).cumprod()
    div = ((f["tr"] - f["pr"]) * price.shift(1)).rolling(12).sum()
    if local:
        rf = fred_monthly("IR3TIB01GBM156N") / 1200.0
        long_rate = fred_monthly("IRLTLT01GBM156N")
        short_rate = fred_monthly("IR3TIB01GBM156N")
    else:
        ff = french_us_factors()
        rf = ff["RF"]
        long_rate = fred_monthly("GS10")
        short_rate = fred_monthly("TB3MS")
    out = pd.DataFrame({"ret": f["tr"], "price": price, "div": div})
    out["rf"] = rf.reindex(out.index).ffill()
    out["term_spread"] = (long_rate - short_rate).reindex(out.index).ffill()
    out["ath"] = out["price"].cummax()
    out["high_52w"] = out["price"].rolling(12).max()
    return out.dropna(subset=["ret", "rf"])


def save_panel(df: pd.DataFrame, name: str = "us_monthly.csv") -> Path:
    path = CACHE_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    panel = us_monthly()
    p = save_panel(panel)
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(panel.tail(6))
        print(panel.loc["1929-06":"1929-12"])
    print(f"Saved {p} ({len(panel)} rows, {panel.index[0].date()} -> {panel.index[-1].date()})")
