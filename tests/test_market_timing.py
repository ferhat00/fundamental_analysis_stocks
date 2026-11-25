"""Offline unit tests for market_timing (synthetic data; no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_timing import allocation as A
from market_timing import backtest as B
from market_timing import evaluate as E
from market_timing import indicators as I


def synthetic_panel(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1950-01-31", periods=n, freq="ME")
    ret = rng.normal(0.007, 0.045, n)
    price = 100 * np.cumprod(1 + ret - 0.003)  # price excludes a 0.3%/m dividend
    cpi = 100 * np.cumprod(1 + rng.normal(0.003, 0.002, n))
    earn = price / 18 * np.exp(rng.normal(0, 0.05, n))
    df = pd.DataFrame({
        "price": price,
        "ret": ret,
        "rf": 0.003,
        "cpi_pit": cpi,
        "earn_pit": earn,
        "div": price * 0.036,
        "gs10": 5 + rng.normal(0, 0.5, n).cumsum() * 0.05,
        "tb3": 4 + rng.normal(0, 0.5, n).cumsum() * 0.05,
        "baa": 7 + rng.normal(0, 0.1, n),
        "aaa": 6 + rng.normal(0, 0.1, n),
        "unrate": 5 + rng.normal(0, 0.2, n).cumsum() * 0.1,
        "sahm": rng.uniform(0, 0.8, n),
        "cfnai": rng.normal(0, 0.5, n),
        "nfci": rng.normal(0, 0.5, n),
        "vix": 20 + rng.normal(0, 3, n),
        "mcap_gdp": 1 + rng.normal(0, 0.05, n).cumsum() * 0.1,
        "aiae": 0.3 + rng.normal(0, 0.01, n),
        "rv_1m": np.abs(rng.normal(0.15, 0.03, n)),
    }, index=idx)
    df["ath"] = df["price"].cummax()
    df["high_52w"] = df["price"].rolling(12, min_periods=1).max()
    df["ma_200d"] = df["price"].rolling(10, min_periods=1).mean()
    return df


# ---------------------------------------------------------------- no look-ahead

def test_indicators_do_not_use_future_data():
    df = synthetic_panel()
    cut = 400
    base = I.compute_all(df, with_bsadf=True, pct_min_periods=60)
    perturbed = df.copy()
    rng = np.random.default_rng(99)
    future = perturbed.index[cut + 1:]
    for col in perturbed.columns:
        perturbed.loc[future, col] = perturbed.loc[future, col] * (1 + rng.normal(0, 0.3, len(future)))
    alt = I.compute_all(perturbed, with_bsadf=True, pct_min_periods=60)
    past = base.index[: cut + 1]
    pd.testing.assert_frame_equal(base.loc[past], alt.loc[past], check_exact=False, rtol=1e-10)


def test_cash_rules_do_not_use_future_data():
    df = synthetic_panel()
    cut = 420
    ind = I.compute_all(df, pct_min_periods=60)
    perturbed = df.copy()
    perturbed.iloc[cut + 1:, perturbed.columns.get_loc("price")] *= 0.5
    perturbed["ath"] = perturbed["price"].cummax()
    ind2 = I.compute_all(perturbed, pct_min_periods=60)
    for fn in (A.composite_cash, A.bank_gains_cash, A.faber_cash, A.growth_trend_cash):
        a, b = fn(ind), fn(ind2)
        pd.testing.assert_series_equal(a.iloc[: cut + 1], b.iloc[: cut + 1])


def test_expanding_percentile_is_expanding_only():
    x = pd.Series([1.0, 2, 3, 4, 5, 0, 10])
    p = I.expanding_percentile(x, min_periods=1)
    # first obs: rank among itself = 0.5
    assert p.iloc[0] == pytest.approx(0.5)
    # value 5 at t=4 is the max of {1..5}: (4 + 0.5)/5
    assert p.iloc[4] == pytest.approx(4.5 / 5)
    # appending later values must not change earlier percentiles
    p2 = I.expanding_percentile(pd.Series([1.0, 2, 3, 4, 5]), min_periods=1)
    np.testing.assert_allclose(p.iloc[:5].to_numpy(), p2.to_numpy())


def test_forward_returns_alignment():
    df = synthetic_panel(50)
    f = E.forward_excess(df, 3)
    xr = np.log1p(df["ret"]) - np.log1p(df["rf"])
    assert f.iloc[10] == pytest.approx(xr.iloc[11:14].sum())
    assert np.isnan(f.iloc[-1])


def test_oos_r2_only_uses_realised_targets():
    df = synthetic_panel(400)
    y = E.forward_excess(df, 12)
    x = pd.Series(np.random.default_rng(3).normal(size=len(df)), index=df.index)
    r1 = E.oos_r2(x, y, h=12, oos_start=str(df.index[200].date()), min_train=60)
    y2 = y.copy()
    y2.iloc[-30:] = 99.0  # targets not yet realised at earlier origins
    r2 = E.oos_r2(x, y2, h=12, oos_start=str(df.index[200].date()), min_train=60)
    common = r1.forecasts.index[r1.forecasts.index < df.index[-30 - 12]]
    pd.testing.assert_series_equal(r1.forecasts.loc[common], r2.forecasts.loc[common])


# ---------------------------------------------------------------- allocation bounds

def test_composite_within_bounds_and_band():
    df = synthetic_panel()
    ind = I.compute_all(df, pct_min_periods=60)
    for rule in (A.CashRule(), A.CashRule(vol_cap=0.1), A.CashRule(tactical_both=0.5, strategic_max=0.5)):
        c = A.composite_cash(ind, rule)
        assert c.min() >= 0.0 - 1e-12
        assert c.max() <= 0.40 + 1e-12
    res = B.run_backtest(df["ret"], df["rf"], A.composite_cash(ind), band=0.05)
    changes = res.cash_weight.diff().abs()
    # either no trade (drift only, which is small) or a move to target of >= 5pp
    tgt = A.composite_cash(ind).shift(1).reindex(res.cash_weight.index)
    traded = res.turnover > 0
    assert ((res.cash_weight[traded] - tgt[traded]).abs() < 1e-12).all()
    assert (res.turnover[traded] >= 0.05 - 1e-12).all()


def test_horizon_adjustment():
    assert A.horizon_adjustment(1)["envelope"] == (1.0, 1.0)
    short = A.horizon_adjustment(3, "late", annual_withdrawal_pct=0.04)
    assert short["envelope"][0] == pytest.approx(0.10)
    assert A.horizon_adjustment(30)["scale"] == 0.5
    assert A.horizon_adjustment(10)["envelope"] == (0.0, 0.40)


# ---------------------------------------------------------------- backtest arithmetic

def test_backtest_matches_hand_calculation():
    idx = pd.date_range("2000-01-31", periods=4, freq="ME")
    eq = pd.Series([0.0, 0.10, -0.10, 0.05], idx)
    rf = pd.Series([0.0, 0.01, 0.01, 0.01], idx)
    tgt = pd.Series([0.5, 0.5, 0.5, 0.5], idx)
    res = B.run_backtest(eq, rf, tgt, band=0.05, cost_bps=0.0)
    # month 2: 50/50 -> 0.5*0.10 + 0.5*0.01 = 0.055
    assert res.returns.iloc[0] == pytest.approx(0.055)
    # drift: equity 0.55, cash 0.505 -> cash weight 0.505/1.055 = 0.4787 (< 5pp from 0.5: no trade)
    c = 0.505 / 1.055
    assert res.cash_weight.iloc[1] == pytest.approx(c)
    assert res.returns.iloc[1] == pytest.approx((1 - c) * -0.10 + c * 0.01)


def test_backtest_costs_and_buy_hold():
    idx = pd.date_range("2000-01-31", periods=3, freq="ME")
    eq = pd.Series([0.0, 0.02, 0.03], idx)
    rf = pd.Series([0.0, 0.0, 0.0], idx)
    bh = B.run_backtest(eq, rf, pd.Series(0.0, idx))
    np.testing.assert_allclose(bh.returns.to_numpy(), [0.02, 0.03])
    tgt = pd.Series([0.0, 0.40, 0.40], idx)
    res = B.run_backtest(eq, rf, tgt, band=0.05, cost_bps=10)
    # first month starts at target 0 (no trade); second month trades 40% -> 4bp cost
    assert res.returns.iloc[1] == pytest.approx(0.6 * 0.03 - 0.40 * 0.001)


def test_metrics_sane():
    df = synthetic_panel()
    m = B.metrics(df["ret"], df["rf"])
    assert -1 < m["MaxDD"] <= 0
    assert m["Vol"] > 0


def test_bsadf_detects_explosive_path():
    rng = np.random.default_rng(5)
    rw = np.cumsum(rng.normal(0, 0.03, 300))
    bubble = rw.copy()
    bubble[240:] = bubble[239] + np.cumsum(np.full(60, 0.03)) * np.linspace(1, 3, 60)
    s_rw = I.bsadf(pd.Series(rw)).iloc[-1]
    s_b = I.bsadf(pd.Series(bubble)).iloc[-1]
    assert s_b > s_rw
    assert s_b > 1.0
