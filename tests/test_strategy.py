import numpy as np
import pandas as pd

from fts.strategy import (backtest, buy_and_hold_weights, channel_breakout_weights,
                          lead_lag_weights, performance, walk_forward_predictions)


def prices_from_returns(returns):
    return np.exp(returns.cumsum())


def test_backtest_applies_weights_to_the_next_day():
    prices = pd.DataFrame({"a": [100.0, 110.0, 99.0, 99.0]}, index=range(1, 5))
    weights = pd.DataFrame({"a": [0.0, 1.0, 0.0, 0.0]}, index=range(1, 5))   # long only at the close of day 2
    daily, _ = backtest(weights, prices, first_day=1)
    assert np.allclose(daily.to_numpy(), [0.0, -0.10, 0.0])   # earns day 3's -10%, not day 2's +10%


def test_costs_are_charged_on_traded_amount():
    prices = pd.DataFrame({"a": [100.0] * 4}, index=range(1, 5))
    weights = pd.DataFrame({"a": [0.0, 1.0, 0.0, 0.0]}, index=range(1, 5))   # buy at close 2, sell at close 3
    daily, turnover = backtest(weights, prices, first_day=1, cost_bps=10)
    # A trade at the close of day t is booked on day t+1, the first day it affects.
    assert np.allclose(daily.to_numpy(), [0.0, -0.001, -0.001])
    assert np.allclose(turnover.to_numpy(), [0.0, 1.0, 1.0])


def test_buy_and_hold_matches_the_average_price_relative():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, (300, 3)), columns=list("abc"), index=range(1, 301))
    prices = prices_from_returns(returns)
    daily, _ = backtest(buy_and_hold_weights(prices, 50), prices, first_day=50, drift=True)
    expected = (prices.loc[300] / prices.loc[50]).mean()
    assert np.isclose((1 + daily).prod(), expected)


def test_channel_breakout_positions_use_past_prices_only():
    prices = pd.DataFrame({"a": [1.0, 2.0, 3.0, 2.5, 10.0, 9.0, 0.5]}, index=range(1, 8))
    weights = channel_breakout_weights(prices, lookback=3)["a"]
    assert weights.loc[4] == 0.0      # 2.5 is inside the range of days 1-3
    assert weights.loc[5] == 1.0      # 10 breaks the high of days 2-4
    assert weights.loc[6] == 1.0      # position is held
    assert weights.loc[7] == -1.0     # 0.5 breaks the low of days 4-6


def test_lead_lag_weights_normalisation():
    predictions = pd.DataFrame({"a": [0.02, -0.01, 0.0], "b": [-0.02, -0.03, 0.0]})
    both = lead_lag_weights(predictions)
    assert np.allclose(both.abs().sum(axis=1), [1.0, 1.0, 0.0])
    long_only = lead_lag_weights(predictions, long_only=True)
    assert long_only.iloc[0].tolist() == [1.0, 0.0]
    assert long_only.iloc[1].tolist() == [0.0, 0.0]     # nothing positive: cash


def test_walk_forward_strategy_profits_from_a_planted_lead_and_not_from_noise():
    rng = np.random.default_rng(0)
    n = 3000
    leader = rng.normal(0, 0.01, n)
    follower = 0.6 * np.roll(leader, 1) + rng.normal(0, 0.008, n)
    noise = rng.normal(0, 0.01, n)
    index = pd.RangeIndex(1, n + 1)

    planted = pd.DataFrame({"leader": leader, "follower": follower}, index=index)
    predictions = walk_forward_predictions(planted, first_day=1500, refit_every=500, lags=2)
    assert predictions.loc[:1499].isna().all().all()
    daily, _ = backtest(lead_lag_weights(predictions), prices_from_returns(planted), first_day=1500)
    assert performance(daily)["sharpe"] > 3

    pure_noise = pd.DataFrame({"x": leader, "y": noise}, index=index)
    predictions = walk_forward_predictions(pure_noise, first_day=1500, refit_every=500, lags=2)
    daily, _ = backtest(lead_lag_weights(predictions), prices_from_returns(pure_noise), first_day=1500)
    assert abs(performance(daily)["sharpe"]) < 1.5


def test_walk_forward_predictions_ignore_future_data():
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0, 0.01, (1200, 2)), columns=["a", "b"], index=pd.RangeIndex(1, 1201))
    base = walk_forward_predictions(returns, first_day=800, refit_every=200, lags=2)
    changed = returns.copy()
    changed.loc[1001:] = rng.normal(0, 0.05, (200, 2))
    after = walk_forward_predictions(changed, first_day=800, refit_every=200, lags=2)
    assert np.allclose(base.loc[800:999], after.loc[800:999])
