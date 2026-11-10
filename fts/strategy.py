"""Daily trading rules on the seven series, with a common backtest.

Every rule returns a table of portfolio weights decided at the close of day t.
``backtest`` applies them to the returns of day t+1, so a rule can only use
information up to and including day t.

Rules:

* ``buy_and_hold_weights``: equal amounts bought on the first day and never
  rebalanced (a teammate's benchmark).
* ``channel_breakout_weights``: long after a new high over the lookback
  window, short after a new low (a teammate's rule).
* ``lead_lag_weights``: positions from walk-forward forecasts of the next
  day's return, which rest on the lead-lag relations between the series.
"""
import numpy as np
import pandas as pd

from .forecast import lagged_features, make_model

DAYS_PER_YEAR = 252  # assumption: the data has no calendar, a "day" is taken as a trading day


def walk_forward_predictions(returns, first_day, refit_every=250, lags=5):
    """Next-day return forecasts for every series, using only past data.

    A ridge model per series is refitted every ``refit_every`` days on all
    days before the refit day. The row for day t is the forecast of the
    return on day t+1, made with the returns up to day t.
    """
    X_all = lagged_features(returns, lags)
    predictions = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    usable_X = X_all.notna().all(axis=1)
    days = returns.index[returns.index >= first_day]
    for start in range(0, len(days), refit_every):
        block = days[start:start + refit_every]
        block = block[usable_X.loc[block].to_numpy()]
        for name in returns.columns:
            y = returns[name].shift(-1)
            # Target day t+1 must lie before the first day of the block.
            train = usable_X & y.notna() & (returns.index + 1 < block[0])
            model = make_model("ridge").fit(X_all[train].to_numpy(), y[train].to_numpy())
            predictions.loc[block, name] = model.predict(X_all.loc[block].to_numpy())
    return predictions


def lead_lag_weights(predictions, long_only=False):
    """Weights proportional to the forecast return, with gross exposure 1.

    With ``long_only`` the negative forecasts are dropped and the rest is
    scaled to sum to 1 (a teammate's "buy" rule); days without a positive
    forecast stay in cash.
    """
    signal = predictions.clip(lower=0.0) if long_only else predictions
    gross = signal.abs().sum(axis=1)
    return signal.div(gross.where(gross > 0), axis=0).fillna(0.0)


def channel_breakout_weights(prices, lookback=150):
    """Equal-weighted long/short positions from Donchian channel breakouts."""
    upper = prices.shift(1).rolling(lookback).max()
    lower = prices.shift(1).rolling(lookback).min()
    position = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    position = position.mask(prices > upper, 1.0).mask(prices < lower, -1.0)
    return position.ffill().fillna(0.0) / prices.shape[1]


def buy_and_hold_weights(prices, first_day):
    """Weights of a portfolio bought in equal amounts on ``first_day`` and left alone."""
    relative = prices / prices.loc[first_day]
    weights = relative.div(relative.sum(axis=1), axis=0)
    weights.loc[weights.index < first_day] = 0.0
    return weights


def backtest(weights, prices, first_day, cost_bps=0.0, drift=False):
    """Daily portfolio returns of weights decided at the close of each day.

    The return of day t+1 is ``sum(weights[t] * simple_return[t+1])`` minus
    trading costs of ``cost_bps`` basis points on the amount traded at the
    close of day t (booked on day t+1). With
    ``drift`` the weights are taken to move with prices between decisions
    (buy and hold), so no trading is charged.
    """
    simple = prices.pct_change()
    held = weights.shift(1)
    gross = (held * simple).sum(axis=1)
    turnover = pd.Series(0.0, index=weights.index) if drift else weights.diff().abs().sum(axis=1).shift(1)
    net = gross - turnover.fillna(0.0) * cost_bps / 1e4
    keep = net.index > first_day
    return net[keep], turnover[keep]


def performance(daily, turnover=None):
    """Annualised return, volatility, Sharpe ratio (zero risk-free rate) and drawdown."""
    value = (1 + daily).cumprod()
    out = {
        "annual_return": value.iloc[-1] ** (DAYS_PER_YEAR / len(daily)) - 1,
        "annual_volatility": daily.std() * np.sqrt(DAYS_PER_YEAR),
        "sharpe": daily.mean() / daily.std() * np.sqrt(DAYS_PER_YEAR),
        "max_drawdown": (value / value.cummax() - 1).min(),
    }
    if turnover is not None:
        out["daily_turnover"] = turnover.mean()
    return out
