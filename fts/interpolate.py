"""Interpolation of a block of missing prices, with uncertainty.

All methods work on log prices and return, for every missing day, a mean and a
standard deviation of the log price. Three methods are provided:

* ``brownian_bridge``: straight line between the two known endpoints, with the
  variance of a random walk tied down at both ends.
* ``local_trend``: the state-space model of the original notebook (local
  linear trend plus an AR(1) component), smoothed with the Kalman smoother.
* ``regression_bridge``: the other series are observed while one series is
  missing. Their returns explain part of the missing returns, and only the
  unexplained remainder is bridged.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.structural import UnobservedComponents


def _gap_steps(log_price, first_day, last_day):
    """Known endpoints of a gap and the number of one-day steps between them."""
    left, right = first_day - 1, last_day + 1
    if left not in log_price.index or right not in log_price.index:
        raise ValueError("the gap must have an observed day on both sides")
    x_left, x_right = log_price.loc[left], log_price.loc[right]
    if np.isnan(x_left) or np.isnan(x_right):
        raise ValueError("the days next to the gap must be observed")
    return x_left, x_right, right - left


def _bridge(x_left, x_right, explained, sigma):
    """Condition a random walk with known increments on its end value.

    ``explained`` holds the T predicted one-day returns from the left endpoint
    to the right endpoint. The remainder ``(x_right - x_left) - sum(explained)``
    is spread evenly over the steps, and the variance at step i is
    ``sigma^2 * i * (T - i) / T``. Returns mean and sd for the T - 1 missing days.
    """
    steps = len(explained)
    i = np.arange(1, steps)
    cumulative = np.cumsum(explained)
    remainder = (x_right - x_left) - cumulative[-1]
    mean = x_left + cumulative[:-1] + i / steps * remainder
    sd = sigma * np.sqrt(i * (steps - i) / steps)
    return mean, sd


def brownian_bridge(log_price, first_day, last_day):
    """Linear interpolation of the log price with random-walk bridge variance."""
    x_left, x_right, steps = _gap_steps(log_price, first_day, last_day)
    sigma = log_price.diff().std()
    mean, sd = _bridge(x_left, x_right, np.zeros(steps), sigma)
    return _frame(mean, sd, first_day, last_day)


def regression_bridge(log_prices, target, first_day, last_day, offsets=(-1, 0, 1)):
    """Bridge the part of the missing returns that the other series do not explain.

    The target's daily return is regressed on the returns of the other series
    at the given day offsets (-1 is the day before, +1 the day after), using
    every day outside the gap where all of them are observed. Inside the gap
    the fitted returns give the shape of the path, and ``_bridge`` makes it
    meet the known price after the gap.
    """
    x_left, x_right, steps = _gap_steps(log_prices[target], first_day, last_day)
    others = [c for c in log_prices.columns if c != target]

    # Isolated missing days in the explanatory series are filled linearly so
    # that they do not remove rows inside the gap. Rows built from filled
    # values are left out of the fit.
    observed = log_prices[others].notna()
    filled_returns = log_prices[others].interpolate(limit_area="inside").diff()
    genuine = (observed & observed.shift(1, fill_value=False))

    features, usable = {}, pd.Series(True, index=log_prices.index)
    for offset in offsets:
        for name in others:
            features[f"{name}@{offset:+d}"] = filled_returns[name].shift(-offset)
            usable &= genuine[name].shift(-offset, fill_value=False)
    X = pd.DataFrame(features)
    y = log_prices[target].diff()

    gap_days = np.arange(first_day, last_day + 2)  # returns into each missing day and the day after
    train = usable & y.notna() & ~X.index.isin(gap_days)
    design = np.column_stack([np.ones(train.sum()), X[train].to_numpy()])
    coef, *_ = np.linalg.lstsq(design, y[train].to_numpy(), rcond=None)
    residual = y[train].to_numpy() - design @ coef
    sigma = residual.std(ddof=design.shape[1])

    X_gap = X.loc[gap_days]
    if X_gap.isna().any().any():
        raise ValueError("an explanatory series is missing inside or next to the gap")
    explained = coef[0] + X_gap.to_numpy() @ coef[1:]
    mean, sd = _bridge(x_left, x_right, explained, sigma)
    return _frame(mean, sd, first_day, last_day)


def local_trend(log_price, first_day, last_day, margin=50):
    """Kalman smoother for a local linear trend plus AR(1) model.

    Same model as the original notebook, fitted on the gap and ``margin``
    observed days on each side. Unlike the original, the interpolated value is
    the full smoothed signal (trend plus AR component), not the trend alone.
    """
    window = log_price.loc[first_day - margin:last_day + margin]
    model = UnobservedComponents(window.to_numpy(), level=True, trend=True, irregular=True,
                                 stochastic_level=True, stochastic_trend=True, autoregressive=1)
    with warnings.catch_warnings():
        # On some short windows the optimiser stops at its iteration limit. The
        # smoother still runs on the last parameters, and the validation scores
        # this method as it is.
        warnings.simplefilter("ignore")
        result = model.fit(disp=False, maxiter=200)
    design = np.asarray(model.ssm["design"]).reshape(1, -1)      # (1, k_states), time-invariant
    states = result.smoothed_state                               # (k_states, n)
    covariances = result.smoother_results.smoothed_state_cov     # (k_states, k_states, n)
    signal = (design @ states).ravel()
    variance = np.einsum("ij,jkn,ik->n", design, covariances, design)
    inside = (window.index >= first_day) & (window.index <= last_day)
    return _frame(signal[inside], np.sqrt(np.maximum(variance[inside], 0.0)), first_day, last_day)


def _frame(mean, sd, first_day, last_day):
    return pd.DataFrame({"mean": mean, "sd": sd}, index=pd.RangeIndex(first_day, last_day + 1, name="day"))


def interval(estimate, level=0.95):
    """Lower and upper log-price bounds of a central interval."""
    z = norm.ppf(0.5 + level / 2)
    return estimate["mean"] - z * estimate["sd"], estimate["mean"] + z * estimate["sd"]


METHODS = {
    "brownian_bridge": lambda lp, target, a, b: brownian_bridge(lp[target], a, b),
    "local_trend": lambda lp, target, a, b: local_trend(lp[target], a, b),
    "regression_bridge": regression_bridge,
}


def candidate_blocks(log_prices, length, margin=50):
    """First days of all blocks of ``length`` days with complete data around them."""
    complete = log_prices.notna().all(axis=1).to_numpy()
    days = log_prices.index.to_numpy()
    window = length + 2 * margin
    ok = np.convolve(complete, np.ones(window, dtype=int), mode="valid") == window
    return days[np.flatnonzero(ok) + margin]


def validate(log_prices, length=50, n_blocks=40, seed=0, level=0.95):
    """Hide blocks of known prices and score each method on recovering them.

    For every series, ``n_blocks`` blocks of ``length`` days are drawn from
    stretches where all series are observed. Each block is hidden in that one
    series and interpolated by every method. Returns one row per (series,
    block, method) with the RMSE of the log price, the share of hidden days
    inside the interval, and the mean interval width.
    """
    rng = np.random.default_rng(seed)
    starts = candidate_blocks(log_prices, length)
    rows = []
    for target in log_prices.columns:
        for first_day in rng.choice(starts, size=n_blocks, replace=False):
            last_day = int(first_day) + length - 1
            truth = log_prices.loc[first_day:last_day, target]
            hidden = log_prices.copy()
            hidden.loc[first_day:last_day, target] = np.nan
            for name, method in METHODS.items():
                estimate = method(hidden, target, int(first_day), last_day)
                lower, upper = interval(estimate, level)
                rows.append({
                    "series": target, "first_day": int(first_day), "method": name,
                    "rmse": float(np.sqrt(np.mean((estimate["mean"] - truth) ** 2))),
                    "coverage": float(((truth >= lower) & (truth <= upper)).mean()),
                    "width": float((upper - lower).mean()),
                })
    return pd.DataFrame(rows)


def fill_gaps(log_prices, method="regression_bridge"):
    """Interpolate every interior gap of every series.

    Returns the filled log prices and a long table of the interpolated days
    with their mean and standard deviation. Gaps are filled from the original
    data, so one series' interpolated values never feed another series' fit.
    """
    from .data import missing_runs

    filled, records = log_prices.copy(), []
    for target in log_prices.columns:
        for first_day, last_day in missing_runs(log_prices[target]):
            if last_day - first_day + 1 == 1:
                estimate = brownian_bridge(log_prices[target], first_day, last_day)
            else:
                estimate = METHODS[method](log_prices, target, first_day, last_day)
            filled.loc[first_day:last_day, target] = estimate["mean"]
            records.append(estimate.assign(series=target).reset_index())
    return filled, pd.concat(records, ignore_index=True)
