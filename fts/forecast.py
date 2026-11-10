"""Multi-step forecasts of the log price with prediction intervals.

Two questions are kept apart:

1. Is the mean return predictable, and for how many days ahead?
   ``direct_r2`` fits one model per horizon on lagged returns of all series
   and scores it out of sample against the historical mean.
2. How uncertain is the price h days ahead? The ``*_paths`` and
   ``*_quantiles`` functions give intervals for the cumulative log return,
   and ``backtest_intervals`` checks their coverage on past data.
"""
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LEVEL = 0.95


# ---------------------------------------------------------------- mean return

def lagged_features(returns, lags):
    """Returns of every series at lags 0..lags-1 relative to the forecast origin.

    Row t holds what is known at the end of day t: the returns of days t,
    t-1, ..., t-lags+1.
    """
    return pd.concat({f"{name}_lag{k}": returns[name].shift(k)
                      for name in returns.columns for k in range(lags)}, axis=1)


def make_model(kind):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
    if kind == "random_forest":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=25, max_features=0.33,
                                     n_jobs=-1, random_state=0)
    raise ValueError(f"unknown model: {kind}")


def direct_r2(returns, horizons=(1, 2, 3, 5, 10), lags=5, kinds=("ridge", "random_forest"),
              n_folds=5, test_share=0.5):
    """Out-of-sample R^2 of direct h-step forecasts of the daily return.

    For horizon h the model predicts the return of day t+h from the features
    known at day t (direct forecasting: one model per horizon, no recursion).
    The last ``test_share`` of the origins is split into ``n_folds``
    consecutive test blocks. For each block the model is fitted on the
    origins whose target day lies before the block, so no training target
    overlaps the test period. R^2 is measured against the mean return of the
    same training data: positive means the model beats "no predictability".
    """
    X_all = lagged_features(returns, lags)
    rows = []
    for target in returns.columns:
        for h in horizons:
            y_all = returns[target].shift(-h)
            ok = X_all.notna().all(axis=1) & y_all.notna()
            X, y = X_all[ok].to_numpy(), y_all[ok].to_numpy()
            days = X_all.index[ok].to_numpy()
            first_test = int(len(y) * (1 - test_share))
            edges = np.linspace(first_test, len(y), n_folds + 1).astype(int)
            sse = {kind: 0.0 for kind in kinds}
            sse_mean = 0.0
            for start, stop in zip(edges[:-1], edges[1:]):
                train = days + h < days[start]
                baseline = y[train].mean()
                sse_mean += np.sum((y[start:stop] - baseline) ** 2)
                for kind in kinds:
                    model = make_model(kind).fit(X[train], y[train])
                    sse[kind] += np.sum((y[start:stop] - model.predict(X[start:stop])) ** 2)
            for kind in kinds:
                rows.append({"series": target, "horizon": h, "model": kind,
                             "r2_oos": 1 - sse[kind] / sse_mean})
    return pd.DataFrame(rows)


def lead_lag_shift(returns, target, steps, lags=5):
    """Expected return minus the historical mean for the next ``steps`` days.

    Ridge direct models fitted on all data, evaluated at the last observed
    day. Used to move the start of a forecast path when ``direct_r2`` shows
    that the first steps are predictable.
    """
    X_all = lagged_features(returns, lags)
    last = X_all.iloc[[-1]]
    if last.isna().any().any():
        raise ValueError("the last days of the history must be observed")
    shifts = []
    for h in range(1, steps + 1):
        y_all = returns[target].shift(-h)
        ok = X_all.notna().all(axis=1) & y_all.notna()
        model = make_model("ridge").fit(X_all[ok].to_numpy(), y_all[ok].to_numpy())
        shifts.append(float(model.predict(last.to_numpy())[0]) - y_all[ok].mean())
    return np.array(shifts)


# ------------------------------------------------------------------ intervals

def _quantiles(cumulative_paths, level):
    lower, median, upper = np.quantile(cumulative_paths, [(1 - level) / 2, 0.5, (1 + level) / 2], axis=0)
    return pd.DataFrame({"lower": lower, "median": median, "upper": upper},
                        index=pd.RangeIndex(1, cumulative_paths.shape[1] + 1, name="horizon"))


def gaussian_quantiles(returns, horizon, level=LEVEL):
    """Random walk with drift and constant volatility: mean*h +/- z*sd*sqrt(h)."""
    x = returns.dropna().to_numpy()
    h = np.arange(1, horizon + 1)
    z = norm.ppf(0.5 + level / 2)
    centre, spread = x.mean() * h, z * x.std(ddof=1) * np.sqrt(h)
    return pd.DataFrame({"lower": centre - spread, "median": centre, "upper": centre + spread},
                        index=pd.RangeIndex(1, horizon + 1, name="horizon"))


def summed_step_quantiles(returns, horizon, level=LEVEL):
    """The interval construction of the original notebook, kept for comparison.

    It takes the quantiles of the one-day return and adds them up over the
    horizon. That assumes every day lands on its own 2.5% (or 97.5%) quantile,
    so the width grows like h instead of sqrt(h) and the interval is far too
    wide at long horizons.
    """
    x = returns.dropna().to_numpy()
    lower, median, upper = np.quantile(x, [(1 - level) / 2, 0.5, (1 + level) / 2])
    h = np.arange(1, horizon + 1)
    return pd.DataFrame({"lower": lower * h, "median": median * h, "upper": upper * h},
                        index=pd.RangeIndex(1, horizon + 1, name="horizon"))


def bootstrap_paths(returns, horizon, n_paths=2000, block=20, seed=0):
    """Circular block bootstrap of past returns.

    Blocks of ``block`` consecutive days are drawn with replacement and joined
    into paths of ``horizon`` days. Blocks keep short-range dependence such as
    volatility clustering, but a path does not depend on the volatility at the
    forecast origin. Returns simulated one-day returns, shape (n_paths, horizon).
    """
    x = returns.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, len(x), size=(n_paths, n_blocks))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]) % len(x)
    return x[index].reshape(n_paths, -1)[:, :horizon]


def garch_paths(returns, horizon, n_paths=2000, seed=0):
    """Filtered historical simulation from a GARCH(1,1) model with constant mean.

    The model is fitted to the past returns. Future paths start from the
    volatility at the forecast origin and draw their shocks from the model's
    own standardised residuals, so fat tails are kept and no distribution is
    assumed. Returns simulated one-day returns, shape (n_paths, horizon).
    """
    x = returns.dropna().to_numpy() * 100.0  # percent, for a well-scaled likelihood
    result = arch_model(x, mean="Constant", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
    forecast = result.forecast(horizon=horizon, method="bootstrap", simulations=n_paths,
                               random_state=np.random.RandomState(seed), reindex=False)
    return forecast.simulations.values[-1] / 100.0


def ewma_t_paths(returns, horizon, n_paths=2000, decay=0.94, drift_shrinkage=0.5, seed=0):
    """Student-t random walk with EWMA volatility (a project teammate's model).

    The volatility at the origin is an exponentially weighted average of past
    squared returns, and it keeps updating with the simulated shocks. Shocks
    are Student-t with the degrees of freedom fitted to the standardised
    returns, scaled to unit variance, and the drift is the historical mean
    times ``drift_shrinkage``. This is the per-series version of the model; the
    original also correlates the shocks across series, which does not change
    the interval of a single series. Returns shape (n_paths, horizon).
    """
    x = returns.dropna().to_numpy()
    mean = x.mean()
    centred = x - mean
    variance = centred.var(ddof=1)
    for value in centred:
        variance = decay * variance + (1 - decay) * value ** 2
    dof = float(np.clip(student_t.fit(centred / centred.std(ddof=1), floc=0.0)[0], 2.1, 100.0))

    rng = np.random.default_rng(seed)
    current = np.full(n_paths, variance)
    paths = np.empty((n_paths, horizon))
    for h in range(horizon):
        shock = np.sqrt(current) * rng.standard_t(dof, size=n_paths) * np.sqrt((dof - 2) / dof)
        paths[:, h] = drift_shrinkage * mean + shock
        current = decay * current + (1 - decay) * shock ** 2
    return paths


METHODS = {
    "summed_step_quantiles": lambda r, h, seed: summed_step_quantiles(r, h),
    "gaussian_random_walk": lambda r, h, seed: gaussian_quantiles(r, h),
    "block_bootstrap": lambda r, h, seed: _quantiles(np.cumsum(bootstrap_paths(r, h, seed=seed), axis=1), LEVEL),
    "garch_fhs": lambda r, h, seed: _quantiles(np.cumsum(garch_paths(r, h, seed=seed), axis=1), LEVEL),
    "ewma_student_t": lambda r, h, seed: _quantiles(np.cumsum(ewma_t_paths(r, h, seed=seed), axis=1), LEVEL),
}


def backtest_intervals(returns, horizon=200, report=(1, 5, 20, 50, 100, 200), first_origin=2000,
                       step=50, methods=None):
    """Rolling-origin check of the 95% intervals for the cumulative log return.

    ``returns`` must have no missing values after ``first_origin``. At every
    origin (each ``step`` rows) the methods see the returns up to and
    including the origin and forecast the next ``horizon`` days; the realised
    cumulative return is then compared with the interval at the horizons in
    ``report``. Returns one row per (series, origin, method, horizon) with
    hit, width, interval score and the error of the interval median. Origins
    closer than ``horizon`` overlap in their test windows, so the rows are not
    independent.
    """
    methods = methods or METHODS
    alpha = 1 - LEVEL
    rows = []
    for name in returns.columns:
        x = returns[name]
        for origin in range(first_origin, len(x) - horizon + 1, step):
            past, future = x.iloc[:origin], x.iloc[origin:origin + horizon]
            if future.isna().any():
                continue
            realised = future.cumsum().to_numpy()
            for method, function in methods.items():
                q = function(past, horizon, origin)
                for h in report:
                    low, mid, high = q.iloc[h - 1][["lower", "median", "upper"]]
                    y = realised[h - 1]
                    score = (high - low) + 2 / alpha * (max(low - y, 0.0) + max(y - high, 0.0))
                    rows.append({"series": name, "origin": int(x.index[origin - 1]), "method": method,
                                 "horizon": h, "hit": low <= y <= high, "width": high - low,
                                 "interval_score": score, "median_error": y - mid, "realised": y})
    return pd.DataFrame(rows)


def forecast_prices(returns, last_price, forecast_days, shift=None, n_paths=5000, seed=0):
    """Price forecast with 95% interval for one series from the GARCH simulation.

    ``shift`` optionally adds expected-return adjustments to the first days
    (see ``lead_lag_shift``).
    """
    horizon = len(forecast_days)
    paths = garch_paths(returns, horizon, n_paths=n_paths, seed=seed)
    if shift is not None:
        paths[:, :len(shift)] += shift
    q = _quantiles(np.cumsum(paths, axis=1), LEVEL)
    out = np.exp(np.log(last_price) + q)
    out.index = pd.Index(forecast_days, name="day")
    return out
