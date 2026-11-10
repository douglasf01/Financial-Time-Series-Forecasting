import numpy as np
import pandas as pd

from fts.forecast import (backtest_intervals, bootstrap_paths, direct_r2, ewma_t_paths,
                          garch_paths, gaussian_quantiles, lagged_features, summed_step_quantiles)


def random_walk_returns(n=3000, sigma=0.01, seed=0, columns=("a",)):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, sigma, (n, len(columns))), columns=list(columns),
                        index=pd.RangeIndex(1, n + 1, name="day"))


def test_lagged_features_use_only_the_past():
    returns = pd.DataFrame({"a": np.arange(10.0)}, index=range(1, 11))
    features = lagged_features(returns, lags=3)
    assert features.loc[5].tolist() == [4.0, 3.0, 2.0]     # days 5, 4, 3


def test_direct_r2_finds_a_planted_one_day_lead_and_nothing_else():
    rng = np.random.default_rng(0)
    n = 4000
    leader = rng.normal(0, 0.01, n)
    follower = 0.6 * np.roll(leader, 1) + rng.normal(0, 0.008, n)
    returns = pd.DataFrame({"leader": leader, "follower": follower}, index=pd.RangeIndex(1, n + 1))
    table = direct_r2(returns, horizons=(1, 2), lags=3, kinds=("ridge",), n_folds=3)
    r2 = table.set_index(["series", "horizon"])["r2_oos"]
    assert r2["follower", 1] > 0.25            # true R^2 is 0.36
    assert abs(r2["follower", 2]) < 0.03
    assert abs(r2["leader", 1]) < 0.03


def test_gaussian_interval_width_grows_with_the_square_root_of_time():
    q = gaussian_quantiles(random_walk_returns()["a"], horizon=100)
    width = q["upper"] - q["lower"]
    assert np.isclose(width.loc[100] / width.loc[1], 10.0)


def test_summed_step_quantiles_grow_linearly():
    q = summed_step_quantiles(random_walk_returns()["a"], horizon=100)
    width = q["upper"] - q["lower"]
    assert np.isclose(width.loc[100] / width.loc[1], 100.0)


def test_path_simulators_shape_and_scale():
    returns = random_walk_returns()["a"]
    for simulate in (bootstrap_paths, garch_paths, ewma_t_paths):
        paths = simulate(returns, 50, n_paths=1500, seed=1)
        assert paths.shape == (1500, 50)
        total_sd = paths.sum(axis=1).std()
        assert 0.7 < total_sd / (0.01 * np.sqrt(50)) < 1.3, simulate.__name__


def test_simulators_are_reproducible():
    returns = random_walk_returns()["a"]
    for simulate in (bootstrap_paths, garch_paths, ewma_t_paths):
        assert np.array_equal(simulate(returns, 20, n_paths=50, seed=3), simulate(returns, 20, n_paths=50, seed=3))


def test_backtest_coverage_on_a_true_random_walk():
    """On i.i.d. Gaussian returns the Gaussian interval covers about 95%,
    and the summed-quantile interval covers far too much at long horizons."""
    returns = random_walk_returns(n=4000, columns=("a", "b", "c", "d"))
    methods = {
        "gaussian": lambda r, h, seed: gaussian_quantiles(r, h),
        "summed": lambda r, h, seed: summed_step_quantiles(r, h),
    }
    scores = backtest_intervals(returns, horizon=50, report=(1, 50), first_origin=1000, step=10, methods=methods)
    coverage = scores.groupby(["method", "horizon"])["hit"].mean()
    assert 0.90 < coverage["gaussian", 1] < 0.99
    assert 0.88 < coverage["gaussian", 50] < 0.99
    assert coverage["summed", 50] == 1.0
    width = scores.groupby(["method", "horizon"])["width"].mean()
    assert width["summed", 50] > 5 * width["gaussian", 50]


def test_backtest_never_shows_the_future_to_a_method():
    returns = random_walk_returns(n=1500)
    seen = []

    def spy(past, horizon, seed):
        seen.append(past.index[-1])
        return gaussian_quantiles(past, horizon)

    scores = backtest_intervals(returns, horizon=20, report=(20,), first_origin=1000, step=100,
                                methods={"spy": spy})
    assert seen == scores["origin"].tolist()
    assert all(origin < 1500 - 19 for origin in seen)
