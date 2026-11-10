import numpy as np
import pandas as pd

from fts.interpolate import _bridge, brownian_bridge, fill_gaps, interval, regression_bridge, validate


def simulated_log_prices(n=3000, seed=0):
    """Two series where b's return is a's return plus small noise."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.01, n)
    b = a + rng.normal(0, 0.002, n)
    c = rng.normal(0, 0.01, n)
    frame = pd.DataFrame({"a": a, "b": b, "c": c}).cumsum()
    frame.index = pd.RangeIndex(1, n + 1, name="day")
    return frame


def test_bridge_meets_both_endpoints():
    mean, sd = _bridge(1.0, 2.0, np.zeros(10), sigma=0.1)
    assert len(mean) == 9
    assert np.allclose(mean, np.linspace(1.0, 2.0, 11)[1:-1])      # straight line
    assert np.isclose(sd[0], sd[-1]) and sd.argmax() == 4          # symmetric, widest in the middle
    assert np.isclose(sd[4], 0.1 * np.sqrt(5 * 5 / 10))


def test_bridge_matches_simulated_conditional_distribution():
    """Mean and sd agree with random walks filtered on their end value."""
    rng = np.random.default_rng(1)
    steps, sigma = 8, 1.0
    walks = rng.normal(0, sigma, (400_000, steps)).cumsum(axis=1)
    kept = walks[np.abs(walks[:, -1] - 2.0) < 0.05]
    mean, sd = _bridge(0.0, 2.0, np.zeros(steps), sigma)
    assert np.allclose(kept[:, :-1].mean(axis=0), mean, atol=0.05)
    assert np.allclose(kept[:, :-1].std(axis=0), sd, atol=0.05)


def test_regression_bridge_uses_the_other_series():
    """Averaged over many hidden blocks: far smaller error, honest intervals."""
    log_prices = simulated_log_prices()
    plain_errors, informed_errors, inside = [], [], []
    for first_day in range(200, 2800, 100):
        last_day = first_day + 49
        truth = log_prices.loc[first_day:last_day, "b"]
        hidden = log_prices.copy()
        hidden.loc[first_day:last_day, "b"] = np.nan
        plain = brownian_bridge(hidden["b"], first_day, last_day)
        informed = regression_bridge(hidden, "b", first_day, last_day)
        plain_errors.append(np.sqrt(np.mean((plain["mean"] - truth) ** 2)))
        informed_errors.append(np.sqrt(np.mean((informed["mean"] - truth) ** 2)))
        lower, upper = interval(informed)
        inside.append(((truth >= lower) & (truth <= upper)).mean())
        assert informed["sd"].max() < 0.3 * plain["sd"].max()
    assert np.mean(informed_errors) < 0.3 * np.mean(plain_errors)
    assert 0.90 < np.mean(inside) <= 1.0


def test_regression_bridge_does_not_read_the_hidden_values():
    log_prices = simulated_log_prices()
    one = log_prices.copy()
    one.loc[500:549, "b"] = np.nan
    first = regression_bridge(one, "b", 500, 549)
    log_prices.loc[500:549, "b"] += 5.0  # changing the hidden truth must not matter
    again = log_prices.copy()
    again.loc[500:549, "b"] = np.nan
    assert np.allclose(first["mean"], regression_bridge(again, "b", 500, 549)["mean"])


def test_fill_gaps_fills_everything_and_keeps_observed_values():
    log_prices = simulated_log_prices(n=1500)
    holes = log_prices.copy()
    holes.loc[300:349, "a"] = np.nan
    holes.loc[700:749, "b"] = np.nan
    holes.loc[900, :] = np.nan
    filled, table = fill_gaps(holes)
    assert not filled.isna().any().any()
    observed = holes.notna()
    assert np.allclose(filled[observed], holes[observed], equal_nan=True)
    assert len(table) == 50 + 50 + 3


def test_validation_intervals_cover_on_simulated_data():
    scores = validate(simulated_log_prices(), length=30, n_blocks=15, seed=0)
    by_method = scores.groupby("method")[["rmse", "coverage"]].mean()
    assert by_method.loc["brownian_bridge", "coverage"] > 0.85
    assert by_method.loc["regression_bridge", "coverage"] > 0.85
