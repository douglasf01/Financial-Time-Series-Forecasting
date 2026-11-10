import numpy as np
import pandas as pd

from fts.data import (SERIES, fill_single_days, load_history, load_prices, log_returns,
                      missing_runs, remove_outliers, split_history)


def test_file_structure():
    prices = load_prices()
    assert list(prices.columns) == SERIES
    assert prices.index[0] == 1 and prices.index[-1] == 5456
    assert (np.diff(prices.index) == 1).all()


def test_outliers_are_the_five_placeholder_days():
    prices = load_prices()
    cleaned, days = remove_outliers(prices)
    assert days == [36, 1194, 2836, 3430, 4118]
    assert (prices.loc[days] == 1000).all().all()
    assert cleaned.loc[days].isna().all().all()
    # The result does not depend on the exact factor.
    for factor in (3, 50, 100):
        assert remove_outliers(prices, factor)[1] == days


def test_history_and_gaps():
    history, forecast_days, _ = load_history()
    assert len(history) == 5256 and len(forecast_days) == 200
    assert forecast_days[0] == 5257
    for name in SERIES:
        long_runs = [(a, b) for a, b in missing_runs(history[name]) if b > a]
        assert len(long_runs) == 1 and long_runs[0][1] - long_runs[0][0] + 1 == 50


def test_returns_never_span_a_gap():
    prices = pd.DataFrame({"a": [1.0, 2.0, np.nan, np.nan, 8.0, 16.0]}, index=range(1, 7))
    returns = log_returns(prices)["a"]
    assert returns.isna().tolist() == [True, False, True, True, True, False]
    assert np.isclose(returns.loc[6], np.log(2))
    # Dropping the missing rows first would create a three-day return of log(4).
    assert not np.isclose(returns.dropna(), np.log(4)).any()


def test_fill_single_days_only_fills_isolated_days():
    prices = pd.DataFrame({"a": [1.0, np.nan, 4.0, np.nan, np.nan, 8.0]}, index=range(1, 7))
    filled = fill_single_days(prices)["a"]
    assert np.isclose(filled.loc[2], 2.0)          # geometric mean of 1 and 4
    assert filled.loc[[4, 5]].isna().all()


def test_split_history_keeps_interior_gaps():
    prices = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": [1.0, 2.0, 3.0, np.nan]}, index=range(1, 5))
    history, future = split_history(prices)
    assert list(history.index) == [1, 2, 3] and list(future) == [4]
