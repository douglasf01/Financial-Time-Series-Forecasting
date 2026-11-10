"""Loading and cleaning of the price data.

The file holds seven daily price series over 5456 days. Three things need
handling before any modelling:

* five days carry the placeholder value 1000 in every series (outliers);
* each series has one interior gap of 50 missing days;
* the last 200 days are missing in every series and are the forecast target.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "spiff_data.csv"
SERIES = ["gurkor", "guitars", "slingshots", "stocks", "sugar", "water", "tranquillity"]


def load_prices(path=DATA_PATH):
    """Return the raw prices as a DataFrame indexed by day (1-based)."""
    df = pd.read_csv(path, index_col=0)
    return df.set_index("day")[SERIES].astype(float)


def remove_outliers(prices, factor=20.0):
    """Set implausible prices to NaN.

    A price is an outlier when it exceeds ``factor`` times the median of its
    series. The placeholder days are about 100 to 300 times the median, while
    the largest genuine price is under 2.5 times it, so any factor between
    roughly 3 and 100 gives the same result.

    Returns the cleaned prices and the list of days that were changed.
    """
    mask = prices > factor * prices.median()
    days = prices.index[mask.any(axis=1)].tolist()
    return prices.mask(mask), days


def split_history(prices):
    """Split into the observed history and the trailing days to forecast.

    The forecast days are the block of rows at the end where every series is
    missing. Returns ``(history, forecast_days)``.
    """
    observed = prices.notna().any(axis=1)
    last = observed[observed].index[-1]
    return prices.loc[:last], prices.loc[last + 1:].index


def log_returns(prices):
    """One-day log returns.

    A return is NaN whenever the price on that day or on the day before is
    missing, so no return ever spans more than one day. (Dropping the missing
    rows first and then differencing would turn every gap into one large
    multi-day "daily" return.)
    """
    return np.log(prices).diff()


def missing_runs(series):
    """Runs of consecutive missing days as a list of ``(first_day, last_day)``."""
    missing = series.isna().to_numpy()
    days = series.index.to_numpy()
    runs, start = [], None
    for i, is_missing in enumerate(missing):
        if is_missing and start is None:
            start = i
        elif not is_missing and start is not None:
            runs.append((int(days[start]), int(days[i - 1])))
            start = None
    if start is not None:
        runs.append((int(days[start]), int(days[-1])))
    return runs


def fill_single_days(prices):
    """Fill missing runs of exactly one day by linear interpolation of the log price.

    Used for the five outlier days, so that a single bad day does not break a
    200-day evaluation window. Longer gaps are left missing.
    """
    filled = prices.copy()
    for name in prices.columns:
        for first_day, last_day in missing_runs(prices[name]):
            inside = first_day - 1 in prices.index and last_day + 1 in prices.index
            if first_day == last_day and inside:
                neighbours = prices.loc[[first_day - 1, first_day + 1], name]
                if neighbours.notna().all():
                    filled.loc[first_day, name] = np.exp(np.log(neighbours).mean())
    return filled


def load_history(path=DATA_PATH):
    """Cleaned observed history, the forecast days and the removed outlier days."""
    prices, outlier_days = remove_outliers(load_prices(path))
    history, forecast_days = split_history(prices)
    return history, forecast_days, outlier_days
