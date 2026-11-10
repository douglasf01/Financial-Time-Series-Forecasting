"""Descriptive tests on the log returns."""
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, bds


def stationarity_table(log_prices, returns):
    """ADF p-values for the log price levels and for the log returns."""
    rows = {}
    for name in returns.columns:
        rows[name] = {
            "adf_p_log_price": adfuller(log_prices[name].dropna())[1],
            "adf_p_log_return": adfuller(returns[name].dropna())[1],
        }
    return pd.DataFrame(rows).T


def dependence_table(returns, lags=10):
    """Tests for serial dependence in each return series.

    * ``ljung_box_p``: autocorrelation in the returns (linear dependence).
    * ``ljung_box_sq_p``: autocorrelation in the squared returns, i.e.
      volatility clustering.
    * ``bds_stat`` / ``bds_p``: BDS test of the i.i.d. hypothesis. It rejects
      for any dependence, including volatility clustering, so a rejection
      alone does not show that the mean is predictable.
    """
    rows = {}
    for name in returns.columns:
        x = returns[name].dropna().to_numpy()
        stat, p = bds(x, max_dim=2)
        rows[name] = {
            "ljung_box_p": acorr_ljungbox(x, lags=[lags])["lb_pvalue"].iloc[0],
            "ljung_box_sq_p": acorr_ljungbox(x ** 2, lags=[lags])["lb_pvalue"].iloc[0],
            "bds_stat": float(np.squeeze(stat)),
            "bds_p": float(np.squeeze(p)),
        }
    return pd.DataFrame(rows).T


def lead_lag_table(returns, max_lag=5):
    """Strongest lagged cross-correlation for every (target, source) pair.

    Entry ``corr`` is ``corr(target[t], source[t - lag])`` at the lag in
    ``1..max_lag`` with the largest absolute value. ``threshold`` is the
    approximate 5% significance bound ``2 / sqrt(n)`` for a single lag.
    """
    rows = []
    for target in returns.columns:
        for source in returns.columns:
            corrs = [returns[target].corr(returns[source].shift(k)) for k in range(1, max_lag + 1)]
            best = int(np.argmax(np.abs(corrs)))
            n = (returns[target].notna() & returns[source].shift(best + 1).notna()).sum()
            rows.append({"target": target, "source": source, "lag": best + 1,
                         "corr": corrs[best], "threshold": 2 / np.sqrt(n)})
    return pd.DataFrame(rows)
