"""Descriptive tests on the log returns.

Writes results/stationarity.csv, results/dependence.csv and results/lead_lag.csv.
"""
from pathlib import Path

import numpy as np

from fts.data import load_history, log_returns
from fts.diagnostics import dependence_table, lead_lag_table, stationarity_table

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    RESULTS.mkdir(exist_ok=True)
    history, forecast_days, outlier_days = load_history()
    returns = log_returns(history)
    print(f"{len(history)} observed days, {len(forecast_days)} to forecast, outlier days {outlier_days}")
    print(f"valid one-day returns per series: {int(returns.notna().sum().min())}")

    stationarity = stationarity_table(np.log(history), returns)
    stationarity.round(4).to_csv(RESULTS / "stationarity.csv")
    print("\nADF p-values (unit root is the null hypothesis):")
    print(stationarity.round(4).to_string())

    dependence = dependence_table(returns)
    dependence.to_csv(RESULTS / "dependence.csv", float_format="%.4g")
    print("\nSerial dependence in the returns:")
    print(dependence.to_string(float_format=lambda v: f"{v:.3g}"))

    lead_lag = lead_lag_table(returns)
    lead_lag.round(4).to_csv(RESULTS / "lead_lag.csv", index=False)
    strong = lead_lag[(lead_lag["corr"].abs() > 2 * lead_lag["threshold"]) & (lead_lag["target"] != lead_lag["source"])]
    print("\nLead-lag pairs with |corr| above twice the 5% bound:")
    print(strong.sort_values("corr", key=abs, ascending=False).round(3).to_string(index=False))
    print("\nSame-day return correlations:")
    print(returns.corr().round(2).to_string())
    returns.corr().round(4).to_csv(RESULTS / "correlation.csv")


if __name__ == "__main__":
    main()
