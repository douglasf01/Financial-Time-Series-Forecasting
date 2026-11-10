"""Predictability check, interval backtest and the final 200-day forecast.

Writes to results/:
  predictability_r2.csv          out-of-sample R^2 of direct h-step models
  interval_backtest.csv          coverage, width and interval score by method and horizon
  interval_backtest_by_series.csv
  point_forecast_backtest.csv    RMSE of "no change" against "historical drift"
  forecast.csv                   median and 95% interval for the 200 missing days
  interval_backtest.png, forecast.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fts.data import fill_single_days, load_history, log_returns
from fts.forecast import backtest_intervals, direct_r2, forecast_prices, lead_lag_shift

RESULTS = Path(__file__).resolve().parent.parent / "results"
R2_THRESHOLD = 0.01  # a horizon counts as predictable when ridge beats the mean by this much


def predictability(returns):
    table = direct_r2(returns)
    wide = table.pivot_table(index=["series", "model"], columns="horizon", values="r2_oos")
    wide.round(4).to_csv(RESULTS / "predictability_r2.csv")
    print("Out-of-sample R^2 of direct h-step return forecasts (columns: horizon in days):")
    print(wide.round(3).to_string())
    return table


def interval_backtest(returns, step):
    scores = backtest_intervals(returns, step=step)
    scores.to_csv(RESULTS / "interval_backtest_raw.csv", index=False, float_format="%.6f")

    def summarise(keys):
        return scores.groupby(keys).agg(coverage=("hit", "mean"), width=("width", "mean"),
                                        interval_score=("interval_score", "mean"),
                                        n=("hit", "size")).round(4)

    by_method = summarise(["method", "horizon"])
    by_method.to_csv(RESULTS / "interval_backtest.csv")
    summarise(["series", "method", "horizon"]).to_csv(RESULTS / "interval_backtest_by_series.csv")
    print("\n95% interval backtest, all series pooled:")
    print(by_method.to_string())

    # Point forecasts: is the historical drift better than "no change"?
    one = scores[scores["method"] == "gaussian_random_walk"]
    drift_error = one["median_error"]
    points = pd.DataFrame({"rmse_no_change": one["realised"] ** 2, "rmse_drift": drift_error ** 2,
                           "series": one["series"], "horizon": one["horizon"]})
    points = np.sqrt(points.groupby(["series", "horizon"])[["rmse_no_change", "rmse_drift"]].mean())
    points.round(5).to_csv(RESULTS / "point_forecast_backtest.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for method, group in by_method.reset_index().groupby("method"):
        axes[0].plot(group["horizon"], group["coverage"], marker="o", ms=3, label=method)
        axes[1].plot(group["horizon"], group["width"], marker="o", ms=3, label=method)
    axes[0].axhline(0.95, color="black", lw=0.8, ls=":")
    axes[0].set(title="Coverage of 95% intervals", xlabel="horizon (days)", ylabel="share of outcomes inside", xscale="log")
    axes[1].set(title="Mean interval width (log return)", xlabel="horizon (days)", xscale="log", yscale="log")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "interval_backtest.png", dpi=130)
    plt.close(fig)
    return by_method


def final_forecast(history, returns, forecast_days, r2):
    ridge = r2[r2["model"] == "ridge"]
    frames = []
    for name in history.columns:
        predictable = ridge[(ridge["series"] == name) & (ridge["r2_oos"] > R2_THRESHOLD)]["horizon"]
        steps = 0
        while steps + 1 in set(predictable):
            steps += 1
        shift = lead_lag_shift(returns, name, steps) if steps else None
        if steps:
            print(f"{name}: first {steps} day(s) adjusted by the lead-lag model: {np.round(shift, 5)}")
        forecast = forecast_prices(returns[name], history[name].iloc[-1], forecast_days, shift=shift)
        frames.append(forecast.assign(series=name).reset_index())
    table = pd.concat(frames, ignore_index=True)[["series", "day", "median", "lower", "upper"]]
    table.to_csv(RESULTS / "forecast.csv", index=False, float_format="%.6f")

    fig, axes = plt.subplots(4, 2, figsize=(11, 10))
    for ax, name in zip(axes.ravel(), history.columns):
        past = history[name].iloc[-600:]
        f = table[table["series"] == name]
        ax.plot(past.index, past, color="black", lw=0.8)
        ax.plot(f["day"], f["median"], color="tab:red", lw=1.2)
        ax.fill_between(f["day"], f["lower"], f["upper"], color="tab:red", alpha=0.2)
        ax.set_title(name, fontsize=10)
    axes.ravel()[-1].axis("off")
    fig.suptitle("200-day forecast: median and 95% interval (GARCH filtered historical simulation)")
    fig.tight_layout()
    fig.savefig(RESULTS / "forecast.png", dpi=130)
    plt.close(fig)
    last = table.groupby("series").tail(1).set_index("series")
    print("\nForecast for the last missing day:")
    print(last.round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=int, default=50, help="days between backtest origins")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    history, forecast_days, outlier_days = load_history()
    print(f"{len(history)} observed days, {len(forecast_days)} days to forecast, "
          f"outlier days removed: {outlier_days}")
    history = fill_single_days(history)
    returns = log_returns(history)

    r2 = predictability(returns)
    interval_backtest(returns, args.step)
    final_forecast(history, returns, forecast_days, r2)


if __name__ == "__main__":
    main()
