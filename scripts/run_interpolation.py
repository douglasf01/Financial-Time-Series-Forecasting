"""Validate the interpolation methods on hidden blocks, then fill the real gaps.

Writes results/interpolation_validation.csv, results/interpolated_gaps.csv and
results/interpolation_<series>.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fts.data import load_history, missing_runs
from fts.interpolate import METHODS, fill_gaps, interval, validate

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    RESULTS.mkdir(exist_ok=True)
    history, _, _ = load_history()
    log_prices = np.log(history)

    scores = validate(log_prices, length=50, n_blocks=40, seed=0)
    summary = scores.groupby(["series", "method"])[["rmse", "coverage", "width"]].mean().round(4)
    overall = scores.groupby("method")[["rmse", "coverage", "width"]].mean().round(4)
    summary.to_csv(RESULTS / "interpolation_validation.csv")
    print("Hidden-block validation, mean over 40 blocks of 50 days per series (log price):")
    print(summary.to_string())
    print("\nAll series pooled:")
    print(overall.to_string())

    _, table = fill_gaps(log_prices, method="regression_bridge")
    lower = np.exp(table["mean"] - 1.96 * table["sd"])
    upper = np.exp(table["mean"] + 1.96 * table["sd"])
    out = table.assign(price=np.exp(table["mean"]), lower_95=lower, upper_95=upper)
    out[["series", "day", "price", "lower_95", "upper_95"]].to_csv(
        RESULTS / "interpolated_gaps.csv", index=False, float_format="%.6f")

    for target in log_prices.columns:
        for first_day, last_day in missing_runs(log_prices[target]):
            if last_day - first_day < 1:
                continue  # single outlier days are not plotted
            fig, ax = plt.subplots(figsize=(9, 3.5))
            window = history.loc[first_day - 50:last_day + 50, target]
            ax.plot(window.index, window, color="black", lw=1, label="observed")
            for name, colour in [("brownian_bridge", "tab:gray"), ("regression_bridge", "tab:red")]:
                estimate = METHODS[name](log_prices, target, first_day, last_day)
                low, high = interval(estimate)
                ax.plot(estimate.index, np.exp(estimate["mean"]), color=colour, lw=1.2, label=name)
                ax.fill_between(estimate.index, np.exp(low), np.exp(high), color=colour, alpha=0.15)
            ax.set(title=f"{target}: interpolated days {first_day} to {last_day} with 95% intervals",
                   xlabel="day", ylabel="price")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(RESULTS / f"interpolation_{target}.png", dpi=130)
            plt.close(fig)


if __name__ == "__main__":
    main()
