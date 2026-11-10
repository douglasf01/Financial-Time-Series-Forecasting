"""Backtest the trading rules on the days after ``FIRST_DAY``.

Writes results/strategy_performance.csv, results/strategy_by_series.csv and
results/strategy.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fts.data import fill_single_days, load_history, log_returns
from fts.strategy import (backtest, buy_and_hold_weights, channel_breakout_weights,
                          lead_lag_weights, performance, walk_forward_predictions)

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIRST_DAY = 2000   # all interior gaps end before day 1448
COSTS = (0, 5, 10, 25, 50)


def main():
    RESULTS.mkdir(exist_ok=True)
    history, _, _ = load_history()
    prices = fill_single_days(history)
    returns = log_returns(prices)

    predictions = walk_forward_predictions(returns, FIRST_DAY)
    rules = {
        "buy_and_hold": (buy_and_hold_weights(prices, FIRST_DAY), True),
        "channel_breakout_150": (channel_breakout_weights(prices, 150), False),
        "lead_lag_long_only": (lead_lag_weights(predictions, long_only=True), False),
        "lead_lag_long_short": (lead_lag_weights(predictions), False),
    }

    rows, curves = [], {}
    for name, (weights, drift) in rules.items():
        for cost in COSTS:
            daily, turnover = backtest(weights, prices, FIRST_DAY, cost_bps=cost, drift=drift)
            rows.append({"rule": name, "cost_bps": cost, **performance(daily, turnover)})
            if cost == 10:
                curves[name] = (1 + daily).cumprod()
    table = pd.DataFrame(rows).set_index(["rule", "cost_bps"]).round(4)
    table.to_csv(RESULTS / "strategy_performance.csv")
    print(f"Days {FIRST_DAY + 1} to {prices.index[-1]}, by trading cost in basis points:")
    print(table.to_string())

    # Where does the long/short result come from? One series at a time.
    weights = lead_lag_weights(predictions)
    simple = prices.pct_change()
    contribution = (weights.shift(1) * simple)[prices.index > FIRST_DAY]
    hit = (np.sign(predictions.shift(1)) == np.sign(returns))[prices.index > FIRST_DAY].mean()
    by_series = pd.DataFrame({
        "share_of_gross_return": contribution.sum() / contribution.sum().sum(),
        "mean_abs_weight": weights[prices.index >= FIRST_DAY].abs().mean(),
        "direction_hit_rate": hit,
    }).round(4)
    by_series.to_csv(RESULTS / "strategy_by_series.csv")
    print("\nLead-lag long/short, by series:")
    print(by_series.to_string())

    fig, ax = plt.subplots(figsize=(9, 4))
    for name, curve in curves.items():
        ax.plot(curve.index, curve, lw=1.1, label=name)
    ax.set(yscale="log", xlabel="day", ylabel="portfolio value (start = 1)",
           title="Trading rules after costs of 10 basis points per trade")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "strategy.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
