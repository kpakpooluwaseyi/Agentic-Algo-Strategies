"""
Cox-Ross-Rubinstein (CRR) Binomial Option Hedging Strategy
===========================================================

This strategy simulates the dynamic hedging of a short European call option
using the CRR binomial tree model. The goal is to replicate the option's
payoff by continuously adjusting a portfolio consisting of the underlying
asset and a risk-free asset.

This backtest only simulates the underlying asset (stock) leg of the hedge.
The Profit & Loss (P&L) shown in the backtest results represents the cost
of maintaining this hedge. The option seller's total profit would be the
initial premium received minus this hedging cost.

NOTE: This strategy is for educational purposes to demonstrate the mechanics
of delta hedging and is not a typical trading strategy. It ignores the
boilerplate ATR/Volume/etc. rules as they are not applicable to this model.
"""

import numpy as np
import pandas as pd
from backtesting import Strategy, Backtest
import json
import os

def crr_binomial_tree_delta(S, K, T, r, sigma, N):
    """
    Calculates the European call option delta using the CRR binomial model.

    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity in years
        r (float): Risk-free interest rate
        sigma (float): Annualized volatility
        N (int): Number of steps in the binomial tree

    Returns:
        float: The option delta, representing the number of shares to hold.
    """
    if T <= 0 or N <= 0:
        return 1.0 if S > K else 0.0

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # If model parameters lead to arbitrage, it's invalid.
    if not (d < np.exp(r * dt) < u):
        return 0.5 # Return a neutral value

    # Asset prices at maturity (vectorized)
    asset_prices = S * d**np.arange(N, -1, -1) * u**np.arange(0, N + 1)

    # Option values at maturity (vectorized)
    option_values = np.maximum(asset_prices - K, 0)

    # Backward induction through the tree to t=1
    for i in range(N, 1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[:-1] + (1 - p) * option_values[1:])

    # The loop finishes at t=1. `option_values` now holds the two possible
    # option values for the next step.
    v_up = option_values[0]
    v_down = option_values[1]

    # Stock prices for the next step
    s_up = S * u
    s_down = S * d

    # Delta is the change in option value over the change in stock price
    delta = (v_up - v_down) / (s_up - s_down)

    return delta


class CrrBinomialOptionHedging(Strategy):
    """
    Implements a delta-hedging strategy based on the CRR Binomial Model.
    """
    # --- Option & Model Parameters ---
    strike_price = 16600
    time_to_maturity_days = 30
    risk_free_rate = 0.05
    volatility_lookback = 30
    n_steps = 30
    n_options_simulated = 10_000

    def init(self):
        close_series = pd.Series(self.data.Close)
        log_returns = np.log(close_series / close_series.shift(1))
        lookback_points = self.volatility_lookback * 24 * 4
        self.sigma = self.I(
            lambda: log_returns.rolling(lookback_points).std() * np.sqrt(365 * 24 * 4),
            name="AnnualizedVolatility"
        )
        self.start_date = self.data.index[lookback_points]
        self.maturity_date = self.start_date + pd.Timedelta(days=self.time_to_maturity_days)
        self.last_rebalance_date = None

    def next(self):
        current_time = self.data.index[-1]

        if not (self.start_date <= current_time <= self.maturity_date):
            if self.position:
                self.position.close()
            return

        current_date = current_time.date()
        if current_date == self.last_rebalance_date:
            return
        self.last_rebalance_date = current_date

        S = self.data.Close[-1]
        sigma_val = self.sigma[-1]
        time_to_maturity_years = (self.maturity_date - current_time).total_seconds() / (365 * 24 * 3600)

        if S <= 0 or np.isnan(sigma_val) or sigma_val <= 0 or time_to_maturity_years <= 0:
            return

        target_delta = crr_binomial_tree_delta(
            S, self.strike_price, time_to_maturity_years, self.risk_free_rate, sigma_val, self.n_steps
        )

        target_units = target_delta * self.n_options_simulated

        # Adjust position to match the target delta.
        # We need to express the target position as a fraction of total equity.
        target_value = target_units * S
        target_fraction = target_value / self.equity

        # First, close any existing position to rebalance.
        if self.position:
            self.position.close()

        # Place a new order with the target size if the delta is positive.
        if target_fraction > 0:
            self.buy(size=min(target_fraction, 0.99))

# ===== STANDALONE MODE =====
if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}.")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data.columns = [c.strip().title() for c in data.columns]

    bt = Backtest(data, CrrBinomialOptionHedging, cash=100_000_000, commission=.001)
    stats = bt.run()
    print(stats)

    os.makedirs('results', exist_ok=True)

    stats_dict = dict(stats)
    for key in ['_strategy', '_equity_curve', '_trades']:
        stats_dict.pop(key, None)

    def default_serializer(o):
        if isinstance(o, (np.integer, np.floating)): return float(o)
        if isinstance(o, (pd.Timestamp, pd.Timedelta)): return str(o)
        if pd.isna(o): return None
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=2, default=default_serializer)

    print("\nBacktest stats saved to results/temp_result.json")
    bt.plot(filename='results/crr_binomial_option_hedging.html', open_browser=False)
    print("Backtest plot saved to results/crr_binomial_option_hedging.html")
