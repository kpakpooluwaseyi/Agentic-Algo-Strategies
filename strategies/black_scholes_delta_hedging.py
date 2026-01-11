"""
Black-Scholes Delta Hedging Strategy
=====================================

This script simulates the delta hedging of a short European call option on a
stock that follows a geometric Brownian motion. This is a simplified model
and has several limitations.
"""

from backtesting import Backtest, Strategy
import numpy as np
import pandas as pd
from scipy.stats import norm


def _black_scholes_call(S, K, T, r, sigma):
    """Calculates the Black-Scholes price of a European call option."""
    if T <= 0:
        return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

def _black_scholes_delta(S, K, T, r, sigma):
    """Calculates the Black-Scholes delta of a European call option."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    return delta

class BlackScholesDeltaHedging(Strategy):
    # Option parameters
    K_pct = 1.05  # Strike price as a percentage of the initial price
    T_days = 30   # Time to maturity in days
    r = 0.05      # Risk-free rate (annualized)
    sigma = 0.5   # Volatility (annualized)
    n_options = 10000 # Number of options being hedged

    def init(self):
        # Calculate static option parameters
        self.initial_price = self.data.Close[0]
        self.K = self.initial_price * self.K_pct
        self.start_date = self.data.index[0]
        self.maturity_date = self.start_date + pd.Timedelta(days=self.T_days)

        # We are hedging a short call, so we hold a long position in the underlying
        # The number of shares to hold is equal to the option's delta
        self.target_shares = 0


    def next(self):
        # Get current variables
        S = self.data.Close[-1]
        current_date = self.data.index[-1]

        # Calculate time to maturity in years
        time_to_maturity = (self.maturity_date - current_date).total_seconds() / (365 * 24 * 60 * 60)

        # If the option has expired, close all positions
        if time_to_maturity <= 0:
            if self.position:
                self.position.close()
            return

        # Calculate the current target delta
        delta = _black_scholes_delta(S, self.K, time_to_maturity, self.r, self.sigma)

        # Calculate the target number of shares to hold
        self.target_shares = delta * self.n_options

        # Adjust position to match target delta
        # Using FractionalBacktest would be more realistic, but for simplicity
        # and compatibility, we trade discrete units.
        # We use a large n_options to make sure the position size is > 1.
        current_holding = self.position.size if self.position else 0
        trade_size = self.target_shares - current_holding

        if abs(trade_size) > 1: # Avoid tiny trades, ensure trade_size is at least 1 unit
            if trade_size > 0:
                self.buy(size=int(trade_size))
            else:
                self.sell(size=int(abs(trade_size)))

if __name__ == '__main__':
    # --- Configuration ---
    DATA_PATH = 'data/BTC-USD-15m.csv'
    INITIAL_CASH = 1_000_000  # Start with a large cash amount for hedging
    COMMISSION = .002         # 0.2% commission

    # --- Data Loading ---
    try:
        data = pd.read_csv(DATA_PATH, index_col='datetime', parse_dates=True)
        # Sanitize column names (e.g., ' open' -> 'Open')
        data.columns = [col.strip().capitalize() for col in data.columns]
        # Drop the unnamed column if it exists
        if 'Unnamed: 6' in data.columns:
            data = data.drop(columns=['Unnamed: 6'])
        # Limit data to a shorter period for a reasonable backtest duration
        data = data['2023-01-01':'2023-03-31']
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        # As a fallback, create synthetic data for demonstration
        print("Generating synthetic data...")
        from backtesting.test import EURUSD
        data = EURUSD.copy()
        data = data.iloc[-3000:]
        data['Close'] = 17000 + (data['Close'] - data['Close'].iloc[0]) * 1000

    # --- Backtest Execution ---
    # Note: We use a large initial cash value because the asset price (BTC) is high,
    # and we need sufficient capital to hold the delta-neutral position.
    bt = Backtest(data, BlackScholesDeltaHedging,
                  cash=INITIAL_CASH, commission=COMMISSION)

    print("Running backtest...")
    stats = bt.run()

    # --- Results & Interpretation ---
    print("\n" + "="*50)
    print("Black-Scholes Delta Hedging Simulation Results")
    print("="*50)
    print(stats)
    print("\n" + "="*50)
    print("IMPORTANT INTERPRETATION NOTES:")
    print("1. The 'Equity Final' and 'Return [%]' ONLY reflect the performance of the")
    print("   hedging portfolio (the spot BTC trades), NOT the full option strategy.")
    print("2. The profit/loss from the initial option premium is NOT included here.")
    print("3. In a perfect Black-Scholes world, the final value of this hedging")
    print("   portfolio should be equal to the option's payoff at expiration.")
    print("   The total P/L of the option seller would be:")
    print("   [Initial Premium] - [Hedging Cost], where Hedging Cost = [Final Equity] - [Initial Cash]")
    print("="*50)

    # --- Plotting ---
    # The plot shows the trades made to rebalance the delta hedge over time.
    bt.plot(filename='results/black_scholes_delta_hedging.html', open_browser=False)
    print("\nPlot saved to 'results/black_scholes_delta_hedging.html'")
