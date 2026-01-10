"""
Option Spread Slope Arbitrage Strategy
========================================
This strategy simulates a call option spread arbitrage based on the slope
restrictions of option pricing, as described in financial theory. Since the
backtesting framework only supports single-instrument trading on spot prices,
this implementation acts as a model or a proxy for the true arbitrage strategy.

Core Logic:
1.  **Black-Scholes Model:** A Black-Scholes formula is used to estimate the
    theoretical prices of two European call options with different strike prices
    (K1 and K2).
2.  **Volatility Calculation:** The historical volatility of the underlying asset
    (BTC-USD) is calculated on a rolling basis, as this is a key input for the
    Black-Scholes model.
3.  **Arbitrage Condition:** The strategy checks for violations of the no-arbitrage
    condition for the slope of the option price curve:
        (C(K2) - C(K1)) / (K2 - K1) < -B(0,T)
    where C(K) is the call price at strike K, and B(0,T) is the present value
    factor for a risk-free bond.
4.  **Simulated Mispricing:** Since a perfect market would not present this
    arbitrage, the strategy simulates these opportunities by introducing a
    probabilistic "shock" or mispricing to the theoretical option prices.
5.  **Proxy Trade:** When an arbitrage opportunity is detected, the strategy
    initiates a SHORT position on the underlying asset. This serves as a proxy for
    the actual arbitrage trade, which would be a "bear call spread" (selling a
    call at K1 and buying a call at K2). A short position is used because a bear
    spread profits from a decrease in the underlying's price, aligning the
    directional exposure.
6.  **Holding Period:** The position is held for a fixed duration, simulating the
    time until the options' common expiration date (T), at which point the
    arbitrage profit would be realized.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from backtesting import Strategy, Backtest
import os
import json

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculates the price of a European call option using the Black-Scholes model.
    S: Spot price of the underlying asset
    K: Strike price of the option
    T: Time to maturity (in years)
    r: Risk-free interest rate
    sigma: Volatility of the underlying asset's returns
    """
    if sigma == 0 or T <= 0:
        return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

def preprocess_data(df, vol_window=30, **params):
    """
    Calculate historical volatility.
    """
    df = df.copy()
    # Calculate daily log returns
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    # Calculate rolling historical volatility (annualized)
    # Assuming 252 trading days in a year. For 15m data, there are 252*24*4 periods/year
    annualization_factor = np.sqrt(252 * 24 * 4)
    df['volatility'] = log_returns.rolling(window=vol_window).std() * annualization_factor
    return df

class OptionSpreadSlopeArbitrage(Strategy):
    # --- Strategy Parameters ---
    time_to_maturity_days = 30  # T in years
    risk_free_rate = 0.05       # r
    strike_distance_pct = 0.05  # K2-K1 as a percentage of current price
    volatility_window = 30      # Lookback window for historical volatility

    # --- Simulation Parameters ---
    mispricing_probability = 0.5 # Chance of an arbitrage opportunity appearing on any given bar
    mispricing_shock_factor = 1.2 # How much the slope is steepened to trigger the arbitrage

    def init(self):
        self.volatility = self.I(lambda: self.data.df['volatility'], name='volatility')
        self.time_to_maturity_years = self.time_to_maturity_days / 365.25
        self.hold_duration_bars = int(self.time_to_maturity_days * 24 * 4) # For 15m data
        self.exit_bar = -1

    def next(self):
        # --- Exit Logic ---
        if self.position:
            if len(self.data) >= self.exit_bar:
                self.position.close()
            return

        # --- Entry Logic ---
        if pd.isna(self.volatility[-1]):
            return

        if np.random.rand() > self.mispricing_probability:
            return

        spot_price = self.data.Close[-1]
        sigma = self.volatility[-1]

        k1 = spot_price * (1 - self.strike_distance_pct)
        k2 = spot_price * (1 + self.strike_distance_pct)

        c1 = black_scholes_call(spot_price, k1, self.time_to_maturity_years, self.risk_free_rate, sigma)
        c2 = black_scholes_call(spot_price, k2, self.time_to_maturity_years, self.risk_free_rate, sigma)

        b_t = np.exp(-self.risk_free_rate * self.time_to_maturity_years)

        slope = (c2 - c1) / (k2 - k1)

        shocked_slope = slope * self.mispricing_shock_factor

        if shocked_slope < -b_t:
            self.sell()
            self.exit_bar = len(self.data) + self.hold_duration_bars

# --- Standalone Execution ---
if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
        if 'Unnamed: 6' in df.columns:
            df = df.drop(columns=['Unnamed: 6'])
        df.columns = [col.strip().capitalize() for col in df.columns]
    except FileNotFoundError:
        print("data/BTC-USD-15m.csv not found. Using sample data.")
        dates = pd.date_range('2023-01-01', periods=5000, freq='15min')
        np.random.seed(42)
        price = 40000 + np.cumsum(np.random.randn(5000) * 20)
        df = pd.DataFrame({
            'Open': price, 'High': price + np.random.rand(5000) * 50,
            'Low': price - np.random.rand(5000) * 50, 'Close': price + np.random.randn(5000) * 10,
            'Volume': np.random.rand(5000) * 100
        }, index=dates)

    df = preprocess_data(df)
    df = df.dropna()

    bt = Backtest(df, OptionSpreadSlopeArbitrage, cash=100_000, commission=.002)
    stats = bt.run()

    print("\n=== Option Spread Slope Arbitrage Strategy Results ===")
    print(stats)

    def sanitize_stats(stats_series):
        sanitized = {}
        for key, value in stats_series.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                sanitized[key] = str(value)
            elif isinstance(value, (np.integer, np.floating)):
                sanitized[key] = float(value)
            # Add a check to skip the Strategy object itself
            elif isinstance(value, Strategy):
                continue
            elif isinstance(value, type) or hasattr(value, 'to_json'):
                continue
            else:
                sanitized[key] = value
        return sanitized

    json_filename = 'results/option_spread_slope_arbitrage.json'
    with open(json_filename, 'w') as f:
        json.dump(sanitize_stats(stats), f, indent=4)
    print(f"\nStats saved to {json_filename}")

    plot_filename = 'results/option_spread_slope_arbitrage.html'
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"\nPlot saved to {plot_filename}")
    except Exception as e:
        print(f"\nCould not save plot: {e}")
