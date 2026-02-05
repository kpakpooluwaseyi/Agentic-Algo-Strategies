"""
Option Butterfly Spread Arbitrage Strategy (Simulation)
=======================================================

This strategy simulates the identification of arbitrage opportunities in
an option butterfly spread, based on violations of convexity restrictions.

**Disclaimer:** The `backtesting.py` framework is designed for single-instrument
spot trading and cannot natively handle multi-leg options positions. This
implementation is therefore a **simulation** of the arbitrage *setup*.

**How it works:**
1.  **Theoretical Pricing:** It uses the Black-Scholes model to calculate the
    theoretical prices of three European call options with different strike
    prices (K1 < K2 < K3) based on the underlying asset's (BTC-USD) price
    and historical volatility.
2.  **Arbitrage Condition Check:** It checks for a violation of the convexity
    principle, which is the condition for a risk-free arbitrage opportunity:
    `(C(K3)-C(K2))/(K3-K2) < (C(K2)-C(K1))/(K2-K1)`
3.  **Simulated Trade:** When this condition is met, the strategy enters a long
    position (`self.buy()`) to mark the event on the chart. This trade does
    **not** represent the actual profit and loss of the spread but merely
    signals that an arbitrage opportunity was identified.
4.  **Holding Period:** The position is held for a fixed duration representing
    the time to maturity of the options, after which it is closed.

The P&L shown in the backtest results is purely indicative of the spot price
movement after the arbitrage signal and should not be interpreted as the
actual return of the options spread. The true arbitrage would yield a small,
risk-free profit based on the initial mispricing.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from backtesting import Strategy, Backtest
import os
import json

# --- Black-Scholes Model for European Options ---
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculates the price of a European call option using the Black-Scholes model.
    S: Underlying asset price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    sigma: Volatility of the underlying asset
    """
    if sigma == 0 or T == 0:
        return np.maximum(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

# --- Data Preparation ---
def preprocess_data(df, **params):
    """
    Prepares the data by calculating historical volatility.
    """
    df = df.copy()
    # Calculate daily log returns
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    # Calculate rolling historical volatility (e.g., 30-day)
    # Annualized by multiplying by sqrt of periods per year (15min data)
    periods_per_day = 24 * 4  # 96 15-minute periods in a day
    annual_factor = np.sqrt(periods_per_day * 365.25)
    df['volatility'] = log_returns.rolling(window=params.get('vol_period', 30 * periods_per_day)).std() * annual_factor
    return df


# --- Strategy Implementation ---
class OptionButterflySpreadArbitrage(Strategy):
    # --- Strategy Parameters ---
    # Strike prices for the butterfly spread (K1 < K2 < K3)
    # These are defined relative to the current price at entry
    k1_delta_pct = 0.95  # K1 is 5% below current price
    k2_delta_pct = 1.00  # K2 is at-the-money
    k3_delta_pct = 1.05  # K3 is 5% above current price

    # Time to maturity for the options (in days)
    time_to_maturity_days = 30

    # Risk-free interest rate (annualized)
    risk_free_rate = 0.02

    # Volatility calculation period
    vol_period = 960 # 10 days * 96 periods/day

    # --- Simulation Parameters ---
    # This parameter introduces a random "shock" to the middle option's price
    # to simulate market inefficiencies, allowing the arbitrage condition to be met.
    mispricing_shock_pct = 0.005 # 0.5% shock

    def init(self):
        # Calculate time to maturity in years and in 15-min bars
        self.T_years = self.time_to_maturity_days / 365.25
        self.hold_duration_bars = self.time_to_maturity_days * (24 * 4)

        # Indicator for historical volatility
        self.volatility = self.I(lambda: self.data.df['volatility'], name='volatility')

        # Variable to track entry bar index
        self.entry_bar = -1

    def next(self):
        # --- Warmup Guard ---
        # Wait for volatility indicator to be ready
        if pd.isna(self.volatility[-1]):
            return

        # --- Exit Logic ---
        # If a position is open, check if it has been held for the required duration
        if self.position:
            if len(self.data) - 1 >= self.entry_bar + self.hold_duration_bars:
                self.position.close()
            return # Don't check for new entries while a position is open

        # --- Entry Logic ---
        S = self.data.Close[-1]
        sigma = self.volatility[-1]

        # Define the three strike prices relative to the current spot price
        K1 = S * self.k1_delta_pct
        K2 = S * self.k2_delta_pct
        K3 = S * self.k3_delta_pct

        # Ensure K2 is the midpoint for a symmetric spread, as per the definition
        K2 = (K1 + K3) / 2

        # Calculate the theoretical prices of the three call options
        C1 = black_scholes_call(S, K1, self.T_years, self.risk_free_rate, sigma)
        C2_theoretical = black_scholes_call(S, K2, self.T_years, self.risk_free_rate, sigma)
        C3 = black_scholes_call(S, K3, self.T_years, self.risk_free_rate, sigma)

        # --- Simulate Market Mispricing ---
        # Introduce a random shock to the middle option's price.
        # This is the key to the simulation, as perfect Black-Scholes prices
        # will never violate the convexity principle.
        shock = (np.random.rand() - 0.5) * 2 * self.mispricing_shock_pct # Random shock between -pct and +pct
        C2 = C2_theoretical * (1 + shock)

        # Check for the convexity violation (arbitrage condition)
        # To avoid division by zero if strikes are the same
        if (K3 - K2 > 0) and (K2 - K1 > 0):
            slope1 = (C2 - C1) / (K2 - K1)
            slope2 = (C3 - C2) / (K3 - K2)

            if slope2 < slope1:
                # Arbitrage opportunity detected.
                # Enter a long position to mark the event.
                self.buy()
                self.entry_bar = len(self.data) - 1

# --- Runnable Main Block ---
if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    try:
        df_raw = pd.read_csv(data_path, index_col=0, parse_dates=True)
        df_raw.columns = [col.strip().capitalize() for col in df_raw.columns]
        data_loaded = True
    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_path}'.")
        data_loaded = False

    if data_loaded:
        print("Preprocessing data...")
        df_processed = preprocess_data(df_raw, vol_period=OptionButterflySpreadArbitrage.vol_period)

        if df_processed.empty:
            print("DataFrame is empty after preprocessing. Cannot run backtest.")
        else:
            print("Running backtest...")
            bt = Backtest(df_processed, OptionButterflySpreadArbitrage, cash=100_000, commission=0.0, finalize_trades=True)
            stats = bt.run()

            print("\n=== Option Butterfly Spread Arbitrage (Simulation) Results ===")
            print(stats)

            os.makedirs('results', exist_ok=True)

            plot_filename = 'results/option_butterfly_spread_arbitrage.html'
            bt.plot(filename=plot_filename, open_browser=False)
            print(f"\nPlot saved to '{plot_filename}'")

            stats_dict = stats.to_dict()
            stats_dict['_strategy'] = str(stats_dict['_strategy'])
            for key, value in stats_dict.items():
                if isinstance(value, pd.DataFrame): stats_dict[key] = None
                elif isinstance(value, (pd.Timestamp, pd.Timedelta)): stats_dict[key] = str(value)
                elif isinstance(value, np.integer): stats_dict[key] = int(value)
                elif isinstance(value, np.floating): stats_dict[key] = float(value)
                elif pd.isna(value): stats_dict[key] = None

            stats_dict_cleaned = {k: v for k, v in stats_dict.items() if v is not None}

            json_filename = 'results/temp_result.json'
            with open(json_filename, 'w') as f:
                json.dump(stats_dict_cleaned, f, indent=4)
            print(f"Stats saved to '{json_filename}'")
