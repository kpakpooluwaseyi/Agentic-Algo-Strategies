
import pandas as pd
from backtesting import Backtest, Strategy
import numpy as np
import json
import os

class ForwardSpotArbitrage(Strategy):
    """
    This strategy attempts to simulate a Forward-Spot Arbitrage opportunity for a
    non-dividend paying asset.

    **IMPORTANT LIMITATIONS:**
    1.  **Data:** This backtest runs on spot price data (`BTC-USD-15m.csv`) only.
        It does NOT have access to a real-world feed of forward/futures prices.
    2.  **Simulation:** To test the entry logic, an "observed" market forward price is
        *synthetically generated*. Arbitrage opportunities are rare. To simulate this,
        the logic only runs when a random event is triggered, controlled by the
        `arbitrage_chance` parameter.
    3.  **P&L Representation:** `backtesting.py` calculates P&L based on the directional
        trades in the spot market (the `self.buy()` and `self.sell()` calls). This
        **DOES NOT** represent the true profit of an arbitrage strategy, which would
        be a small, risk-free return locked in at entry. The backtest results will
        show the performance of the spot leg only and will appear as a standard
        directional strategy, not a risk-free arbitrage.

    The purpose of this backtest is to validate the *entry logic* under simulated,
    event-driven conditions, not to accurately model arbitrage profitability.
    """
    # --- Strategy Parameters ---
    # Time to maturity for the synthetic forward contract (in bars)
    # e.g., 240 bars = 2.5 days on 15-min data. A short-term contract.
    time_to_maturity = 240

    # Annualized risk-free interest rate (e.g., 5%)
    risk_free_rate = 0.05

    # This parameter simulates the magnitude of the random market mispricing.
    # A value of 0.001 means the observed forward price can deviate by up to 0.1%.
    mispricing_magnitude = 0.001

    # Probability (e.g., 1%) of an arbitrage opportunity appearing on any given bar.
    arbitrage_chance = 0.01

    def init(self):
        """
        Initialize the strategy's state. No indicators are needed as we
        are simulating the arbitrage based on the raw price feed.
        """
        self.trade_entry_bar = None
        # Seed for repeatable random noise generation
        np.random.seed(0)

    def next(self):
        """
        Define the trading logic for each bar.
        """
        current_bar = len(self.data) - 1

        # --- 1. Check for an active position ---
        if self.position:
            # Time-based exit: close position if held until synthetic maturity
            if current_bar - self.trade_entry_bar >= self.time_to_maturity:
                self.position.close()
                self.trade_entry_bar = None
            return # Don't open new trades while one is active

        # --- 2. Simulate a rare arbitrage opportunity ---
        if np.random.rand() < self.arbitrage_chance:
            # NOTE: This is a SIMULATION. The provided data only has spot prices.
            # To test the arbitrage logic, we must synthetically create a
            # "market forward price" that can temporarily deviate from the
            # theoretical price.

            # Use the actual close price as the current Spot Price S(t)
            spot_price = self.data.Close[-1]

            # (T-t) must be in years. Our time_to_maturity is in bars.
            bars_in_year = 365 * 24 * 4
            time_to_maturity_years = self.time_to_maturity / bars_in_year

            # Calculate the no-arbitrage theoretical forward price
            theoretical_forward = spot_price * np.exp(self.risk_free_rate * time_to_maturity_years)

            # Create a synthetic "observed" forward price by adding a small, random
            # deviation to the THEORETICAL forward price. This correctly simulates
            # a market price that is temporarily misaligned with its theoretical value.
            noise = (np.random.rand() - 0.5) * 2 * self.mispricing_magnitude * theoretical_forward
            observed_forward = theoretical_forward + noise

            # --- 3. Arbitrage Entry Conditions ---
            # Case 1: Forward is Overpriced (Sell high, buy low)
            if observed_forward > theoretical_forward:
                self.buy() # Represents buying the spot asset
                self.trade_entry_bar = current_bar

            # Case 2: Forward is Underpriced (Buy low, sell high)
            elif observed_forward < theoretical_forward:
                self.sell() # Represents shorting the spot asset
                self.trade_entry_bar = current_bar

if __name__ == '__main__':
    # --- Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    strategy_name = 'forward_spot_arbitrage_no_dividend'

    # --- Data Loading and Preprocessing ---
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # Clean and capitalize column names
    data.columns = [col.strip().title() for col in data.columns]
    # Drop any unnamed columns that may exist
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # --- Backtesting ---
    print(f"Running backtest for {strategy_name}...")
    bt = Backtest(data, ForwardSpotArbitrage, cash=100_000, commission=.002)

    stats = bt.run()
    print(stats)

    # --- Save Results ---
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Sanitize stats for JSON output
    def sanitize_stats(stats_obj):
        sanitized = {}
        for key, value in stats_obj.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                continue # Skip non-serializable types
            if key.startswith('_'):
                continue # Skip internal objects
            if pd.isna(value):
                sanitized[key] = None
            elif isinstance(value, (np.int64, np.int32)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                sanitized[key] = float(value)
            else:
                sanitized[key] = value
        return sanitized

    clean_stats = sanitize_stats(stats)

    results_path = os.path.join(results_dir, 'temp_result.json')
    with open(results_path, 'w') as f:
        json.dump(clean_stats, f, indent=4)

    print(f"\nBacktest statistics saved to {results_path}")

    # --- Plotting ---
    plot_path = os.path.join(results_dir, f'{strategy_name}.html')
    try:
        bt.plot(filename=plot_path, open_browser=False)
        print(f"Backtest plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
