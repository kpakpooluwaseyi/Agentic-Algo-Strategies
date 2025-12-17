import pandas as pd
from backtesting import Backtest, Strategy
import json
import os

# --- Data Preprocessing ---

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates synthetic VIX-related data from the base BTC data to simulate
    the conditions required for the VIX Futures Carry strategy.
    Resamples to daily, creates spot/futures proxies, and a settlement cycle.
    """
    # Resample to daily frequency, taking the last observation of the day
    daily_df = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # 1. Create VIX Spot and Futures Proxies
    # Proxy for VIX Spot Price: Short-term (e.g., 5-day) moving average
    daily_df['vix_spot'] = daily_df['Close'].rolling(window=5).mean()
    # Proxy for VIX Futures Price: Long-term (e.g., 20-day) moving average
    daily_df['vix_futures'] = daily_df['Close'].rolling(window=20).mean()

    # 2. Create 'Days Until Settlement'
    # Simulate a monthly futures contract cycle (e.g., 21 trading days)
    cycle_length = 21
    settlement_counter = [cycle_length - (i % cycle_length) for i in range(len(daily_df))]
    daily_df['days_to_settlement'] = settlement_counter

    # Ensure no division by zero
    daily_df['days_to_settlement'] = daily_df['days_to_settlement'].replace(0, 1)

    # Clean up NaNs created by rolling windows
    daily_df.dropna(inplace=True)

    return daily_df

# --- Strategy Definition ---

class VixFuturesCarryHedgedStrategy(Strategy):
    """
    Implements a simplified version of a VIX Futures Carry strategy.
    Since we don't have actual VIX data, we simulate it.
    - 'VIX Spot' is proxied by a short-term moving average.
    - 'VIX Futures' is proxied by a long-term moving average.
    - The difference creates a synthetic 'contango' or 'backwardation'.
    - The strategy enters short on high contango and long on high backwardation.
    - The 'hedged' aspect with S&P futures is omitted as we only have one asset.
    """
    # Optimizable parameter
    roll_threshold = 0.10  # Daily roll magnitude to trigger a trade

    def init(self):
        """
        Initialize the strategy. Calculate the daily roll indicator.
        """
        # The pre-processing function has already added the necessary columns.
        # We can access them directly via self.data, e.g., self.data.vix_spot

        # Calculate the Daily Roll as a custom indicator
        self.daily_roll = self.I(
            lambda: (self.data.vix_futures - self.data.vix_spot) / self.data.days_to_settlement
        )

    def next(self):
        """
        Define the trading logic. The strategy re-evaluates its position daily.
        """
        # Get the most recent daily roll value
        current_roll = self.daily_roll[-1]

        # --- Daily Re-evaluation ---
        # If a position exists, close it at the start of the new day's evaluation.
        # A new position will be opened if the signal persists.
        if self.position:
            self.position.close()

        # --- Entry Rules ---
        # Long VIX Entry (Backwardation)
        if current_roll < -self.roll_threshold:
            # In our simulation, this is a bet on 'VIX' (BTC price volatility) increasing.
            # The strategy document says to go LONG ES (S&P) in this case.
            # So, we will BUY the underlying asset (BTC).
            self.buy()

        # Short VIX Entry (Contango)
        elif current_roll > self.roll_threshold:
            # In our simulation, this is a bet on 'VIX' (BTC price volatility) decreasing.
            # The strategy document says to go SHORT ES (S&P) in this case.
            # So, we will SELL the underlying asset (BTC).
            self.sell()

# --- Backtesting Execution ---

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure you have the required CSV file.")
    else:
        # Load data
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Clean column names: strip whitespace and convert to TitleCase
        data.columns = [x.strip().title() for x in data.columns]

        # Preprocess data to create synthetic VIX futures structure
        daily_data = preprocess_data(data)


        # Initialize and run the backtest
        bt = Backtest(daily_data, VixFuturesCarryHedgedStrategy, cash=100_000, commission=.002)

        print("Running backtest...")
        stats = bt.run()
        print(stats)

        # Sanitize results for JSON output
        results = {
            'strategy_name': 'vix_futures_carry_hedged',
            'return': stats.get('Return [%]', None),
            'sharpe': stats.get('Sharpe Ratio', None),
            'max_drawdown': stats.get('Max. Drawdown [%]', None),
            'win_rate': stats.get('Win Rate [%]', None),
            'total_trades': stats.get('# Trades', 0)
        }

        # Clean results from potential NaNs
        for key, value in results.items():
            if pd.isna(value):
                results[key] = None

        # Ensure the results directory exists
        os.makedirs('results', exist_ok=True)

        # Save the results
        with open('results/temp_result.json', 'w') as f:
            json.dump(results, f, indent=4)

        print("\nBacktest stats saved to results/temp_result.json")

        # Generate the plot
        try:
            plot_filename = 'results/vix_futures_carry_hedged.html'
            bt.plot(filename=plot_filename)
            print(f"Plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot: {e}")
