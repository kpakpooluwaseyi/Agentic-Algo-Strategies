from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
import os

# --- Data Generation and Preprocessing ---

def generate_synthetic_data(days=1000):
    """
    Generates synthetic daily data that trends upwards with some pullbacks,
    suitable for testing a 52-week high strategy.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range(start='2020-01-01', periods=days, freq='D')

    # Base trend with some noise
    price = 100
    prices = [price]
    for _ in range(len(dates) - 1):
        price += rng.normal(0.1, 1.5) + 0.1 # Add a slight upward drift
        prices.append(max(10, price)) # Ensure price doesn't go below 10

    df = pd.DataFrame({'Close': prices}, index=dates)

    # Introduce some pullbacks
    for i in range(5, days, 100):
        pullback_length = rng.integers(10, 30)
        start_index = max(0, i - pullback_length)
        df.iloc[start_index:i] *= rng.uniform(0.85, 0.95)

    df['Open'] = df['Close'].shift(1).fillna(df['Close'])
    df['High'] = df[['Open', 'Close']].max(axis=1) + rng.uniform(0, 2, size=len(df))
    df['Low'] = df[['Open', 'Close']].min(axis=1) - rng.uniform(0, 2, size=len(df))

    # Resample to 15-min to match the target data format
    df_15m = df.resample('15min').ffill()
    df_15m['Volume'] = rng.integers(100, 1000, size=len(df_15m))

    return df_15m[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


def preprocess_data(df):
    """
    Calculates the 52-week high and merges it back into the main DataFrame.
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Resample to daily to get the maximum high for each day
    daily_high = df['High'].resample('D').max()

    # Calculate 52-week (365 days) rolling high on the daily data
    rolling_high = daily_high.rolling(window=365, min_periods=1).max()

    # Map the daily rolling high back to the 15-min dataframe's date
    df['52_week_high'] = df.index.normalize().map(rolling_high)

    # Forward fill the mapped values to apply to all bars within a day
    df['52_week_high'] = df['52_week_high'].ffill()

    df = df.dropna(subset=['52_week_high'])

    return df

# --- Strategy Definition ---

def passthrough(data):
    return data

class FiftyTwoWeekHighRelativeStrength(Strategy):
    """
    A strategy that enters long positions when the price is near its 52-week high.
    """
    # Strategy parameters
    proximity_pct = 0.05  # How close to the 52w high to trigger an entry (e.g., 5%)
    sl_pct = 0.10         # Stop loss percentage
    tp_pct = 0.20         # Take profit percentage

    def init(self):
        """
        Initialize indicators and state variables.
        """
        # Custom indicator for the 52-week high
        self.fifty_two_week_high = self.I(passthrough, self.data.df['52_week_high'].values)

    def next(self):
        """
        The main strategy logic that is executed on each bar.
        """
        # Don't open new trades if one is already open
        if self.position:
            return

        current_price = self.data.Close[-1]
        high_52_week = self.fifty_two_week_high[-1]

        # Entry condition: Price is within proximity_pct of the 52-week high
        if current_price >= high_52_week * (1 - self.proximity_pct):

            # Calculate stop loss and take profit levels
            sl = current_price * (1 - self.sl_pct)
            tp = current_price * (1 + self.tp_pct)

            # Place the buy order
            self.buy(sl=sl, tp=tp)


# --- Backtesting Execution ---

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Clean column names: strip whitespace, remove trailing commas, and capitalize
        data.columns = [c.strip().title() for c in data.columns]
        data.rename(columns={'Volume,': 'Volume'}, inplace=True) # Address trailing comma issue
    else:
        print("Data file not found. Generating synthetic data...")
        data = generate_synthetic_data(days=1500)

    # Preprocess the data
    data = preprocess_data(data)

    # Run the backtest
    bt = Backtest(data, FiftyTwoWeekHighRelativeStrength, cash=100000, commission=.002, finalize_trades=True)

    print("Running backtest...")
    stats = bt.run()

    # Save results to a JSON file
    os.makedirs('results', exist_ok=True)

    # Sanitize stats for JSON output
    sanitized_stats = {}
    for key, value in stats.items():
        if key.startswith('_'):  # Exclude internal objects like _strategy, _equity_curve, _trades
            continue
        if isinstance(value, (pd.Series, pd.DataFrame)):
            continue
        if pd.isna(value):
            sanitized_stats[key] = None
        elif isinstance(value, (np.int64, np.int32)):
            sanitized_stats[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            sanitized_stats[key] = float(value)
        elif isinstance(value, pd.Timestamp):
            sanitized_stats[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            sanitized_stats[key] = str(value)
        else:
            sanitized_stats[key] = value

    result_data = {
        'strategy_name': 'fifty_two_week_high_relative_strength',
        'parameters': {
            'proximity_pct': FiftyTwoWeekHighRelativeStrength.proximity_pct,
            'sl_pct': FiftyTwoWeekHighRelativeStrength.sl_pct,
            'tp_pct': FiftyTwoWeekHighRelativeStrength.tp_pct
        },
        'stats': sanitized_stats
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(result_data, f, indent=2)

    print("Backtest stats saved to results/temp_result.json")
    print(stats)

    # Generate the plot
    try:
        plot_filename = 'results/fifty_two_week_high_relative_strength.html'
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
