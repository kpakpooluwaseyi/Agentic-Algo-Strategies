
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply
import numpy as np
from sklearn.linear_model import LinearRegression
import json
import os

# --- Helper Functions ---

def sanitize_stats(stats):
    """
    Sanitizes the stats object by converting non-serializable types to strings or basic types.
    Removes the _strategy object to avoid serialization issues.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(key, str) and key.startswith('_'):
            continue
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = float(value)
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            # Decide how you want to represent these; maybe just a summary
            sanitized[key] = 'DataFrame/Series object'
        elif hasattr(value, 'to_dict'):
            sanitized[key] = value.to_dict()
        else:
            sanitized[key] = value
    if '_strategy' in sanitized:
        del sanitized['_strategy']
    return sanitized

# --- Strategy Definition ---

class FamaFrenchStrategy(Strategy):
    """
    A simplified proxy of the Fama-French Three-Factor Model for a single asset (BTC-USD).
    This strategy is for educational purposes to demonstrate the concept, as a true
    Fama-French model requires a universe of stocks and actual Fama-French factor data.

    Factors are proxied as follows:
    - Mkt-RF (Market Risk Premium): Proxied by the asset's own daily return.
    - SMB (Small Minus Big): Proxied by a comparison of recent volume to long-term average volume.
      Higher relative volume is treated as a 'small cap' characteristic (higher retail interest).
    - HML (High Minus Low): Proxied by a momentum indicator (e.g., RSI). High RSI is treated
      as a 'growth' stock characteristic (low B/M ratio), and low RSI as a 'value' stock (high B/M).

    The strategy trains a rolling linear regression model to predict the next day's return
    based on these proxy factors.
    - If predicted return > 0, it goes long.
    - If predicted return < 0, it goes short.
    - Positions are held for exactly one day.
    """

    # --- Strategy Parameters ---
    train_window = 100  # Number of past days to use for training the model

    def init(self):
        # Pre-calculate indicators here if needed, but the main logic is in next()
        # due to the rolling training nature of the model.
        pass

    def next(self):
        # --- Rebalancing: Close position at the start of each new bar ---
        if self.position:
            self.position.close()

        # --- Data Preparation & Model Training ---
        # Ensure we have enough data to train the model
        if len(self.data.Close) < self.train_window:
            return

        # Get the training data for the rolling window
        # Features: Mkt_Proxy, SMB_Proxy, HML_Proxy for the last `train_window` days
        # Target: Next_Day_Return for the same period

        # Note: self.data.df is not available here. Access columns directly.
        mkt_proxy = self.data.Mkt_Proxy
        smb_proxy = self.data.SMB_Proxy
        hml_proxy = self.data.HML_Proxy
        target = self.data.Next_Day_Return

        # We use data up to the second-to-last bar for training,
        # to predict the return for the *current* bar (which is `Next_Day_Return` of the previous bar)
        end_idx = len(self.data.Close) - 1
        start_idx = max(0, end_idx - self.train_window)

        X_train_df = pd.DataFrame({
            'Mkt_Proxy': mkt_proxy[start_idx:end_idx],
            'SMB_Proxy': smb_proxy[start_idx:end_idx],
            'HML_Proxy': hml_proxy[start_idx:end_idx]
        })
        y_train = target[start_idx:end_idx]

        # Drop rows with NaN values that might exist from indicator calculations
        X_train_df['y'] = y_train
        X_train_df = X_train_df.dropna()

        if len(X_train_df) < 20: # Ensure we have a reasonable amount of data to train on
            return

        y_train = X_train_df['y']
        X_train = X_train_df.drop(columns=['y'])

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # --- Prediction ---
        # Predict the next day's return using the most recent data
        latest_features = np.array([
            mkt_proxy[-1],
            smb_proxy[-1],
            hml_proxy[-1]
        ]).reshape(1, -1)

        predicted_return = model.predict(latest_features)[0]

        # --- Entry Logic ---
        if predicted_return > 0:
            self.buy()
        elif predicted_return < 0:
            self.sell()


# --- Main Execution Block ---

if __name__ == '__main__':
    # --- Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    results_dir = 'results'
    results_file = os.path.join(results_dir, 'temp_result.json')
    plot_file = os.path.join(results_dir, 'fama_french_three_factor_long_short_equity.html')

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # --- Data Loading and Preprocessing ---
    try:
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        # Fix for CSVs with inconsistent column name spacing
        df.columns = df.columns.str.strip()
        # Rename to a consistent format before any processing
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        # As a fallback, generate synthetic data for demonstration
        print("Generating synthetic data for demonstration...")
        n_points = 50000
        dates = pd.date_range(start='2020-01-01', periods=n_points, freq='15min')
        price = 20000 + np.random.randn(n_points).cumsum()
        volume = np.random.randint(100, 1000, size=n_points)
        df = pd.DataFrame({
            'open': price,
            'high': price + np.random.rand(n_points) * 10,
            'low': price - np.random.rand(n_points) * 10,
            'close': price + np.random.randn(n_points),
            'volume': volume
        }, index=dates)
        df.index.name = 'datetime'

    # Resample to daily timeframe
    daily_df = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # --- Feature Engineering (Proxy Factors) ---
    # 1. Mkt-RF (Market Risk Premium Proxy): Daily return
    daily_df['Mkt_Proxy'] = daily_df['Close'].pct_change()

    # 2. SMB (Size Proxy): Ratio of short-term volume to long-term volume
    daily_df['SMB_Proxy'] = (daily_df['Volume'].rolling(window=20).mean() /
                             daily_df['Volume'].rolling(window=100).mean())

    # 3. HML (Value Proxy): Using RSI as a momentum proxy
    # We need a function to calculate RSI because we can't use self.I here
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    daily_df['HML_Proxy'] = rsi(daily_df['Close'], period=14)

    # 4. Target Variable: Next Day's Return
    daily_df['Next_Day_Return'] = daily_df['Close'].pct_change().shift(-1)

    # Clean up data
    daily_df = daily_df.dropna()

    # --- Backtesting ---
    if not daily_df.empty:
        bt = Backtest(daily_df, FamaFrenchStrategy, cash=100_000, commission=.002)

        print("Running backtest...")
        stats = bt.run()

        print("\n--- Backtest Stats ---")
        print(stats)

        # --- Save Results ---
        print(f"\nSaving results to {results_file}...")
        sanitized = sanitize_stats(stats.to_dict())
        with open(results_file, 'w') as f:
            json.dump(sanitized, f, indent=4)

        # --- Generate Plot ---
        print(f"Saving plot to {plot_file}...")
        try:
            bt.plot(filename=plot_file, open_browser=False)
        except Exception as e:
            print(f"Could not generate plot due to an error: {e}")
    else:
        print("Could not run backtest because the processed data is empty.")
