
import pandas as pd
from backtesting import Backtest, Strategy
import pandas_ta as ta
import numpy as np
import json
import os
import sys

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df: pd.DataFrame):
    """
    Applies indicator calculations and multi-timeframe analysis.
    Resamples 15m data to 4h and 30m, calculates indicators,
    and merges them back into the 15m dataframe.
    """
    df = df.copy()

    # --- Aggregation rules for resampling ---
    agg_rules = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}

    # --- 1. 4-Hour Environmental Timeframe ---
    df_4h = df.resample('4H').agg(agg_rules).dropna()

    # Market Cipher B on 4h
    df_4h = cipher_b(df_4h)

    # Market Cipher A (Ribbon 5) Proxy: 21 EMA
    df_4h['ema21'] = ta.ema(df_4h['Close'], length=21)

    # Wolfpack ID Proxy: VWAP crossing its 5-period SMA
    df_4h.ta.vwap(append=True)
    df_4h['vwap_sma5'] = ta.sma(df_4h[df_4h.columns[-1]], length=5) # df.columns[-1] is the vwap column
    df_4h['wolfpack_cross_up'] = (df_4h[df_4h.columns[-2]] > df_4h[df_4h.columns[-1]]).astype(int)

    # Prefix 4h columns
    df_4h.columns = [f"4h_{col}" for col in df_4h.columns]

    # --- 2. 30-Minute Execution Timeframe (Proxy for 24m) ---
    df_30m = df.resample('30min').agg(agg_rules).dropna()

    # Convert to Heikin Ashi candles
    ha_cols = ['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']
    df_30m[ha_cols] = ta.ha(df_30m['Open'], df_30m['High'], df_30m['Low'], df_30m['Close'])

    # Use Heikin Ashi candles for subsequent calculations
    exec_df = df_30m.copy()
    exec_df['Open'], exec_df['High'], exec_df['Low'], exec_df['Close'] = df_30m['HA_Open'], df_30m['HA_High'], df_30m['HA_Low'], df_30m['HA_Close']

    # Market Cipher B on 30m Heikin Ashi
    exec_df = cipher_b(exec_df)

    # Wolfpack ID Proxy on 30m Heikin Ashi
    exec_df.ta.vwap(append=True)
    exec_df['vwap_sma5'] = ta.sma(exec_df[exec_df.columns[-1]], length=5)
    exec_df['wolfpack_cross_up'] = (exec_df[exec_df.columns[-2]] > exec_df[exec_df.columns[-1]]).astype(int)

    # Add Heikin Ashi doji detection
    body_size = abs(exec_df['HA_Close'] - exec_df['HA_Open'])
    range_size = exec_df['HA_High'] - exec_df['HA_Low']
    # Avoid division by zero
    range_size[range_size == 0] = np.nan
    exec_df['is_doji'] = (body_size / range_size) < 0.1 # Doji if body is less than 10% of range

    # Prefix 30m columns
    exec_df.columns = [f"30m_{col}" for col in exec_df.columns]

    # --- 3. Merge back into 15m DataFrame ---
    df = pd.merge(df, df_4h, how='left', left_index=True, right_index=True)
    df = pd.merge(df, exec_df, how='left', left_index=True, right_index=True)

    # Forward-fill the multi-timeframe data
    df.ffill(inplace=True)

    # Add 15m ATR for risk management
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    return df.dropna()

class MarketCipher4h24mSwingLong(Strategy):
    """
    Market Cipher 4H/24M Swing Long Strategy
    A multi-timeframe strategy using proxies for Market Cipher indicators.
    """

    def init(self):
        """
        Initialize indicators.
        """
        # Indicators will be initialized here in a later step
        pass

    def next(self):
        """
        Define the trading logic.
        """
        # Trading logic will be implemented here in a later step
        pass

def sanitize_stats(stats):
    """
    Sanitizes the stats object by converting non-serializable types.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = float(value)
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            # Skip DataFrame/Series objects like _equity_curve and _trades
            continue
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    return sanitized

if __name__ == '__main__':
    # Configuration
    data_path = 'data/BTC-USD-15m.csv'
    output_json_path = 'results/temp_result.json'
    output_plot_path = 'results/market_cipher_4h_24m_swing_long.html'

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Load data
    try:
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        # Sanitize column names
        data.columns = [c.strip().title() for c in data.columns]
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        # As a fallback, create synthetic data for verification
        print("Generating synthetic data...")
        from backtesting.test import GOOG
        data = GOOG.copy()
        data = data.resample('15min').ffill()
        data.index.name = 'datetime'


    # Preprocess data
    data = preprocess_data(data)

    if data.empty:
        print("Error: Data is empty after preprocessing. Cannot run backtest.")
    else:
        # Run backtest
        print("Running backtest...")
        bt = Backtest(data, MarketCipher4h24mSwingLong, cash=100_000, commission=.002)

        try:
            stats = bt.run()
            print("Backtest complete. Stats:")
            print(stats)

            # Sanitize and save stats to JSON
            sanitized_stats = sanitize_stats(stats)
            with open(output_json_path, 'w') as f:
                json.dump(sanitized_stats, f, indent=4)
            print(f"Stats saved to {output_json_path}")

            # Generate and save plot
            try:
                bt.plot(filename=output_plot_path, open_browser=False)
                print(f"Plot saved to {output_plot_path}")
            except Exception as e:
                print(f"Could not generate plot: {e}")

        except Exception as e:
            print(f"An error occurred during backtest execution: {e}")
