
import pandas as pd
import pandas_ta as ta
from backtesting import Strategy, Backtest
import numpy as np
import json
import os

def sanitize_stats(stats):
    """
    Sanitizes the stats object by converting non-serializable types to native Python types.
    Removes DataFrames ('_equity_curve', '_trades') to avoid excessive data in JSON.
    """
    if stats is None:
        return {}

    # Convert pandas Series to dictionary
    sanitized = stats.to_dict()

    # List of keys to remove that contain DataFrame objects
    keys_to_remove = ['_equity_curve', '_trades']
    for key in keys_to_remove:
        if key in sanitized:
            del sanitized[key]

    # Recursively sanitize the rest of the dictionary
    for key, value in sanitized.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.int64)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            sanitized[key] = float(value)
        elif isinstance(value, (np.ndarray, pd.Series)):
             sanitized[key] = value.tolist() # Convert arrays/series to lists
        elif pd.isna(value):
            sanitized[key] = None
        # If the key is '_strategy', we might want to represent it as a string or a simplified dict
        elif key == '_strategy':
            sanitized[key] = str(value)


    return sanitized


def preprocess_data(df, htf_period=50, ema_fast_period=20, ema_slow_period=50, atr_period=14, volume_ma_period=20):
    """
    Adds all indicators to the DataFrame.
    """
    # Ensure index is a DatetimeIndex
    df.index = pd.to_datetime(df.index)

    # Sanitize column names
    df.columns = [col.strip().title() for col in df.columns]

    # Higher timeframe trend (4H)
    df_4h = df.resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # Calculate 4H EMA
    df_4h['ema_htf'] = ta.ema(df_4h['Close'], length=htf_period)

    # Determine HTF trend (1 for up, -1 for down)
    df_4h['htf_trend'] = np.where(df_4h['Close'] > df_4h['ema_htf'], 1, -1)

    # Forward fill to 15m
    df['htf_trend'] = df_4h['htf_trend'].reindex(df.index, method='ffill')
    df['htf_trend'] = df['htf_trend'].bfill()

    # 15m EMA Cloud
    df['ema_fast'] = ta.ema(df['Close'], length=ema_fast_period)
    df['ema_slow'] = ta.ema(df['Close'], length=ema_slow_period)

    # ATR for risk management
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=atr_period)

    # Volume Moving Average
    df['volume_ma'] = ta.sma(df['Volume'], length=volume_ma_period)

    # DEBUG: Inspect DataFrame state
    # print("After indicators, before dropna:")
    # print(df.head(10))
    # print(df.tail(10))
    # print(df.isnull().sum())

    # Drop rows with NaN values resulting from indicator calculations
    #df.dropna(inplace=True)

    # DEBUG: Inspect DataFrame state after dropna
    # print("\nAfter dropna:")
    # print(df.head(10))
    # print(f"DataFrame empty: {df.empty}")


    return df

class EmaCloudTrendContinuation(Strategy):
    """
    EMA Cloud Trend Continuation Strategy
    - Uses a higher timeframe (4H) EMA to establish trend bias.
    - Enters on pullbacks to a lower timeframe (15m) EMA cloud.
    - All entries are confirmed with above-average volume.
    - Risk management is based on ATR.
    """
    # Optimizable parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    def init(self):
        # Indicators
        self.htf_trend = self.I(lambda: self.data.htf_trend, name='htf_trend')
        self.ema_fast = self.I(lambda: self.data.ema_fast, name='ema_fast')
        self.ema_slow = self.I(lambda: self.data.ema_slow, name='ema_slow')
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')

    def next(self):
        # Ensure all indicators have valid values
        if np.isnan(self.htf_trend[-1]) or np.isnan(self.ema_fast[-1]) or np.isnan(self.ema_slow[-1]) or \
           np.isnan(self.atr[-1]) or np.isnan(self.volume_ma[-1]):
            return

        # --- Filters ---
        # 1. Volume Filter: Only trade when volume is above its moving average
        if self.data.Volume[-1] < self.volume_ma[-1]:
            return

        # 2. Trend Filter: Ensure we are trading in the direction of the higher timeframe trend
        is_htf_uptrend = self.htf_trend[-1] == 1
        is_htf_downtrend = self.htf_trend[-1] == -1

        # --- Entry Logic ---
        if not self.position:
            # Long Entry Conditions
            if is_htf_uptrend:
                # Price must be above the fast EMA (in the "buy zone")
                is_above_cloud = self.data.Close[-1] > self.ema_fast[-1]
                # A pullback is identified if the low of the candle touches or goes into the cloud (between fast and slow EMA)
                is_pullback = self.data.Low[-1] <= self.ema_fast[-1]

                if is_above_cloud and is_pullback:
                    sl = self.data.Close[-1] - (self.atr[-1] * self.atr_sl_multiplier)
                    tp = self.data.Close[-1] + (self.atr[-1] * self.atr_tp_multiplier)
                    self.buy(sl=sl, tp=tp)

            # Short Entry Conditions
            elif is_htf_downtrend:
                # Price must be below the fast EMA (in the "sell zone")
                is_below_cloud = self.data.Close[-1] < self.ema_fast[-1]
                # A pullback is identified if the high of the candle touches or goes into the cloud
                is_pullback = self.data.High[-1] >= self.ema_fast[-1]

                if is_below_cloud and is_pullback:
                    sl = self.data.Close[-1] + (self.atr[-1] * self.atr_sl_multiplier)
                    tp = self.data.Close[-1] - (self.atr[-1] * self.atr_tp_multiplier)
                    self.sell(sl=sl, tp=tp)

# --- Main execution block ---
if __name__ == '__main__':
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found. Please ensure the data file is in the correct directory.")
        # As a fallback for testing, create some synthetic data
        data = pd.DataFrame(
            np.random.rand(2000, 5) * 100 + 40000,
            columns=['Open', 'High', 'Low', 'Close', 'Volume'],
            index=pd.to_datetime(pd.date_range('2022-01-01', periods=2000, freq='15min'))
        )
        data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.rand(2000) * 10
        data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.rand(2000) * 10


    # Preprocess the data
    data = preprocess_data(data)

    # Initialize and run the backtest
    bt = Backtest(data, EmaCloudTrendContinuation, cash=100_000, commission=.002)
    stats = bt.run()

    print("--- Backtest Stats ---")
    print(stats)

    # Save stats to JSON
    sanitized_stats = sanitize_stats(stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    print("\\n--- Plotting ---")
    plot_filename = 'results/strategy_41ffa6925853.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Plot saved to {plot_filename}")
