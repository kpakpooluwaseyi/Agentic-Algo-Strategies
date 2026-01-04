
import pandas as pd
import numpy as np
import talib
from backtesting import Strategy, Backtest
import sys
import os
import json
from scipy.signal import find_peaks

# Add parent directory to path for custom imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

# --- Strategy Implementation ---

def preprocess_data(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Prepares the data by adding required indicators and multi-timeframe analysis.
    """
    # 1. Calculate indicators on the base 15m timeframe
    df = cipher_b(df)
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=20)

    # 2. Resample to 4H to get higher-timeframe trend
    df_4h = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # Calculate Cipher B on the 4H data
    df_4h = cipher_b(df_4h)

    # 3. Define the 4H trend conditions based on strategy rules
    # Long trend: Trigger wave up (wt1 > wt2), positive money flow, and not overbought (wt1 < 53)
    long_trend_condition = (df_4h['wt1'] > df_4h['wt2']) & (df_4h['rsimfi'] > 0) & (df_4h['wt1'] < 53)
    # Short trend: Trigger wave down (wt1 < wt2) and negative money flow
    short_trend_condition = (df_4h['wt1'] < df_4h['wt2']) & (df_4h['rsimfi'] < 0)

    # Create a numerical trend column (2 for long, 1 for short, 0 for neutral)
    df_4h['htf_trend'] = 0
    df_4h.loc[long_trend_condition, 'htf_trend'] = 2
    df_4h.loc[short_trend_condition, 'htf_trend'] = 1

    # 4. Detect Bearish Divergence on the 4H timeframe
    price_peaks_idx, _ = find_peaks(df_4h['Close'], distance=5, prominence=df_4h['Close'].std() / 2)
    wt1_peaks_idx, _ = find_peaks(df_4h['wt1'], distance=5, prominence=df_4h['wt1'].std() / 2)

    df_4h['bearish_divergence_signal'] = False

    if len(price_peaks_idx) >= 2 and len(wt1_peaks_idx) >= 2:
        # Check last two peaks
        if df_4h['Close'].iloc[price_peaks_idx[-1]] > df_4h['Close'].iloc[price_peaks_idx[-2]]:
            # Find corresponding wt1 peaks
            corresponding_wt1_peaks = df_4h['wt1'].iloc[wt1_peaks_idx[np.searchsorted(wt1_peaks_idx, price_peaks_idx[-2]):np.searchsorted(wt1_peaks_idx, price_peaks_idx[-1])+1]]
            if len(corresponding_wt1_peaks) >= 2:
                if corresponding_wt1_peaks.iloc[-1] < corresponding_wt1_peaks.iloc[-2]:
                    # Bearish divergence detected. Signal from the first peak to the second.
                    signal_start_index = df_4h.index[price_peaks_idx[-2]]
                    signal_end_index = df_4h.index[price_peaks_idx[-1]]
                    df_4h.loc[signal_start_index:signal_end_index, 'bearish_divergence_signal'] = True

    # 5. Merge the 4H trend and divergence signal back into the 15m dataframe
    df['htf_trend'] = df_4h['htf_trend'].reindex(df.index, method='ffill').fillna(0).astype(int)
    df['bearish_divergence_signal'] = df_4h['bearish_divergence_signal'].reindex(df.index, method='ffill').fillna(False)

    # Convert boolean signals from the 15m cipher_b to int for backtesting.py
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    return df


class MarketCipher4h24mTrend(Strategy):
    """
    Implements the Market Cipher 4h/24m Trend Following strategy.
    The 24m timeframe is proxied by the 15m data provided.
    """
    # Optimizable parameters based on development guidelines
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_ma_period = 20

    def init(self):
        """
        Initialize indicators using self.I to make them accessible in `next()`
        """
        self.htf_trend = self.I(lambda: self.data.htf_trend.astype(int), name='htf_trend')
        self.buy_signal = self.I(lambda: self.data.buy_signal.astype(int), name='buy_signal')
        self.sell_signal = self.I(lambda: self.data.sell_signal.astype(int), name='sell_signal')
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')
        self.bearish_divergence = self.I(lambda: self.data.bearish_divergence_signal.astype(int), name='bearish_divergence')

    def next(self):
        """
        Define the trading logic.
        """
        # --- FILTERS ---
        # 1. Volume Filter: Only trade if volume is above its moving average
        volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1]

        # --- ENTRY CONDITIONS ---
        if not self.position:
            # LONG ENTRY
            # 4H Trend must be UP (htf_trend == 2)
            # 15m must have a buy signal
            # Volume must be confirmed
            if self.htf_trend[-1] == 2 and self.buy_signal[-1] == 1 and volume_confirmed:
                sl = self.data.Close[-1] - (self.atr[-1] * self.atr_sl_multiplier)
                tp = self.data.Close[-1] + (self.atr[-1] * self.atr_tp_multiplier)
                self.buy(sl=sl, tp=tp)

            # SHORT ENTRY
            # 4H must show bearish divergence
            # 4H Trend must be DOWN (htf_trend == 1)
            # 15m must have a sell signal
            # Volume must be confirmed
            elif self.bearish_divergence[-1] == 1 and self.htf_trend[-1] == 1 and self.sell_signal[-1] == 1 and volume_confirmed:
                sl = self.data.Close[-1] + (self.atr[-1] * self.atr_sl_multiplier)
                tp = self.data.Close[-1] - (self.atr[-1] * self.atr_tp_multiplier)
                self.sell(sl=sl, tp=tp)

# --- Backtesting Execution ---

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to be JSON serializable.
    Removes non-serializable types like DataFrames, Timestamps, etc.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.int64)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            sanitized[key] = float(value)
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            # Skip DataFrames and Series as they are not easily serializable
            continue
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    # Ensure _strategy is a string if it's an object
    if '_strategy' in sanitized and not isinstance(sanitized['_strategy'], str):
        sanitized['_strategy'] = str(sanitized['_strategy'])
    return sanitized


if __name__ == '__main__':
    # Configuration
    data_path = 'data/BTC-USD-15m.csv'
    output_json_path = 'results/temp_result.json'
    output_plot_path = 'results/strategy_78da4c98448c.html'

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Load data
    try:
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        # Sanitize column names (e.g., 'open' -> 'Open', '  volume  ' -> 'Volume')
        data.columns = [col.strip().title() for col in data.columns]
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        # Create dummy data for testing if file not found
        data = pd.DataFrame({
            'Open': np.random.rand(2000) + 100,
            'High': np.random.rand(2000) + 101,
            'Low': np.random.rand(2000) + 99,
            'Close': np.random.rand(2000) + 100,
            'Volume': np.random.rand(2000) * 100
        }, index=pd.to_datetime(pd.date_range('2022-01-01', periods=2000, freq='15min')))

    # Preprocess data
    data = preprocess_data(data)

    # Run backtest
    bt = Backtest(data, MarketCipher4h24mTrend, cash=100_000, commission=.002)
    stats = bt.run()

    print("--- Backtest Results ---")
    print(stats)

    # Sanitize and save stats
    sanitized_stats = sanitize_stats(stats)
    with open(output_json_path, 'w') as f:
        json.dump(sanitized_stats, f, indent=4)
    print(f"Saved stats to {output_json_path}")

    # Save plot
    try:
        bt.plot(filename=output_plot_path, open_browser=False)
        print(f"Saved plot to {output_plot_path}")
    except Exception as e:
        print(f"Could not save the plot due to an error: {e}")
