
import json
import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# --- Data Preprocessing ---
def preprocess_data(df, fast_ema_period=20, slow_ema_period=50, trend_ema_period=200, atr_period=14, volume_ma_period=20):
    """
    Adds all necessary indicators to the DataFrame.
    """
    # Sanitize column names
    df.columns = [col.strip().title() for col in df.columns]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Calculate indicators
    df.ta.ema(length=fast_ema_period, append=True)
    df.ta.ema(length=slow_ema_period, append=True)
    df.ta.ema(length=trend_ema_period, append=True)
    df.ta.atr(length=atr_period, append=True)
    df.ta.sma(close=df['Volume'], length=volume_ma_period, append=True)

    # Multi-Timeframe Trend Filter (4H)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_4h.ta.ema(length=trend_ema_period, append=True)
    htf_ema_col = f'EMA_{trend_ema_period}'
    df_4h['htf_uptrend'] = (df_4h['Close'] > df_4h[htf_ema_col]).astype(int)

    df['htf_uptrend'] = df_4h['htf_uptrend'].reindex(df.index, method='ffill')

    # Rename columns for clarity
    df.rename(columns={
        f'EMA_{fast_ema_period}': 'fast_ema',
        f'EMA_{slow_ema_period}': 'slow_ema',
        f'EMA_{trend_ema_period}': 'trend_ema',
        f'ATRr_{atr_period}': 'atr',
        f'SMA_{volume_ma_period}': 'volume_ma'
    }, inplace=True)

    return df

# --- Strategy Class ---
class EmaCrossTrendFilter(Strategy):
    """
    A trend-following strategy based on EMA crossovers with a higher-timeframe
    trend filter, volume confirmation, and ATR-based risk management.
    """
    # Optimizable parameters
    fast_ema_period = 20
    slow_ema_period = 50
    trend_ema_period = 200
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_ma_multiplier = 1.0 # Trades only when volume is above its MA

    def init(self):
        # Initialize indicators using the preprocessed data columns
        self.fast_ema = self.I(lambda: self.data.fast_ema, name='Fast EMA')
        self.slow_ema = self.I(lambda: self.data.slow_ema, name='Slow EMA')
        self.trend_ema = self.I(lambda: self.data.trend_ema, name='Trend EMA')
        self.atr = self.I(lambda: self.data.atr, name='ATR')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='Volume MA')
        self.htf_uptrend = self.I(lambda: self.data.htf_uptrend, name='HTF Uptrend')

    def next(self):
        # If a position is already open, do nothing.
        if self.position:
            return

        price = self.data.Close[-1]

        # --- FILTERS ---
        # 1. Volume Filter: Current volume must be above its moving average
        volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_ma_multiplier
        if not volume_confirmed:
            return

        # 2. Higher-Timeframe Trend Filter
        is_htf_uptrend = self.htf_uptrend[-1] == 1

        # --- ENTRY CONDITIONS ---
        # Long Entry: Fast EMA crosses above Slow EMA, with price above Trend EMA and HTF confirmation
        if is_htf_uptrend and self.slow_ema[-1] > self.trend_ema[-1] and crossover(self.fast_ema, self.slow_ema):
            sl = price - self.atr[-1] * self.atr_sl_multiplier
            tp = price + self.atr[-1] * self.atr_tp_multiplier
            self.buy(sl=sl, tp=tp)

        # Short Entry: Fast EMA crosses below Slow EMA, with price below Trend EMA and HTF confirmation
        elif not is_htf_uptrend and self.slow_ema[-1] < self.trend_ema[-1] and crossover(self.slow_ema, self.fast_ema):
            sl = price + self.atr[-1] * self.atr_sl_multiplier
            tp = price - self.atr[-1] * self.atr_tp_multiplier
            self.sell(sl=sl, tp=tp)

def sanitize_stats(stats):
    """Converts a backtesting stats Series to a JSON-serializable dictionary."""
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = float(value)
        elif isinstance(value, (pd.Series, pd.DataFrame, Strategy)):
            continue  # Skip complex or non-serializable objects
        else:
            sanitized[key] = value
    return sanitized

# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        data = pd.read_csv('src/data/rbi/BTC-USD-15m.csv', parse_dates=['datetime'], index_col='datetime')
    except FileNotFoundError:
        print("Error: Data file not found. Make sure 'src/data/rbi/BTC-USD-15m.csv' exists.")
        exit()

    # Preprocess data with default strategy parameters
    processed_data = preprocess_data(
        data.copy(),
        fast_ema_period=EmaCrossTrendFilter.fast_ema_period,
        slow_ema_period=EmaCrossTrendFilter.slow_ema_period,
        trend_ema_period=EmaCrossTrendFilter.trend_ema_period
    )
    processed_data.dropna(inplace=True)

    print("Data preprocessed successfully. Running backtest...")

    bt = Backtest(processed_data, EmaCrossTrendFilter, cash=100_000, commission=.002)
    stats = bt.run()

    print("\n--- Backtest Stats ---")
    print(stats)

    # --- Save Results ---
    stats_dict = sanitize_stats(stats)

    # Save stats to JSON
    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print("\n[+] Backtest statistics saved to results/temp_result.json")

    # Save plot to HTML
    bt.plot(filename='results/ema_cross_trend_filter.html', open_browser=False)
    print("[+] Backtest plot saved to results/ema_cross_trend_filter.html")
