"""
Mean Reversion Crypto Strategy
==============================
A mean-reversion strategy tailored for cryptocurrency markets, incorporating
multi-timeframe analysis and ATR-based risk management as per repository rules.
"""

from backtesting import Strategy, Backtest
import talib
import numpy as np
import pandas as pd

def preprocess_data(df, **params):
    """
    Adds all necessary indicators and higher-timeframe features to the data,
    ensuring compliance with the agent development rules.
    """
    # Validate required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain {required_cols}")

    # 1. Higher Timeframe Trend Filter (4H EMA)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)
    df_4h['htf_uptrend'] = df_4h['Close'] > df_4h['ema_200']

    # Map 4H trend back to the original timeframe
    df['htf_uptrend'] = df_4h['htf_uptrend'].reindex(df.index, method='ffill').fillna(False)

    # 2. ATR for Risk Management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # 3. Volume Confirmation
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=20)

    # 4. Moving Averages for Crossover Signal
    df['sma_fast'] = talib.SMA(df['Close'], timeperiod=params.get('sma_fast', 20))
    df['sma_slow'] = talib.SMA(df['Close'], timeperiod=params.get('sma_slow', 50))

    return df

class MeanReversionCrypto(Strategy):
    """
    A moving average crossover strategy with a higher-timeframe trend filter,
    volume confirmation, and ATR-based risk management.
    """
    # Optimizable parameters
    sma_fast = 20
    sma_slow = 50
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_ma_multiplier = 1.0

    def init(self):
        # Indicators from preprocessed data
        self.htf_uptrend = self.I(lambda: self.data.htf_uptrend, name='htf_uptrend')
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')
        self.sma_fast = self.I(lambda: self.data.sma_fast, name='sma_fast')
        self.sma_slow = self.I(lambda: self.data.sma_slow, name='sma_slow')

    def next(self):
        price = self.data.Close[-1]

        # Exit logic for existing positions
        if self.position:
            return

        # Entry Conditions
        volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_ma_multiplier

        # Long Entry: Fast SMA crosses above Slow SMA
        if self.htf_uptrend[-1] and volume_confirmed:
            if self.sma_fast[-1] > self.sma_slow[-1] and self.sma_fast[-2] <= self.sma_slow[-2]:
                sl = price - self.atr[-1] * self.atr_sl_multiplier
                tp = price + self.atr[-1] * self.atr_tp_multiplier
                self.buy(sl=sl, tp=tp)

        # Short Entry: Fast SMA crosses below Slow SMA
        elif not self.htf_uptrend[-1] and volume_confirmed:
            if self.sma_fast[-1] < self.sma_slow[-1] and self.sma_fast[-2] >= self.sma_slow[-2]:
                sl = price + self.atr[-1] * self.atr_sl_multiplier
                tp = price - self.atr[-1] * self.atr_tp_multiplier
                self.sell(sl=sl, tp=tp)

# Standalone backtesting block.
if __name__ == '__main__':
    import json
    import os

    # --- Configuration ---
    DATA_PATH = 'data/BTC-USD-15m.csv'
    CASH = 100_000
    COMMISSION = 0.002

    # --- Data Loading ---
    try:
        df = pd.read_csv(DATA_PATH, index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        # As a fallback, create synthetic data for demonstration
        print("Generating synthetic data...")
        from backtesting.test import EURUSD
        df = EURUSD.copy().resample('15min').agg('last').ffill()
        df = df.rename(columns={'Last': 'Close'})
        df['Open'] = df['Close'].shift(1)
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.005, len(df)))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.005, len(df)))
        df['Volume'] = np.random.randint(100, 1000, len(df))

    # --- Preprocessing ---
    df = preprocess_data(df, sma_fast=MeanReversionCrypto.sma_fast_period, sma_slow=MeanReversionCrypto.sma_slow_period)
    df = df.dropna()

    # --- Backtesting ---
    bt = Backtest(df, MeanReversionCrypto, cash=CASH, commission=COMMISSION)

    print("Running backtest...")
    stats = bt.run()
    print(stats)

    # --- Results ---
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save stats to JSON
    stats_dict = dict(stats)
    # Sanitize stats for JSON serialization
    for key, value in stats_dict.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_dict[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            stats_dict[key] = float(value)
        elif hasattr(value, 'to_dict'): # Handle pandas Series/DataFrame
            stats_dict[key] = value.to_dict()

    # Remove non-serializable items
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)

    # Save plot
    plot_filename = 'results/mean_reversion_crypto.html'
    bt.plot(filename=plot_filename, open_browser=False)

    print(f"\nBacktest complete. Results saved.")
    print(f"Stats: results/temp_result.json")
    print(f"Plot: {plot_filename}")
