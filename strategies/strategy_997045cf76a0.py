"""
strategy_997045cf76a0: vumanchu_scalping_5min
"""

import json
import numpy as np
import pandas as pd
import talib
import os
import sys
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df, **params):
    """
    Adds all necessary indicators to the DataFrame for the strategy.
    """
    # Primary timeframe indicators
    df['ema_50'] = talib.EMA(df['Close'], timeperiod=50)
    df['ema_200'] = talib.EMA(df['Close'], timeperiod=200)

    # VuManchu Cipher B indicator
    df = cipher_b(df)

    # Mandatory Guideline Indicators
    # 1. Higher Timeframe Trend Filter (4H)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)
    df_4h['htf_trend_up'] = df_4h['Close'] > df_4h['ema_200']

    df['htf_trend_up'] = df_4h['htf_trend_up'].reindex(df.index, method='ffill')
    df['htf_trend_up'].fillna(False, inplace=True) # Fill initial NaNs

    # 2. Volume Confirmation
    df['volume_sma_20'] = talib.SMA(df['Volume'], timeperiod=20)

    # 3. ATR for Risk Management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Convert boolean signals for backtesting.py compatibility
    df['buy_signal'] = df['buy_signal'].astype(bool)
    df['sell_signal'] = df['sell_signal'].astype(bool)

    return df

class VumanchuScalping(Strategy):
    """
    Implements the VuManchu Scalping strategy with mandatory guideline adaptations.
    """
    # Optimizable parameters
    ema_fast_period = 50
    ema_slow_period = 200
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    max_trades_per_trend = 5

    def init(self):
        # Indicators
        self.ema_fast = self.I(talib.EMA, self.data.Close, self.ema_fast_period)
        self.ema_slow = self.I(talib.EMA, self.data.Close, self.ema_slow_period)
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.volume_sma = self.I(lambda: self.data.volume_sma_20, name='volume_sma_20')
        self.htf_trend_up = self.I(lambda: self.data.htf_trend_up, name='htf_trend_up')
        self.vumanchu_buy = self.I(lambda: self.data.buy_signal, name='vumanchu_buy')
        self.vumanchu_sell = self.I(lambda: self.data.sell_signal, name='vumanchu_sell')

        # State variables
        self.trade_count = 0
        self.trend = 0 # 1 for bullish, -1 for bearish, 0 for neutral

    def next(self):
        # Wait for indicator warmup
        if len(self.data) < self.ema_slow_period or np.isnan(self.atr[-1]):
            return

        price = self.data.Close[-1]

        # Detect trend change (Golden/Death Cross)
        if crossover(self.ema_fast, self.ema_slow):
            self.trend = 1
            self.trade_count = 0
        elif crossover(self.ema_slow, self.ema_fast):
            self.trend = -1
            self.trade_count = 0

        # Don't trade if a position is already open
        if self.position:
            return

        # Entry logic
        if self.trade_count < self.max_trades_per_trend:
            # Bullish trend entry
            if self.trend == 1 and self.vumanchu_buy[-1]:
                # Guideline checks
                if self.htf_trend_up[-1] and self.data.Volume[-1] > self.volume_sma[-1]:
                    sl = price - (self.atr[-1] * self.atr_sl_multiplier)
                    tp = price + (self.atr[-1] * self.atr_tp_multiplier)

                    # Additional check to prevent invalid SL/TP
                    if tp > price and sl < price:
                        self.buy(sl=sl, tp=tp)
                        self.trade_count += 1

            # Bearish trend entry
            elif self.trend == -1 and self.vumanchu_sell[-1]:
                # Guideline checks
                if not self.htf_trend_up[-1] and self.data.Volume[-1] > self.volume_sma[-1]:
                    sl = price + (self.atr[-1] * self.atr_sl_multiplier)
                    tp = price - (self.atr[-1] * self.atr_tp_multiplier)

                    # Additional check to prevent invalid SL/TP
                    if tp < price and sl > price:
                        self.sell(sl=sl, tp=tp)
                        self.trade_count += 1

if __name__ == '__main__':
    # Load data
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found. Please ensure the data file is in the correct directory.")
        exit()

    # Preprocess data
    df.columns = [col.strip().capitalize() for col in df.columns]
    df = preprocess_data(df)

    # Run backtest
    bt = Backtest(df, VumanchuScalping, cash=100_000, commission=.002)
    stats = bt.run()

    print("Backtest Results:")
    print(stats)

    # Save results to a JSON file
    results_path = 'results/temp_result.json'
    stats_dict = dict(stats)

    def sanitize_for_json(obj):
        if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        if pd.isna(obj):
            return None
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize_for_json(i) for i in obj]
        return obj

    # Clean the stats dictionary
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)
    if '_strategy' in stats_dict:
        stats_dict['_strategy'] = str(stats_dict['_strategy'])

    sanitized_stats = sanitize_for_json(stats_dict)

    with open(results_path, 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    print(f"\nResults saved to {results_path}")

    # Generate plot
    plot_filename = 'results/strategy_997045cf76a0.html'
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
