"""
CipherB With Filters Strategy
=============================

This strategy uses the VuManchu Cipher B indicator for entry signals,
filtered by a higher-timeframe trend indicator, and uses ATR-based risk management,
adhering to the repository's development guidelines.

Entry Logic:
- Long: Cipher B buy signal (`buy_signal` > 0) + Positive MFI (`rsimfi` > 0)
- Short: Cipher B sell signal (`sell_signal` > 0) + Negative MFI (`rsimfi` < 0)

Filters (Mandatory):
- Higher Timeframe Trend: Only take longs if the 4H trend is up (Close > 4H EMA).

Risk Management (Mandatory):
- Stop Loss: ATR-based (2x ATR by default).
- Take Profit: ATR-based (3x ATR by default).
"""

import pandas as pd
import talib
from backtesting import Strategy, Backtest
import numpy as np
import os
import sys
import json

# Add parent directory to path to allow for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b


# --- Preprocessing Function ---

def preprocess_data(df: pd.DataFrame, **params):
    """
    Applies all necessary indicators and filters to the raw OHLCV data.
    """
    df = df.copy()

    # 1. Add VuManchu Cipher B indicator suite
    df = cipher_b(df)
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    # 2. Add Higher Timeframe (HTF) Trend Filter (4h EMA)
    df_4h = df.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)
    df_4h['htf_trend_up'] = (df_4h['Close'] > df_4h['ema_200']).astype(int)

    # Map 4h trend back to the original timeframe
    df['htf_trend_up'] = df_4h['htf_trend_up'].reindex(df.index, method='ffill')
    df['htf_trend_up'] = df['htf_trend_up'].fillna(0)

    # 3. Add ATR for Risk Management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    return df


# --- Strategy Class ---

class CipherBWithFiltersStrategy(Strategy):
    """
    Implements the CipherB with Filters strategy.
    """
    # --- Optimizable Parameters ---
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    def init(self):
        self.buy_sig = self.I(lambda: self.data.buy_signal, name='buy_signal')
        self.sell_sig = self.I(lambda: self.data.sell_signal, name='sell_signal')
        self.htf_trend_up = self.I(lambda: self.data.htf_trend_up, name='htf_trend_up')
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.rsimfi = self.I(lambda: self.data.rsimfi, name='rsimfi')

    def next(self):
        if np.isnan(self.atr[-1]) or np.isnan(self.rsimfi[-1]):
            return

        price = self.data.Close[-1]

        if not self.position:
            if self.buy_sig[-1] > 0 and self.htf_trend_up[-1] > 0 and self.rsimfi[-1] > 0:
                sl = price - (self.atr_sl_multiplier * self.atr[-1])
                tp = price + (self.atr_tp_multiplier * self.atr[-1])
                if sl < price and tp > price:
                    self.buy(sl=sl, tp=tp)

            elif self.sell_sig[-1] > 0 and self.htf_trend_up[-1] == 0 and self.rsimfi[-1] < 0:
                sl = price + (self.atr_sl_multiplier * self.atr[-1])
                tp = price - (self.atr_tp_multiplier * self.atr[-1])
                if sl > price and tp < price:
                    self.sell(sl=sl, tp=tp)


# --- Standalone Runner ---

if __name__ == '__main__':
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
        # Sanitize column names: remove whitespace and capitalize
        df.columns = [col.strip().capitalize() for col in df.columns]
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found.")
        sys.exit(1)

    df_processed = preprocess_data(df)

    bt = Backtest(
        df_processed,
        CipherBWithFiltersStrategy,
        cash=100_000,
        commission=.002
    )

    stats = bt.run()
    print("--- CipherB With Filters Strategy Results ---")
    print(stats)

    if not os.path.exists('results'):
        os.makedirs('results')

    def sanitize_stats(stats_obj):
        if isinstance(stats_obj, pd.Series):
            stats_obj = stats_obj.to_dict()
        sanitized = {}
        for k, v in stats_obj.items():
            if isinstance(v, (pd.Timestamp, pd.Timedelta)):
                sanitized[k] = str(v)
            elif isinstance(v, np.integer):
                sanitized[k] = int(v)
            elif isinstance(v, np.floating):
                sanitized[k] = float(v)
            elif isinstance(v, (int, float, str, bool, type(None))):
                sanitized[k] = v
        return sanitized

    final_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(final_stats, f, indent=4)
    print("\nSaved stats to results/temp_result.json")

    plot_filename = 'results/cipher_b_with_filters.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Saved plot to {plot_filename}")
