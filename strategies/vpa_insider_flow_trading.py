"""
VPA Insider Flow Trading Strategy
=================================

A strategy inspired by Volume Price Analysis (VPA) but implemented using the
VuManchu Cipher B indicator and adhering to the repository's strict development guidelines.

Entry Logic:
- Long: Cipher B buy signal (`buy_signal` > 0)
- Short: Cipher B sell signal (`sell_signal` > 0)

Filters (Mandatory):
- Higher Timeframe Trend: Only take longs if the 4H trend is up (Close > 4H EMA).
- Volume Confirmation: Entry bar's volume must be above its 20-period moving average.

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
    df['htf_trend_up'] = df['htf_trend_up'].fillna(0) # Avoid chained assignment warning

    # 3. Add Volume Confirmation Filter (20-period SMA)
    df['volume_ma20'] = talib.SMA(df['Volume'], timeperiod=20)

    # 4. Add ATR for Risk Management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    return df


# --- Strategy Class ---

class VpaInsiderFlowTrading(Strategy):
    """
    Implements the VPA/Cipher B hybrid strategy.
    """
    # --- Optimizable Parameters ---
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    def init(self):
        """
        Initialize all indicators. The data is preprocessed, so we just need
        to create references using self.I() for plotting and access.
        """
        self.buy_sig = self.I(lambda: self.data.buy_signal, name='buy_signal')
        self.sell_sig = self.I(lambda: self.data.sell_signal, name='sell_signal')
        self.htf_trend_up = self.I(lambda: self.data.htf_trend_up, name='htf_trend_up')
        self.volume_ma = self.I(lambda: self.data.volume_ma20, name='volume_ma20')
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.rsimfi = self.I(lambda: self.data.rsimfi, name='rsimfi')

    def next(self):
        """
        Main trading logic.
        """
        # --- Guard Clause for Warmup Period ---
        # Wait for all indicators to have valid values
        if np.isnan(self.atr[-1]) or np.isnan(self.volume_ma[-1]) or np.isnan(self.rsimfi[-1]):
            return

        price = self.data.Close[-1]

        # --- FILTERS ---
        volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1]

        # --- ENTRY LOGIC ---
        if not self.position:
            # Long entry: Cipher B buy signal + HTF uptrend + Volume confirmation
            if self.buy_sig[-1] > 0 and self.htf_trend_up[-1] > 0 and volume_confirmed:
                sl = price - (self.atr_sl_multiplier * self.atr[-1])
                tp = price + (self.atr_tp_multiplier * self.atr[-1])
                self.buy(sl=sl, tp=tp)

            # Short entry: Cipher B sell signal + HTF downtrend + Volume confirmation
            elif self.sell_sig[-1] > 0 and self.htf_trend_up[-1] == 0 and volume_confirmed:
                sl = price + (self.atr_sl_multiplier * self.atr[-1])
                tp = price - (self.atr_tp_multiplier * self.atr[-1])
                self.sell(sl=sl, tp=tp)


# --- Standalone Runner ---

if __name__ == '__main__':
    # --- Data Loading and Preprocessing ---
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found. Please ensure the data file is in the correct location.")
        sys.exit(1)

    df_processed = preprocess_data(df)

    # --- Backtest Execution ---
    bt = Backtest(
        df_processed,
        VpaInsiderFlowTrading,
        cash=100_000,
        commission=.002
    )

    stats = bt.run()
    print(stats)

    # --- Results / Plotting ---
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Sanitize stats for JSON output
    stats_json = {k: v for k, v in stats.items() if isinstance(v, (int, float, str))}
    with open('results/temp_result.json', 'w') as f:
        import json
        json.dump(stats_json, f, indent=4)

    bt.plot(filename='results/vpa_insider_flow_trading.html', open_browser=False)
