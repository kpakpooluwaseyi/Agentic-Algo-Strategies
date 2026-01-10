"""
Strategy: EMA 200 Reversal (Hold the Mayo)
Author: MoonDev
Description: This strategy identifies M/W reversal patterns that occur at or near the 200 EMA,
             a key level often representing a dynamic support/resistance zone. It uses a
             higher-timeframe trend filter and volume confirmation for robust entry signals.

Confluence Factors:
- Pattern: M (short) or W (long) reversal pattern.
- Level: Pattern forms at or very near the 15m 200 EMA.
- HTF Trend: Entry must be in the direction of the 4H trend (above/below 4H 200 EMA).
- Volume: Entry candle must have above-average volume.
- TDI/Cipher: VuManchu Cipher B provides entry confirmation.
"""

import pandas as pd
import talib
from scipy.signal import find_peaks
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b
from backtesting import Strategy

def preprocess_data(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Adds all necessary indicators and features to the DataFrame for the strategy.
    """
    # --- Multi-Timeframe (HTF) Trend Filter (4H) ---
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)
    df_4h['htf_uptrend'] = df_4h['Close'] > df_4h['ema_200']

    # Map 4H trend back to the 15m DataFrame, forward-filling values
    df['htf_uptrend'] = df_4h['htf_uptrend'].reindex(df.index, method='ffill').fillna(False)

    # --- Core 15m Indicators ---
    df['ema_200'] = talib.EMA(df['Close'], timeperiod=200)
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=20)

    # --- Confluence Indicators ---
    # Apply VuManchu Cipher B for TDI/market sentiment signals
    df = cipher_b(df)
    df['cipher_buy'] = (df['wt1'] > df['wt2']) & (df['wt1'] < -20)
    df['cipher_sell'] = (df['wt1'] < df['wt2']) & (df['wt1'] > 20)

    # --- M/W Pattern Detection (Mathematical Proxy) ---
    # Use scipy to find significant swing highs (M-tops) and lows (W-bottoms)
    # The 'distance' parameter is crucial for filtering out minor noise
    peak_indices, _ = find_peaks(df['High'], distance=params.get('peak_distance', 10))
    trough_indices, _ = find_peaks(-df['Low'], distance=params.get('peak_distance', 10))

    df['is_peak'] = False
    if len(peak_indices) > 0:
        df.iloc[peak_indices, df.columns.get_loc('is_peak')] = True

    df['is_trough'] = False
    if len(trough_indices) > 0:
        df.iloc[trough_indices, df.columns.get_loc('is_trough')] = True

    return df.dropna()

class Ema200ReversalHoldTheMayo(Strategy):
    # --- Optimizable Parameters ---
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.5
    ema_proximity_pct = 0.005  # Price must be within 0.5% of the 200 EMA
    peak_distance = 10

    def init(self):
        # --- Pre-calculated Indicators ---
        self.htf_uptrend = self.I(lambda: self.data.htf_uptrend, name="htf_uptrend")
        self.ema_200 = self.I(lambda: self.data.ema_200, name="ema_200")
        self.atr = self.I(lambda: self.data.atr, name="atr")
        self.volume_ma = self.I(lambda: self.data.volume_ma, name="volume_ma")
        self.cipher_buy = self.I(lambda: self.data.cipher_buy, name="cipher_buy")
        self.cipher_sell = self.I(lambda: self.data.cipher_sell, name="cipher_sell")
        self.is_peak = self.I(lambda: self.data.is_peak, name="is_peak")
        self.is_trough = self.I(lambda: self.data.is_trough, name="is_trough")

    def next(self):
        # --- Parameter Shortcuts ---
        price = self.data.Close[-1]

        # --- Mandatory Filters ---
        # 1. Volume Confirmation
        if self.data.Volume[-1] < self.volume_ma[-1]:
            return

        # 2. Check if already in a position
        if self.position:
            return

        # --- Long Entry (W-Pattern Bounce) ---
        # 1. HTF trend must be up
        if self.htf_uptrend[-1]:
            # 2. A trough (W-bottom) must be detected on the current bar
            if self.is_trough[-1]:
                # 3. The low of the trough candle must be close to the 200 EMA
                is_near_ema = abs(self.data.Low[-1] - self.ema_200[-1]) / self.ema_200[-1] < self.ema_proximity_pct
                if is_near_ema:
                    # 4. Cipher B must confirm a buy signal
                    if self.cipher_buy[-1]:
                        sl = price - (self.atr[-1] * self.atr_sl_multiplier)
                        tp = price + (self.atr[-1] * self.atr_tp_multiplier)
                        self.buy(sl=sl, tp=tp)

        # --- Short Entry (M-Pattern Rejection) ---
        # 1. HTF trend must be down
        if not self.htf_uptrend[-1]:
            # 2. A peak (M-top) must be detected on the current bar
            if self.is_peak[-1]:
                # 3. The high of the peak candle must be close to the 200 EMA
                is_near_ema = abs(self.data.High[-1] - self.ema_200[-1]) / self.ema_200[-1] < self.ema_proximity_pct
                if is_near_ema:
                    # 4. Cipher B must confirm a sell signal
                    if self.cipher_sell[-1]:
                        sl = price + (self.atr[-1] * self.atr_sl_multiplier)
                        tp = price - (self.atr[-1] * self.atr_tp_multiplier)
                        self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    from backtesting import Backtest
    import json

    # --- Data Loading and Preparation ---
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found. Please ensure the data file is in the correct directory.")
        sys.exit(1)

    # --- Preprocessing ---
    # Pass strategy-specific parameters to the preprocessing function
    data = preprocess_data(df, peak_distance=Ema200ReversalHoldTheMayo.peak_distance)

    # --- Backtesting ---
    bt = Backtest(data, Ema200ReversalHoldTheMayo, cash=100_000, commission=.002)
    stats = bt.run()

    print(stats)

    # --- Results and Output ---
    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)

    # Save statistics to a JSON file
    stats_dict = dict(stats)
    # The _strategy object is not JSON serializable, so we remove it.
    if '_strategy' in stats_dict:
        del stats_dict['_strategy']

    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)

    # Save the plot to an HTML file
    bt.plot(filename='results/ema_200_reversal_hold_the_mayo.html', open_browser=False)

    print("\nBacktest complete.")
    print(f"Stats saved to results/temp_result.json")
    print(f"Plot saved to results/ema_200_reversal_hold_the_mayo.html")
