
from backtesting import Strategy
import pandas as pd
import pandas_ta as ta
import numpy as np
import sys
import os

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df: pd.DataFrame, **params):
    """
    Adds all indicators to the dataframe.
    """
    # Use a copy to avoid modifying the original DataFrame
    df = df.copy()

    # VuManchu Cipher B for Stochastic RSI
    df = cipher_b(df)

    # MACD
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    # Rename MACD columns for clarity
    df.rename(columns={'MACD_12_26_9': 'macd', 'MACDh_12_26_9': 'macdh', 'MACDs_12_26_9': 'macds'}, inplace=True)


    # ATR for risk management
    df.ta.atr(length=14, append=True)
    df.rename(columns={'ATRr_14': 'atr'}, inplace=True)

    # Volume MA
    df['volume_ma'] = df['Volume'].rolling(20).mean()

    # Higher Timeframe Trend (4H)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).copy()
    df_4h['ema_200'] = ta.ema(df_4h['Close'], length=200)
    df_4h['htf_uptrend'] = (df_4h['Close'] > df_4h['ema_200']).astype(int)

    # Forward fill the trend signal to the original timeframe
    df['htf_uptrend'] = df_4h['htf_uptrend'].reindex(df.index, method='ffill')
    df['htf_uptrend'].fillna(0, inplace=True) # Fill initial NaNs


    # Support Zone
    # A simple approach using rolling min
    rolling_window = params.get('rolling_window', 50)
    df['support'] = df['Low'].rolling(rolling_window, min_periods=10).min().shift(1)

    return df


class SupportResistanceReversal(Strategy):
    # Optimizable parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    support_proximity_pct = 0.01  # 1% proximity to support
    rolling_window = 50

    def init(self):
        # State management for delayed entry
        self.reversal_candle_index = None

        # Indicators
        self.stoch_k = self.I(lambda: self.data.stoch_rsi_k, name='stoch_k')
        self.stoch_d = self.I(lambda: self.data.stoch_rsi_d, name='stoch_d')
        self.macd = self.I(lambda: self.data.macd, name='macd')
        self.macds = self.I(lambda: self.data.macds, name='macds')
        self.macdh = self.I(lambda: self.data.macdh, name='macdh')
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')
        self.htf_uptrend = self.I(lambda: self.data.htf_uptrend, name='htf_uptrend')
        self.support = self.I(lambda: self.data.support, name='support')

    def next(self):
        # Wait for enough data to avoid lookahead bias and NaN errors
        if len(self.data) < self.rolling_window or pd.isna(self.support[-1]) or pd.isna(self.volume_ma[-1]):
            return

        price = self.data.Close[-1]

        # --- Delayed Entry Trigger ---
        if self.reversal_candle_index is not None:
            # Check if it's time to enter (2 bars after setup)
            if len(self.data) - 1 == self.reversal_candle_index + 2:
                sl = price - (self.atr[-1] * self.atr_sl_multiplier)
                tp = price + (self.atr[-1] * self.atr_tp_multiplier)
                self.buy(sl=sl, tp=tp)
                self.reversal_candle_index = None # Reset after entry
            # Invalidate setup if price moves too far away before entry
            elif price > self.support[-1] * (1 + self.support_proximity_pct * 2):
                 self.reversal_candle_index = None
            return # Don't check for new setups while waiting for entry

        # Exit logic is handled by SL/TP orders placed at entry
        if self.position:
            return

        # --- Entry Conditions ---

        # 1. Higher Timeframe Filter
        if not self.htf_uptrend[-1]:
            return

        # 2. Price is near support
        is_near_support = price <= self.support[-1] * (1 + self.support_proximity_pct)
        if not is_near_support:
            return

        # 3. Reversal Candlestick (Bullish Engulfing)
        # A simple check for a bullish engulfing candle
        prev_open = self.data.Open[-2]
        prev_close = self.data.Close[-2]
        curr_open = self.data.Open[-1]
        curr_close = self.data.Close[-1]
        is_bullish_engulfing = (prev_close < prev_open and # Previous candle is bearish
                                curr_close > curr_open and # Current candle is bullish
                                curr_close > prev_open and
                                curr_open < prev_close)
        if not is_bullish_engulfing:
            return

        # 4. Stochastic RSI is oversold and crossed
        stoch_k = self.stoch_k[-1]
        stoch_d = self.stoch_d[-1]
        is_stoch_oversold_cross = stoch_k > stoch_d and stoch_k < 30 and stoch_d < 30
        if not is_stoch_oversold_cross:
            return

        # 5. MACD shows potential for a cross (histogram is increasing)
        is_macd_turning = self.macdh[-1] > self.macdh[-2]
        if not is_macd_turning:
            return

        # 6. Volume confirmation (Guideline: Volume > MA)
        is_volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1]
        if not is_volume_confirmed:
            return

        # --- Set up Delayed Entry ---
        if (is_near_support and is_bullish_engulfing and is_stoch_oversold_cross and
            is_macd_turning and is_volume_confirmed and self.reversal_candle_index is None):

            self.reversal_candle_index = len(self.data) - 1

if __name__ == '__main__':
    from backtesting import Backtest

    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
        # Standardize column names: strip spaces and capitalize
        df.columns = [col.strip().capitalize() for col in df.columns]
    except FileNotFoundError:
        print("Data file not found. Please ensure 'data/BTC-USD-15m.csv' exists.")
        # As a fallback, create some synthetic data
        data_range = pd.date_range('2023-01-01', periods=2000, freq='15min')
        df = pd.DataFrame(np.random.randn(2000, 5),
                          columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                          index=data_range)
        df['Open'] = 20000 + df['Open'].cumsum()
        df['High'] = df['Open'] + abs(np.random.randn(2000))
        df['Low'] = df['Open'] - abs(np.random.randn(2000))
        df['Close'] = df['Open'] + np.random.randn(2000)
        df['Volume'] = np.random.randint(100, 1000, size=2000)


    processed_df = preprocess_data(df)

    bt = Backtest(processed_df, SupportResistanceReversal, cash=100000, commission=.002)
    stats = bt.run()
    print(stats)
    bt.plot(filename='results/strategy_f65cd837a7a0.html', open_browser=False)

    # Save results to temp_result.json
    import json
    results_dict = dict(stats)

    # Remove non-serializable items
    results_dict.pop('_strategy', None)
    results_dict.pop('_equity_curve', None)
    results_dict.pop('_trades', None)

    # Sanitize the results for JSON serialization
    for key, value in results_dict.items():
        if isinstance(value, pd.Timestamp):
            results_dict[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            results_dict[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            results_dict[key] = value.item()
        elif pd.isna(value):
            results_dict[key] = None

    with open('results/temp_result.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

    print("Backtest finished and results saved.")
