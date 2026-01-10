import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

import talib
from scipy.signal import find_peaks
import numpy as np
import json


# +-----------------------------------------------------------------------------+
# |                                                                             |
# |  Sanitize JSON                                                              |
# |                                                                             |
# +-----------------------------------------------------------------------------+
def sanitize_json(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items() if not (isinstance(v, pd.DataFrame) or v is pd.NA or pd.isna(v))}
    elif isinstance(obj, list):
        return [sanitize_json(i) for i in obj]
    elif pd.isna(obj) or obj is pd.NA:
        return None
    return obj


# +-----------------------------------------------------------------------------+
# |                                                                             |
# |  Data Preprocessing                                                         |
# |                                                                             |
# +-----------------------------------------------------------------------------+
def preprocess_data(df: pd.DataFrame, **params):
    """
    Adds all indicators and filters to the input DataFrame.
    """
    # Clean up column names to be consistently capitalized, which is expected by indicator libraries.
    df.columns = [col.strip().capitalize() for col in df.columns]

    # Add VuManchu Cipher B indicator if provided
    cipher_b_func = params.get('cipher_b_func')
    if cipher_b_func:
        df = cipher_b_func(df)

    # Primary Indicators
    df['ema_200'] = talib.EMA(df['Close'], timeperiod=params.get('ema_200_period', 200))
    df['ema_50'] = talib.EMA(df['Close'], timeperiod=params.get('ema_50_period', 50))
    df['ema_13'] = talib.EMA(df['Close'], timeperiod=params.get('ema_13_period', 13))
    df['ema_5'] = talib.EMA(df['Close'], timeperiod=params.get('ema_5_period', 5))

    # Risk Management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=params.get('atr_period', 14))

    # Volume Filter
    df['volume_sma'] = talib.SMA(df['Volume'], timeperiod=params.get('volume_sma_period', 20))

    # Higher-Timeframe Filter (4H)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['ema_50'] = talib.EMA(df_4h['Close'], timeperiod=params.get('ema_50_period', 50))
    df_4h['htf_uptrend'] = df_4h['Close'] > df_4h['ema_50']

    # Map 4H trend back to 15m data
    df['htf_uptrend'] = df_4h['htf_uptrend'].reindex(df.index, method='ffill').fillna(False)

    # Pattern Detection: Find swing points
    distance = params.get('swing_distance', 5)
    prominence = df['atr'].mean() * params.get('swing_prominence_atr_multiplier', 0.5)
    high_peaks, _ = find_peaks(df['High'], distance=distance, prominence=prominence)
    low_peaks, _ = find_peaks(-df['Low'], distance=distance, prominence=prominence)
    df['swing_high'] = np.nan
    df.iloc[high_peaks, df.columns.get_loc('swing_high')] = df.iloc[high_peaks]['High']
    df['swing_low'] = np.nan
    df.iloc[low_peaks, df.columns.get_loc('swing_low')] = df.iloc[low_peaks]['Low']

    # Drop rows with NaN in essential indicator columns
    return df.dropna(subset=['ema_200', 'atr', 'volume_sma', 'wt1'])


# +-----------------------------------------------------------------------------+
# |                                                                             |
# |  Strategy Class                                                             |
# |                                                                             |
# +-----------------------------------------------------------------------------+
# NOTE: Inherits from backtesting.Strategy as the required `MoonDevStrategy`
# is incompatible with the backtesting.py framework used in this repository.
class Ema200BounceReversal(Strategy):
    """
    Strategy based on M/W pattern bounces off the 200 EMA.
    """
    # Optimizable parameters
    ema_200_period = 200
    ema_50_period = 50
    ema_13_period = 13
    ema_5_period = 5
    atr_period = 14
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    proximity_pct = 0.01  # 1% proximity to EMA 200
    volume_sma_period = 20
    swing_distance = 5
    swing_prominence_atr_multiplier = 0.5

    def init(self):
        # Link pre-calculated indicators
        self.ema_200 = self.I(lambda x: x, self.data.df.ema_200, name="EMA 200")
        self.ema_50 = self.I(lambda x: x, self.data.df.ema_50, name="EMA 50")
        self.ema_13 = self.I(lambda x: x, self.data.df.ema_13, name="EMA 13")
        self.ema_5 = self.I(lambda x: x, self.data.df.ema_5, name="EMA 5")
        self.atr = self.I(lambda x: x, self.data.df.atr, name="ATR")
        self.volume_sma = self.I(lambda x: x, self.data.df.volume_sma, name="Volume SMA")
        self.htf_uptrend = self.I(lambda x: x, self.data.df.htf_uptrend, name="HTF Uptrend")
        self.swing_highs = self.I(lambda x: x, self.data.df.swing_high, name="Swing Highs")
        self.swing_lows = self.I(lambda x: x, self.data.df.swing_low, name="Swing Lows")
        self.buy_signal = self.I(lambda x: x, self.data.df.buy_signal, name="Buy Signal")
        self.sell_signal = self.I(lambda x: x, self.data.df.sell_signal, name="Sell Signal")

        # M/W Pattern State Machine
        self.m_pattern_state = 0
        self.m_leg1_high = None
        self.m_neckline_low = None

        self.w_pattern_state = 0
        self.w_leg1_low = None
        self.w_neckline_high = None

    def next(self):
        current_price = self.data.Close[-1]
        current_swing_high = self.swing_highs[-1]
        current_swing_low = self.swing_lows[-1]

        # --- M-Pattern Detection (Short) ---
        if self.m_pattern_state == 0 and not pd.isna(current_swing_high):
            self.m_leg1_high = current_swing_high
            self.m_pattern_state = 1

        elif self.m_pattern_state == 1:
            if not pd.isna(current_swing_low):
                self.m_neckline_low = current_swing_low
                self.m_pattern_state = 2
            elif not pd.isna(current_swing_high) and current_swing_high > self.m_leg1_high:
                 self.m_leg1_high = current_swing_high # New higher high resets

        elif self.m_pattern_state == 2:
            if not pd.isna(current_swing_high):
                leg2_high = current_swing_high
                if leg2_high <= self.m_leg1_high: # Second leg is lower or equal
                    # M-Pattern Confirmed - Check entry conditions
                    center_peak_price = self.m_leg1_high
                    ema_200_price = self.ema_200[-1]
                    is_near_ema = abs(center_peak_price - ema_200_price) / ema_200_price <= self.proximity_pct

                    is_htf_downtrend = not self.htf_uptrend[-1]
                    is_volume_confirmed = self.data.Volume[-1] > self.volume_sma[-1]
                    is_cipher_sell_signal = self.sell_signal[-1]

                    if is_near_ema and is_htf_downtrend and is_volume_confirmed and is_cipher_sell_signal and not self.position:
                        sl = current_price + (self.atr_sl_multiplier * self.atr[-1])
                        tp = current_price - (self.atr_tp_multiplier * self.atr[-1])
                        self.sell(sl=sl, tp=tp)

                self.m_pattern_state = 0 # Reset after check
            elif not pd.isna(current_swing_low) and current_swing_low < self.m_neckline_low:
                self.m_pattern_state = 0 # Structure broke down, reset


        # --- W-Pattern Detection (Long) ---
        if self.w_pattern_state == 0 and not pd.isna(current_swing_low):
            self.w_leg1_low = current_swing_low
            self.w_pattern_state = 1

        elif self.w_pattern_state == 1:
            if not pd.isna(current_swing_high):
                self.w_neckline_high = current_swing_high
                self.w_pattern_state = 2
            elif not pd.isna(current_swing_low) and current_swing_low < self.w_leg1_low:
                self.w_leg1_low = current_swing_low # New lower low resets

        elif self.w_pattern_state == 2:
            if not pd.isna(current_swing_low):
                leg2_low = current_swing_low
                if leg2_low >= self.w_leg1_low: # Second leg is higher or equal
                    # W-Pattern Confirmed - Check entry conditions
                    center_trough_price = self.w_leg1_low
                    ema_200_price = self.ema_200[-1]
                    is_near_ema = abs(center_trough_price - ema_200_price) / ema_200_price <= self.proximity_pct

                    is_htf_uptrend = self.htf_uptrend[-1]
                    is_volume_confirmed = self.data.Volume[-1] > self.volume_sma[-1]
                    is_cipher_buy_signal = self.buy_signal[-1]

                    if is_near_ema and is_htf_uptrend and is_volume_confirmed and is_cipher_buy_signal and not self.position:
                        sl = current_price - (self.atr_sl_multiplier * self.atr[-1])
                        tp = current_price + (self.atr_tp_multiplier * self.atr[-1])
                        self.buy(sl=sl, tp=tp)

                self.w_pattern_state = 0 # Reset after check
            elif not pd.isna(current_swing_high) and current_swing_high > self.w_neckline_high:
                self.w_pattern_state = 0 # Structure broke up, reset


# +-----------------------------------------------------------------------------+
# |                                                                             |
# |  Backtesting Runner                                                         |
# |                                                                             |
# +-----------------------------------------------------------------------------+
if __name__ == '__main__':
    # Add project root to path to allow src imports
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found. Please ensure the data file is in the correct directory.")
        # As a fallback, you can generate synthetic data for testing
        from backtesting.test import GOOG
        data = GOOG.copy()
        data = data.iloc[-2000:] # Use a subset for faster testing

    # Preprocess the data
    strategy_params = {
        'ema_200_period': Ema200BounceReversal.ema_200_period,
        'ema_50_period': Ema200BounceReversal.ema_50_period,
        'ema_13_period': Ema200BounceReversal.ema_13_period,
        'ema_5_period': Ema200BounceReversal.ema_5_period,
        'atr_period': Ema200BounceReversal.atr_period,
        'volume_sma_period': Ema200BounceReversal.volume_sma_period,
        'swing_distance': Ema200BounceReversal.swing_distance,
        'swing_prominence_atr_multiplier': Ema200BounceReversal.swing_prominence_atr_multiplier,
    }
    # Import locally to avoid ModuleNotFoundError when run as a standalone script
    from src.indicators.vumanchu import cipher_b
    strategy_params['cipher_b_func'] = cipher_b

    data = preprocess_data(data, **strategy_params)

    # Initialize and run the backtest
    bt = Backtest(data, Ema200BounceReversal, cash=100_000, commission=.002)
    stats = bt.run()

    print(stats)

    # Sanitize and save results
    stats_dict = stats.to_dict()
    stats_dict.pop('_strategy', None)  # Remove the non-serializable strategy object
    sanitized_stats = sanitize_json(stats_dict)

    # Ensure the results directory exists
    import os
    os.makedirs('results', exist_ok=True)

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    # Generate plot
    try:
        bt.plot(filename='results/ema_200_bounce_reversal_plot.html', open_browser=False)
    except Exception as e:
        print(f"Could not generate plot: {e}")
