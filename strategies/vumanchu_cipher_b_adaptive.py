"""
VuManchu Cipher B Adaptive Strategy
===================================
A backtesting strategy using the VuManchu Cipher B indicator, adapted to be
compliant with the MoonDev strategy development guidelines.

This strategy includes:
- ATR-based risk management (stop loss and take profit).
- A higher-timeframe (4H) trend filter.
- Volume confirmation for entries.
"""

import pandas as pd
import numpy as np
import talib
from backtesting import Strategy, Backtest
import os
import sys
import json


# Add parent directory to path for custom imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df, **params):
    """
    Applies all necessary indicators and filters to the dataframe.
    """
    df = df.copy()

    # 1. Add VuManchu Cipher B indicators
    df = cipher_b(df)
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    # 2. Add ATR for risk management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # 3. Add higher-timeframe (4H) trend filter
    # Note: Using 'h' for frequency is the modern pandas syntax
    df_4h = df.resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)

    # Forward-fill the trend status. The initial NaNs are expected and handled by the strategy logic.
    df['htf_uptrend_signal'] = (df_4h['Close'] > df_4h['ema_200']).reindex(df.index, method='ffill')

    # 4. Add volume confirmation filter
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=20)

    return df


class VuManchuCipherBAdaptive(Strategy):
    """
    VuManchu Cipher B Trading Strategy with adaptive risk management and filters.

    Entry (Long):
    - 4H trend is up (Close > 4H EMA200)
    - Volume is above its 20-period moving average
    - Cipher B buy signal is present
    - Money Flow (rsimfi) is positive

    Entry (Short):
    - 4H trend is down (Close < 4H EMA200)
    - Volume is above its 20-period moving average
    - Cipher B sell signal is present
    - Money Flow (rsimfi) is negative

    Exit:
    - Stop loss or take profit determined by ATR multiples.
    """

    # Optimizable parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 4.0

    def init(self):
        # Indicators are pre-calculated in preprocess_data.
        # Here, we create references to them for easier access.
        # The `self.I` function converts pandas Series to numpy arrays for speed.
        self.buy_sig = self.I(self.data.df.buy_signal)
        self.sell_sig = self.I(self.data.df.sell_signal)
        self.mf = self.I(self.data.df.rsimfi)
        self.atr = self.I(self.data.df.atr)
        self.htf_uptrend = self.I(self.data.df.htf_uptrend_signal)
        self.volume_ma = self.I(self.data.df.volume_ma)

    def next(self):
        """
        Main trading logic executed on each bar.
        """
        # --- Warmup Guard ---
        # Ensure all indicators have valid data before proceeding.
        # This is crucial because of the long lookback period of the 4H EMA.
        if pd.isna(self.htf_uptrend[-1]) or pd.isna(self.atr[-1]) or pd.isna(self.volume_ma[-1]):
            return

        current_price = self.data.Close[-1]

        # --- Pre-trade Filters ---
        # 1. Volume Confirmation: Only consider entries if volume is above average.
        if self.data.Volume[-1] < self.volume_ma[-1]:
            return

        # --- Entry Logic ---
        # Only check for entries if we don't already have a position.
        if not self.position:
            is_htf_up = self.htf_uptrend[-1] > 0  # Convert the signal (1.0 or 0.0) to a boolean

            # 2. Long Entry Conditions:
            #    - Higher timeframe trend is bullish.
            #    - Cipher B issues a buy signal.
            #    - Money Flow is positive, confirming buying pressure.
            if is_htf_up and self.buy_sig[-1] == 1 and self.mf[-1] > 0:
                sl = current_price - (self.atr_sl_multiplier * self.atr[-1])
                tp = current_price + (self.atr_tp_multiplier * self.atr[-1])
                self.buy(sl=sl, tp=tp)

            # 3. Short Entry Conditions:
            #    - Higher timeframe trend is bearish.
            #    - Cipher B issues a sell signal.
            #    - Money Flow is negative, confirming selling pressure.
            elif not is_htf_up and self.sell_sig[-1] == 1 and self.mf[-1] < 0:
                sl = current_price + (self.atr_sl_multiplier * self.atr[-1])
                tp = current_price - (self.atr_tp_multiplier * self.atr[-1])
                self.sell(sl=sl, tp=tp)


# --- Runnable Main Block ---
if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    # --- Data Loading and Preparation ---
    try:
        df_raw = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Sanitize column names (e.g., " open" -> "Open")
        df_raw.columns = [col.strip().capitalize() for col in df_raw.columns]
        data_loaded_successfully = True
    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_path}'.")
        print("Please ensure the file exists and the path is correct.")
        data_loaded_successfully = False

    # --- Backtest Execution ---
    if data_loaded_successfully:
        print("Preprocessing data...")
        df_processed = preprocess_data(df_raw)

        if df_processed.empty:
            print("DataFrame is empty after preprocessing. Cannot run backtest.")
        else:
            print("Running backtest...")
            bt = Backtest(df_processed, VuManchuCipherBAdaptive, cash=100_000, commission=.001)
            stats = bt.run()

            print("\n=== VuManchu Cipher B Adaptive Strategy Results ===")
            print(stats)

            # --- Results and Output ---
            # Ensure the 'results' directory exists
            os.makedirs('results', exist_ok=True)

            # Save the plot
            plot_filename = 'results/vumanchu_cipher_b_adaptive.html'
            bt.plot(filename=plot_filename, open_browser=False)
            print(f"\nPlot saved to '{plot_filename}'")

            # Save the stats to a JSON file
            stats_dict = stats.to_dict()

            # Sanitize stats for JSON serialization
            stats_dict['_strategy'] = str(stats_dict['_strategy'])
            for key, value in stats_dict.items():
                if isinstance(value, pd.DataFrame):
                    stats_dict[key] = None
                elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                    stats_dict[key] = str(value)
                elif isinstance(value, np.integer):
                    stats_dict[key] = int(value)
                elif isinstance(value, np.floating):
                    stats_dict[key] = float(value)
                elif pd.isna(value):
                    stats_dict[key] = None

            stats_dict_cleaned = {k: v for k, v in stats_dict.items() if v is not None}

            json_filename = 'results/temp_result.json'
            with open(json_filename, 'w') as f:
                json.dump(stats_dict_cleaned, f, indent=4)
            print(f"Stats saved to '{json_filename}'")
