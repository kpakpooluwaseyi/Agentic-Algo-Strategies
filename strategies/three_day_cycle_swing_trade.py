
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import talib
import backtesting
from backtesting import Backtest, Strategy
from src.indicators.vumanchu import cipher_b
import os
import json

def sanitize_stats(stats):
    """
    Sanitizes the stats object from a backtest run to make it JSON serializable.
    Removes non-serializable types like DataFrame, Timestamps, etc.
    """
    sanitized = {}
    for key, value in stats.items():
        # Handle non-serializable objects first
        if isinstance(value, (pd.DataFrame, pd.Series)):
            continue
        if isinstance(value, backtesting.Strategy):
            sanitized[key] = value.name if hasattr(value, 'name') else str(value)
            continue

        # Handle Timestamps and Timedeltas
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        # Handle NaN/NA values gracefully before numeric checks
        elif pd.isna(value):
            sanitized[key] = None
        # Handle numpy numeric types
        elif isinstance(value, (np.integer, np.int64)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            sanitized[key] = float(value)
        # Handle standard Python types that are already serializable
        elif isinstance(value, (str, int, float, bool, type(None))):
            sanitized[key] = value
        else:
            # Fallback for any other types, convert to string
            sanitized[key] = str(value)
    return sanitized

def preprocess_data(df, **params):
    """
    Adds all indicators required for the strategy.
    - VuManchu Cipher B for entry signals
    - Higher timeframe (4H) trend filter
    - Volume moving average for confirmation
    - ATR for risk management
    """
    # First, sanitize column names to prevent KeyErrors
    df.columns = [col.strip().title() for col in df.columns]

    # Ensure 'Datetime' column is used as the index
    if 'Datetime' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.set_index('Datetime')

    # 1. Add VuManchu Cipher B Indicator
    df = cipher_b(df)

    # 2. Add Higher Timeframe (4H) Trend Filter
    df_4h = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)
    df_4h['htf_trend_up'] = df_4h['Close'] > df_4h['ema_200']

    # Map 4H trend to original timeframe and forward-fill
    df['htf_trend_up'] = df_4h['htf_trend_up'].reindex(df.index, method='ffill')
    df['htf_trend_up'] = df['htf_trend_up'].fillna(False) # Fill initial NaNs

    # 3. Add Volume Confirmation
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=20)

    # 4. Add ATR for Risk Management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Drop rows where essential indicators are NaN, especially ATR
    df.dropna(subset=['atr'], inplace=True)

    if df.empty:
        raise ValueError("DataFrame is empty after preprocessing and dropping NaNs. "
                         "Check indicator lookback periods and data length.")

    return df


class ThreeDayCycleSwingTrade(Strategy):
    """
    Strategy based on the 3-Day Market Maker Cycle concept.
    This implementation uses VuManchu Cipher B's signals as a proxy for
    W/M formations at peak cycle levels. It adheres strictly to the
    quantitative rules provided (ATR risk management, HTF filter, etc.).

    NOTE on Simplifications: The original strategy concept included highly
    discretionary elements like Forex session-based logic (e.g., "London open
    stop hunt") and scaling into positions. These have been intentionally omitted
    to create a purely quantitative, non-discretionary, and backtestable
    algorithm that complies with the repository's development guidelines.

    NOTE on Inheritance: The request specified inheriting from `MoonDevStrategy`.
    However, that base class is incompatible with the `backtesting.py` framework
    and its `init()`/`next()` methods. To produce a runnable, verifiable
    backtest as requested, inheriting from `backtesting.Strategy` is necessary.
    """
    # Optimizable parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    breakeven_threshold_pct = 0.5 # Percentage of initial TP target to trigger BE

    def init(self):
        # State variables for trade management
        self.trade_level = 0  # 0: No trade, 1: Initial trade, 2: SL moved to BE
        self.initial_tp_price = None
        self.entry_atr = None

        # Initialize indicators calculated in preprocess_data
        self.buy_signal = self.I(lambda: self.data.buy_signal, name='buy_signal')
        self.sell_signal = self.I(lambda: self.data.sell_signal, name='sell_signal')
        self.htf_trend_up = self.I(lambda: self.data.htf_trend_up, name='htf_trend_up')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')
        self.atr = self.I(lambda: self.data.atr, name='atr')

    def next(self):
        price = self.data.Close[-1]

        # --- FILTERS (as per development guidelines) ---
        # 1. Higher Timeframe Trend Filter
        is_htf_uptrend = self.htf_trend_up[-1]
        is_htf_downtrend = not is_htf_uptrend

        # 2. Volume Confirmation Filter
        is_volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1]

        # --- STATE MACHINE & TRADE MANAGEMENT ---

        # If a trade is active, manage it based on its current level
        if self.position:
            entry_price = self.trades[0].entry_price

            # Level 1: Initial trade placed, waiting to move SL to Breakeven
            if self.trade_level == 1:
                if self.position.is_long:
                    # Calculate the price at which we move SL to BE
                    breakeven_trigger_price = entry_price + (self.initial_tp_price - entry_price) * self.breakeven_threshold_pct
                    if price >= breakeven_trigger_price:
                        self.trades[0].sl = entry_price  # Move SL to BE
                        self.trade_level = 2  # Advance to next level
                else: # Short position
                    breakeven_trigger_price = entry_price - (entry_price - self.initial_tp_price) * self.breakeven_threshold_pct
                    if price <= breakeven_trigger_price:
                        self.trades[0].sl = entry_price # Move SL to BE
                        self.trade_level = 2

            # Level 2: SL is at BE, waiting for an exit signal (end of cycle)
            elif self.trade_level == 2:
                if self.position.is_long and self.sell_signal[-1]:
                    self.position.close()
                    self.trade_level = 0
                elif self.position.is_short and self.buy_signal[-1]:
                    self.position.close()
                    self.trade_level = 0

        # --- ENTRY LOGIC ---
        # If no position is open and not in a trade cycle, look for a new entry
        elif self.trade_level == 0:
            current_atr = self.atr[-1]
            if pd.isna(current_atr) or current_atr == 0:
                return # Skip if ATR is not available

            # Long Entry
            if self.buy_signal[-1] and is_htf_uptrend and is_volume_confirmed:
                sl = price - (self.atr_sl_multiplier * current_atr)
                # Calculate the initial TP price, which now acts as the BE trigger
                tp_price = price + (self.atr_tp_multiplier * current_atr)

                self.buy(sl=sl) # TP is managed programmatically
                self.trade_level = 1
                self.initial_tp_price = tp_price
                self.entry_atr = current_atr

            # Short Entry
            elif self.sell_signal[-1] and is_htf_downtrend and is_volume_confirmed:
                sl = price + (self.atr_sl_multiplier * current_atr)
                tp_price = price - (self.atr_tp_multiplier * current_atr)

                self.sell(sl=sl)
                self.trade_level = 1
                self.initial_tp_price = tp_price
                self.entry_atr = current_atr

        # Reset state if position was closed by SL or manually
        if not self.position and self.trade_level != 0:
            self.trade_level = 0
            self.initial_tp_price = None
            self.entry_atr = None

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
    else:
        # Load data, letting the preprocess function handle column selection and cleaning
        data = pd.read_csv(data_path)

        # Preprocess data
        data = preprocess_data(data)

        # Initialize Backtest
        bt = Backtest(data, ThreeDayCycleSwingTrade, cash=100000, commission=.002, finalize_trades=True)

        # Run backtest
        stats = bt.run()

        # Print stats
        print(stats)

        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)

        # Save plot
        plot_filename = 'results/three_day_cycle_swing_trade.html'
        bt.plot(filename=plot_filename)
        print(f"Plot saved to {plot_filename}")

        # Save stats to JSON
        stats_filename = 'results/temp_result.json'
        # Sanitize stats before saving
        sanitized_stats = sanitize_stats(stats)
        # Add strategy name to stats
        sanitized_stats['strategy_name'] = 'ThreeDayCycleSwingTrade'
        with open(stats_filename, 'w') as f:
            json.dump(sanitized_stats, f, indent=4)
        print(f"Stats saved to {stats_filename}")
