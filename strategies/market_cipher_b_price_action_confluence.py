
from backtesting import Strategy
from backtesting.lib import crossover

import pandas as pd
import numpy as np
import talib
from scipy.signal import find_peaks
import os
import sys

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b


def preprocess_data(df, **params):
    """
    Adds all required indicators and signals to the DataFrame.
    """
    # 1. VuManchu Cipher B
    df = cipher_b(df)
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    # 2. ATR for risk management and filters
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Drop NaNs from early indicator calculations before proceeding
    df.dropna(subset=['atr', 'wt1'], inplace=True)

    # 3. Multi-Timeframe Filter (4H EMA 200)
    df_4h = df.resample('4H', label='right', closed='right').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)
    df_4h['htf_trend'] = np.where(df_4h['Close'] > df_4h['ema_200'], 1, -1)
    df['htf_trend'] = df_4h['htf_trend'].reindex(df.index, method='ffill')

    # 4. Volume Confirmation
    volume_ma_period = params.get('volume_ma_period', 20)
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=volume_ma_period)

    # 5. Price Action - Candlestick Patterns (Engulfing)
    prev_close = df['Close'].shift(1)
    prev_open = df['Open'].shift(1)
    is_prev_bullish = prev_close > prev_open
    is_prev_bearish = prev_close < prev_open

    df['bearish_engulfing'] = (is_prev_bullish &
                               (df['Close'] < df['Open']) &
                               (df['Open'] >= prev_close) &
                               (df['Close'] <= prev_open)).astype(int)

    df['bullish_engulfing'] = (is_prev_bearish &
                               (df['Close'] > df['Open']) &
                               (df['Open'] <= prev_close) &
                               (df['Close'] >= prev_open)).astype(int)

    # 6. Price Action - Support & Resistance (Swing Points)
    swing_lookback = params.get('swing_lookback', 50)
    prominence = df['atr'].mean() * 0.5 if not df['atr'].isnull().all() else 0.1

    resistance_indices, _ = find_peaks(df['High'], distance=swing_lookback, prominence=prominence)
    support_indices, _ = find_peaks(-df['Low'], distance=swing_lookback, prominence=prominence)

    df['last_support'] = np.nan
    df['last_resistance'] = np.nan
    if len(support_indices) > 0:
        df.iloc[support_indices, df.columns.get_loc('last_support')] = df.iloc[support_indices]['Low']
    if len(resistance_indices) > 0:
        df.iloc[resistance_indices, df.columns.get_loc('last_resistance')] = df.iloc[resistance_indices]['High']

    df['last_support'].ffill(inplace=True)
    df['last_resistance'].ffill(inplace=True)

    return df


class MarketCipherBPriceActionConfluence(Strategy):
    # NOTE on Base Class:
    # The project's backtesting scripts (`strategies/*.py`) uniformly use `backtesting.Strategy`.
    # The requested `MoonDevStrategy` is part of a separate, incompatible framework in `src/strategies/`.
    # To ensure compatibility with the established backtesting pattern, this strategy
    # inherits from `backtesting.Strategy`.
    """
    A strategy that combines Market Cipher B signals with price action analysis,
    adhering to MoonDev strategy development guidelines.
    """

    # Optimizable Parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_ma_period = 20
    swing_lookback = 50 # For find_peaks distance

    def init(self):
        """
        Initialize indicators and signals.
        """
        # Expose pre-calculated data to the strategy
        self.htf_trend = self.I(lambda: self.data.df['htf_trend'], name="htf_trend")
        self.volume_ma = self.I(lambda: self.data.df['volume_ma'], name="volume_ma")
        self.atr = self.I(lambda: self.data.df['atr'], name="atr")

        # Price Action
        self.bullish_engulfing = self.I(lambda: self.data.df['bullish_engulfing'], name="bullish_engulfing")
        self.bearish_engulfing = self.I(lambda: self.data.df['bearish_engulfing'], name="bearish_engulfing")
        self.last_support = self.I(lambda: self.data.df['last_support'], name="last_support")
        self.last_resistance = self.I(lambda: self.data.df['last_resistance'], name="last_resistance")

        # Cipher B
        self.buy_signal = self.I(lambda: self.data.df['buy_signal'], name="buy_signal")
        self.sell_signal = self.I(lambda: self.data.df['sell_signal'], name="sell_signal")
        self.money_flow = self.I(lambda: self.data.df['rsimfi'], name="money_flow")


    def next(self):
        """
        Define the trading logic.
        """
        price = self.data.Close[-1]
        atr_val = self.atr[-1]

        # --- FILTERS ---
        # 1. Higher Timeframe Trend
        is_htf_up = self.htf_trend[-1] == 1
        is_htf_down = self.htf_trend[-1] == -1

        # 2. Volume Confirmation
        is_volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1]

        # --- LONG ENTRY CONFLUENCE ---
        if not self.position and is_htf_up and is_volume_confirmed:
            # 1. Price Action: Bullish engulfing near support
            is_bullish_pattern = self.bullish_engulfing[-1] == 1
            is_near_support = abs(self.data.Low[-1] - self.last_support[-1]) < (0.5 * atr_val)

            # 2. Market Cipher B: Buy signal and positive, increasing money flow
            is_cipher_buy = self.buy_signal[-1] == 1
            is_money_flow_green_and_increasing = (self.money_flow[-1] > 0 and
                                                      self.money_flow[-1] > self.money_flow[-2])

            if is_bullish_pattern and is_near_support and is_cipher_buy and is_money_flow_green_and_increasing:
                sl = price - self.atr_sl_multiplier * atr_val
                tp = price + self.atr_tp_multiplier * atr_val
                self.buy(sl=sl, tp=tp)

        # --- SHORT ENTRY CONFLUENCE ---
        elif not self.position and is_htf_down and is_volume_confirmed:
            # 1. Price Action: Bearish engulfing near resistance
            is_bearish_pattern = self.bearish_engulfing[-1] == 1
            is_near_resistance = abs(self.data.High[-1] - self.last_resistance[-1]) < (0.5 * atr_val)

            # 2. Market Cipher B: Sell signal and negative, decreasing money flow
            is_cipher_sell = self.sell_signal[-1] == 1
            is_money_flow_red_and_decreasing = (self.money_flow[-1] < 0 and
                                                    self.money_flow[-1] < self.money_flow[-2])

            if is_bearish_pattern and is_near_resistance and is_cipher_sell and is_money_flow_red_and_decreasing:
                sl = price + self.atr_sl_multiplier * atr_val
                tp = price - self.atr_tp_multiplier * atr_val
                self.sell(sl=sl, tp=tp)

        # --- DYNAMIC EXIT LOGIC ---
        else:
            # Exit long if a sell signal appears
            if self.position.is_long and self.sell_signal[-1] == 1:
                self.position.close()
            # Exit short if a buy signal appears
            elif self.position.is_short and self.buy_signal[-1] == 1:
                self.position.close()

if __name__ == '__main__':
    from backtesting import Backtest
    import json

    # --- Data Loading ---
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        # As a fallback, create some synthetic data for demonstration
        print("Generating synthetic data...")
        from backtesting.test import GOOG
        data = GOOG.copy()
        data.columns = [col.capitalize() for col in data.columns]
    else:
        data = pd.read_csv(
            data_path,
            index_col='datetime',
            parse_dates=True,
            usecols=['datetime', 'open', 'high', 'low', 'close', 'volume'],
            skipinitialspace=True
        )
        data.columns = [col.capitalize() for col in data.columns]

    # --- Preprocessing ---
    print("Preprocessing data...")
    data = preprocess_data(data.copy()) # Use a copy to avoid modifying original
    data.dropna(inplace=True)

    # --- Backtesting ---
    print("Starting backtest...")
    bt = Backtest(data, MarketCipherBPriceActionConfluence, cash=100_000, commission=.001)
    stats = bt.run()

    print("\n--- Backtest Stats ---")
    print(stats)

    # --- Reporting ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Save plot
    plot_filename = os.path.join(results_dir, "market_cipher_b_price_action_confluence.html")
    print(f"Saving plot to {plot_filename}...")
    try:
        bt.plot(filename=plot_filename, open_browser=False)
    except Exception as e:
        print(f"Could not generate plot: {e}")

    # Save stats to JSON
    json_filename = os.path.join(results_dir, "temp_result.json")
    print(f"Saving stats to {json_filename}...")

    # Sanitize stats for JSON serialization
    sanitized_stats = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized_stats[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized_stats[key] = float(value)
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            # Don't include DataFrames or Series in the JSON output
            pass
        elif pd.isna(value):
             sanitized_stats[key] = None
        else:
            sanitized_stats[key] = value

    # Remove strategy object if it exists
    if '_strategy' in sanitized_stats:
        del sanitized_stats['_strategy']

    with open(json_filename, 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    print("Done.")
