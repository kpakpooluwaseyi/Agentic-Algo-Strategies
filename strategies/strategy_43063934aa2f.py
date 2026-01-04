
import json
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df: pd.DataFrame, htf_period=200) -> pd.DataFrame:
    """
    Adds required indicators to the DataFrame.
    """
    # Add VuManchu Cipher B indicators
    df = cipher_b(df)

    # Add ATR for risk management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Add multi-timeframe trend filter (4h EMA)
    df_4h = df.resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
    }).dropna()
    df_4h['ema'] = talib.EMA(df_4h['Close'], timeperiod=htf_period)
    df_4h['htf_trend_up'] = df_4h['Close'] > df_4h['ema']

    # Map the 4h trend back to the original timeframe and ensure it's a Series
    htf_map = df.index.floor('4h').map(df_4h['htf_trend_up'])
    htf_series = pd.Series(htf_map, index=df.index)
    df['htf_trend_up'] = htf_series.ffill()

    # Add volume moving average for confirmation
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=20)

    # Add ATR moving average for consolidation check
    df['atr_ma'] = talib.SMA(df['atr'], timeperiod=20)

    df.dropna(inplace=True)
    return df


class MarketCipherBStrategy(Strategy):
    # --- Strategy Parameters ---
    # 1 = Momentum Continuation, 2 = Consolidation Reversal
    strategy_mode = 1

    # --- Indicator Parameters ---
    wt_ob_level = 60
    wt_os_level = -60

    # --- Risk Management Parameters ---
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    def init(self):
        # Indicators from preprocess_data
        self.wt1 = self.I(lambda: self.data.wt1, name="wt1")
        self.wt2 = self.I(lambda: self.data.wt2, name="wt2")
        self.htf_trend_up = self.I(lambda: self.data.htf_trend_up, name="htf_trend_up")
        self.atr = self.I(lambda: self.data.atr, name="atr")
        self.volume_ma = self.I(lambda: self.data.volume_ma, name="volume_ma")
        self.atr_ma = self.I(lambda: self.data.atr_ma, name="atr_ma")
        self.buy_signal = self.I(lambda: self.data.buy_signal, name="buy_signal")
        self.sell_signal = self.I(lambda: self.data.sell_signal, name="sell_signal")

    def next(self):
        price = self.data.Close[-1]
        atr_val = self.atr[-1]

        # --- Volume Confirmation Filter ---
        if self.data.Volume[-1] < self.volume_ma[-1]:
            return

        # --- Strategy Logic ---
        if self.strategy_mode == 1:  # Momentum Continuation
            # Long Entry: Overbought wave with a red dot (sell_signal), indicating a pullback.
            if self.wt1[-1] > self.wt_ob_level and self.sell_signal[-1] and self.htf_trend_up[-1]:
                if not self.position:
                    sl = price - self.atr_sl_multiplier * atr_val
                    tp = price + self.atr_tp_multiplier * atr_val
                    self.buy(sl=sl, tp=tp)

            # Short Entry: Oversold wave with a green dot (buy_signal), indicating a bounce.
            elif self.wt1[-1] < self.wt_os_level and self.buy_signal[-1] and not self.htf_trend_up[-1]:
                if not self.position:
                    sl = price + self.atr_sl_multiplier * atr_val
                    tp = price - self.atr_tp_multiplier * atr_val
                    self.sell(sl=sl, tp=tp)

            # Exit logic for Momentum Continuation
            if self.position.is_long and self.wt1[-1] < self.wt_ob_level:
                self.position.close()
            elif self.position.is_short and self.wt1[-1] > self.wt_os_level:
                self.position.close()

        elif self.strategy_mode == 2:  # Consolidation Reversal
            is_oscillator_neutral = abs(self.wt1[-1]) < self.wt_ob_level
            is_price_consolidating = self.atr[-1] < self.atr_ma[-1]

            if is_oscillator_neutral and is_price_consolidating:
                # Long Entry: Reversal from consolidation near the zero line, aligned with HTF trend.
                if self.wt1[-1] > self.wt2[-1] and self.wt1[-2] < self.wt2[-2] and self.htf_trend_up[-1]:
                     if not self.position:
                        sl = price - self.atr_sl_multiplier * atr_val
                        tp = price + self.atr_tp_multiplier * atr_val
                        self.buy(sl=sl, tp=tp)

                # Short Entry: Reversal from consolidation near the zero line, aligned with HTF trend.
                elif self.wt1[-1] < self.wt2[-1] and self.wt1[-2] > self.wt2[-2] and not self.htf_trend_up[-1]:
                    if not self.position:
                        sl = price + self.atr_sl_multiplier * atr_val
                        tp = price - self.atr_tp_multiplier * atr_val
                        self.sell(sl=sl, tp=tp)

            # Exit logic for Consolidation Reversal (Take profit when momentum accelerates)
            if self.position.is_long and self.wt1[-1] > self.wt_ob_level:
                self.position.close()
            elif self.position.is_short and self.wt1[-1] < self.wt_os_level:
                self.position.close()

def save_stats(stats, filename="results/temp_result.json"):
    """Saves backtest stats to a JSON file, handling non-serializable data."""
    # Create a serializable dictionary from the stats Series
    stats_dict = {}
    for key, value in stats.items():
        # Skip non-serializable objects
        if key in ['_equity_curve', '_trades', '_strategy']:
            continue

        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_dict[key] = str(value)
        elif pd.isna(value) or value is pd.NA:
            stats_dict[key] = None
        elif isinstance(value, (np.integer, np.floating)):
            stats_dict[key] = value.item()
        else:
            stats_dict[key] = value

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(stats_dict, f, indent=4)

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    try:
        # Load data, skipping the malformed header and explicitly naming columns
        # This handles leading spaces and the trailing comma in the header row.
        df = pd.read_csv(
            data_path,
            header=0,
            names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume'],
            index_col='datetime',
            parse_dates=True,
            usecols=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
    except FileNotFoundError:
        print(f"Data file not found at {data_path}. Please check the path.")
        print("Generating synthetic data as a fallback...")
        from backtesting.test import GOOG
        df = GOOG.copy()
        df = df.iloc[-5000:]

    # Preprocess the data
    df = preprocess_data(df)

    # --- Backtest Execution ---
    print("Running backtest for Momentum Continuation (strategy_mode=1)...")
    bt_momentum = Backtest(df, MarketCipherBStrategy, cash=100_000, commission=.002)
    stats_momentum = bt_momentum.run(strategy_mode=1)

    print("\n--- Momentum Continuation Results ---")
    print(stats_momentum)
    save_stats(stats_momentum, "results/strategy_43063934aa2f_momentum.json")
    try:
        bt_momentum.plot(filename="results/strategy_43063934aa2f_momentum.html", open_browser=False)
        print("Momentum plot saved to results/strategy_43063934aa2f_momentum.html")
    except Exception as e:
        print(f"Could not generate plot for momentum strategy: {e}")


    print("\nRunning backtest for Consolidation Reversal (strategy_mode=2)...")
    bt_reversal = Backtest(df, MarketCipherBStrategy, cash=100_000, commission=.002)
    stats_reversal = bt_reversal.run(strategy_mode=2)

    print("\n--- Consolidation Reversal Results ---")
    print(stats_reversal)
    save_stats(stats_reversal, "results/temp_result.json")
    try:
        bt_reversal.plot(filename="results/strategy_43063934aa2f_reversal.html", open_browser=False)
        print("Reversal plot saved to results/strategy_43063934aa2f_reversal.html")
    except Exception as e:
        print(f"Could not generate plot for reversal strategy: {e}")
