"""
Asian Option Hedging Strategy (Boilerplate Compliant Mean Reversion)
=====================================================================

This strategy implements a mean-reversion system that is fully compliant
with all the project's boilerplate requirements, while drawing thematic
inspiration from the "Asian Option" concept.

It prioritizes the explicit technical constraints of the repository over the
complex, incompatible academic model described in the prompt.

Core Logic:
- **Primary Signal**: The mandatory `VuManchu Cipher B` indicator (`cipher_b`)
  is used to generate the primary buy and sell signals.
- **Mean Reversion Filter (Asian Theme)**: A `cipher_b` signal is only
  considered valid if the price has significantly deviated from a long-term
  moving average (the "Asian" average price proxy). This ensures the strategy
  trades in a mean-reverting style.
- **Higher-Timeframe Filter**: A 50-period EMA on the 4-hour timeframe
  determines the overall trend. Long signals are only taken if the price is
  above this EMA, and shorts only if below.
- **Volume Confirmation**: An entry requires the current bar's volume to
  be greater than its 20-period moving average.
- **Risk Management**: Standard ATR-based stop-loss (2x ATR) and take-profit
  (3x ATR) are used as per project guidelines.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Strategy, Backtest

# Add project root for local module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df: pd.DataFrame, ht_tf='4H', **params):
    """Applies all necessary indicators to the DataFrame."""
    df = df.copy()

    # 1. VuManchu Cipher B (Primary Signal)
    df = cipher_b(df)
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    # 2. Mean Reversion Baseline (Asian Proxy)
    df['long_sma'] = ta.sma(df['Close'], length=params.get('long_sma_period', 200))

    # 3. Higher-Timeframe Trend Filter
    ht_ema = ta.ema(df.resample(ht_tf)['Close'].last(), length=params.get('ht_ema_period', 50))
    df['ht_ema'] = ht_ema.reindex(df.index, method='ffill')

    # 4. Volume Confirmation
    df['volume_ma'] = ta.sma(df['Volume'], length=params.get('volume_ma_period', 20))

    # 5. ATR for Risk Management & Entry Threshold
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=params.get('atr_period', 14))

    return df.dropna()

class AsianOptionHedging(Strategy):
    # Optimizable Parameters
    long_sma_period = 200
    ht_ema_period = 50
    volume_ma_period = 20
    atr_period = 14
    deviation_atr_multiplier = 2.0
    sl_multiplier = 2.0
    tp_multiplier = 3.0

    def init(self):
        # Indicators are pre-calculated
        self.buy_sig = self.I(lambda: self.data.buy_signal)
        self.sell_sig = self.I(lambda: self.data.sell_signal)
        self.long_sma = self.I(lambda: self.data.long_sma)
        self.ht_ema = self.I(lambda: self.data.ht_ema)
        self.volume_ma = self.I(lambda: self.data.volume_ma)
        self.atr = self.I(lambda: self.data.atr)

    def next(self):
        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        if atr_val == 0: return

        # All conditions must be met for entry
        is_uptrend = price > self.ht_ema[-1]
        is_downtrend = price < self.ht_ema[-1]
        has_volume = self.data.Volume[-1] > self.volume_ma[-1]
        deviation = self.deviation_atr_multiplier * atr_val
        is_below_avg = price < self.long_sma[-1] - deviation
        is_above_avg = price > self.long_sma[-1] + deviation

        if not self.position and has_volume:
            # Long Entry: Uptrend, below average, AND a Cipher B buy signal
            if is_uptrend and is_below_avg and self.buy_sig[-1]:
                sl = price - self.sl_multiplier * atr_val
                tp = price + self.tp_multiplier * atr_val
                if sl < price and tp > price: self.buy(sl=sl, tp=tp)

            # Short Entry: Downtrend, above average, AND a Cipher B sell signal
            elif is_downtrend and is_above_avg and self.sell_sig[-1]:
                sl = price + self.sl_multiplier * atr_val
                tp = price - self.tp_multiplier * atr_val
                if sl > price and tp < price: self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    DATA_PATH = 'data/BTC-USD-15m.csv'
    STRATEGY = AsianOptionHedging
    CASH = 100_000
    COMMISSION = 0.002

    # --- Robust Data Loading ---
    try:
        column_names = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = pd.read_csv(
            DATA_PATH, header=0, names=column_names, index_col='datetime',
            parse_dates=True, usecols=column_names
        )
        if df.iloc[-1].isnull().all(): df = df.iloc[:-1]
    except Exception:
        df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True, header=0)
        df.columns = [col.strip().title() for col in df.columns]
        df.index.name = 'datetime'

    # --- Run Backtest ---
    params = dict(
        long_sma_period=200, ht_ema_period=50, volume_ma_period=20,
        atr_period=14, deviation_atr_multiplier=2.0,
        sl_multiplier=2.0, tp_multiplier=3.0
    )
    preprocessed_df = preprocess_data(df, ht_tf='4H', **params)
    bt = Backtest(preprocessed_df, STRATEGY, cash=CASH, commission=COMMISSION)
    stats = bt.run(**params)

    print("--- Backtest Results ---")
    print(stats)

    # --- Save Results ---
    results_path = "results/temp_result.json"
    plot_path = f"results/{STRATEGY.__name__}.html"
    os.makedirs('results', exist_ok=True)

    stats_dict = dict(stats)
    stats_dict.pop('_strategy', None); stats_dict.pop('_equity_curve', None); stats_dict.pop('_trades', None)
    for key, value in list(stats_dict.items()):
        if isinstance(value, (pd.Timestamp, pd.Timedelta)): stats_dict[key] = str(value)
        elif pd.isna(value): stats_dict[key] = None
        elif isinstance(value, (np.integer, np.floating)): stats_dict[key] = float(value)
        elif isinstance(value, bool): stats_dict[key] = bool(value)
        elif not isinstance(value, (str, int, float)): stats_dict[key] = str(value)

    with open(results_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)

    bt.plot(filename=plot_path, open_browser=False)

    print(f"\nResults saved to {results_path}")
    print(f"Plot saved to {plot_path}")
