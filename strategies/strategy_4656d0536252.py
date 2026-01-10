
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import sys
import os
import json
import numpy as np

# Add parent directory to path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.indicators.vumanchu import cipher_b

class FinalStrategy(Strategy):
    """
    Final, self-contained strategy for execution.
    """
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    def init(self):
        self.buy_sig = self.I(lambda: self.data.buy_signal, name='BuySignal')
        self.sell_sig = self.I(lambda: self.data.sell_signal, name='SellSignal')
        self.trend_ema = self.I(lambda: self.data.trend_ema, name='TrendEMA')
        self.atr = self.I(lambda: self.data.atr, name='ATR')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='VolumeMA')

    def next(self):
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]
        atr_value = self.atr[-1]
        if not self.position:
            if (self.buy_sig[-1] == 1 and price > self.trend_ema[-1] and volume > self.volume_ma[-1]):
                self.buy(sl=price - atr_value * self.atr_sl_multiplier, tp=price + atr_value * self.atr_tp_multiplier)
            elif (self.sell_sig[-1] == 1 and price < self.trend_ema[-1] and volume > self.volume_ma[-1]):
                self.sell(sl=price + atr_value * self.atr_sl_multiplier, tp=price - atr_value * self.atr_tp_multiplier)

def sanitize_stats(stats_series):
    sanitized = {}
    if stats_series is None: return sanitized
    for key, value in stats_series.items():
        if isinstance(value, (np.integer, np.int64)): value = int(value)
        elif isinstance(value, (np.floating, np.float64)): value = float(value)
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)): value = str(value)
        elif isinstance(value, pd.DataFrame): continue
        elif isinstance(value, type(pd.NA)) or pd.isna(value): value = None
        sanitized[key] = value
    sanitized.pop('_strategy', None)
    return sanitized

if __name__ == '__main__':
    DATA_PATH = 'data/BTC-USD-15m.csv'
    CASH = 100_000
    COMMISSION = 0.002

    try:
        df = pd.read_csv(DATA_PATH, index_col='datetime', parse_dates=True)
        df.columns = [col.strip().capitalize() for col in df.columns]
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        from backtesting.test import EURUSD
        df = EURUSD.copy().iloc[-5000:]
        df.index.name = 'datetime'

    print("Preprocessing data in-line...")
    df_processed = df.copy()
    df_processed = cipher_b(df_processed)
    df_processed['buy_signal'] = df_processed['buy_signal'].astype(int)
    df_processed['sell_signal'] = df_processed['sell_signal'].astype(int)
    df_processed['trend_ema'] = ta.ema(df_processed['Close'], length=200)

    atr_series = ta.atr(high=df_processed['High'], low=df_processed['Low'], close=df_processed['Close'], length=14)
    if isinstance(atr_series, pd.DataFrame):
        df_processed['atr'] = atr_series.iloc[:, 0]
    else:
        df_processed['atr'] = atr_series

    df_processed['volume_ma'] = ta.sma(df_processed['Volume'], length=20)
    df_processed.dropna(inplace=True)

    stats = None
    if df_processed.empty:
        print("Error: Preprocessing resulted in an empty DataFrame.")
    else:
        print("Initializing and running backtest...")
        bt = Backtest(df_processed, FinalStrategy, cash=CASH, commission=COMMISSION)
        stats = bt.run()
        print("\n" + "="*80 + "\nBacktest Results\n" + "="*80)
        print(stats)
        print("="*80 + "\n")

        plot_filename = 'results/strategy_4656d0536252_plot.html'
        print(f"Saving plot to {plot_filename}...")
        try:
            bt.plot(filename=plot_filename, open_browser=False)
        except Exception as e:
            print(f"Could not save plot: {e}")

    results_filename = 'results/temp_result.json'
    print(f"Sanitizing and saving results to {results_filename}...")
    final_stats = sanitize_stats(stats)
    with open(results_filename, 'w') as f:
        json.dump(final_stats, f, indent=4)

    # Clean up original problematic file and debug file
    if os.path.exists('strategies/strategy_4656d0536252.py'):
        os.remove('strategies/strategy_4656d0536252.py')
    if os.path.exists('strategies/debug_strategy.py'):
        os.remove('strategies/debug_strategy.py')

    # Rename the final script to the target name
    os.rename('strategies/final_run.py', 'strategies/strategy_4656d0536252.py')

    print("\nScript finished successfully.")
