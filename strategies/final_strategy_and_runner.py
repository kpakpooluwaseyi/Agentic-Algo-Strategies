
import pandas as pd
import pandas_ta as ta
from backtesting import Strategy, Backtest
import sys
import os
import json
import numpy as np

# Add parent directory for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.indicators.vumanchu import cipher_b

# ==============================================================================
# 1. PURE STRATEGY MODULE CODE
# ==============================================================================

def preprocess_data(df: pd.DataFrame, **params):
    df = df.copy()
    df = cipher_b(df)
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)
    df['trend_ema'] = ta.ema(df['Close'], length=200)
    atr_series = ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=14)
    if isinstance(atr_series, pd.DataFrame):
        df['atr'] = atr_series.iloc[:, 0]
    else:
        df['atr'] = atr_series
    df['volume_ma'] = ta.sma(df['Volume'], length=20)
    df.dropna(inplace=True)
    return df

class Strategy_4656d0536252(Strategy):
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

# ==============================================================================
# 2. VALIDATION RUNNER CODE
# ==============================================================================

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
        df = pd.DataFrame() # Fallback to empty df if not found

    stats = None
    if not df.empty:
        df_processed = preprocess_data(df)
        if not df_processed.empty:
            bt = Backtest(df_processed, Strategy_4656d0536252, cash=CASH, commission=COMMISSION)
            stats = bt.run()
            print(stats)
            os.makedirs('results', exist_ok=True)
            bt.plot(filename='results/strategy_4656d0536252_plot.html', open_browser=False)

    results_filename = 'results/temp_result.json'
    final_stats = sanitize_stats(stats)
    with open(results_filename, 'w') as f:
        json.dump(final_stats, f, indent=4)
    print("\nValidation run finished.")
