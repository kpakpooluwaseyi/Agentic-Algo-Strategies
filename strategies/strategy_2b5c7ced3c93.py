# coding: utf-8
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

import pandas as pd
import numpy as np
import talib
from src.indicators.vumanchu import cipher_b

# Developer Note:
# The initial request specified inheriting from `src.strategies.base.MoonDevStrategy`.
# However, that base class is designed for a signal-generation framework and is
# incompatible with the backtesting workflow (e.g., `init()`, `next()`, `bt.run()`)
# also required by the request.
# Following the architecture of the provided reference implementation
# (`strategies/vumanchu_cipher_b.py`), this strategy inherits from
# `backtesting.Strategy` to create a functional, standalone backtest.

def preprocess_data(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Adds all necessary indicators to the DataFrame for the strategy.
    """
    # Apply VuManchu Cipher B indicator
    df = cipher_b(df)

    # Multi-Timeframe Trend Filter (4H EMA 200)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)
    df_4h['htf_trend_up'] = df_4h['Close'] > df_4h['ema_200']

    # Map the 4H trend back to the 15m timeframe
    df['htf_trend_up'] = df_4h['htf_trend_up'].reindex(df.index, method='ffill').fillna(False)

    # Volume Confirmation
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=20)

    # ATR for Risk Management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    return df

class Strategy2b5c7ced3c93(Strategy):
    # Optimizable parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.5
    volume_ma_multiplier = 1.0

    def init(self):
        # Indicators from preprocessed data
        self.buy_signal = self.I(lambda: self.data.buy_signal, name='buy_signal')
        self.sell_signal = self.I(lambda: self.data.sell_signal, name='sell_signal')
        self.htf_trend_up = self.I(lambda: self.data.htf_trend_up, name='htf_trend_up')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')
        self.atr = self.I(lambda: self.data.atr, name='atr')

    def next(self):
        # --- FILTERS ---
        # Volume filter
        volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_ma_multiplier

        # --- ENTRY LOGIC ---
        # If we are not in a position, check for entry signals
        if not self.position:
            # Long entry
            if self.buy_signal[-1] and self.htf_trend_up[-1] and volume_confirmed:
                sl = self.data.Close[-1] - (self.atr_sl_multiplier * self.atr[-1])
                tp = self.data.Close[-1] + (self.atr_tp_multiplier * self.atr[-1])
                self.buy(sl=sl, tp=tp)

            # Short entry
            elif self.sell_signal[-1] and not self.htf_trend_up[-1] and volume_confirmed:
                sl = self.data.Close[-1] + (self.atr_sl_multiplier * self.atr[-1])
                tp = self.data.Close[-1] - (self.atr_tp_multiplier * self.atr[-1])
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    try:
        data = pd.read_csv('data/BTC-USD-15m.csv')
    except FileNotFoundError:
        print("Data file not found. Using synthetic data for demonstration.")
        # Generate some synthetic data if the file is not available
        rng = pd.date_range('2020-01-01', periods=2000, freq='15min')
        data = pd.DataFrame(np.random.randn(2000, 4),
                              index=rng,
                              columns=['Open', 'High', 'Low', 'Close'])
        data['Volume'] = np.random.randint(100, 1000, size=len(data))
        # Ensure OHLC properties
        data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.rand(len(data))
        data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.rand(len(data))


    # --- Data Preprocessing ---
    # Sanitize column names
    data.columns = [x.capitalize() for x in data.columns]
    if 'Datetime' in data.columns:
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data = data.set_index('Datetime')

    # Ensure index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    data = preprocess_data(data)
    data.dropna(inplace=True)

    print("Running Backtest...")
    bt = Backtest(data, Strategy2b5c7ced3c93, cash=100_000, commission=.002)
    stats = bt.run()
    print(stats)

    # --- Save Results ---
    # Sanitize stats for JSON output
    stats_serializable = {key: (str(value) if isinstance(value, (pd.Timestamp, pd.Timedelta)) else value)
                          for key, value in stats.items() if '_equity_curve' not in key and '_trades' not in key}

    with open('results/temp_result.json', 'w') as f:
        import json
        json.dump(stats_serializable, f, indent=4)

    print("Backtest finished. Plot saved to results/strategy_2b5c7ced3c93.html and stats to results/temp_result.json")
    bt.plot(filename='results/strategy_2b5c7ced3c93.html')
