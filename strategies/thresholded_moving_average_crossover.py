
from backtesting import Strategy

import numpy as np
import pandas as pd
import pandas_ta as ta


def preprocess_data(df):
    df_4h = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    df_4h['MA_4H'] = ta.sma(df_4h.Close, length=50)

    df = df.copy()
    df['MA_4H'] = df_4h['MA_4H'].reindex(df.index, method='ffill')
    return df

class ThresholdedMovingAverageCrossover(Strategy):
    long_term_lookback = 50
    short_term_lookback_pct = 20
    long_threshold = 0.001
    short_threshold = 0.001

    atr_period = 10
    atr_multiplier_tp = 5
    atr_multiplier_sl = 2

    volume_ma_period = 20

    def init(self):
        self.log_close = np.log(self.data.Close)
        self.short_term_lookback = int(self.long_term_lookback * (self.short_term_lookback_pct / 100))

        self.short_ma = self.I(ta.sma, pd.Series(self.log_close), length=self.short_term_lookback)
        self.long_ma = self.I(ta.sma, pd.Series(self.log_close), length=self.long_term_lookback)

        self.atr = self.I(ta.atr, high=pd.Series(self.data.High), low=pd.Series(self.data.Low), close=pd.Series(self.data.Close), length=self.atr_period)
        self.volume_ma = self.I(ta.sma, pd.Series(self.data.Volume), length=self.volume_ma_period)

        self.ma_4h = self.I(lambda x: self.data.MA_4H, self.data.Close)

    def next(self):
        ratio = self.short_ma[-1] / self.long_ma[-1] - 1.0

        if not self.position:
            if ratio > self.long_threshold and self.data.Close[-1] > self.ma_4h[-1] and self.data.Volume[-1] > self.volume_ma[-1]:
                sl = self.data.Close[-1] - self.atr[-1] * self.atr_multiplier_sl
                tp = self.data.Close[-1] + self.atr[-1] * self.atr_multiplier_tp
                self.buy(sl=sl, tp=tp)

            elif ratio < -self.short_threshold and self.data.Close[-1] < self.ma_4h[-1] and self.data.Volume[-1] > self.volume_ma[-1]:
                sl = self.data.Close[-1] + self.atr[-1] * self.atr_multiplier_sl
                tp = self.data.Close[-1] - self.atr[-1] * self.atr_multiplier_tp
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    from backtesting import Backtest
    import pandas as pd
    import json

    try:
        df = pd.read_csv(
            'data/BTC-USD-15m.csv',
            index_col=0,
            parse_dates=True,
            header=0,
            names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume'],
            usecols=[0, 1, 2, 3, 4, 5]
        )
    except FileNotFoundError:
        print("No data file found. Create data/BTC-USD-15m.csv or modify the path.")
        exit(1)

    df = preprocess_data(df)

    bt = Backtest(df, ThresholdedMovingAverageCrossover, cash=100000, commission=.002)

    stats = bt.run()
    print(stats)

    def sanitize_stats(stats):

        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                sanitized[key] = str(value)
            elif isinstance(value, (np.int64, np.float64)):
                sanitized[key] = value.item()
            elif isinstance(value, (int, float, str, bool)) or value is None:
                sanitized[key] = value
        return sanitized

    # Save stats to a JSON file
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitize_stats(stats), f, indent=4)

    bt.plot(filename='results/thresholded_moving_average_crossover.html')
