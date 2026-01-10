"""
Strategy: Thresholded Mean Reversion
"""
from backtesting import Strategy
import talib
import numpy as np
import pandas as pd
import json
import os

def sanitize_stats(stats):
    """Prepares the backtesting stats object for JSON serialization."""
    if isinstance(stats, pd.Series):
        stats = stats.to_dict()
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = float(value)
        elif not isinstance(value, (str, int, float, bool)) and value is not None:
            continue
        else:
            sanitized[key] = value
    return sanitized

def preprocess_data(df, **params):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df_4h = df.resample('4H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)
    htf_trend_signal = df_4h['Close'] > df_4h['ema_200']
    df['htf_trend'] = df.index.floor('4H').map(htf_trend_signal)
    df['htf_trend'] = df['htf_trend'].ffill()
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['log_close'] = np.log(df['Close'])
    return df

class ThresholdedMeanReversion(Strategy):
    lookback = 1000
    rise_threshold = 0.005
    drop_threshold = 0.0005
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    def init(self):
        self.atr = self.I(lambda: self.data.atr, name="atr")
        self.htf_trend = self.I(lambda: self.data.htf_trend, name="htf_trend")
        self.volume_ma = self.I(lambda: self.data.volume_ma, name="volume_ma")
        self.log_close = self.I(lambda: self.data.log_close, name="log_close")

    def next(self):
        # Correct boundary check to prevent IndexError
        if len(self.data) < self.lookback + 1:
            return
        is_uptrend = self.htf_trend[-1] > 0
        is_high_volume = self.data.Volume[-1] > self.volume_ma[-1]
        # Correct off-by-one error in lookback indexing
        long_term_trend = self.log_close[-1] - self.log_close[-(self.lookback + 1)]
        is_strong_trend = long_term_trend > self.rise_threshold
        short_term_drop = self.log_close[-2] - self.log_close[-1]
        is_sharp_drop = short_term_drop > self.drop_threshold
        if not self.position:
            if is_uptrend and is_high_volume and is_strong_trend and is_sharp_drop:
                sl = self.data.Close[-1] - (self.atr_sl_multiplier * self.atr[-1])
                tp = self.data.Close[-1] + (self.atr_tp_multiplier * self.atr[-1])
                self.buy(sl=sl, tp=tp)

if __name__ == '__main__':
    from backtesting import Backtest
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
        # Sanitize column names: strip spaces and capitalize
        df.columns = [col.strip().capitalize() for col in df.columns]
        # Select only the required columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except FileNotFoundError:
        print("Data file 'data/BTC-USD-15m.csv' not found. Generating sample data...")
        dates = pd.date_range('2023-01-01', periods=4000, freq='15min')
        np.random.seed(42)
        price = 20000 + np.cumsum(np.random.randn(4000) * 10)
        df = pd.DataFrame({
            'Open': price, 'High': price + np.random.rand(4000) * 20,
            'Low': price - np.random.rand(4000) * 20, 'Close': price + np.random.randn(4000) * 5,
            'Volume': np.random.rand(4000) * 1000000
        }, index=dates)

    df = preprocess_data(df).dropna()
    bt = Backtest(df, ThresholdedMeanReversion, cash=100000, commission=0.001)
    stats = bt.run()
    print("\n=== ThresholdedMeanReversion Strategy Results ===")
    print(stats)
    if not os.path.exists('results'):
        os.makedirs('results')
    bt.plot(filename='results/thresholded_mean_reversion.html', open_browser=False)
    print("\nPlot saved to results/thresholded_mean_reversion.html")
    sanitized = sanitize_stats(stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized, f, indent=4)
    print("\nStats saved to results/temp_result.json")
