import json
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from backtesting import Backtest, Strategy

def generate_synthetic_data():
    n = 2500
    rng = np.random.default_rng(42)
    price = 100 + np.cumsum(rng.normal(0, 0.1, n))
    for i in range(100, n - 150, 300):
        price[i+20:i+40] = price[i+20] + np.linspace(0, 5, 20); price[i+40:i+60] = price[i+59] - np.linspace(0, 3, 20)
        price[i+60:i+80] = price[i+60] + np.linspace(0, 2.5, 20); price[i+80:i+100] = price[i+99] - np.linspace(0, 6, 20)
    for i in range(250, n - 150, 300):
        price[i+20:i+40] = price[i+20] - np.linspace(0, 5, 20); price[i+40:i+60] = price[i+59] + np.linspace(0, 3, 20)
        price[i+60:i+80] = price[i+60] - np.linspace(0, 2.5, 20); price[i+80:i+100] = price[i+99] + np.linspace(0, 6, 20)
    index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=n, freq='min'))
    data = pd.DataFrame({'Open': price, 'High': price, 'Low': price, 'Close': price, 'Volume': rng.integers(100, 1000, n)}, index=index)
    data['High'] += rng.uniform(0, 0.2, n); data['Low'] -= rng.uniform(0, 0.2, n)
    data['Open'] = (data['High'] + data['Low']) / 2 + rng.normal(0, 0.05, n)
    data['Close'] = (data['High'] + data['Low']) / 2 + rng.normal(0, 0.05, n)
    return data

def preprocess_data(data_1m):
    data_15m = data_1m['Close'].resample('15min').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    peaks, _ = find_peaks(data_15m['High'], prominence=0.5, width=1)
    troughs, _ = find_peaks(-data_15m['Low'], prominence=0.5, width=1)
    data_15m['swing_high'] = np.nan; data_15m['swing_low'] = np.nan
    data_15m.iloc[peaks, data_15m.columns.get_loc('swing_high')] = data_15m.iloc[peaks]['High']
    data_15m.iloc[troughs, data_15m.columns.get_loc('swing_low')] = data_15m.iloc[troughs]['Low']
    data_15m = data_15m.ffill()
    fib_range = data_15m['swing_high'] - data_15m['swing_low']
    data_15m['fib_382'] = data_15m['swing_high'] - fib_range * 0.382
    data_15m['fib_500'] = data_15m['swing_high'] - fib_range * 0.5
    data_15m['fib_618'] = data_15m['swing_high'] - fib_range * 0.618
    data_15m['fib_786'] = data_15m['swing_high'] - fib_range * 0.786
    for col in ['swing_high', 'swing_low', 'fib_382', 'fib_500', 'fib_618', 'fib_786']:
        data_1m[f'{col}_15m'] = data_15m[col].reindex(data_1m.index, method='ffill')
    data_1m['ema_20'] = data_1m['Close'].ewm(span=20, adjust=False).mean()
    return data_1m.dropna()

class FibonacciCenterPeakScalpStrategy(Strategy):
    ema_period = 20
    rr_ratio = 5

    def init(self):
        self.ema = self.I(lambda x, n: pd.Series(x).ewm(span=n, adjust=False).mean(), self.data.Close, self.ema_period)

    def next(self):
        if self.position:
            return

        price = self.data.Close[-1]
        fib_382 = self.data.df.iloc[-1]['fib_382_15m']
        fib_500 = self.data.df.iloc[-1]['fib_500_15m']
        fib_786 = self.data.df.iloc[-1]['fib_786_15m']

        is_setup_active = self.data.Close[-1] < fib_382 and self.data.Close[-1] > fib_786
        is_confluence = abs(fib_500 - self.ema[-1]) < 1.0

        if is_setup_active and is_confluence:
            # Short entry
            if price > fib_500 and abs(price - fib_500) < 0.5 and \
               self.data.Close[-1] < self.data.Open[-1] and self.data.Open[-1] > self.data.Close[-2] and self.data.Close[-1] < self.data.Open[-2]:

                stop_loss = self.data.High[-1] + 0.1
                swing_low = self.data.df.iloc[-1]['swing_low_15m']
                take_profit = self.data.High[-1] - ((self.data.High[-1] - swing_low) * 0.5)
                risk = stop_loss - price
                reward = price - take_profit

                if risk > 0 and reward / risk >= self.rr_ratio:
                    self.sell(sl=stop_loss, tp=take_profit)

            # Long entry
            elif price < fib_500 and abs(price - fib_500) < 0.5 and \
                 self.data.Close[-1] > self.data.Open[-1] and self.data.Open[-1] < self.data.Close[-2] and self.data.Close[-1] > self.data.Open[-2]:

                stop_loss = self.data.Low[-1] - 0.1
                swing_high = self.data.df.iloc[-1]['swing_high_15m']
                take_profit = self.data.Low[-1] + ((swing_high - self.data.Low[-1]) * 0.5)
                risk = price - stop_loss
                reward = take_profit - price

                if risk > 0 and reward / risk >= self.rr_ratio:
                    self.buy(sl=stop_loss, tp=take_profit)

if __name__ == '__main__':
    data = generate_synthetic_data()
    data = preprocess_data(data)
    bt = Backtest(data, FibonacciCenterPeakScalpStrategy, cash=10000, commission=.002, finalize_trades=True)
    stats = bt.optimize(ema_period=range(10, 30, 5), rr_ratio=range(3, 7, 1), maximize='Sharpe Ratio', constraint=lambda p: p.rr_ratio > 2)
    print(stats)
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        stats_dict = {'strategy_name': 'fibonacci_center_peak_scalp', 'return': stats.get('Return [%]', 0.0), 'sharpe': stats.get('Sharpe Ratio', 0.0), 'max_drawdown': stats.get('Max. Drawdown [%]', 0.0), 'win_rate': stats.get('Win Rate [%]', 0.0), 'total_trades': stats.get('# Trades', 0)}
        for key, value in stats_dict.items():
            if key != 'strategy_name':
                stats_dict[key] = (int(value) if isinstance(value, (np.int64, np.int32)) else float(value)) if pd.notna(value) else None
        json.dump(stats_dict, f, indent=2)
    bt.plot(filename='results/fibonacci_center_peak_scalp.html')
