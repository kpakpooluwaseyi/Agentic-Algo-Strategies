from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
import enum

class Trend(enum.Enum):
    UP = 1
    DOWN = 2

def generate_synthetic_data(trend=Trend.UP):
    """
    Generates synthetic data simulating a textbook trend, pullback, and continuation.
    """
    if trend == Trend.UP:
        data = [{'Open': 100, 'High': 102, 'Low': 99.8, 'Close': 101}, {'Open': 101, 'High': 103, 'Low': 100.8, 'Close': 102.5}, {'Open': 102.5, 'High': 105, 'Low': 102.3, 'Close': 104.5}]
        data.extend([{'Open': 104.5, 'High': 104.7, 'Low': 103, 'Close': 103.5}, {'Open': 103.5, 'High': 103.7, 'Low': 102, 'Close': 102.5}])
        data.extend([{'Open': 102.5, 'High': 105, 'Low': 102.3, 'Close': 104}])
    else:
        data = [{'Open': 104.5, 'High': 105, 'Low': 102.3, 'Close': 103.5}, {'Open': 103.5, 'High': 103.7, 'Low': 102, 'Close': 102.5}, {'Open': 102.5, 'High': 103, 'Low': 100.8, 'Close': 101}]
        data.extend([{'Open': 101, 'High': 102.5, 'Low': 100.8, 'Close': 102}, {'Open': 102, 'High': 103, 'Low': 101.8, 'Close': 102.5}])
        data.extend([{'Open': 102.5, 'High': 102.7, 'Low': 100, 'Close': 100.5}])
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(df), freq='15min'))
    return df

class RangeBarPullbackContinuationStrategy(Strategy):
    pip_size = 0.01
    risk_pct = 0.01
    trend_window = 3

    def init(self):
        self.trend = None
        self.nhc_candle = None
        self.nlc_candle = None
        self.pullback_high = -np.inf
        self.pullback_low = np.inf

    def next(self):
        if len(self.data) < self.trend_window:
            return

        is_up = all(self.data.Close[i] > self.data.Close[i-1] for i in range(-self.trend_window, 0))
        is_down = all(self.data.Close[i] < self.data.Close[i-1] for i in range(-self.trend_window, 0))

        if is_up:
            self.trend = Trend.UP
            if self.nhc_candle is None or self.data.Close[-1] > self.nhc_candle['Close']:
                self.nhc_candle = {'Open': self.data.Open[-1], 'Close': self.data.Close[-1]}
                self.pullback_low = np.inf
        elif is_down:
            self.trend = Trend.DOWN
            if self.nlc_candle is None or self.data.Close[-1] < self.nlc_candle['Close']:
                self.nlc_candle = {'Open': self.data.Open[-1], 'Close': self.data.Close[-1]}
                self.pullback_high = -np.inf

        if self.position:
            return

        if self.trend == Trend.UP and self.nhc_candle:
            if self.data.Close[-1] < self.data.Open[-1]:
                self.pullback_low = min(self.pullback_low, self.data.Low[-1])
            elif self.data.Close[-1] > self.data.Open[-1] and self.data.Close[-1] > self.data.Close[-2] and self.pullback_low != np.inf:
                entry_price = self.data.Close[-1]
                stop_loss = self.pullback_low - self.pip_size
                take_profit = self.nhc_candle['Open']
                if entry_price > stop_loss and take_profit > entry_price and (take_profit - entry_price) / (entry_price - stop_loss) >= 1.0:
                    size = (self.equity * self.risk_pct) / (entry_price - stop_loss)
                    if size * entry_price <= self.equity:
                        self.buy(sl=stop_loss, tp=take_profit, size=int(size))
                self.nhc_candle = None
                self.pullback_low = np.inf
        elif self.trend == Trend.DOWN and self.nlc_candle:
            if self.data.Close[-1] > self.data.Open[-1]:
                self.pullback_high = max(self.pullback_high, self.data.High[-1])
            elif self.data.Close[-1] < self.data.Open[-1] and self.data.Close[-1] < self.data.Close[-2] and self.pullback_high != -np.inf:
                entry_price = self.data.Close[-1]
                stop_loss = self.pullback_high + self.pip_size
                take_profit = self.nlc_candle['Open']
                if entry_price < stop_loss and take_profit < entry_price and (entry_price - take_profit) / (stop_loss - entry_price) >= 1.0:
                    size = (self.equity * self.risk_pct) / (stop_loss - entry_price)
                    if size * entry_price <= self.equity:
                        self.sell(sl=stop_loss, tp=take_profit, size=int(size))
                self.nlc_candle = None
                self.pullback_high = -np.inf

if __name__ == '__main__':
    from backtesting.test import GOOG
    data = GOOG.iloc[-500:]

    bt = Backtest(data, RangeBarPullbackContinuationStrategy, cash=100_000, commission=.002)

    stats = bt.optimize(
        pip_size=[0.01, 0.02, 0.03],
        risk_pct=[0.01, 0.02, 0.03],
        trend_window=range(3, 10),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.risk_pct < 0.1
    )

    import os
    os.makedirs('results', exist_ok=True)

    results_dict = {
        'strategy_name': 'range_bar_pullback_continuation',
        'return': stats.get('Return [%]'),
        'sharpe': stats.get('Sharpe Ratio'),
        'max_drawdown': stats.get('Max. Drawdown [%]'),
        'win_rate': stats.get('Win Rate [%]'),
        'total_trades': stats.get('# Trades')
    }

    for key, value in results_dict.items():
        if isinstance(value, (np.integer, np.floating)) and pd.isna(value):
            results_dict[key] = None
        elif isinstance(value, (np.integer, int)):
            results_dict[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            results_dict[key] = float(value)

    with open('results/temp_result.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    try:
        bt.plot(filename='results/range_bar_pullback_continuation.html')
    except TypeError:
        print("Warning: bt.plot() failed. Skipping plot generation.")
