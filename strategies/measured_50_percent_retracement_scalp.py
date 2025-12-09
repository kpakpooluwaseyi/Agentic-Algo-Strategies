from backtesting import Backtest, Strategy, lib
import pandas as pd
import numpy as np
import json
import os
from enum import Enum

def EMA(series, n):
    """Returns the EMA of a series."""
    return pd.Series(series).ewm(span=n, min_periods=n).mean()

class StrategyState(Enum):
    SEARCHING = 1
    IN_AOI = 2
    SHORT_OPEN = 3
    LONG_OPEN = 4

def generate_synthetic_data(periods=5000):
    rng = np.random.default_rng(42)
    index = pd.date_range(start='2023-01-01', periods=periods, freq='1min')
    price = np.zeros(periods)
    price[0] = 100

    p_start_high, p_low, p_retrace_high = 49, 249, 449

    drop_1 = np.linspace(0, -20, p_low - p_start_high + 1)
    price[p_start_high:p_low+1] = price[p_start_high-1] + drop_1 + rng.normal(0, 0.2, len(drop_1)).cumsum()

    start_price_retrace = price[p_low]
    fib_50 = price[p_start_high] - (price[p_start_high] - price[p_low]) * 0.5
    retrace = np.linspace(0, fib_50 - start_price_retrace, p_retrace_high - p_low + 1)
    price[p_low:p_retrace_high+1] = start_price_retrace + retrace + rng.normal(0, 0.15, len(retrace)).cumsum()
    price[p_retrace_high] = fib_50

    start_price_drop2 = price[p_retrace_high]
    drop_2 = np.linspace(0, -25, periods - p_retrace_high - 1)
    price[p_retrace_high+1:] = start_price_drop2 + drop_2 + rng.normal(0, 0.2, len(drop_2)).cumsum()

    price[0:p_start_high] = price[p_start_high] + rng.normal(0, 0.1, p_start_high).cumsum()[::-1]

    price = np.maximum(0.01, price)
    df = pd.DataFrame(index=index, data={'Close': price})
    df['Open'] = df['Close'].shift(1).bfill()
    df['High'] = df[['Open', 'Close']].max(axis=1) + rng.uniform(0, 0.5, periods)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - rng.uniform(0, 0.5, periods)
    df.clip(lower=0.01, inplace=True)
    df['Volume'] = rng.integers(100, 1000, periods)

    df['swing_high'] = np.nan
    df['swing_low'] = np.nan
    df.at[df.index[p_start_high], 'swing_high'] = price[p_start_high]
    df.at[df.index[p_low], 'swing_low'] = price[p_low]

    return df

def preprocess_data(df):
    df[['setup_high', 'setup_low']] = df[['swing_high', 'swing_low']].ffill()
    df['fib_50_aoi'] = df['setup_high'] - (df['setup_high'] - df['setup_low']) * 0.5
    df.dropna(subset=['fib_50_aoi'], inplace=True)
    return df

def ID(series): return series

class Measured50PercentRetracementScalpStrategy(Strategy):
    rr_ratio = 5
    aoi_proximity = 0.01
    ema_period = 200

    def init(self):
        self.aoi = self.I(ID, self.data.df['fib_50_aoi'])
        self.setup_low = self.I(ID, self.data.df['setup_low'])
        self.ema = self.I(EMA, self.data.Close, self.ema_period)
        self.state = StrategyState.SEARCHING
        self.retracement_high = None
        self.last_short_tp = None

    def next(self):
        if self.state == StrategyState.SHORT_OPEN and not self.position.is_short:
            if self.last_short_tp and abs(self.data.Close[-1] - self.last_short_tp) < 0.1:
                if self.data.Close[-1] > self.data.Open[-1]:
                    sl = self.data.Close[-1] * 0.98
                    tp = self.data.Close[-1] * 1.04
                    self.buy(sl=sl, tp=tp)
                    self.state = StrategyState.LONG_OPEN
                    return
            self.state = StrategyState.SEARCHING

        elif self.state == StrategyState.LONG_OPEN and not self.position.is_long:
            self.state = StrategyState.SEARCHING

        if self.state == StrategyState.SEARCHING:
            is_below_ema = self.data.Close[-1] < self.ema[-1]
            if is_below_ema and abs(self.data.High[-1] - self.aoi[-1]) / self.aoi[-1] <= self.aoi_proximity:
                self.state = StrategyState.IN_AOI
                self.retracement_high = self.data.High[-1]

        elif self.state == StrategyState.IN_AOI:
            self.retracement_high = max(self.retracement_high, self.data.High[-1])
            if self.data.Close[-1] < self.retracement_high and not self.position:
                sl = self.retracement_high * 1.002
                tp = self.retracement_high - (self.retracement_high - self.setup_low[-1]) * 0.5
                risk = sl - self.data.Close[-1]
                reward = self.data.Close[-1] - tp

                if risk > 0 and reward / risk >= self.rr_ratio and tp > 0:
                    self.sell(sl=sl, tp=tp)
                    self.last_short_tp = tp
                    self.state = StrategyState.SHORT_OPEN

            if abs(self.data.Close[-1] - self.aoi[-1]) / self.aoi[-1] > self.aoi_proximity * 3:
                 self.state = StrategyState.SEARCHING

if __name__ == '__main__':
    data = preprocess_data(generate_synthetic_data(periods=5000))
    bt = Backtest(data, Measured50PercentRetracementScalpStrategy, cash=100_000, commission=.002)
    stats = bt.optimize(rr_ratio=range(3, 8, 1), aoi_proximity=[x * 0.01 for x in range(1, 11)], ema_period=[100, 200], maximize='Sharpe Ratio')

    os.makedirs('results', exist_ok=True)

    sharpe = stats.get('Sharpe Ratio', 0.0)
    if np.isnan(sharpe): sharpe = 0.0

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'measured_50_percent_retracement_scalp',
            'return': stats.get('Return [%]', 0.0), 'sharpe': sharpe,
            'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
            'win_rate': stats.get('Win Rate [%]', 0.0), 'total_trades': int(stats.get('# Trades', 0))
        }, f, indent=2)

    bt.plot(filename='results/measured_50_percent_retracement_scalp.html', open_browser=False)
    print("--- Best Run Stats ---")
    print(stats)
