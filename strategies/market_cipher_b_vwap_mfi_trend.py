from backtesting import Strategy, Backtest
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import json
import numpy as np

# Wrapper functions for pandas-ta indicators
def mfi_indicator(high, low, close, volume, length):
    return ta.mfi(high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), volume=pd.Series(volume), length=length).values

def macd_histogram(close, fast, slow, signal):
    macd = ta.macd(pd.Series(close), fast=fast, slow=slow, signal=signal)
    return macd.iloc[:, 1].values

class MarketCipherBVwapMfiTrendStrategy(Strategy):
    # Optimizable parameters
    vwap_trigger_period = 20
    mfi_period = 14
    mfi_exit_threshold_high = 80
    mfi_exit_threshold_low = 20
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    risk_reward_ratio = 2.0
    sl_lookback = 20
    sl_buffer_pct = 0.01

    def init(self):
        # Pre-calculated indicators
        self.vwap = self.data.VWAP
        self.vwap_trigger = self.data.VWAP_Trigger

        # Indicators calculated within the framework
        self.mfi = self.I(mfi_indicator, self.data.High, self.data.Low, self.data.Close, self.data.Volume, self.mfi_period)
        self.macd_hist = self.I(macd_histogram, self.data.Close, self.macd_fast, self.macd_slow, self.macd_signal)

    def next(self):
        price = self.data.Close[-1]

        # --- Exit Logic ---
        if self.position.is_long and self.mfi[-1] > self.mfi_exit_threshold_high:
            self.position.close()
        elif self.position.is_short and self.mfi[-1] < self.mfi_exit_threshold_low:
            self.position.close()

        # --- Entry Logic ---
        if self.position:
            return

        # Long entry conditions
        vwap_trending_up = self.vwap[-1] > self.vwap[-2]
        is_above_vwap = price > self.vwap[-1]
        pullback_to_vwap = self.data.Low[-1] <= self.vwap[-1]
        mfi_turning_up = self.mfi[-1] > self.mfi[-2]
        macd_positive = self.macd_hist[-1] > 0

        # Short entry conditions
        vwap_trending_down = self.vwap[-1] < self.vwap[-2]
        is_below_vwap = price < self.vwap[-1]
        bounce_to_vwap = self.data.High[-1] >= self.vwap[-1]
        mfi_turning_down = self.mfi[-1] < self.mfi[-2]
        macd_negative = self.macd_hist[-1] < 0

        if self.vwap[-1] > self.vwap_trigger[-1] and vwap_trending_up and is_above_vwap and pullback_to_vwap and mfi_turning_up and macd_positive:
            recent_low = np.min(self.data.Low[-self.sl_lookback:])
            stop_loss = recent_low * (1 - self.sl_buffer_pct)
            take_profit = price + (price - stop_loss) * self.risk_reward_ratio
            self.buy(sl=stop_loss, tp=take_profit)
        elif self.vwap[-1] < self.vwap_trigger[-1] and vwap_trending_down and is_below_vwap and bounce_to_vwap and mfi_turning_down and macd_negative:
            recent_high = np.max(self.data.High[-self.sl_lookback:])
            stop_loss = recent_high * (1 + self.sl_buffer_pct)
            take_profit = price - (stop_loss - price) * self.risk_reward_ratio
            self.sell(sl=stop_loss, tp=take_profit)

if __name__ == '__main__':
    data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    data.columns = [c.strip().title() for c in data.columns]
    data.sort_index(inplace=True)

    # Pre-calculate VWAP and its trigger since it requires a DatetimeIndex
    data['VWAP'] = ta.vwap(data.High, data.Low, data.Close, data.Volume, anchor="D")
    data['VWAP_Trigger'] = ta.sma(data.VWAP, MarketCipherBVwapMfiTrendStrategy.vwap_trigger_period)

    bt = Backtest(data, MarketCipherBVwapMfiTrendStrategy, cash=100000, commission=.002)
    stats = bt.run()
    print(stats)

    stats_dict = dict(stats)
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    for key, value in stats_dict.items():
        if isinstance(value, pd.Timestamp):
            stats_dict[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            stats_dict[key] = str(value)
        elif pd.isna(value):
            stats_dict[key] = None

    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)

    bt.plot(filename='results/market_cipher_b_vwap_mfi_trend.html')
