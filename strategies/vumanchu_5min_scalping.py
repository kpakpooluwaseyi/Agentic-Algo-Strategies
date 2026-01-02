import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def vumanchu_indicator(high, low, close, volume):
    """
    Proxy for the VuManchu Cipher B+ Divergences indicator, as src.indicators.vumanchu is not available.
    This implementation uses MACD to generate signals based on the strategy description.
    - Green dot (1): Bullish MACD crossover while the MACD line is above zero.
    - Red dot (-1): Bearish MACD crossover while the MACD line is above zero.
    """
    # Convert numpy arrays to pandas Series for pandas-ta compatibility
    high, low, close, volume = (pd.Series(x) for x in (high, low, close, volume))

    # Calculate MACD
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    macd_line = macd['MACD_12_26_9']
    signal_line = macd['MACDs_12_26_9']

    # Generate signals
    signal = np.zeros(len(close))

    # Bullish crossover: MACD line crosses above signal line
    bullish_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))

    # Bearish crossover: MACD line crosses below signal line
    bearish_cross = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    # Green dot: Bullish crossover while MACD line is above zero
    signal[bullish_cross & (macd_line > 0)] = 1

    # Red dot: Bearish crossover while MACD line is above zero
    signal[bearish_cross & (macd_line > 0)] = -1

    return signal

class Vumanchu5MinScalping(Strategy):
    # NOTE: Default EMA periods from the strategy description may result in
    # zero trades on the provided 15m dataset due to the longer timeframe.
    ema_fast_period = 50
    ema_slow_period = 200
    swing_lookback = 20
    max_consecutive_trades = 5
    sl_ema_proximity_pct = 0.01

    def init(self):
        self.ema_fast = self.I(ta.ema, pd.Series(self.data.Close), length=self.ema_fast_period)
        self.ema_slow = self.I(ta.ema, pd.Series(self.data.Close), length=self.ema_slow_period)
        self.vumanchu = self.I(vumanchu_indicator,
                               self.data.High,
                               self.data.Low,
                               self.data.Close,
                               self.data.Volume)
        self.long_trend = False
        self.short_trend = False
        self.long_trade_counter = 0
        self.short_trade_counter = 0

    def next(self):
        if not self.position:
            # Golden Cross: 50 EMA crosses above 200 EMA
            if crossover(self.ema_fast, self.ema_slow):
                self.long_trend = True
                self.short_trend = False
                self.long_trade_counter = 0
                self.short_trade_counter = 0
            # Death Cross: 50 EMA crosses below 200 EMA
            elif crossover(self.ema_slow, self.ema_fast):
                self.long_trend = False
                self.short_trend = True
                self.long_trade_counter = 0
                self.short_trade_counter = 0

        # Trend weakening logic
        if self.long_trend and self.data.Close[-1] < self.ema_slow[-1]:
            self.long_trend = False
        if self.short_trend and self.data.Close[-1] > self.ema_slow[-1]:
            self.short_trend = False

        # Long Entry
        if self.long_trend and self.vumanchu == 1 and self.long_trade_counter < self.max_consecutive_trades:
            if not self.position:
                swing_low = self.data.Low[-self.swing_lookback:].min()
                sl = swing_low
                # If the swing low is close to the EMA, place the SL just underneath the EMA
                if abs(swing_low - self.ema_slow[-1]) / self.ema_slow[-1] < self.sl_ema_proximity_pct:
                    sl = self.ema_slow[-1] * 0.99
                tp = self.data.Close[-1] + (self.data.Close[-1] - sl)
                self.buy(sl=sl, tp=tp)
                self.long_trade_counter += 1

        # Short Entry
        elif self.short_trend and self.vumanchu == -1 and self.short_trade_counter < self.max_consecutive_trades:
            if not self.position:
                swing_high = self.data.High[-self.swing_lookback:].max()
                sl = swing_high
                # If the swing high is too far, use the EMA as the stop loss level
                if abs(swing_high - self.ema_slow[-1]) / self.ema_slow[-1] > self.sl_ema_proximity_pct:
                    sl = self.ema_slow[-1] * 1.01
                tp = self.data.Close[-1] - (sl - self.data.Close[-1])
                self.sell(sl=sl, tp=tp)
                self.short_trade_counter += 1

if __name__ == '__main__':
    # Load data
    data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    # Correctly format column names
    data.columns = [col.strip().capitalize() for col in data.columns]
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]


    # Backtest
    bt = Backtest(data, Vumanchu5MinScalping, cash=1_000_000, commission=.002, finalize_trades=True)
    stats = bt.run()
    print(stats)
    bt.plot(filename='results/vumanchu_5min_scalping.html')

    # Save results
    import json

    def sanitize_stats(stats):
        if isinstance(stats, dict):
            return {k: sanitize_stats(v) for k, v in stats.items() if not isinstance(v, (pd.DataFrame, Strategy))}
        elif isinstance(stats, list):
            return [sanitize_stats(i) for i in stats]
        elif isinstance(stats, pd.Timestamp):
            return stats.isoformat()
        elif isinstance(stats, pd.Timedelta):
            return str(stats)
        elif pd.isna(stats):
            return None
        elif isinstance(stats, (np.integer, np.int64)):
            return int(stats)
        elif isinstance(stats, (np.floating, np.float64)):
            return float(stats)
        else:
            return stats

    with open('results/temp_result.json', 'w') as f:
        sanitized_stats = sanitize_stats(stats.to_dict())
        json.dump(sanitized_stats, f, indent=4)
