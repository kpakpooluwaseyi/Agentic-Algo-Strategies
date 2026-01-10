
from backtesting import Strategy
from backtesting.lib import crossover

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import pandas_ta as ta

def preprocess_data(df, atr_period=14, volatility_threshold_multiplier=2.0, **params):
    df = df.copy()
    return df


class VolatilitySpikeMeanReversion(Strategy):
    short_atr_period = 10
    long_atr_period = 50
    volatility_multiplier = 2.5
    reversal_confirmation_window = 3
    stop_loss_atr_multiplier = 2.0
    take_profit_atr_multiplier = 3.5

    def init(self):
        self.short_atr = self.I(ta.atr, pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), length=self.short_atr_period)
        self.long_atr = self.I(ta.atr, pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), length=self.long_atr_period)
        self.spike_detected = False
        self.spike_direction = 0  # 1 for bullish, -1 for bearish
        self.spike_bar_index = 0


    def next(self):
        if self.spike_detected:
            # Reversal logic
            if self.spike_direction == 1 and self.data.Close[-1] < self.data.Open[-1]:  # Bearish reversal candle
                self.sell(
                    sl=self.data.Close[-1] + self.short_atr[-1] * self.stop_loss_atr_multiplier,
                    tp=self.data.Close[-1] - self.short_atr[-1] * self.take_profit_atr_multiplier
                )
                self.spike_detected = False
            elif self.spike_direction == -1 and self.data.Close[-1] > self.data.Open[-1]:  # Bullish reversal candle
                self.buy(
                    sl=self.data.Close[-1] - self.short_atr[-1] * self.stop_loss_atr_multiplier,
                    tp=self.data.Close[-1] + self.short_atr[-1] * self.take_profit_atr_multiplier
                )
                self.spike_detected = False

            # Timeout for the spike signal
            if len(self.data.Close) - self.spike_bar_index > self.reversal_confirmation_window:
                self.spike_detected = False

        # Spike detection logic
        if not self.position and not self.spike_detected:
            if self.short_atr[-1] > self.long_atr[-1] * self.volatility_multiplier:
                self.spike_detected = True
                self.spike_bar_index = len(self.data.Close) -1
                if self.data.Close[-1] > self.data.Open[-1]:
                    self.spike_direction = 1  # Bullish spike
                else:
                    self.spike_direction = -1  # Bearish spike


if __name__ == '__main__':
    import pandas as pd
    from backtesting import Backtest
    import json

    # Load sample data
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("No data file found. Create data/BTC-USD-15m.csv or modify the path.")
        exit(1)

    # Preprocess if needed
    df = preprocess_data(df)

    # Run backtest
    bt = Backtest(df, VolatilitySpikeMeanReversion, cash=100_000, commission=.002)
    stats = bt.run()

    # Save results
    with open('results/temp_result.json', 'w') as f:
        json.dump(stats.to_dict(), f, indent=4)

    print(stats)

    # Show plot
    bt.plot(filename='results/volatility_spike_mean_reversion.html', open_browser=False)
