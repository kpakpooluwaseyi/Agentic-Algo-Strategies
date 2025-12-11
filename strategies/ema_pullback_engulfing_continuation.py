from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
import numpy as np
import json
from scipy.signal import find_peaks

def EMA(series, n):
    """Returns the EMA of a given series."""
    return pd.Series(series).ewm(span=n, adjust=False).mean().values

class EmaPullbackEngulfingContinuationStrategy(Strategy):
    ema_period = 50
    risk_reward_ratio = 2
    swing_lookback = 100

    def init(self):
        self.ema = self.I(EMA, self.data.Close, self.ema_period)
        self.highs = self.data.High
        self.lows = self.data.Low
        self.closes = self.data.Close
        self.opens = self.data.Open
        # Pre-calculate swing points
        self.swing_highs = self.I(self._swing_points, self.data.High, distance=10, is_high=True)
        self.swing_lows = self.I(self._swing_points, self.data.Low, distance=10, is_high=False)
        # State tracking for Market Structure Break
        self.msb_up = False
        self.msb_down = False

    def _swing_points(self, series, distance, is_high):
        if is_high:
            peaks, _ = find_peaks(series, distance=distance)
            points = np.zeros_like(series, dtype=np.float64)
            points[peaks] = series[peaks]
        else:
            troughs, _ = find_peaks(-series, distance=distance)
            points = np.zeros_like(series, dtype=np.float64)
            points[troughs] = series[troughs]
        return points

    def next(self):
        # Find the most recent swing high/low
        previous_swing_high = self.swing_highs[-2] if len(self.swing_highs) > 1 and self.swing_highs[-2] > 0 else -np.inf
        previous_swing_low = self.swing_lows[-2] if len(self.swing_lows) > 1 and self.swing_lows[-2] > 0 else np.inf

        # Detect new Market Structure Breaks and set state
        if self.closes[-1] > previous_swing_high:
            self.msb_up = True
            self.msb_down = False
        elif self.closes[-1] < previous_swing_low:
            self.msb_down = True
            self.msb_up = False

        # Long Entry Conditions
        is_uptrend = self.closes[-1] > self.ema[-1]
        is_pullback_long = self.closes[-2] < self.opens[-2] and self.closes[-3] < self.opens[-3] # Two red candles

        is_bullish_engulfing = (
            self.closes[-1] > self.opens[-1] and
            self.closes[-2] < self.opens[-2] and
            self.closes[-1] > self.opens[-2] and
            self.opens[-1] < self.closes[-2]
        )

        # Check if the engulfing pattern forms the swing low of the pullback
        is_swing_low = self.lows[-2] < self.lows[-3] and self.lows[-1] >= self.lows[-2]

        if self.msb_up and is_uptrend and is_pullback_long and is_bullish_engulfing and is_swing_low and not self.position:
            stop_loss = min(self.lows[-1], self.lows[-2])
            take_profit = self.closes[-1] + (self.closes[-1] - stop_loss) * self.risk_reward_ratio
            if take_profit > self.closes[-1]:
                self.buy(sl=stop_loss, tp=take_profit)
                self.msb_up = False # Reset state after entry

        # Short Entry Conditions
        is_downtrend = self.closes[-1] < self.ema[-1]
        is_pullback_short = self.closes[-2] > self.opens[-2] and self.closes[-3] > self.opens[-3] # Two green candles

        is_bearish_engulfing = (
            self.closes[-1] < self.opens[-1] and
            self.closes[-2] > self.opens[-2] and
            self.opens[-1] > self.closes[-2] and
            self.closes[-1] < self.opens[-2]
        )

        # Check if the engulfing pattern forms the swing high of the pullback
        is_swing_high = self.highs[-2] > self.highs[-3] and self.highs[-1] <= self.highs[-2]

        if self.msb_down and is_downtrend and is_pullback_short and is_bearish_engulfing and is_swing_high and not self.position:
            stop_loss = max(self.highs[-1], self.highs[-2])
            take_profit = self.closes[-1] - (stop_loss - self.closes[-1]) * self.risk_reward_ratio
            if take_profit < self.closes[-1]:
                self.sell(sl=stop_loss, tp=take_profit)
                self.msb_down = False # Reset state after entry

if __name__ == '__main__':
    from backtesting.test import GOOG
    data = GOOG.copy()

    bt = Backtest(data, EmaPullbackEngulfingContinuationStrategy, cash=10000, commission=.002)

    stats = bt.optimize(
        ema_period=range(50, 101, 50),
        swing_lookback=range(100, 201, 100),
        risk_reward_ratio=[2],
        maximize='Sharpe Ratio'
    )

    import os
    os.makedirs('results', exist_ok=True)

    results_dict = {
        'strategy_name': 'ema_pullback_engulfing_continuation',
        'return': stats.get('Return [%]', 0.0),
        'sharpe': stats.get('Sharpe Ratio', 0.0),
        'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
        'win_rate': stats.get('Win Rate [%]', 0.0),
        'total_trades': stats.get('# Trades', 0)
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    try:
        bt.plot(filename='results/plot.html')
    except Exception as e:
        print(f"Could not generate plot: {e}")