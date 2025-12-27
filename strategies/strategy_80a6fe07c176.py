import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import json
import os

class BeatTheMarketMakerStrategy(Strategy):
    ema_period = 50
    peak_lookback = 10
    risk_reward_ratio = 2.0
    sl_buffer_pct = 0.01

    def init(self):
        # Indicators
        self.ema = self.I(ta.ema, pd.Series(self.data.Close), length=self.ema_period)

        # State machine variables for M-pattern (bearish)
        self.m_peak1 = None
        self.m_trough = None
        self.m_peak2 = None

        # State machine variables for W-pattern (bullish)
        self.w_trough1 = None
        self.w_peak = None
        self.w_trough2 = None

    def _is_swing_high(self, index):
        """More robustly check if the candle at `index` is a swing high."""
        if index < self.peak_lookback or index >= len(self.data.High) - self.peak_lookback:
            return False

        window = self.data.High[index - self.peak_lookback : index + self.peak_lookback + 1]
        return self.data.High[index] == max(window)

    def _is_swing_low(self, index):
        """More robustly check if the candle at `index` is a swing low."""
        if index < self.peak_lookback or index >= len(self.data.Low) - self.peak_lookback:
            return False

        window = self.data.Low[index - self.peak_lookback : index + self.peak_lookback + 1]
        return self.data.Low[index] == min(window)

    def next(self):
        # Alias current index for readability
        current_index = len(self.data.Close) - 1 - self.peak_lookback
        if current_index < 0:
            return

        # --- M-Pattern (Bearish Reversal) Detection ---
        # State 1: Look for the first peak above the EMA
        if self.m_peak1 is None:
            if self._is_swing_high(current_index) and self.data.High[current_index] > self.ema[current_index]:
                self.m_peak1 = (current_index, self.data.High[current_index])

        # State 2: After peak 1, look for a swing low
        elif self.m_trough is None:
            if self._is_swing_low(current_index) and self.data.Low[current_index] > self.m_peak1[1]:
                 # Invalid pattern, reset
                self.m_peak1 = None
            elif self._is_swing_low(current_index):
                self.m_trough = (current_index, self.data.Low[current_index])

        # State 3: After the trough, look for a second peak, ideally near the first one
        elif self.m_peak2 is None:
            if self._is_swing_high(current_index) and self.data.High[current_index] > self.ema[current_index]:
                if self.data.High[current_index] >= self.m_peak1[1] * 1.01: # Ensure second peak is a significant new high
                    self.m_peak2 = (current_index, self.data.High[current_index])
                else: # second peak is not high enough, reset
                    self.m_peak1, self.m_trough = None, None
            elif self.data.Low[current_index] < self.m_trough[1]: # Price made a new low, invalidating the M
                self.m_peak1, self.m_trough = None, None

        # --- W-Pattern (Bullish Reversal) Detection ---
        # State 1: Look for the first trough below the EMA
        if self.w_trough1 is None:
            if self._is_swing_low(current_index) and self.data.Low[current_index] < self.ema[current_index]:
                self.w_trough1 = (current_index, self.data.Low[current_index])

        # State 2: After trough 1, look for a swing high
        elif self.w_peak is None:
            if self._is_swing_high(current_index) and self.data.High[current_index] < self.w_trough1[1]:
                # Invalid pattern, reset
                self.w_trough1 = None
            elif self._is_swing_high(current_index):
                self.w_peak = (current_index, self.data.High[current_index])

        # State 3: After the peak, look for a second trough, ideally near the first one
        elif self.w_trough2 is None:
            if self._is_swing_low(current_index) and self.data.Low[current_index] < self.ema[current_index]:
                if self.data.Low[current_index] <= self.w_trough1[1] * 0.99: # Ensure second trough is a significant new low
                    self.w_trough2 = (current_index, self.data.Low[current_index])
                else: # second trough is not low enough, reset
                    self.w_trough1, self.w_peak = None, None
            elif self.data.High[current_index] > self.w_peak[1]: # Price made a new high, invalidating the W
                self.w_trough1, self.w_peak = None, None

        # --- Trade Execution ---
        if self.position:
            return # Don't enter a new trade if one is already open

        # M-Pattern trade entry
        if self.m_peak2 and self.data.Close[-1] < self.m_trough[1]:
            # Entry confirmed, place short trade
            sl_price = max(self.m_peak1[1], self.m_peak2[1]) * (1 + self.sl_buffer_pct)
            tp_price = self.data.Close[-1] - (sl_price - self.data.Close[-1]) * self.risk_reward_ratio

            if sl_price > self.data.Close[-1] and tp_price < self.data.Close[-1]:
                 self.sell(sl=sl_price, tp=tp_price)

            # Reset state for both patterns
            self.m_peak1, self.m_trough, self.m_peak2 = None, None, None
            self.w_trough1, self.w_peak, self.w_trough2 = None, None, None

        # W-Pattern trade entry
        if self.w_trough2 and self.data.Close[-1] > self.w_peak[1]:
            # Entry confirmed, place long trade
            sl_price = min(self.w_trough1[1], self.w_trough2[1]) * (1 - self.sl_buffer_pct)
            tp_price = self.data.Close[-1] + (self.data.Close[-1] - sl_price) * self.risk_reward_ratio

            if sl_price < self.data.Close[-1] and tp_price > self.data.Close[-1]:
                self.buy(sl=sl_price, tp=tp_price)

            # Reset state for both patterns
            self.w_trough1, self.w_peak, self.w_trough2 = None, None, None
            self.m_peak1, self.m_trough, self.m_peak2 = None, None, None

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Ensure column names are in the format Backtesting.py expects
        data.columns = [c.strip().title() for c in data.columns]
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')] # Drop unnamed columns
    else:
        raise FileNotFoundError(f"Data file not found at {data_path}")

    bt = Backtest(data, BeatTheMarketMakerStrategy, cash=100_000, commission=.002)

    stats = bt.run()
    print(stats)

    # Save results
    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        """Sanitizes the stats dictionary for JSON serialization."""
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Series, pd.DataFrame, Strategy)):
                continue
            elif pd.isna(value):
                sanitized[key] = None
            elif isinstance(value, (int, float, str, bool)):
                sanitized[key] = value
            elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                 sanitized[key] = str(value)
            else:
                try:
                    # Attempt to convert numpy types to python native types
                    sanitized[key] = value.item()
                except (AttributeError, TypeError):
                    sanitized[key] = str(value)
        return sanitized

    clean_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(clean_stats, f, indent=4)

    print("Backtest results saved to results/temp_result.json")

    try:
        bt.plot(filename='results/strategy_80a6fe07c176.html')
    except Exception as e:
        print(f"Could not generate plot: {e}")
