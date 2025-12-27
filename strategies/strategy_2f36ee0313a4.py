
import pandas as pd
import numpy as np
import json
import os
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks

# --- INDICATOR HELPER FUNCTIONS ---

def fvg_indicator(high: pd.Series, low: pd.Series) -> tuple:
    """
    Identifies Fair Value Gaps (FVGs) using pandas Series for safe shifting.
    The input arrays from self.I() must be converted to Series.
    """
    high = pd.Series(high)
    low = pd.Series(low)

    high_prev = high.shift(1)
    low_prev = low.shift(1)
    high_next = high.shift(-1)
    low_next = low.shift(-1)

    bullish_mask = high_prev < low_next
    bearish_mask = low_prev > high_next

    bullish_fvg_top = np.where(bullish_mask, low_next, np.nan)
    bullish_fvg_bottom = np.where(bullish_mask, high_prev, np.nan)
    bearish_fvg_top = np.where(bearish_mask, low_prev, np.nan)
    bearish_fvg_bottom = np.where(bearish_mask, high_next, np.nan)

    return bullish_fvg_top, bullish_fvg_bottom, bearish_fvg_top, bearish_fvg_bottom


# --- STRATEGY CLASS ---

class PullbackTradingStrategy(Strategy):
    swing_distance = 15
    lookback_window = 250
    min_slope_pct = 0.5
    pullback_timeout = 50
    sl_buffer_pct = 1.0

    def init(self):
        self.bullish_fvg_top, self.bullish_fvg_bottom, \
        self.bearish_fvg_top, self.bearish_fvg_bottom = self.I(
            fvg_indicator, self.data.High, self.data.Low, name="FVG"
        )
        self.break_direction = None
        self.breakout_bar = None
        self.fvg_spotted = None
        self.trend_start_price = None
        self.fib_levels = {}
        self.tp_level = 1

    def _find_trendline(self, data, trend_type):
        prices = data['High'] if trend_type == 'down' else data['Low']
        find_peaks_arg = prices if trend_type == 'down' else -prices

        peaks_idx, _ = find_peaks(find_peaks_arg, distance=self.swing_distance)
        points = [{'index': i, 'price': prices[i]} for i in peaks_idx]

        if len(points) < 2: return None, None
        points.sort(key=lambda x: x['index'], reverse=True)

        p1 = points[0]
        for i in range(1, len(points)):
            p2 = points[i]
            price_condition = p1['price'] < p2['price'] if trend_type == 'down' else p1['price'] > p2['price']
            if price_condition:
                if (p1['index'] - p2['index']) == 0: continue
                slope = (p1['price'] - p2['price']) / (p1['index'] - p2['index'])
                min_slope_val = (p2['price'] * self.min_slope_pct / 100) / (p1['index'] - p2['index']) if (p1['index'] - p2['index']) != 0 else 0
                slope_condition = slope < -min_slope_val if trend_type == 'down' else slope > min_slope_val
                if slope_condition:
                    trendline_val = p1['price'] + slope * (len(data['Close']) - 1 - p1['index'])
                    return trendline_val, p2['price']
        return None, None

    def next(self):
        current_bar_index = len(self.data.Close) - 1

        if not self.position and self.fib_levels:
            self._reset_trade_vars()

        if self.position:
            if self.position.is_long:
                if self.tp_level == 1 and self.data.High[-1] >= self.fib_levels[0.5]:
                    self.position.close(portion=0.25)
                    self.tp_level = 2
                elif self.tp_level == 2 and self.data.High[-1] >= self.fib_levels[0.618]:
                    self.position.close(portion=0.5)
                    self.tp_level = 3
                elif self.tp_level == 3 and self.data.High[-1] >= self.fib_levels[0.78]:
                    self.position.close()
            else: # Short
                if self.tp_level == 1 and self.data.Low[-1] <= self.fib_levels[0.5]:
                    self.position.close(portion=0.25)
                    self.tp_level = 2
                elif self.tp_level == 2 and self.data.Low[-1] <= self.fib_levels[0.618]:
                    self.position.close(portion=0.5)
                    self.tp_level = 3
                elif self.tp_level == 3 and self.data.Low[-1] <= self.fib_levels[0.78]:
                    self.position.close()
            return

        if current_bar_index < self.lookback_window:
            return

        window_data = {'High': self.data.High[-self.lookback_window:], 'Low': self.data.Low[-self.lookback_window:], 'Close': self.data.Close[-self.lookback_window:]}

        if self.break_direction is None:
            downtrend_line, trend_start = self._find_trendline(window_data, 'down')
            if downtrend_line and self.data.Close[-1] > downtrend_line:
                self.break_direction = 'long'
                self.breakout_bar = current_bar_index
                self.trend_start_price = trend_start

            uptrend_line, trend_start = self._find_trendline(window_data, 'up')
            if uptrend_line and self.data.Close[-1] < uptrend_line:
                self.break_direction = 'short'
                self.breakout_bar = current_bar_index
                self.trend_start_price = trend_start
        else:
            if current_bar_index > self.breakout_bar + self.pullback_timeout:
                self._reset_setup_vars()
                return

            if self.break_direction == 'long':
                if pd.notna(self.bullish_fvg_top[-2]) and self.bullish_fvg_top[-2] < self.data.Close[self.breakout_bar]:
                    self.fvg_spotted = (self.bullish_fvg_top[-2], self.bullish_fvg_bottom[-2])
                if self.fvg_spotted and self.data.Low[-1] <= self.fvg_spotted[0]:
                    sl = self.fvg_spotted[1] * (1 - self.sl_buffer_pct / 100)
                    if self.data.Close[-1] > sl:
                        self.buy(sl=sl)
                        self._calculate_fib_levels(self.trend_start_price, self.data.Low[-1], 'long')
                        self._reset_setup_vars()
            elif self.break_direction == 'short':
                if pd.notna(self.bearish_fvg_bottom[-2]) and self.bearish_fvg_bottom[-2] > self.data.Close[self.breakout_bar]:
                    self.fvg_spotted = (self.bearish_fvg_top[-2], self.bearish_fvg_bottom[-2])
                if self.fvg_spotted and self.data.High[-1] >= self.fvg_spotted[1]:
                    sl = self.fvg_spotted[0] * (1 + self.sl_buffer_pct / 100)
                    if self.data.Close[-1] < sl:
                        self.sell(sl=sl)
                        self._calculate_fib_levels(self.trend_start_price, self.data.High[-1], 'short')
                        self._reset_setup_vars()

    def _calculate_fib_levels(self, start_price, end_price, direction):
        if start_price is None or end_price is None: return
        price_range = abs(start_price - end_price)
        if direction == 'long':
            self.fib_levels = {0.5: end_price + price_range * 0.5, 0.618: end_price + price_range * 0.618, 0.78: end_price + price_range * 0.78}
        else:
            self.fib_levels = {0.5: end_price - price_range * 0.5, 0.618: end_price - price_range * 0.618, 0.78: end_price - price_range * 0.78}

    def _reset_setup_vars(self):
        self.break_direction = None
        self.breakout_bar = None
        self.fvg_spotted = None
        self.trend_start_price = None

    def _reset_trade_vars(self):
        self.fib_levels = {}
        self.tp_level = 1

# --- MAIN EXECUTION BLOCK ---

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        print(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        data.columns = [col.strip().capitalize() for col in data.columns]

        print("Running backtest...")
        bt = Backtest(data, PullbackTradingStrategy, cash=100_000, commission=.002)

        stats = bt.run()
        print(stats)

        os.makedirs('results', exist_ok=True)
        plot_filename = 'results/strategy_2f36ee0313a4.html'
        json_filename = 'results/temp_result.json'

        try:
            bt.plot(filename=plot_filename, open_browser=False)
            print(f"Backtest plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot: {e}")

        stats_dict = stats.to_dict()
        for key in ['_strategy', '_equity_curve', '_trades']:
            if key in stats_dict:
                del stats_dict[key]

        for key, value in list(stats_dict.items()):
            if isinstance(value, pd.Timestamp):
                stats_dict[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                stats_dict[key] = str(value)
            elif pd.isna(value):
                stats_dict[key] = None
            elif isinstance(value, (np.integer, np.int64)):
                stats_dict[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                stats_dict[key] = float(value)

        with open(json_filename, 'w') as f:
            json.dump(stats_dict, f, indent=4)
        print(f"Backtest stats saved to {json_filename}")
