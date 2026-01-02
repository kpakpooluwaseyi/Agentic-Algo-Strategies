import pandas as pd
from backtesting import Strategy
from backtesting.lib import FractionalBacktest
import pandas_ta as ta
import json
import numpy as np

def SMA(array, n):
    """Return short moving average of `array` of period `n`."""
    return pd.Series(array).rolling(n).mean().to_numpy()

def BBANDS(close_array, length, std):
    close = pd.Series(close_array)
    bbands = ta.bbands(close=close, length=length, std=std)
    # Return a writable copy of the lower, middle, and upper bands
    return bbands.iloc[:, :3].values.T.copy()

class BollingerBandsReversal(Strategy):
    bb_length = 20
    bb_std = 2.0
    sl_buffer_pct = 0.01
    min_rr = 1.5
    trade_size = 0.1  # Trade 10% of equity
    slope_lookback = 10
    slope_threshold = 0.0005 # Normalized slope must be less than this

    def init(self):
        self.bbands = self.I(BBANDS, self.data.Close, self.bb_length, self.bb_std)
        self.bb_lower = self.bbands[0]
        self.bb_middle = self.bbands[1]
        self.bb_upper = self.bbands[2]

    def next(self):
        if self.position:
            return

        # Market condition filter: Check if the market is sideways/flat
        if len(self.data) < self.slope_lookback:
            return

        y = self.bb_middle[-self.slope_lookback:]
        x = np.arange(self.slope_lookback)
        slope = np.polyfit(x, y, 1)[0]
        normalized_slope = slope / self.data.Close[-1]

        if abs(normalized_slope) > self.slope_threshold:
            return

        price = self.data.Close[-1]

        # Sell setup
        prev_high = self.data.High[-2]
        prev_low = self.data.Low[-2]
        prev_open = self.data.Open[-2]
        prev_close = self.data.Close[-2]

        curr_high = self.data.High[-1]
        curr_low = self.data.Low[-1]
        curr_open = self.data.Open[-1]
        curr_close = self.data.Close[-1]

        # Bearish Engulfing Reversal from Upper Band
        touched_upper = prev_high >= self.bb_upper[-2]
        is_prev_bullish = prev_close > prev_open
        is_curr_bearish_engulfing = curr_close < prev_open and curr_open > prev_close

        if touched_upper and is_prev_bullish and is_curr_bearish_engulfing:
            sl = curr_high * (1 + self.sl_buffer_pct)
            tp = self.bb_lower[-1]

            # Validate SL/TP and R:R
            if sl > price and tp < price:
                risk = sl - price
                reward = price - tp
                if reward / risk >= self.min_rr:
                    self.sell(sl=sl, tp=tp, size=self.trade_size)

        # Bullish Engulfing Reversal from Lower Band
        touched_lower = prev_low <= self.bb_lower[-2]
        is_prev_bearish = prev_close < prev_open
        is_curr_bullish_engulfing = curr_close > prev_open and curr_open < prev_close

        if touched_lower and is_prev_bearish and is_curr_bullish_engulfing:
            sl = curr_low * (1 - self.sl_buffer_pct)
            tp = self.bb_upper[-1]

            # Validate SL/TP and R:R
            if sl < price and tp > price:
                risk = price - sl
                reward = tp - price
                if reward / risk >= self.min_rr:
                    self.buy(sl=sl, tp=tp, size=self.trade_size)

if __name__ == '__main__':
    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Data file not found. Please ensure 'data/BTC-USD-15m.csv' exists.")
        # As a fallback, create some synthetic data to allow the script to run
        print("Generating synthetic data for demonstration.")
        from backtesting.test import GOOG
        data = GOOG.copy()
        data = data.iloc[-2000:]

    # Sanitize column names
    data.columns = [col.strip().title() for col in data.columns]

    bt = FractionalBacktest(data, BollingerBandsReversal, cash=10000, commission=.002)

    stats = bt.run()

    print(stats)

    # Save the stats to a JSON file
    stats_dict = stats.to_dict()

    # Sanitize the stats dictionary
    def sanitize_stats(stats_obj):
        sanitized = {}
        for key, value in stats_obj.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                sanitized[key] = str(value)
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                # Skip pandas objects that are not easily serializable
                continue
            elif pd.isna(value):
                sanitized[key] = None
            elif isinstance(value, (int, float, str, bool, type(None))):
                sanitized[key] = value
            else:
                # Convert other numpy types to standard Python types
                try:
                    sanitized[key] = value.item()
                except AttributeError:
                    sanitized[key] = str(value) # Fallback to string
        return sanitized

    # Sanitize the main stats and the _strategy object if it exists
    final_stats = sanitize_stats(stats_dict)
    if '_strategy' in stats and hasattr(stats['_strategy'], '__dict__'):
        final_stats['_strategy'] = 'Strategy object is not serializable'

    # Remove non-serializable objects from stats
    for key in ['_equity_curve', '_trades']:
        if key in final_stats:
            del final_stats[key]

    with open('results/temp_result.json', 'w') as f:
        json.dump(final_stats, f, indent=4)

    try:
        bt.plot(filename='results/_bollinger_bands_reversal_.html', open_browser=False)
    except Exception as e:
        print(f"Could not generate plot: {e}")