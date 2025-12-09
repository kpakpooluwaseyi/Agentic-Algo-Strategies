
import json
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks

# Custom indicator functions must be defined at the module level.
def pass_through(series):
    return series

def bearish_engulfing(df):
    """Identifies bearish engulfing patterns."""
    prev_row = df.shift(1)
    return (
        (prev_row['Close'] > prev_row['Open']) &  # Previous candle is bullish
        (df['Open'] > prev_row['Close']) &       # Current candle opens higher
        (df['Close'] < prev_row['Open'])         # Current candle closes lower
    )

class FibonacciCenterPeakScalpStrategy(Strategy):
    # Optimization parameters
    peak_distance = 20
    fib_tolerance = 0.01
    min_rr_ratio = 5
    sl_buffer_pct = 0.01
    ema_period = 50

    # State tracking variables
    point_a = None
    point_b = None
    setup_active = False

    def init(self):
        # Pre-calculate swing points and EMA
        self.highs = self.I(pass_through, self.data.df['highs'])
        self.lows = self.I(pass_through, self.data.df['lows'])
        self.ema = self.I(pass_through, self.data.df['ema'])

    def next(self):
        current_price_high = self.data.High[-1]
        current_price_low = self.data.Low[-1]
        current_price_close = self.data.Close[-1]

        # Find the most recent confirmed swing high (A) and low (B)
        # Look back to find the last peak/trough that is not NaN
        high_peaks = np.where(~np.isnan(self.highs))[0]
        low_troughs = np.where(~np.isnan(self.lows))[0]

        if len(high_peaks) == 0 or len(low_troughs) == 0:
            return

        last_high_idx = high_peaks[-1]
        last_low_idx = low_troughs[-1]

        # Ensure A (high) happened before B (low) for a short setup
        if last_high_idx >= last_low_idx:
            return

        point_a_price = self.data.High[last_high_idx]
        point_b_price = self.data.Low[last_low_idx]

        # Calculate Fibonacci levels for the A-B move
        fib_50 = point_a_price - (point_a_price - point_b_price) * 0.5
        fib_61_8 = point_a_price - (point_a_price - point_b_price) * 0.618

        # --- Invalidation and Setup Logic ---
        # If we had a setup, but price went above the 61.8 level, invalidate
        if self.setup_active and current_price_high > fib_61_8:
            self.setup_active = False
            self.point_a = None
            self.point_b = None

        # Check if the price is touching the 50% fib level (Point C area)
        # And ensure the EMA is also in this area for confluence
        is_at_fib_50 = abs(current_price_high - fib_50) / fib_50 < self.fib_tolerance
        is_at_ema = abs(current_price_high - self.ema[-1]) / self.ema[-1] < self.fib_tolerance

        if is_at_fib_50 and is_at_ema:
            self.setup_active = True
            self.point_a = point_a_price
            self.point_b = point_b_price

        # --- Entry Logic ---
        if self.setup_active and self.position.is_flat:
            # Check for 15M bearish engulfing as 1M confirmation proxy
            if bearish_engulfing(self.data.df.iloc[-2:]).iloc[-1]:
                point_c_price = current_price_high

                # Risk Management
                stop_loss = point_c_price * (1 + self.sl_buffer_pct)

                # Exit Logic: TP is 50% of the B-C move
                take_profit = point_c_price - (point_c_price - self.point_b) * 0.5

                # Ensure R:R is met and SL/TP are valid
                risk = stop_loss - current_price_close
                reward = current_price_close - take_profit

                if risk > 0 and reward / risk >= self.min_rr_ratio:
                    if take_profit < current_price_close:
                        self.sell(sl=stop_loss, tp=take_profit)
                        # Reset state after entry
                        self.setup_active = False
                        self.point_a = None
                        self.point_b = None


def generate_synthetic_data():
    """Generates synthetic data with a clear A-B-C pattern for shorting."""
    # A-B Drop
    price = np.linspace(100, 90, 25)
    # B-C Retracement to 50%
    price = np.append(price, np.linspace(90, 95, 15))
    # Engulfing pattern at C
    price = np.append(price, [95.1, 93]) # Open higher, close much lower
    # C-D Drop to TP
    price = np.append(price, np.linspace(93, 92.5, 15)) # TP is 95 - (95-90)*0.5 = 92.5
    # Continued weakness (optional hold)
    price = np.append(price, np.linspace(92.5, 85, 25))

    dates = pd.to_datetime(pd.date_range('2023-01-01', periods=len(price), freq='15min'))

    df = pd.DataFrame(index=dates)
    df['Open'] = price
    df['High'] = price * 1.001
    df['Low'] = price * 0.999
    df['Close'] = price
    df['Volume'] = np.random.randint(100, 1000, size=len(price))

    # Create the bearish engulfing at the peak C
    peak_c_index = 40
    df.at[df.index[peak_c_index], 'Open'] = 95.1
    df.at[df.index[peak_c_index], 'High'] = 95.2
    df.at[df.index[peak_c_index], 'Close'] = 93
    df.at[df.index[peak_c_index], 'Low'] = 92.9

    # Make previous candle bullish for the pattern
    df.at[df.index[peak_c_index-1], 'Open'] = 94.8
    df.at[df.index[peak_c_index-1], 'Close'] = 95.0

    return df

if __name__ == '__main__':
    data = generate_synthetic_data()

    # --- Pre-processing ---
    # Find swing highs and lows using scipy
    highs = find_peaks(data['High'], distance=15)[0]
    lows = find_peaks(data['Low'] * -1, distance=15)[0]

    data['highs'] = np.nan
    data.iloc[highs, data.columns.get_loc('highs')] = data.iloc[highs]['High']

    data['lows'] = np.nan
    data.iloc[lows, data.columns.get_loc('lows')] = data.iloc[lows]['Low']

    # Add EMA
    data['ema'] = data['Close'].ewm(span=50, adjust=False).mean()

    bt = Backtest(data, FibonacciCenterPeakScalpStrategy, cash=100_000, commission=.002)

    stats = bt.optimize(
        peak_distance=range(15, 26, 5),
        fib_tolerance=[0.01, 0.015],
        min_rr_ratio=range(4, 6, 1),
        sl_buffer_pct=[0.01, 0.015],
        ema_period=range(40, 61, 10),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.min_rr_ratio > 2 # Example constraint
    )

    print(stats)

    # --- JSON Sanitization ---
    def sanitize_stats(stats_obj):
        sanitized = {}
        for key, value in stats_obj.items():
            if isinstance(value, (pd.Series, pd.DataFrame)):
                continue # Skip pandas objects
            if isinstance(value, (np.int64, np.int32)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                sanitized[key] = float(value)
            elif isinstance(value, (str, bool, int, float)) or value is None:
                sanitized[key] = value
        return sanitized

    clean_stats = sanitize_stats(stats)

    # --- Output ---
    import os
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'fibonacci_center_peak_scalp',
            'return': clean_stats.get('Return [%]', 0.0),
            'sharpe': clean_stats.get('Sharpe Ratio', 0.0),
            'max_drawdown': clean_stats.get('Max. Drawdown [%]', 0.0),
            'win_rate': clean_stats.get('Win Rate [%]', 0.0),
            'total_trades': clean_stats.get('# Trades', 0)
        }, f, indent=2)

    try:
        bt.plot(filename="results/fibonacci_center_peak_scalp.html")
    except TypeError as e:
        print(f"Could not generate plot due to a known backtesting.py issue: {e}")
