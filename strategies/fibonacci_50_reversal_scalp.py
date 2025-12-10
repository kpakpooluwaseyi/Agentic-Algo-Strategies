from backtesting import Backtest, Strategy
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import json

def generate_synthetic_data():
    """
    Generates synthetic data that specifically creates a textbook
    'Level Drop -> 50% Retracement -> Reversal' pattern for short entries.
    """
    n_points = 200
    index = pd.to_datetime(pd.date_range('2023-01-01', periods=n_points, freq='1min'))

    # Start with some noise
    close_np = 100 + np.random.randn(n_points).cumsum() * 0.1

    # Create the 'Level Drop'
    # High Point (A)
    close_np[45:50] = np.linspace(close_np[44], 105, 5)
    # Low Point (B)
    close_np[50:60] = np.linspace(105, 95, 10)

    # Create the Retracement to the 50% level
    # 50% level is 95 + (105-95)*0.5 = 100
    # High Point of Retracement (C)
    close_np[60:70] = np.linspace(95, 100, 10)

    # Create a reversal pattern (bearish engulfing)
    # Previous candle close is ~100.
    # Engulfing candle: Open > 100, Close < previous open (~99)
    close_np[70] = 100.1 # Open of engulfing
    close_np[71] = 98   # Close of engulfing

    # Continue the downtrend to hit a potential TP
    close_np[72:82] = np.linspace(98, 96, 10)

    # Fill the rest with noise
    close_np[82:] = close_np[81] + np.random.randn(n_points - 82).cumsum() * 0.1

    # Create DataFrame and OHLC columns correctly
    data = pd.DataFrame(index=index)
    data['Close'] = close_np
    data['Open'] = data['Close'].shift(1).bfill()
    data['High'] = data[['Open', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Close']].min(axis=1)

    # Inject wicks for the pattern
    data.at[data.index[50], 'High'] = 105.2 # Peak of Level Drop (A)
    data.at[data.index[60], 'Low'] = 94.8  # Low of Level Drop (B)
    data.at[data.index[70], 'High'] = 100.3 # Peak of Retracement / Rejection (C)

    return data

# Custom indicator function to find peaks (swing highs/lows)
def find_swings(array, distance):
    # scipy.signal.find_peaks finds local maxima.
    # To find swing lows, we invert the series.
    peaks, _ = find_peaks(array, distance=distance)
    troughs, _ = find_peaks(-array, distance=distance)

    # Create a signal array: 1 for peak, -1 for trough, 0 otherwise
    signals = np.zeros(len(array))
    signals[peaks] = 1
    signals[troughs] = -1
    return signals

class Fibonacci50ReversalScalpStrategy(Strategy):
    # Optimizable parameters
    peak_distance = 15 # Simulates 15M swing detection on 1M data
    fib_tolerance = 0.03 # 3% tolerance for hitting the 50% fib level
    sl_buffer = 0.001 # Buffer for stop-loss in percentage
    min_rr = 4.0 # Minimum Risk-to-Reward ratio

    def init(self):
        # Use a custom indicator to find swing points
        self.swing_points = self.I(find_swings, self.data.Close, self.peak_distance)

        # State machine variables
        self.level_drop_high = None
        self.level_drop_low = None
        self.reversal_confirmation = False

    def next(self):
        # If a position is already open, do nothing.
        if self.position:
            return

        current_index = len(self.data.Close) - 1

        # === STATE 1: Find a Level Drop (Swing High followed by Swing Low) ===
        if self.level_drop_high is None:
            swings = np.where(self.swing_points != 0)[0]
            if len(swings) >= 2:
                last_swing_idx = swings[-1]
                prev_swing_idx = swings[-2]

                if self.swing_points[prev_swing_idx] == 1 and self.swing_points[last_swing_idx] == -1:
                    # A swing high followed by a swing low is found
                    self.level_drop_high = (prev_swing_idx, self.data.High[prev_swing_idx])
                    self.level_drop_low = (last_swing_idx, self.data.Low[last_swing_idx])
            return

        # === STATE 2: Wait for retracement to the 50% Fib level ===
        if not self.reversal_confirmation:
            level_high_price = self.level_drop_high[1]
            level_low_price = self.level_drop_low[1]

            fib_50 = level_low_price + (level_high_price - level_low_price) * 0.5
            upper_bound = fib_50 * (1 + self.fib_tolerance)
            lower_bound = fib_50 * (1 - self.fib_tolerance)

            current_high = self.data.High[-1]

            # Invalidate if price breaks the initial high before confirmation
            if current_high > level_high_price:
                self.level_drop_high = None
                self.level_drop_low = None
                return

            # Check if price has hit the 50% zone
            if current_high >= lower_bound:
                 self.reversal_confirmation = True
            return

        # === STATE 3: Look for bearish reversal pattern (Engulfing) ===
        if self.reversal_confirmation:
            prev_open = self.data.Open[-2]
            prev_close = self.data.Close[-2]
            curr_open = self.data.Open[-1]
            curr_close = self.data.Close[-1]

            is_bullish_prev = prev_close > prev_open
            is_bearish_curr = curr_close < curr_open
            is_engulfing = curr_open >= prev_close and curr_close <= prev_open

            if is_bullish_prev and is_bearish_curr and is_engulfing:
                entry_price = self.data.Close[-1]
                rejection_high = self.data.High[-1]

                stop_loss = rejection_high * (1 + self.sl_buffer)

                tp_fib_high = rejection_high
                tp_fib_low = self.level_drop_low[1]
                take_profit = tp_fib_high - (tp_fib_high - tp_fib_low) * 0.5

                if stop_loss > entry_price and take_profit < entry_price:
                    rr = (entry_price - take_profit) / (stop_loss - entry_price)
                    if rr >= self.min_rr:
                        self.sell(sl=stop_loss, tp=take_profit)

            # If the current candle closes higher than the entry zone, invalidate
            if self.data.Close[-1] > self.level_drop_high[1]:
                 self.level_drop_high = None
                 self.level_drop_low = None
                 self.reversal_confirmation = False

if __name__ == '__main__':
    # Generate synthetic data
    data = generate_synthetic_data()

    # Initialize the backtest
    bt = Backtest(data, Fibonacci50ReversalScalpStrategy, cash=10000, commission=.002)

    # Optimize the strategy
    stats = bt.optimize(
        peak_distance=range(10, 25, 5),
        fib_tolerance=[i/100 for i in range(1, 6)], # 1% to 5%
        sl_buffer=[i/1000 for i in range(1, 5)], # 0.1% to 0.4%
        min_rr=[i/2 for i in range(4, 11)], # 2.0 to 5.0
        maximize='Sharpe Ratio',
        constraint=lambda p: p.min_rr > 1 # Ensure RR is at least 1
    )

    print("Best stats:")
    print(stats)

    # Sanitize results for JSON output
    results = {
        'strategy_name': 'fibonacci_50_reversal_scalp',
        'return': stats.get('Return [%]', None),
        'sharpe': stats.get('Sharpe Ratio', None),
        'max_drawdown': stats.get('Max. Drawdown [%]', None),
        'win_rate': stats.get('Win Rate [%]', None),
        'total_trades': stats.get('# Trades', 0)
    }

    # Ensure results directory exists
    import os
    os.makedirs('results', exist_ok=True)

    # Save the results to a JSON file
    with open('results/temp_result.json', 'w') as f:
        # Replace NaN with None for valid JSON
        cleaned_results = {k: (None if pd.isna(v) else v) for k, v in results.items()}
        json.dump(cleaned_results, f, indent=2)

    print("Results saved to results/temp_result.json")

    # Generate the plot
    try:
        bt.plot(filename='results/fibonacci_50_reversal_scalp_plot.html')
        print("Plot saved to results/fibonacci_50_reversal_scalp_plot.html")
    except Exception as e:
        print(f"Could not generate plot: {e}")
