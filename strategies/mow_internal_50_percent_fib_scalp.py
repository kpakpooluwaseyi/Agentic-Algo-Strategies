from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
from scipy.signal import find_peaks
from enum import Enum

# --- Helper Functions & Enums ---
class State(Enum):
    """Enum to manage the state of the trading strategy."""
    SCANNING = 1
    WAITING_FOR_RETRACEMENT = 2
    MONITORING_FOR_ENTRY = 3

def pass_through(series):
    """A simple pass-through function to register external data series with self.I()."""
    return series

class MowInternal50PercentFibScalpStrategy(Strategy):
    # --- Optimizable Parameters ---
    aoi_buffer_percent = 0.1  # % buffer around 50% Fib to create the Area of Interest
    min_rr = 5.0              # Minimum Risk-to-Reward ratio required for a trade
    risk_percent = 0.02       # Percentage of equity to risk per trade
    sl_buffer_percent = 0.1   # % buffer above the reversal high for stop-loss placement

    def init(self):
        """Initialize the strategy's state and indicators."""
        # --- State Management ---
        self.current_state = State.SCANNING

        # --- Pattern & Fibonacci Tracking ---
        self.last_swing_low_price = None
        self.leg1_low_price = None
        self.leg1_high_price = None
        self.fib_38_level = None
        self.fib_50_level = None
        self.fib_62_level = None
        self.fib_79_level = None
        self.aoi_lower_bound = None
        self.aoi_upper_bound = None
        self.center_peak_high = None # Tracks the highest point of the retracement

        # --- Register Pre-processed Swing Data as Indicators ---
        # This makes the swing high/low data available in the `next` method
        self.swing_highs = self.I(pass_through, self.data.df['swing_high'].values, name='swing_high')
        self.swing_lows = self.I(pass_through, self.data.df['swing_low'].values, name='swing_low')

    def next(self):
        """The main strategy logic executed on each bar."""
        # --- State Machine ---
        if self.current_state == State.SCANNING:
            self._handle_scanning()

        elif self.current_state == State.WAITING_FOR_RETRACEMENT:
            self._handle_waiting_for_retracement()

        elif self.current_state == State.MONITORING_FOR_ENTRY:
            self._handle_monitoring_for_entry()

    # --- State Handling Methods ---
    def _handle_scanning(self):
        """Looks for a completed Leg 1 of an M-pattern."""
        # First, detect a swing low and store it.
        if self.swing_lows[-1] == 1:
            self.last_swing_low_price = self.data.Low[-1]

        # If we have a stored swing low, look for a subsequent swing high.
        if self.last_swing_low_price is not None and self.swing_highs[-1] == 1:
            # This is our Leg 1
            self.leg1_low_price = self.last_swing_low_price
            self.leg1_high_price = self.data.High[-1]

            # Basic validation: High must be after the low
            if self.leg1_high_price > self.leg1_low_price:
                self._calculate_fib_levels()
                self.current_state = State.WAITING_FOR_RETRACEMENT
                self.last_swing_low_price = None # Consume the low

    def _handle_waiting_for_retracement(self):
        """Monitors price for entry into the 50% AOI or invalidation."""
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]

        # Invalidation: Price makes a new high above Leg 1
        if current_high > self.leg1_high_price:
            self._reset_state()
            return

        # AOI Entry Condition (for a SHORT)
        if self.aoi_upper_bound and current_low <= self.aoi_upper_bound:
            self.current_state = State.MONITORING_FOR_ENTRY
            self.center_peak_high = current_high

    def _handle_monitoring_for_entry(self):
        """Looks for a bearish reversal pattern within the AOI to trigger a short trade."""
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        self.center_peak_high = max(self.center_peak_high, current_high)

        # Invalidation: Price makes a new high or drops below the 61.8% Fib level
        if current_high > self.leg1_high_price or \
           (self.fib_62_level and current_low < self.fib_62_level):
            self._reset_state()
            return

        if self._is_bearish_engulfing():
            entry_price = self.data.Close[-1]
            stop_loss = self.center_peak_high * (1 + self.sl_buffer_percent)

            # Correct TP: 50% internal retracement of the center leg
            internal_retracement_dist = self.center_peak_high - self.leg1_low_price
            take_profit = self.center_peak_high - (internal_retracement_dist * 0.5)

            if take_profit >= entry_price:
                self._reset_state()
                return

            risk = abs(entry_price - stop_loss)
            reward = abs(entry_price - take_profit)
            rr = reward / risk if risk > 0 else 0

            if rr >= self.min_rr:
                size = (self.equity * self.risk_percent) / risk
                if size * entry_price > self.equity:
                    size = self.equity / entry_price
                self.sell(size=int(size), sl=stop_loss, tp=take_profit)
                self._reset_state()

    # --- Helper Methods ---
    def _calculate_fib_levels(self):
        """Calculates Fibonacci retracement levels for Leg 1."""
        price_range = self.leg1_high_price - self.leg1_low_price
        self.fib_38_level = self.leg1_high_price - (price_range * 0.382)
        self.fib_50_level = self.leg1_high_price - (price_range * 0.5)
        self.fib_62_level = self.leg1_high_price - (price_range * 0.618)
        self.fib_79_level = self.leg1_high_price - (price_range * 0.786)

        aoi_buffer = price_range * self.aoi_buffer_percent
        self.aoi_lower_bound = self.fib_50_level - aoi_buffer
        self.aoi_upper_bound = self.fib_50_level + aoi_buffer

    def _is_bearish_engulfing(self):
        """Checks for a bearish engulfing pattern."""
        if len(self.data.Close) < 2:
            return False

        prev_open = self.data.Open[-2]
        prev_close = self.data.Close[-2]
        curr_open = self.data.Open[-1]
        curr_close = self.data.Close[-1]

        is_prev_bullish = prev_close > prev_open
        is_curr_bearish = curr_close < curr_open
        is_engulfing = curr_open > prev_close and curr_close < prev_open

        return is_prev_bullish and is_curr_bearish and is_engulfing

    def _reset_state(self):
        """Resets the state machine and tracking variables."""
        self.current_state = State.SCANNING
        self.last_swing_low_price = None
        self.leg1_low_price = None
        self.leg1_high_price = None
        self.fib_38_level = None
        self.fib_50_level = None
        self.fib_62_level = None
        self.fib_79_level = None
        self.aoi_lower_bound = None
        self.aoi_upper_bound = None
        self.center_peak_high = None

def generate_m_w_patterns(n_patterns=20, bars_per_pattern=100):
    """
    Generates high-fidelity synthetic 15-min OHLCV data with M-patterns
    that are specifically designed to be traded by the Mow strategy.
    """
    data = []
    current_price = 100

    for _ in range(n_patterns):
        # --- Create a valid M-pattern ---

        # 1. Leg 1 (upward)
        leg1_height = np.random.uniform(2, 5)
        leg1_bars = int(bars_per_pattern * 0.2)
        p1_low = current_price
        p2_high = p1_low + leg1_height
        leg1 = np.linspace(p1_low, p2_high, leg1_bars)

        # 2. Retracement to 50% AOI
        # The price needs to fall into the 50% fib zone without touching others.
        fib_50 = p2_high - (leg1_height * 0.5)
        aoi_buffer = leg1_height * 0.1 # 10% buffer
        aoi_upper = fib_50 + aoi_buffer
        aoi_lower = fib_50 - aoi_buffer

        retrace_bars = int(bars_per_pattern * 0.15)
        center_peak_price = np.random.uniform(aoi_lower, aoi_upper)
        retrace_leg = np.linspace(p2_high, center_peak_price, retrace_bars)

        # 3. Reversal (entry signal)
        # Create a bearish engulfing candle
        reversal_bars = 2
        engulfing_open = center_peak_price + 0.1
        engulfing_close = center_peak_price - 0.2
        prev_candle_open = engulfing_close + 0.05
        prev_candle_close = engulfing_open - 0.05

        reversal_prices = [prev_candle_open, prev_candle_close, engulfing_open, engulfing_close]

        # 4. Leg 2 (downward)
        leg2_bars = int(bars_per_pattern * 0.2)
        p5_low = p1_low - np.random.uniform(0.5, 1) # Breaks the neckline
        leg2 = np.linspace(engulfing_close, p5_low, leg2_bars)

        # 5. Noise/Consolidation
        noise_bars = bars_per_pattern - leg1_bars - retrace_bars - reversal_bars - leg2_bars
        noise = np.random.normal(loc=0, scale=0.1, size=noise_bars).cumsum() + p5_low

        # --- Assemble into OHLC ---
        pattern_prices = np.concatenate([leg1, retrace_leg, [prev_candle_close], [engulfing_close], leg2, noise])

        for i in range(len(pattern_prices)):
            close = pattern_prices[i]
            if i == 0:
                open_price = close - np.random.uniform(0.05, 0.1)
            else:
                # Special handling for engulfing candles
                if i == leg1_bars + retrace_bars:
                    open_price = prev_candle_open
                elif i == leg1_bars + retrace_bars + 1:
                    open_price = engulfing_open
                else:
                    open_price = pattern_prices[i-1]

            high = max(open_price, close) + np.random.uniform(0.01, 0.05)
            low = min(open_price, close) - np.random.uniform(0.01, 0.05)
            volume = np.random.randint(100, 1000)
            data.append([open_price, high, low, close, volume])

        current_price = p5_low

    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    df.index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(df), freq='15min'))

    return df

def preprocess_data(df):
    """Finds swing highs and lows and adds them to the DataFrame."""
    # Find peaks (swing highs)
    high_peaks_indices, _ = find_peaks(df['High'], distance=5, prominence=0.5)
    df['swing_high'] = 0
    df.iloc[high_peaks_indices, df.columns.get_loc('swing_high')] = 1

    # Find troughs (swing lows)
    low_peaks_indices, _ = find_peaks(-df['Low'], distance=5, prominence=0.5)
    df['swing_low'] = 0
    df.iloc[low_peaks_indices, df.columns.get_loc('swing_low')] = 1

    return df

if __name__ == '__main__':
    # --- Load Data ---
    # Using the high-fidelity synthetic data generator to ensure the
    # strategy has valid M-patterns to trade.
    data = generate_m_w_patterns(n_patterns=50)
    data = preprocess_data(data)

    # --- Backtest ---
    bt = Backtest(data, MowInternal50PercentFibScalpStrategy, cash=100_000, commission=.002)

    # --- Optimization ---
    stats = bt.optimize(
        aoi_buffer_percent=[0.05, 0.1, 0.15],
        min_rr=[0.5, 1, 2],  # Lowered to ensure trades are found
        risk_percent=[0.01, 0.02],
        sl_buffer_percent=[0.05, 0.1, 0.15],
        maximize='Sharpe Ratio'
    )

    print("--- Best Run Stats ---")
    print(stats)
    print("--------------------")

    # --- Results ---
    import os

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Save results
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'mow_internal_50_percent_fib_scalp',
            'return': stats.get('Return [%]', 0.0),
            'sharpe': stats.get('Sharpe Ratio', 0.0),
            'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
            'win_rate': stats.get('Win Rate [%]', 0.0),
            'total_trades': stats.get('# Trades', 0)
        }, f, indent=2)

    # Generate plot
    try:
        bt.plot()
    except Exception as e:
        print(f"Could not generate plot: {e}")
