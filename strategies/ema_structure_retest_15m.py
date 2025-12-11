from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import json

# Custom Indicator and Candlestick Pattern Functions

def EMA(series, period):
    """Calculates the Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, adjust=False).mean()

def is_bullish_engulfing(df, i):
    """Checks for a bullish engulfing pattern at index i."""
    if i == 0: return False
    current_candle = df.iloc[i]
    previous_candle = df.iloc[i - 1]
    # Previous candle must be bearish, current must be bullish
    if not (previous_candle['Close'] < previous_candle['Open'] and current_candle['Close'] > current_candle['Open']):
        return False
    # Current candle's body must engulf the previous candle's body
    if not (current_candle['Open'] < previous_candle['Close'] and current_candle['Close'] > previous_candle['Open']):
        return False
    return True

def is_bearish_engulfing(df, i):
    """Checks for a bearish engulfing pattern at index i."""
    if i == 0: return False
    current_candle = df.iloc[i]
    previous_candle = df.iloc[i - 1]
    # Previous candle must be bullish, current must be bearish
    if not (previous_candle['Close'] > previous_candle['Open'] and current_candle['Close'] < current_candle['Open']):
        return False
    # Current candle's body must engulf the previous candle's body
    if not (current_candle['Open'] > previous_candle['Close'] and current_candle['Close'] < previous_candle['Open']):
        return False
    return True

def is_hammer(df, i):
    """Checks for a hammer pattern (bullish reversal)."""
    candle = df.iloc[i]
    body_size = abs(candle['Close'] - candle['Open'])
    lower_wick = candle['Open'] - candle['Low'] if candle['Open'] < candle['Close'] else candle['Close'] - candle['Low']
    upper_wick = candle['High'] - candle['Close'] if candle['Open'] < candle['Close'] else candle['High'] - candle['Open']
    # Body should be small, lower wick long, upper wick short
    return lower_wick > 2 * body_size and upper_wick < body_size and body_size > 0

def is_doji(df, i):
    """Checks for a Doji pattern (indecision)."""
    candle = df.iloc[i]
    body_size = abs(candle['Close'] - candle['Open'])
    total_range = candle['High'] - candle['Low']
    # Body size must be very small relative to the total range
    return total_range > 0 and body_size / total_range < 0.1

def is_railroad_track(df, i, is_bullish=True):
    """Checks for a Railroad Track pattern."""
    if i == 0: return False
    current_candle = df.iloc[i]
    previous_candle = df.iloc[i - 1]

    # Check for opposing candles
    if is_bullish:
        # Previous bearish, current bullish
        if not (previous_candle['Close'] < previous_candle['Open'] and current_candle['Close'] > current_candle['Open']):
            return False
    else: # is_bearish
        # Previous bullish, current bearish
        if not (previous_candle['Close'] > previous_candle['Open'] and current_candle['Close'] < current_candle['Open']):
            return False

    # Check for similar body size (e.g., within 30% of each other)
    current_body = abs(current_candle['Close'] - current_candle['Open'])
    previous_body = abs(previous_candle['Close'] - previous_candle['Open'])
    if not (min(current_body, previous_body) / max(current_body, previous_body) > 0.7):
        return False

    return True

def is_bullish_reversal_pattern(df, i):
    """Checks for any of the specified bullish reversal patterns."""
    return is_bullish_engulfing(df, i) or is_hammer(df, i) or is_doji(df, i) or is_railroad_track(df, i, is_bullish=True)

def is_bearish_reversal_pattern(df, i):
    """Checks for any of the specified bearish reversal patterns."""
    return is_bearish_engulfing(df, i) or is_doji(df, i) or is_railroad_track(df, i, is_bullish=False)


class EmaStructureRetest15MStrategy(Strategy):
    # --- Strategy Parameters ---
    ema_fast_period = 50
    ema_slow_period = 200
    risk_reward_ratio = 1.5

    # --- Optimization Parameters ---
    # Lookback period for identifying swing points.
    swing_lookback = 10

    def init(self):
        # --- Indicators ---
        self.ema_fast = self.I(EMA, self.data.Close, self.ema_fast_period)
        self.ema_slow = self.I(EMA, self.data.Close, self.ema_slow_period)

        # --- Swing Point Detection (Causal) ---
        # These lists will be populated dynamically in the next() method.
        self.swing_highs = []
        self.swing_lows = []

        # --- State Machine Variables ---
        self.long_setup_state = None # Can be 'BOS_CONFIRMED', 'WAITING_FOR_RETEST'
        self.short_setup_state = None # Can be 'BOS_CONFIRMED', 'WAITING_FOR_RETEST'

        self.long_structure_low = None # SL for long trades
        self.short_structure_high = None # SL for short trades

    def next(self):
        # --- Causal Swing Point Detection ---
        # We need enough data to look back.
        # The lookback period is centered, so we need `swing_lookback` on both sides.
        lookback_range = self.swing_lookback * 2 + 1
        if len(self.data) < lookback_range:
            return

        current_index = len(self.data.Close) - 1
        # The index we're checking is in the middle of our lookback window
        # This introduces a lag of `swing_lookback` bars, which is realistic.
        idx_to_check = current_index - self.swing_lookback

        window = self.data.High[idx_to_check - self.swing_lookback : idx_to_check + self.swing_lookback + 1]
        if len(window) == lookback_range and window[self.swing_lookback] == max(window):
            if idx_to_check not in self.swing_highs:
                self.swing_highs.append(idx_to_check)

        window = self.data.Low[idx_to_check - self.swing_lookback : idx_to_check + self.swing_lookback + 1]
        if len(window) == lookback_range and window[self.swing_lookback] == min(window):
            if idx_to_check not in self.swing_lows:
                self.swing_lows.append(idx_to_check)

        # If a position is already open, do nothing.
        if self.position:
            return

        # --- LONG LOGIC ---
        # 1. Detect Bullish Break of Structure (BOS)
        # Find the two most recent swing highs
        if len(self.swing_highs) >= 2:
            last_high_idx, prev_high_idx = self.swing_highs[-1], self.swing_highs[-2]

            # Check for BOS: last high is higher than the previous high
            if self.data.High[last_high_idx] > self.data.High[prev_high_idx]:
                # Find the swing low between these two highs
                lows_between = [l for l in self.swing_lows if prev_high_idx < l < last_high_idx]
                if lows_between:
                    self.long_structure_low = self.data.Low[lows_between[-1]]
                    self.long_setup_state = 'BOS_CONFIRMED'
                    # Invalidate short setup if a bullish BOS occurs
                    self.short_setup_state = None

        # 2. Wait for EMA Retest and Entry Signal (if BOS confirmed)
        if self.long_setup_state == 'BOS_CONFIRMED':
            # Condition: Price must be near the 50 EMA
            price_touches_ema = self.data.Low[-1] <= self.ema_fast[-1] <= self.data.High[-1]

            # Condition: 50 EMA must be above 200 EMA
            ema_cross_valid = self.ema_fast[-1] > self.ema_slow[-1]

            # Condition: Bullish reversal pattern
            is_reversal = is_bullish_reversal_pattern(self.data.df, current_index)

            if price_touches_ema and ema_cross_valid and is_reversal:
                sl = self.long_structure_low
                entry_price = self.data.Close[-1]
                tp = entry_price + (entry_price - sl) * self.risk_reward_ratio

                if entry_price > sl: # Basic validation
                    self.buy(sl=sl, tp=tp)
                    self.long_setup_state = None # Reset state

        # --- SHORT LOGIC (Inverse of Long) ---
        # 1. Detect Bearish Break of Structure (BOS)
        if len(self.swing_lows) >= 2:
            last_low_idx, prev_low_idx = self.swing_lows[-1], self.swing_lows[-2]

            if self.data.Low[last_low_idx] < self.data.Low[prev_low_idx]:
                highs_between = [h for h in self.swing_highs if prev_low_idx < h < last_low_idx]
                if highs_between:
                    self.short_structure_high = self.data.High[highs_between[-1]]
                    self.short_setup_state = 'BOS_CONFIRMED'
                    self.long_setup_state = None

        # 2. Wait for EMA Retest and Entry Signal
        if self.short_setup_state == 'BOS_CONFIRMED':
            price_touches_ema = self.data.Low[-1] <= self.ema_fast[-1] <= self.data.High[-1]
            ema_cross_valid = self.ema_fast[-1] < self.ema_slow[-1]
            is_reversal = is_bearish_reversal_pattern(self.data.df, current_index)

            if price_touches_ema and ema_cross_valid and is_reversal:
                sl = self.short_structure_high
                entry_price = self.data.Close[-1]
                tp = entry_price - (sl - entry_price) * self.risk_reward_ratio

                if entry_price < sl and tp > 0: # Ensure TP is a valid price
                    self.sell(sl=sl, tp=tp)
                    self.short_setup_state = None

if __name__ == '__main__':
    # Load or generate data
    from backtesting.test import GOOG
    # The strategy is designed for 15-minute data, GOOG is daily.
    # We will proceed with GOOG for demonstration purposes as required.
    data = GOOG.copy()

    # It's a good practice to resample daily data to a proxy for intraday,
    # though it won't replicate true intraday price action.
    # This step is illustrative. For a real test, 15-min data is needed.
    # data = data.resample('4H').last().ffill() # Example resampling

    # Run backtest
    bt = Backtest(data, EmaStructureRetest15MStrategy, cash=10000, commission=.002)

    # Optimize
    stats = bt.optimize(
        swing_lookback=range(5, 30, 5),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.swing_lookback > 0
    )

    # Save results
    import os
    os.makedirs('results', exist_ok=True)

    # A robust way to sanitize stats for JSON serialization
    def sanitize_stats(stats):
        # This handles cases where metrics might be missing (e.g., no trades executed)
        # or have non-serializable types like NaN or numpy floats.
        sanitized = {
            'strategy_name': 'ema_structure_retest_15m',
            'return': stats.get('Return [%]', 0.0),
            'sharpe': stats.get('Sharpe Ratio', 0.0),
            'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
            'win_rate': stats.get('Win Rate [%]', 0.0),
            'total_trades': stats.get('# Trades', 0)
        }
        for key, value in sanitized.items():
            if isinstance(value, (np.floating, np.integer)):
                sanitized[key] = float(value) if np.isfinite(value) else None
            elif isinstance(value, int):
                 sanitized[key] = int(value)
            elif pd.isna(value):
                sanitized[key] = None
        return sanitized

    final_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(final_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")

    # Generate plot
    try:
        bt.plot(filename='results/ema_structure_retest_15m_plot.html')
        print("Backtest plot saved to results/ema_structure_retest_15m_plot.html")
    except Exception as e:
        print(f"Could not generate plot: {e}")
