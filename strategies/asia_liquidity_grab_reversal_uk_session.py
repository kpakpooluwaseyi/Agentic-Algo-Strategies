
import json
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from enum import Enum

# --- Session and State Definitions ---
# Use an Enum for clear, explicit state management within the strategy
class StrategyState(Enum):
    WAITING_FOR_ASIA_CLOSE = 1
    WAITING_FOR_GRAB = 2
    WAITING_FOR_CONFIRMATION = 3

# Define session times in UTC for clarity and consistency
ASIA_START_H, ASIA_START_M = 0, 0   # 00:00 UTC
ASIA_END_H, ASIA_END_M = 7, 0     # 07:00 UTC
UK_START_H, UK_START_M = 7, 0     # 07:00 UTC

def generate_synthetic_data(days=200):
    """
    Generates realistic synthetic 15-minute data with a deterministic pattern
    for validating the strategy's logic without lookahead bias.
    """
    n_points = days * 96
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='15min')

    open_p, high_p, low_p, close_p = (np.full(n_points, 1.5) for _ in range(4))
    np.random.seed(42)

    for i in range(days):
        day_start_idx = i * 96

        # --- Asia Session (00:00-07:00 UTC): Tight range ---
        asia_start, asia_end = day_start_idx, day_start_idx + 28
        base = 1.5000
        for j in range(asia_start, asia_end):
            open_p[j], close_p[j] = base, base + np.random.uniform(-0.0003, 0.0003)
            high_p[j] = max(open_p[j], close_p[j]) + np.random.uniform(0, 0.0001)
            low_p[j] = min(open_p[j], close_p[j]) - np.random.uniform(0, 0.0001)
            base = close_p[j]

        asia_high_price = np.max(high_p[asia_start:asia_end])
        asia_low_price = np.min(low_p[asia_start:asia_end])

        # --- UK Session (starts 07:00 UTC): Engineered pattern ---
        # Candle 1: Liquidity Grab (07:15 UTC)
        grab_idx = day_start_idx + 29
        open_p[grab_idx] = asia_high_price - 0.0002
        close_p[grab_idx] = asia_high_price + 0.0001
        high_p[grab_idx] = asia_high_price + 0.0005
        low_p[grab_idx] = open_p[grab_idx] - 0.0001

        # Candle 2: Bearish Engulfing (07:30 UTC)
        engulf_idx = grab_idx + 1
        open_p[engulf_idx] = close_p[grab_idx] + 0.0001
        close_p[engulf_idx] = open_p[grab_idx] - 0.0001
        high_p[engulf_idx] = open_p[engulf_idx] + 0.0001
        low_p[engulf_idx] = close_p[engulf_idx] - 0.0001

        # Post-pattern: drift down towards the Asia Low
        trend_start_idx = engulf_idx + 1
        trend_end_idx = day_start_idx + 48 # around midday
        drift = np.linspace(0, (asia_low_price - close_p[engulf_idx]), trend_end_idx - trend_start_idx)
        for j, step in enumerate(drift):
             price = close_p[engulf_idx] + step + np.random.normal(0, 0.0001)
             open_p[trend_start_idx+j] = price
             close_p[trend_start_idx+j] = price + np.random.uniform(-0.0001, 0.0001)
             high_p[trend_start_idx+j] = max(open_p[trend_start_idx+j], close_p[trend_start_idx+j])
             low_p[trend_start_idx+j] = min(open_p[trend_start_idx+j], close_p[trend_start_idx+j])

    return pd.DataFrame({
        'Open': open_p, 'High': high_p, 'Low': low_p, 'Close': close_p, 'Volume': 100
    }, index=dates)


class AsiaLiquidityGrabReversalUkSessionStrategy(Strategy):
    max_asia_range_perc = 2.0  # Optimization parameter

    def init(self):
        # --- Daily State Variables ---
        self.current_day = None
        self.state = StrategyState.WAITING_FOR_ASIA_CLOSE

        # --- Asia Session Data ---
        self.asia_candles = []
        self.asia_high = None
        self.asia_low = None

        # --- Pattern Detection ---
        self.grab_candle_high = None

    def next(self):
        # --- Time and Session Management ---
        current_time = self.data.index[-1]
        today = current_time.date()

        # Reset daily state at the beginning of a new day
        if self.current_day != today:
            self.current_day = today
            self.state = StrategyState.WAITING_FOR_ASIA_CLOSE
            self.asia_candles = []
            self.asia_high = self.asia_low = None
            self.grab_candle_high = None

        if self.position:
            return

        # --- State Machine Logic ---

        # State 1: Accumulate Asia session candles
        if self.state == StrategyState.WAITING_FOR_ASIA_CLOSE:
            if ASIA_START_H <= current_time.hour < ASIA_END_H:
                self.asia_candles.append(self.data.df.iloc[-1])

            # Transition: Once Asia session is over, calculate levels
            elif current_time.hour >= ASIA_END_H:
                if self.asia_candles:
                    asia_df = pd.DataFrame(self.asia_candles)
                    self.asia_high = asia_df['High'].max()
                    self.asia_low = asia_df['Low'].min()

                    # Validate Asia range before proceeding
                    if ((self.asia_high - self.asia_low) / self.asia_low) * 100 < self.max_asia_range_perc:
                        self.state = StrategyState.WAITING_FOR_GRAB
                    else:
                        # Range too large, wait for the next day
                        self.state = None
                else:
                    self.state = None # No candles, wait for next day

        # State 2: Wait for a liquidity grab during the UK session
        elif self.state == StrategyState.WAITING_FOR_GRAB:
            if current_time.hour >= UK_START_H and self.data.High[-1] > self.asia_high:
                self.grab_candle_high = self.data.High[-1]
                self.state = StrategyState.WAITING_FOR_CONFIRMATION

        # State 3: On the next candle, check for a bearish engulfing confirmation
        elif self.state == StrategyState.WAITING_FOR_CONFIRMATION:
            if len(self.data.Close) < 2:
                self.state = StrategyState.WAITING_FOR_GRAB # Reset state
                return

            prev_open, prev_close = self.data.Open[-2], self.data.Close[-2]
            curr_open, curr_close = self.data.Open[-1], self.data.Close[-1]

            is_prev_bullish = prev_close > prev_open
            is_curr_bearish = curr_close < curr_open
            engulfs = curr_open > prev_close and curr_close < prev_open

            if is_prev_bullish and is_curr_bearish and engulfs:
                sl = self.grab_candle_high
                tp = self.asia_low
                if tp < curr_close and sl > curr_close:
                    self.sell(sl=sl, tp=tp, size=1)

            # Whether a trade was placed or not, the setup is now invalid for today.
            self.state = None # Halt further action for today

if __name__ == '__main__':
    data = generate_synthetic_data(days=200)

    # Use UTC for timezone-aware data
    data.index = data.index.tz_localize('UTC')

    bt = Backtest(data, AsiaLiquidityGrabReversalUkSessionStrategy,
                  cash=100000, commission=.0002)

    print("Optimizing strategy...")
    stats = bt.optimize(
        max_asia_range_perc=list(np.arange(0.5, 3.0, 0.5)),
        maximize='Sharpe Ratio',
        constraint=lambda param: param.max_asia_range_perc > 0
    )

    print("Best run stats:")
    print(stats)

    # --- Correctly Access and Save Results ---
    import os
    os.makedirs('results', exist_ok=True)

    # The `stats` object is a Series with metrics. Handle potential NaN for Sharpe.
    sharpe = stats.get('Sharpe Ratio')

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'asia_liquidity_grab_reversal_uk_session',
            'return': float(stats['Return [%]']),
            'sharpe': sharpe if pd.notna(sharpe) else None,  # Convert NaN to None for valid JSON (null)
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),  # Use value directly from stats
            'total_trades': int(stats['# Trades'])   # Use value directly from stats
        }, f, indent=2)

    plot_filename = 'results/asia_liquidity_grab_reversal_uk_session.html'
    bt.plot(filename=plot_filename, open_browser=False)

    print(f"Backtest complete. Results saved to results/temp_result.json and plot saved to {plot_filename}")
