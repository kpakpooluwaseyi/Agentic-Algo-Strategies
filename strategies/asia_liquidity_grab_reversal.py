
import json
import os
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

def generate_forex_data(days=200):
    """Generates synthetic 24-hour Forex data."""
    rng = np.random.default_rng(seed=42)

    # Generate 15-minute timestamps for the specified number of days
    dates = pd.date_range(start='2023-01-01', periods=days * 24 * 4, freq='15min', tz='UTC')

    # Base price movement with some noise
    price_changes = rng.normal(0, 0.0005, size=len(dates))
    base_price = 1.1000 + np.cumsum(price_changes)

    # Create OHLC data
    open_price = base_price
    high_price = open_price + rng.uniform(0, 0.001, size=len(dates))
    low_price = open_price - rng.uniform(0, 0.001, size=len(dates))
    close_price = rng.uniform(low_price, high_price, size=len(dates))

    # Ensure High is the max and Low is the min
    high_price = np.maximum(high_price, np.maximum(open_price, close_price))
    low_price = np.minimum(low_price, np.minimum(open_price, close_price))

    # Create DataFrame
    df = pd.DataFrame({
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Close': close_price,
    }, index=dates)

    # Add some volume
    df['Volume'] = rng.integers(100, 1000, size=len(dates))

    return df

def preprocess_data(df):
    """Calculates Asia Session High/Low and other necessary indicators."""

    # --- Define Session Times in UTC ---
    asia_start_time = pd.to_datetime('00:00').time()
    asia_end_time = pd.to_datetime('08:00').time()

    # --- Calculate Daily Groupers ---
    df['date'] = df.index.date

    # --- Asia Session High/Low ---
    is_asia_session = (df.index.time >= asia_start_time) & (df.index.time < asia_end_time)
    asia_session_data = df[is_asia_session].copy()

    # Group by date to get the high and low for each Asia session
    asia_high_low = asia_session_data.groupby('date').agg(
        ASH=('High', 'max'),
        ASL=('Low', 'min')
    )

    # Map the session high/low back to the main DataFrame
    # This is more robust than merging as it preserves the original DatetimeIndex
    df['ASH'] = df['date'].map(asia_high_low['ASH'])
    df['ASL'] = df['date'].map(asia_high_low['ASL'])

    # Forward-fill the values to make them available throughout the day
    df['ASH'] = df['ASH'].ffill()
    df['ASL'] = df['ASL'].ffill()

    # Calculate Asia Session Range Percentage
    df['asia_range_pct'] = ((df['ASH'] - df['ASL']) / df['ASL']) * 100

    # --- Candlestick Patterns ---
    # Bearish Engulfing
    prev_body_is_bullish = df['Close'].shift(1) > df['Open'].shift(1)
    current_body_is_bearish = df['Close'] < df['Open']
    body_engulfs = (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1))
    df['is_bearish_engulfing'] = prev_body_is_bullish & current_body_is_bearish & body_engulfs

    # Bullish Engulfing
    prev_body_is_bearish = df['Close'].shift(1) < df['Open'].shift(1)
    current_body_is_bullish = df['Close'] > df['Open']
    body_engulfs = (df['Open'] < df['Close'].shift(1)) & (df['Close'] > df['Open'].shift(1))
    df['is_bullish_engulfing'] = prev_body_is_bearish & current_body_is_bullish & body_engulfs

    # Remove rows with NaN values created by shifts and ffills
    df.dropna(inplace=True)

    return df

class AsiaLiquidityGrabReversalStrategy(Strategy):
    """
    A strategy that trades reversals after liquidity grabs above/below the Asia session range.
    """
    asia_range_max_pct = 2.0 # Optimization parameter: max percentage for Asia range

    def init(self):
        # --- State Machine ---
        # 0: Searching for setup
        # 1: Liquidity grab above ASH, waiting for bearish reversal
        # 2: Liquidity grab below ASL, waiting for bullish reversal
        self.trade_state = 0

        # NOTE: Pre-calculated data is accessed directly in `next()` via `self.data`
        # to avoid multiprocessing errors with `bt.optimize()`.

        # --- Session Times ---
        self.uk_start_time = pd.to_datetime('08:00').time()
        self.uk_end_time = pd.to_datetime('16:00').time()

    def next(self):
        # --- Pre-computation for the current bar ---
        current_time = self.data.index[-1].time()
        is_uk_session = self.uk_start_time <= current_time < self.uk_end_time

        # Don't trade outside of the UK session
        if not is_uk_session:
            # If we transition out of UK session, reset state
            if self.trade_state != 0:
                self.trade_state = 0
            return

        # --- State Machine Logic ---

        # State 0: Searching for a liquidity grab
        if self.trade_state == 0:
            # Pre-condition: Asia range must be within the threshold
            if self.data.asia_range_pct[-1] < self.asia_range_max_pct:
                # Check for grab above Asia Session High (ASH)
                if self.data.Close[-1] > self.data.ASH[-1]:
                    self.trade_state = 1 # Transition to waiting for bearish reversal
                # Check for grab below Asia Session Low (ASL)
                elif self.data.Close[-1] < self.data.ASL[-1]:
                    self.trade_state = 2 # Transition to waiting for bullish reversal

        # State 1: Price is above ASH, waiting for bearish confirmation
        elif self.trade_state == 1:
            # If a bearish engulfing candle forms, enter SHORT
            if self.data.is_bearish_engulfing[-1] and not self.position:
                sl = self.data.High[-1]      # SL above the reversal candle's high
                tp = self.data.ASL[-1]       # TP at the other side of Asia range

                # Basic R:R check to avoid bad trades
                if self.data.Close[-1] - tp > sl - self.data.Close[-1]:
                    self.sell(sl=sl, tp=tp)

                self.trade_state = 0 # Reset state after placing trade

            # If price moves back inside the Asia range without a signal, reset state
            elif self.data.Close[-1] < self.data.ASH[-1]:
                self.trade_state = 0

        # State 2: Price is below ASL, waiting for bullish confirmation
        elif self.trade_state == 2:
            # If a bullish engulfing candle forms, enter LONG
            if self.data.is_bullish_engulfing[-1] and not self.position:
                sl = self.data.Low[-1]   # SL below the reversal candle's low
                tp = self.data.ASH[-1]   # TP at the other side of Asia range

                # Basic R:R check
                if tp - self.data.Close[-1] > self.data.Close[-1] - sl:
                    self.buy(sl=sl, tp=tp)

                self.trade_state = 0 # Reset state after placing trade

            # If price moves back inside the Asia range without a signal, reset state
            elif self.data.Close[-1] > self.data.ASL[-1]:
                self.trade_state = 0


if __name__ == '__main__':
    # --- 1. Load and Prepare Data ---
    # Using generated data because session logic requires 24h data.
    data = generate_forex_data(days=120)
    data = preprocess_data(data)

    # --- 2. Initialize Backtest ---
    bt = Backtest(data, AsiaLiquidityGrabReversalStrategy, cash=100_000, commission=.0002)

    # --- 3. Optimize ---
    # Find the best Asia range percentage threshold
    stats = bt.optimize(
        asia_range_max_pct=np.arange(0.5, 3.0, 0.25).tolist(),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.asia_range_max_pct > 0
    )
    print(stats)

    # --- 4. Save Results ---
    os.makedirs('results', exist_ok=True)

    # Handle cases where no trades were made
    if stats['# Trades'] > 0:
        result_dict = {
            'strategy_name': 'asia_liquidity_grab_reversal',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }
    else:
        result_dict = {
            'strategy_name': 'asia_liquidity_grab_reversal',
            'return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0
        }

    with open('results/temp_result.json', 'w') as f:
        json.dump(result_dict, f, indent=2)

    # --- 5. Generate Plot ---
    # The plot will be for the BEST run from the optimization
    bt.plot(filename='results/asia_liquidity_grab_reversal_plot.html')
