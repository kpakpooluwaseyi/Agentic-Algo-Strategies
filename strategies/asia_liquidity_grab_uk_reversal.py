from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json

# Custom indicator functions must be defined at the module level.
def is_bearish_engulfing(df):
    """Checks for a bearish engulfing pattern."""
    prev_row = df.iloc[-2]
    curr_row = df.iloc[-1]

    # Previous candle must be bullish
    if prev_row['Close'] <= prev_row['Open']:
        return False

    # Current candle must be bearish and engulf the previous one
    return (curr_row['Open'] > prev_row['Close'] and
            curr_row['Close'] < prev_row['Open'])

def is_bullish_engulfing(df):
    """Checks for a bullish engulfing pattern."""
    prev_row = df.iloc[-2]
    curr_row = df.iloc[-1]

    # Previous candle must be bearish
    if prev_row['Close'] >= prev_row['Open']:
        return False

    # Current candle must be bullish and engulf the previous one
    return (curr_row['Open'] < prev_row['Close'] and
            curr_row['Close'] > prev_row['Open'])

def pass_through(series):
    return series

class AsiaLiquidityGrabUkReversalStrategy(Strategy):
    # Optimizable parameters
    asia_range_max_perc = 0.5 # Max Asia range in percentage
    sl_buffer_perc = 0.05 # SL buffer in percentage

    def init(self):
        df = self.data.df.copy()

        # --- Session Identification ---
        df['hour'] = df.index.hour
        df['is_asia'] = (df['hour'] >= 0) & (df['hour'] < 8)
        df['is_uk'] = (df['hour'] >= 8) & (df['hour'] < 16)

        # --- Daily Asia High/Low Calculation ---
        df['date'] = df.index.date
        asia_session_data = df[df['is_asia']].groupby('date')

        daily_asia_high = asia_session_data['High'].max()
        daily_asia_low = asia_session_data['Low'].min()

        df['asia_high'] = df['date'].map(daily_asia_high).ffill()
        df['asia_low'] = df['date'].map(daily_asia_low).ffill()

        # --- Asia Range Check ---
        df['asia_range'] = ((df['asia_high'] - df['asia_low']) / df['asia_low']) * 100
        df['is_valid_asia_range'] = df['asia_range'] < self.asia_range_max_perc

        # --- Engulfing Patterns ---
        # Note: self.I requires a function that returns a Series/array
        # We'll calculate this within next() for simplicity with current bar context,
        # but for performance, a proper vectorized indicator would be better.

        # --- Add processed data as indicators ---
        self.is_asia = self.I(pass_through, df['is_asia'].values)
        self.is_uk = self.I(pass_through, df['is_uk'].values)
        self.asia_high = self.I(pass_through, df['asia_high'].values)
        self.asia_low = self.I(pass_through, df['asia_low'].values)
        self.is_valid_asia_range = self.I(pass_through, df['is_valid_asia_range'].values)


    def next(self):
        # NOTE: TP2+ logic is not implemented in this version. The strategy
        # currently only targets the opposite side of the Asia range for TP1.

        # Wait for at least 2 bars to have data
        if len(self.data) < 2:
            return

        # --- Pre-computation for readability ---
        price = self.data.Close[-1]

        # --- Conditions for trade entry ---
        if not self.position and self.is_uk[-1] and self.is_valid_asia_range[-1]:

            # --- SHORT Entry Logic ("Immediately After" Check) ---
            # 1. Previous bar grabbed liquidity above HOA
            prev_high = self.data.High[-2]
            grabbed_hoa = prev_high > self.asia_high[-2]

            # 2. Current bar is a bearish engulfing that closes back below HOA
            is_engulfing_bearish = is_bearish_engulfing(self.data.df.iloc[-2:])
            closed_below_hoa = price < self.asia_high[-1]

            if grabbed_hoa and is_engulfing_bearish and closed_below_hoa:
                sl = self.data.High[-2] * (1 + self.sl_buffer_perc) # SL above the grab candle's high
                tp = self.asia_low[-1]

                if price > tp and price < sl:
                    self.sell(sl=sl, tp=tp)
                return # Exit after placing a trade

            # --- LONG Entry Logic ("Immediately After" Check) ---
            # 1. Previous bar grabbed liquidity below LOA
            prev_low = self.data.Low[-2]
            grabbed_loa = prev_low < self.asia_low[-2]

            # 2. Current bar is a bullish engulfing that closes back above LOA
            is_engulfing_bullish = is_bullish_engulfing(self.data.df.iloc[-2:])
            closed_above_loa = price > self.asia_low[-1]

            if grabbed_loa and is_engulfing_bullish and closed_above_loa:
                sl = self.data.Low[-2] * (1 - self.sl_buffer_perc) # SL below the grab candle's low
                tp = self.asia_high[-1]

                if price < tp and price > sl:
                    self.buy(sl=sl, tp=tp)

def generate_synthetic_data(days=90):
    """
    Generates synthetic 24-hour data with specific patterns for the strategy.
    This function creates textbook long and short setups.
    """
    total_bars = days * 24 * 4  # 15-min bars
    rng = pd.date_range('2023-01-01', periods=total_bars, freq='15min')
    df = pd.DataFrame(index=rng)

    # Base price movement with some noise
    base_price = 100
    price = base_price + np.random.randn(total_bars).cumsum() * 0.1

    df['Open'] = price
    df['High'] = price + np.random.uniform(0, 0.1, total_bars)
    df['Low'] = price - np.random.uniform(0, 0.1, total_bars)
    df['Close'] = price + np.random.uniform(-0.05, 0.05, total_bars)
    df['Volume'] = np.random.randint(100, 1000, total_bars)

    # Engineer the specific patterns
    for day in range(1, days - 1):
        day_start_idx = day * 96

        # --- Asia Session (00:00 - 08:00 UTC) ---
        asia_start_idx = day_start_idx
        asia_end_idx = day_start_idx + 32  # 8 hours * 4 bars/hour

        # Force a tight range
        asia_mid_price = df['Close'].iloc[asia_start_idx]
        ohlc_columns = ['Open', 'High', 'Low', 'Close']
        df.loc[df.index[asia_start_idx:asia_end_idx], ohlc_columns] = asia_mid_price
        noise = np.random.uniform(-0.05, 0.05, size=(asia_end_idx - asia_start_idx, 4))
        df.loc[df.index[asia_start_idx:asia_end_idx], ohlc_columns] += noise

        asia_high = df.iloc[asia_start_idx:asia_end_idx]['High'].max()
        asia_low = df.iloc[asia_start_idx:asia_end_idx]['Low'].min()

        # Alternate between creating a short and a long setup each day
        if day % 2 == 0:
            # --- UK Session SHORT Setup (2-bar pattern) ---
            engulfing_idx = asia_end_idx + 5 # 1h 15m into UK session
            grab_idx = engulfing_idx - 1

            # Bar 1: Grabs liquidity above HOA
            df.loc[df.index[grab_idx], 'Open'] = asia_high - 0.1
            df.loc[df.index[grab_idx], 'Close'] = asia_high + 0.05 # Closes slightly bullish
            df.loc[df.index[grab_idx], 'High'] = asia_high + 0.15  # The grab
            df.loc[df.index[grab_idx], 'Low'] = asia_high - 0.1

            # Bar 2: Bearish engulfing, closing below HOA
            df.loc[df.index[engulfing_idx], 'Open'] = asia_high + 0.1 # Opens above prev close
            df.loc[df.index[engulfing_idx], 'High'] = asia_high + 0.12
            df.loc[df.index[engulfing_idx], 'Close'] = asia_high - 0.2 # Closes below HOA & engulfs
            df.loc[df.index[engulfing_idx], 'Low'] = asia_high - 0.22
        else:
            # --- UK Session LONG Setup (2-bar pattern) ---
            engulfing_idx = asia_end_idx + 5 # 1h 15m into UK session
            grab_idx = engulfing_idx - 1

            # Bar 1: Grabs liquidity below LOA
            df.loc[df.index[grab_idx], 'Open'] = asia_low + 0.1
            df.loc[df.index[grab_idx], 'Close'] = asia_low - 0.05 # Closes slightly bearish
            df.loc[df.index[grab_idx], 'High'] = asia_low + 0.1
            df.loc[df.index[grab_idx], 'Low'] = asia_low - 0.15   # The grab

            # Bar 2: Bullish engulfing, closing above LOA
            df.loc[df.index[engulfing_idx], 'Open'] = asia_low - 0.1 # Opens below prev close
            df.loc[df.index[engulfing_idx], 'Low'] = asia_low - 0.12
            df.loc[df.index[engulfing_idx], 'Close'] = asia_low + 0.2 # Closes above LOA & engulfs
            df.loc[df.index[engulfing_idx], 'High'] = asia_low + 0.22

    return df.dropna()

if __name__ == '__main__':
    # Load or generate data
    data = generate_synthetic_data()

    # Run backtest
    bt = Backtest(data, AsiaLiquidityGrabUkReversalStrategy, cash=10000, commission=.002)

    # Optimize
    stats = bt.optimize(
        asia_range_max_perc=np.arange(0.2, 2.2, 0.2).tolist(),
        sl_buffer_perc=np.arange(0.01, 0.1, 0.01).tolist(),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.asia_range_max_perc > 0 and p.sl_buffer_perc > 0
    )

    # Save results
    import os
    os.makedirs('results', exist_ok=True)

    # Sanitize results for JSON serialization
    sanitized_stats = {}
    for key, value in stats.items():
        if isinstance(value, np.integer):
            sanitized_stats[key] = int(value)
        elif isinstance(value, np.floating):
            sanitized_stats[key] = float(value)
        elif isinstance(value, np.bool_):
            sanitized_stats[key] = bool(value)
        elif isinstance(value, (pd.Series, pd.DataFrame)):
             # Exclude non-serializable pandas objects
            sanitized_stats[key] = None
        elif pd.isna(value):
            sanitized_stats[key] = None
        else:
            sanitized_stats[key] = value

    # Ensure all required keys are present
    result_data = {
        'strategy_name': 'asia_liquidity_grab_uk_reversal',
        'return': sanitized_stats.get('Return [%]', None),
        'sharpe': sanitized_stats.get('Sharpe Ratio', None),
        'max_drawdown': sanitized_stats.get('Max. Drawdown [%]', None),
        'win_rate': sanitized_stats.get('Win Rate [%]', None),
        'total_trades': sanitized_stats.get('# Trades', 0)
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(result_data, f, indent=2)

    # Generate plot
    try:
        bt.plot(filename='results/asia_liquidity_grab_uk_reversal_plot.html')
    except TypeError as e:
        print(f"Could not generate plot due to a known issue with the plotting library: {e}")
        print("Continuing without the plot.")
