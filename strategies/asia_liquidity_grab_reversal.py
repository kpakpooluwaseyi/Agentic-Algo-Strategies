from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json

def generate_synthetic_data(days=50):
    """
    Generates synthetic 15-minute OHLC data simulating the Asia session liquidity grab pattern.
    Alternates between bullish and bearish setups each day.
    """
    rng = np.random.default_rng(42)
    periods = days * 24 * 4  # 15-minute periods in a day
    dt_index = pd.date_range(start='2023-01-01', periods=periods, freq='15min', tz='UTC')

    price = 1.0
    prices = []

    for i in range(periods):
        # Determine the time of day to create session-specific patterns
        hour = dt_index[i].hour

        # New day, reset the pattern
        if i % 96 == 0:
            day_of_week = dt_index[i].dayofweek
            is_bullish_day = (i // 96) % 2 == 0 # Alternate pattern each day

            # Asia Session (00:00 - 08:00 UTC) - Consolidation
            asia_high = price + rng.uniform(0.0005, 0.0010)
            asia_low = price - rng.uniform(0.0005, 0.0010)

        # Asia Session - range-bound
        if 0 <= hour < 8:
            open_price = price
            high = min(price + rng.uniform(0, 0.0003), asia_high)
            low = max(price - rng.uniform(0, 0.0003), asia_low)
            close = rng.uniform(low, high)
            price = close

        # London Session (08:00 - 16:00 UTC) - Liquidity Grab & Reversal
        elif 8 <= hour < 16:
            # First bar of London - The Grab
            if i % 96 == 32: # 8:00 UTC
                if is_bullish_day:
                    open_price = asia_low
                    high = open_price + 0.0002
                    low = asia_low - 0.0015 # Spike below LOA
                    close = low + 0.0004
                    price = close
                else: # Bearish day
                    open_price = asia_high
                    low = open_price - 0.0002
                    high = asia_high + 0.0015 # Spike above HOA
                    close = high - 0.0004
                    price = close
            # Second bar of London - The Engulfing Candle Confirmation
            elif i % 96 == 33: # 8:15 UTC
                if is_bullish_day:
                    open_price = price
                    low = open_price - 0.0001
                    high = asia_low + 0.0010 # Close back above LOA
                    close = high - 0.0002
                    price = close
                else: # Bearish day
                    open_price = price
                    high = open_price + 0.0001
                    low = asia_high - 0.0010 # Close back below HOA
                    close = low + 0.0002
                    price = close
            # Rest of London - Trend in reversal direction
            else:
                if is_bullish_day:
                    price += 0.00015
                else:
                    price -= 0.00015
                open_price = price
                change = rng.uniform(-0.0002, 0.0002)
                high = price + abs(change) if change > 0 else price
                low = price - abs(change) if change < 0 else price
                price = price + change
                close = price

        # NY Session and rest of day - continuation or noise
        else:
            open_price = price
            change = rng.uniform(-0.0003, 0.0003)
            high = max(open_price, open_price + change) + rng.uniform(0, 0.0002)
            low = min(open_price, open_price + change) - rng.uniform(0, 0.0002)
            close = open_price + change
            price = close

        prices.append({'Open': open_price, 'High': high, 'Low': low, 'Close': close})

    df = pd.DataFrame(prices, index=dt_index)
    return df


def preprocess_data(df):
    """
    Augments the data with session information and Asia session range metrics.
    - Asia Session: 00:00 - 08:00 UTC
    - London Session: 08:00 - 16:00 UTC
    - New York Session: 13:00 - 21:00 UTC (overlaps with London)
    """
    df['hour'] = df.index.hour

    # Identify sessions
    df['session'] = 'Other'
    df.loc[(df['hour'] >= 0) & (df['hour'] < 8), 'session'] = 'Asia'
    df.loc[(df['hour'] >= 8) & (df['hour'] < 16), 'session'] = 'London'
    # Note: NY session can overlap, London takes precedence here for simplicity
    df.loc[(df['hour'] >= 13) & (df['hour'] < 21) & (df['session'] != 'London'), 'session'] = 'NY'

    # Calculate Asia session High and Low for each day
    asia_session_data = df[df['session'] == 'Asia'].copy()
    asia_session_data['date'] = asia_session_data.index.date

    daily_asia_high = asia_session_data.groupby('date')['High'].max()
    daily_asia_low = asia_session_data.groupby('date')['Low'].min()

    # Map HOA and LOA to the main dataframe
    df['date'] = df.index.date
    df['HOA'] = df['date'].map(daily_asia_high)
    df['LOA'] = df['date'].map(daily_asia_low)

    # Calculate Asia Range Percentage
    df['asia_range_percent'] = ((df['HOA'] - df['LOA']) / df['LOA']) * 100

    # Forward-fill the HOA/LOA and range values to make them available throughout the day
    df[['HOA', 'LOA', 'asia_range_percent']] = df[['HOA', 'LOA', 'asia_range_percent']].ffill()

    # Clean up helper columns
    df = df.drop(columns=['hour', 'date'])

    # Drop rows with NaN values that were created during mapping and ffilling
    df = df.dropna()

    return df

def identity(series):
    return series

class AsiaLiquidityGrabReversalStrategy(Strategy):
    """
    This strategy identifies the Asia trading session's high and low, then waits for a
    liquidity grab (a false breakout) during the London or New York session. It enters
    on a reversal candlestick pattern (engulfing) that closes back inside the Asia range.
    """
    max_asia_range_percent = 2.0  # Optimization parameter

    def init(self):
        # Map pre-calculated data to indicators to access them in `next()`
        self.session = self.I(identity, self.data.df['session'])
        self.hoa = self.I(identity, self.data.df['HOA'])
        self.loa = self.I(identity, self.data.df['LOA'])
        self.asia_range_percent = self.I(identity, self.data.df['asia_range_percent'])

        # State machine variables
        self.current_day = None
        self.daily_hoa = None
        self.daily_loa = None
        self.liquidity_grab_high = None
        self.liquidity_grab_low = None
        self.state = "WAITING_FOR_ASIA_CLOSE" # Initial state

    def next(self):
        # --- Daily State Reset ---
        current_date = self.data.index[-1].date()
        if self.current_day != current_date:
            self.current_day = current_date
            self.state = "WAITING_FOR_ASIA_CLOSE"
            self.daily_hoa = None
            self.daily_loa = None
            self.liquidity_grab_high = None
            self.liquidity_grab_low = None

        # --- State Machine Logic ---

        # 1. Waiting for Asia session to end to define the range
        if self.state == "WAITING_FOR_ASIA_CLOSE":
            if self.session[-1] == 2.0: # London session starts (assuming 2.0 is London's code)
                # Check pre-condition: Asia range must be within the threshold
                if self.asia_range_percent[-1] < self.max_asia_range_percent:
                    self.daily_hoa = self.hoa[-1]
                    self.daily_loa = self.loa[-1]
                    self.state = "WAITING_FOR_GRAB"

        # 2. Waiting for a liquidity grab above HOA or below LOA
        elif self.state == "WAITING_FOR_GRAB":
            # Bearish setup: Price grabs liquidity above HOA
            if self.data.High[-1] > self.daily_hoa:
                self.liquidity_grab_high = self.data.High[-1]
                self.state = "WAITING_FOR_SHORT_CONFIRMATION"
            # Bullish setup: Price grabs liquidity below LOA
            elif self.data.Low[-1] < self.daily_loa:
                self.liquidity_grab_low = self.data.Low[-1]
                self.state = "WAITING_FOR_LONG_CONFIRMATION"

        # 3a. Waiting for a bearish engulfing confirmation for a SHORT trade
        elif self.state == "WAITING_FOR_SHORT_CONFIRMATION":
            # Continuously update the highest point of the grab
            self.liquidity_grab_high = max(self.liquidity_grab_high, self.data.High[-1])

            # Bearish Engulfing check
            is_bearish_candle = self.data.Close[-1] < self.data.Open[-1]
            prev_body_size = abs(self.data.Close[-2] - self.data.Open[-2])
            curr_body_size = abs(self.data.Close[-1] - self.data.Open[-1])

            is_engulfing = (is_bearish_candle and
                            curr_body_size > prev_body_size and
                            self.data.Close[-1] < self.data.Open[-2] and
                            self.data.Open[-1] > self.data.Close[-2])

            # Entry condition: Engulfing candle that closes back below HOA
            if is_engulfing and self.data.Close[-1] < self.daily_hoa:
                sl = self.liquidity_grab_high * 1.0005 # SL with a small buffer
                tp = self.daily_loa
                # Ensure SL/TP are valid before placing trade
                if sl > self.data.Close[-1] and tp < self.data.Close[-1]:
                    self.sell(sl=sl, tp=tp)
                    self.state = "IN_TRADE" # Move to a neutral state until next day

        # 3b. Waiting for a bullish engulfing confirmation for a LONG trade
        elif self.state == "WAITING_FOR_LONG_CONFIRMATION":
            # Continuously update the lowest point of the grab
            self.liquidity_grab_low = min(self.liquidity_grab_low, self.data.Low[-1])

            # Bullish Engulfing check
            is_bullish_candle = self.data.Close[-1] > self.data.Open[-1]
            prev_body_size = abs(self.data.Close[-2] - self.data.Open[-2])
            curr_body_size = abs(self.data.Close[-1] - self.data.Open[-1])

            is_engulfing = (is_bullish_candle and
                            curr_body_size > prev_body_size and
                            self.data.Open[-1] < self.data.Close[-2] and
                            self.data.Close[-1] > self.data.Open[-2])

            # Entry condition: Engulfing candle that closes back above LOA
            if is_engulfing and self.data.Close[-1] > self.daily_loa:
                sl = self.liquidity_grab_low * 0.9995 # SL with a small buffer
                tp = self.daily_hoa
                # Ensure SL/TP are valid before placing trade
                if sl < self.data.Close[-1] and tp > self.data.Close[-1]:
                    self.buy(sl=sl, tp=tp)
                    self.state = "IN_TRADE"


if __name__ == '__main__':
    data = generate_synthetic_data(days=50)
    data = preprocess_data(data)

    # The 'session' column is categorical, convert to integer codes for backtesting.py
    data['session'] = pd.Categorical(data['session'], categories=['Other', 'Asia', 'London', 'NY'], ordered=True)
    session_codes = data['session'].cat.codes
    data['session'] = session_codes

    bt = Backtest(data, AsiaLiquidityGrabReversalStrategy, cash=100_000, commission=.002)

    # Optimize
    stats = bt.optimize(
        max_asia_range_percent=np.arange(0.5, 3.1, 0.5).tolist(),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.max_asia_range_percent > 0
    )

    print("Best stats:")
    print(stats)

    # Save results
    import os
    os.makedirs('results', exist_ok=True)

    # Sanitize stats for JSON serialization
    results = {
        'strategy_name': 'asia_liquidity_grab_reversal',
        'return': float(stats.get('Return [%]', 0.0)),
        'sharpe': float(stats.get('Sharpe Ratio', 0.0)),
        'max_drawdown': float(stats.get('Max. Drawdown [%]', 0.0)),
        'win_rate': float(stats.get('Win Rate [%]', 0.0)),
        'total_trades': int(stats.get('# Trades', 0))
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to results/temp_result.json")

    # Generate plot
    try:
        bt.plot(filename="results/asia_liquidity_grab_reversal.html", open_browser=False)
        print("Plot saved to results/asia_liquidity_grab_reversal.html")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")
