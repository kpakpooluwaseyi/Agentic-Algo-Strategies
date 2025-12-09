import json
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

# Define session times in UTC hours
ASIA_START_HOUR = 20
ASIA_END_HOUR = 9
UK_START_HOUR = 9
UK_END_HOUR = 17

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Asia session High/Low (HOA/LOA) to the DataFrame.
    The Asia session for a given trading day (e.g., Tuesday) is defined
    as running from 20:00 on the previous calendar day (Monday) to
    08:00 on the current calendar day (Tuesday).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    # Ensure timezone is set, defaulting to UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    # Create a unique ID for each Asia session. A new session starts at 20:00 UTC.
    # We can identify the session by finding the date after shifting the time back by 20 hours.
    session_date = (df.index - pd.Timedelta(hours=ASIA_START_HOUR)).date
    df['asia_session_id'] = pd.factorize(session_date)[0]

    # Identify all candles that fall within the Asia session time window
    asia_session_mask = (df.index.hour >= ASIA_START_HOUR) | (df.index.hour < ASIA_END_HOUR)

    # Calculate HOA and LOA for each session
    asia_df = df[asia_session_mask].copy()
    if not asia_df.empty:
        # Group by the session ID and find the max high and min low
        session_stats = asia_df.groupby('asia_session_id').agg(
            HOA=('High', 'max'),
            LOA=('Low', 'min')
        )

        # Map these stats back to the main dataframe using a more index-friendly method
        hoa_map = session_stats['HOA']
        loa_map = session_stats['LOA']
        df['HOA'] = df['asia_session_id'].map(hoa_map)
        df['LOA'] = df['asia_session_id'].map(loa_map)
    else:
        df['HOA'] = np.nan
        df['LOA'] = np.nan

    # Forward-fill the HOA/LOA values to make them available throughout the day
    df['HOA'] = df['HOA'].ffill()
    df['LOA'] = df['LOA'].ffill()

    # Drop rows where we don't have session data yet (e.g., the very first session)
    df.dropna(subset=['HOA', 'LOA'], inplace=True)

    df['Asia_Range'] = df['HOA'] - df['LOA']
    df['Asia_Range_Pct'] = (df['Asia_Range'] / df['LOA']) * 100

    return df

class AsiaLiquidityGrabReversalStrategy(Strategy):
    """
    A strategy that trades reversals after a liquidity grab above/below the
    Asia session high/low during the London session.
    """
    # Optimization parameters
    risk_pct = 1.0
    asia_range_max_pct = 2.0

    def init(self):
        # State variables to track liquidity grabs
        self.current_day = None
        self.grabbed_hoa = False
        self.grabbed_loa = False

    def position_sizer(self, entry_price, sl_price):
        """Calculates position size based on fixed fractional risk."""
        if self.equity <= 0 or abs(entry_price - sl_price) == 0:
            return 0

        risk_per_trade = self.equity * (self.risk_pct / 100)
        sl_distance_dollars = abs(entry_price - sl_price)
        size = risk_per_trade / sl_distance_dollars

        # In a real forex scenario, you wouldn't need to floor this.
        # But backtesting.py requires integer sizes for non-margin accounts.
        return int(size)

    def next(self):
        current_time = self.data.index[-1]

        # Reset state at the start of the UK session
        if current_time.hour == UK_START_HOUR and current_time.hour != self.data.index[-2].hour:
            self.grabbed_hoa = False
            self.grabbed_loa = False

        # Only trade during the UK session
        if not (UK_START_HOUR <= current_time.hour < UK_END_HOUR):
            return

        # Ensure we have valid HOA/LOA data
        hoa = self.data.HOA[-1]
        loa = self.data.LOA[-1]
        if np.isnan(hoa) or np.isnan(loa):
            return

        # --- Liquidity Grab Detection ---
        if self.data.High[-1] > hoa:
            self.grabbed_hoa = True
        if self.data.Low[-1] < loa:
            self.grabbed_loa = True

        # --- Entry Logic ---
        if self.position or len(self.data.Close) < 2:
            return

        is_asia_range_valid = self.data.Asia_Range_Pct[-1] < self.asia_range_max_pct

        # --- SHORT ENTRY ---
        if self.grabbed_hoa and is_asia_range_valid:
            # Bearish Engulfing Pattern
            is_prev_bullish = self.data.Close[-2] > self.data.Open[-2]
            is_curr_bearish = self.data.Close[-1] < self.data.Open[-1]
            is_engulfing = self.data.Open[-1] >= self.data.Close[-2] and self.data.Close[-1] < self.data.Open[-2]

            if is_prev_bullish and is_curr_bearish and is_engulfing and self.data.Close[-1] < hoa:
                sl_price = self.data.High[-1]
                tp_price = loa

                if tp_price < self.data.Close[-1]:
                    size = self.position_sizer(self.data.Close[-1], sl_price)
                    if size > 0:
                        self.sell(sl=sl_price, tp=tp_price, size=size)
                        self.grabbed_hoa = False

        # --- LONG ENTRY ---
        if self.grabbed_loa and is_asia_range_valid:
            # Bullish Engulfing Pattern
            is_prev_bearish = self.data.Close[-2] < self.data.Open[-2]
            is_curr_bullish = self.data.Close[-1] > self.data.Open[-1]
            is_engulfing = self.data.Open[-1] <= self.data.Close[-2] and self.data.Close[-1] > self.data.Open[-2]

            if is_prev_bearish and is_curr_bullish and is_engulfing and self.data.Close[-1] > loa:
                sl_price = self.data.Low[-1]
                tp_price = hoa

                if tp_price > self.data.Close[-1]:
                    size = self.position_sizer(self.data.Close[-1], sl_price)
                    if size > 0:
                        self.buy(sl=sl_price, tp=tp_price, size=size)
                        self.grabbed_loa = False

if __name__ == '__main__':
    def generate_synthetic_data(days=30):
        """
        Generates synthetic 24-hour OHLCV data for testing the strategy.
        This function creates predictable patterns of Asia session consolidation,
        followed by a liquidity grab and reversal during the London session.
        """
        n_periods = days * 24 * 4  # 15-min periods in a day
        index = pd.date_range(start='2023-01-01', periods=n_periods, freq='15min', tz='UTC')

        base_price = 1.0
        data = []

        np.random.seed(42) # for reproducibility

        for day in range(days):
            # --- ASIA SESSION (20:00 day-1 to 09:00 day) ---
            # Create a consolidation range
            asia_mid_price = base_price + np.random.uniform(-0.005, 0.005)
            asia_range = np.random.uniform(0.002, 0.01) # ~0.2% to 1% range
            hoa = asia_mid_price + asia_range / 2
            loa = asia_mid_price - asia_range / 2

            day_start_idx = day * 96

            # --- LIQUIDITY GRAB & REVERSAL (UK SESSION) ---
            # Randomly decide to do a bullish, bearish, or no-trade day
            scenario = np.random.choice(['bullish', 'bearish', 'no_trade'])

            for i in range(96): # 96 periods in a day
                current_idx = day_start_idx + i
                current_time = index[current_idx]

                open_price = data[-1]['Close'] if data else asia_mid_price

                if ASIA_START_HOUR <= current_time.hour or current_time.hour < ASIA_END_HOUR:
                    # In Asia Session: Consolidate
                    close = open_price + np.random.uniform(-asia_range/4, asia_range/4)
                    high = max(open_price, close) + np.random.uniform(0, asia_range/8)
                    low = min(open_price, close) - np.random.uniform(0, asia_range/8)
                    # Clamp to HOA/LOA
                    high = min(high, hoa)
                    low = max(low, loa)
                    close = min(max(close, low), high)

                elif UK_START_HOUR <= current_time.hour < UK_END_HOUR:
                    # UK Session: Potential for grab and reversal

                    # Grab candle (e.g., at 09:15)
                    if current_time.hour == 9 and current_time.minute == 15:
                        if scenario == 'bearish':
                            high = hoa + 0.001 # Grab liquidity above HOA
                            low = open_price
                            close = high - 0.0005
                        elif scenario == 'bullish':
                            low = loa - 0.001 # Grab liquidity below LOA
                            high = open_price
                            close = low + 0.0005
                        else: # no_trade
                             close, high, low = open_price, open_price*1.0005, open_price*0.9995

                    # Reversal candle (e.g., at 09:30)
                    elif current_time.hour == 9 and current_time.minute == 30:
                        prev_candle = data[-1]
                        if scenario == 'bearish' and prev_candle['High'] > hoa:
                            # Bearish engulfing
                            open_price = prev_candle['Close'] + 0.0001
                            close = prev_candle['Open'] - 0.0005
                            high = open_price
                            low = close
                        elif scenario == 'bullish' and prev_candle['Low'] < loa:
                            # Bullish engulfing
                            open_price = prev_candle['Close'] - 0.0001
                            close = prev_candle['Open'] + 0.0005
                            low = open_price
                            high = close
                        else: # no_trade or failed grab
                            close, high, low = open_price, open_price*1.0005, open_price*0.9995

                    # Trend towards TP
                    elif current_time.hour >= 10:
                        if scenario == 'bearish':
                            # Move towards LOA
                            close = open_price - np.random.uniform(0.0001, 0.001)
                            low = close
                            high = open_price
                        elif scenario == 'bullish':
                             # Move towards HOA
                            close = open_price + np.random.uniform(0.0001, 0.001)
                            high = close
                            low = open_price
                        else:
                             close, high, low = open_price, open_price*1.0005, open_price*0.9995
                    else:
                        # Other UK session times
                        close, high, low = open_price, open_price*1.001, open_price*0.999
                else:
                    # Outside of key sessions
                    close = open_price + np.random.uniform(-0.001, 0.001)
                    high = max(open_price, close)
                    low = min(open_price, close)

                data.append({'Open': open_price, 'High': high, 'Low': low, 'Close': close, 'Volume': np.random.randint(100, 1000)})

            # Set base for next day
            base_price = data[-1]['Close']

        df = pd.DataFrame(data, index=index)
        return df

    # Generate synthetic data
    # NOTE: Using real data is preferable, but for a self-contained example,
    # we generate data that should trigger the strategy's logic.
    data = generate_synthetic_data(days=90)

    # Preprocess the data to add session info
    data = preprocess_data(data)

    # Initialize and run the backtest
    # Setting a high cash value to ensure position sizer can always open trades
    bt = Backtest(data, AsiaLiquidityGrabReversalStrategy, cash=1_000_000, commission=.002)

    # Optimize the strategy
    stats = bt.optimize(
        asia_range_max_pct=np.arange(1.0, 3.0, 0.5).tolist(),
        risk_pct=[1.0], # Keep risk fixed for this optimization
        maximize='Sharpe Ratio',
        constraint=lambda p: p.asia_range_max_pct > 0
    )

    print("Best stats:", stats)

    # --- Output Requirements ---
    import os
    os.makedirs('results', exist_ok=True)

    # Handle cases with no trades
    if stats['# Trades'] > 0:
        result_dict = {
            'strategy_name': 'asia_liquidity_grab_reversal',
            'return': float(stats.get('Return [%]', 0.0)),
            'sharpe': float(stats.get('Sharpe Ratio', 0.0)),
            'max_drawdown': float(stats.get('Max. Drawdown [%]', 0.0)),
            'win_rate': float(stats.get('Win Rate [%]', 0.0)),
            'total_trades': int(stats.get('# Trades', 0))
        }
    else:
        result_dict = {
            'strategy_name': 'asia_liquidity_grab_reversal',
            'return': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0,
            'win_rate': 0.0, 'total_trades': 0
        }

    with open('results/temp_result.json', 'w') as f:
        json.dump(result_dict, f, indent=2)

    print("Results saved to results/temp_result.json")

    # Generate plot
    bt.plot(filename="results/asia_liquidity_grab_reversal.html")
    print("Plot saved to results/asia_liquidity_grab_reversal.html")
