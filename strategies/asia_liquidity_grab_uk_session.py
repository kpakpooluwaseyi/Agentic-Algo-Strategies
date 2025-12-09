
import json
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

"""
Strategy: Asia Liquidity Grab during UK Session
"""

def preprocess_data(df: pd.DataFrame, asia_start_hour=0, asia_end_hour=8, uk_start_hour=7, uk_end_hour=16) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame to add session information and daily Asia range data.
    - Identifies Asia and UK session bars.
    - Calculates the high and low of the daily Asia session.
    - Calculates the Asia session range in percentage.
    - Forwards fills the session data to be available throughout the day.
    """
    df = df.copy()

    # Ensure the index is a DatetimeIndex and localized to UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    # 1. Identify sessions based on UTC hour
    df['hour'] = df.index.hour
    df['is_asia'] = (df['hour'] >= asia_start_hour) & (df['hour'] < asia_end_hour)
    df['is_uk'] = (df['hour'] >= uk_start_hour) & (df['hour'] < uk_end_hour)

    # 2. Calculate daily Asia high and low
    # Create a daily grouping key
    df['day'] = df.index.date

    # Calculate high/low for Asia session only
    asia_session_data = df[df['is_asia']].copy()
    daily_asia_high = asia_session_data.groupby('day')['High'].max()
    daily_asia_low = asia_session_data.groupby('day')['Low'].min()

    # Map daily values back to the original dataframe
    df['asia_high'] = df['day'].map(daily_asia_high)
    df['asia_low'] = df['day'].map(daily_asia_low)

    # Forward-fill the daily high/low to make it available throughout the day
    df['asia_high'].ffill(inplace=True)
    df['asia_low'].ffill(inplace=True)

    # 3. Calculate Asia range percentage
    df['asia_range_perc'] = ((df['asia_high'] - df['asia_low']) / df['asia_low']) * 100

    # 4. Identify the first bar of the UK session
    df['is_first_uk_bar'] = (df['is_uk'] & ~df['is_uk'].shift(1).fillna(False))

    # 5. Calculate daily 50% level for TP2
    daily_high = df.groupby('day')['High'].transform('max')
    daily_low = df.groupby('day')['Low'].transform('min')
    df['daily_50_level'] = daily_low + (daily_high - daily_low) * 0.5

    # Clean up helper columns
    df.drop(columns=['hour', 'day'], inplace=True)

    # Drop rows with NaN values resulting from initial calculations
    df.dropna(inplace=True)

    return df

class AsiaLiquidityGrabUkSessionStrategy(Strategy):
    # Strategy parameters to be optimized
    asia_range_max_perc = 2.0
    sl_buffer_pips = 10 # Stop loss buffer in pips

    def init(self):
        # Indicators are pre-calculated and accessed directly from the data frame
        self.asia_high = self.data.asia_high
        self.asia_low = self.data.asia_low
        self.asia_range_perc = self.data.asia_range_perc
        self.is_uk_session = self.data.is_uk
        self.daily_50_level = self.data.daily_50_level

    def next(self):
        # Pre-computation of values for the current step
        price = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]

        # Candlestick pattern detection
        # Bullish Engulfing
        is_bullish_engulfing = (self.data.Open[-2] > self.data.Close[-2] and  # Previous bar is bearish
                                self.data.Open[-1] < self.data.Close[-1] and  # Current bar is bullish
                                self.data.Open[-1] < self.data.Close[-2] and
                                self.data.Close[-1] > self.data.Open[-2])

        # Bearish Engulfing
        is_bearish_engulfing = (self.data.Open[-2] < self.data.Close[-2] and  # Previous bar is bullish
                                self.data.Open[-1] > self.data.Close[-1] and  # Current bar is bearish
                                self.data.Open[-1] > self.data.Close[-2] and
                                self.data.Close[-1] < self.data.Open[-2])

        # Hammer (fixed): long lower wick, small body relative to candle range
        full_range = self.data.High[-1] - self.data.Low[-1]
        body_size = abs(self.data.Close[-1] - self.data.Open[-1])
        lower_wick = self.data.Open[-1] - self.data.Low[-1] if self.data.Close[-1] > self.data.Open[-1] else self.data.Close[-1] - self.data.Low[-1]
        is_hammer = full_range > 0 and lower_wick > full_range * 0.6 and body_size < full_range * 0.3

        # Condition checks
        is_valid_asia_range = self.asia_range_perc[-1] < self.asia_range_max_perc
        sl_buffer = self.sl_buffer_pips * 0.0001 # Convert pips to price

        # === Entry Logic ===
        if self.is_uk_session[-1] and is_valid_asia_range and not self.position:
            # SHORT entry logic
            liquidity_grab_above = current_high > self.asia_high[-1]
            if liquidity_grab_above and is_bearish_engulfing:
                sl = current_high + sl_buffer
                tp1 = self.asia_low[-1]
                tp2 = self.daily_50_level[-1]
                # Validate TPs before placing trades
                if price > tp1:
                    self.sell(sl=sl, tp=tp1, size=0.5)
                if price > tp2 and tp2 < tp1: # Ensure TP2 is a further target
                    self.sell(sl=sl, tp=tp2, size=0.5)

            # LONG entry logic
            liquidity_grab_below = current_low < self.asia_low[-1]
            if liquidity_grab_below and (is_bullish_engulfing or is_hammer):
                sl = current_low - sl_buffer
                tp1 = self.asia_high[-1]
                tp2 = self.daily_50_level[-1]
                # Validate TPs before placing trades
                if price < tp1:
                    self.buy(sl=sl, tp=tp1, size=0.5)
                if price < tp2 and tp2 > tp1: # Ensure TP2 is a further target
                    self.buy(sl=sl, tp=tp2, size=0.5)

if __name__ == '__main__':
    def generate_perfect_test_data():
        """Generates a small, specific dataset designed to trigger the strategy's logic."""
        # Day 1: Setup for a SHORT trade
        # Asia Session (00:00 - 07:45): Low volatility, range between 1.1010 and 1.1020
        # UK Session (08:00 - ...): Liquidity grab above Asia high, then bearish engulfing
        day1_asia = pd.DataFrame({
            'Open': [1.1015, 1.1012, 1.1018], 'High': [1.1020, 1.1018, 1.1019],
            'Low': [1.1011, 1.1010, 1.1015], 'Close': [1.1012, 1.1018, 1.1016]
        }, index=pd.to_datetime(['2023-01-02 04:00', '2023-01-02 05:00', '2023-01-02 06:00'], utc=True))

        day1_uk = pd.DataFrame({
            'Open': [1.1016, 1.1025, 1.1035], 'High': [1.1022, 1.1030, 1.1038], # Previous candle is bullish
            'Low': [1.1015, 1.1021, 1.1010], 'Close': [1.1025, 1.1032, 1.1012]  # Bearish engulfing
        }, index=pd.to_datetime(['2023-01-02 07:45', '2023-01-02 08:00', '2023-01-02 08:15'], utc=True)) # Grab at 08:00, Engulfing at 08:15

        # Day 2: Setup for a LONG trade
        # Asia Session: Range between 1.0950 and 1.0960
        # UK Session: Liquidity grab below Asia low, then hammer
        day2_asia = pd.DataFrame({
            'Open': [1.0955, 1.0952, 1.0958], 'High': [1.0960, 1.0958, 1.0959],
            'Low': [1.0951, 1.0950, 1.0955], 'Close': [1.0952, 1.0958, 1.0956]
        }, index=pd.to_datetime(['2023-01-03 04:00', '2023-01-03 05:00', '2023-01-03 06:00'], utc=True))

        day2_uk = pd.DataFrame({
            'Open': [1.0956, 1.0945, 1.0951], 'High': [1.0958, 1.0950, 1.0965],
            'Low': [1.0950, 1.0935, 1.0948], 'Close': [1.0950, 1.0948, 1.0962] # Hammer at 08:15
        }, index=pd.to_datetime(['2023-01-03 07:45', '2023-01-03 08:00', '2023-01-03 08:15'], utc=True))

        # Combine and fill missing data
        full_data = pd.concat([day1_asia, day1_uk, day2_asia, day2_uk])
        date_range = pd.date_range(start=full_data.index.min(), end=full_data.index.max(), freq='15min')
        return full_data.reindex(date_range).fillna(method='ffill')

    def generate_forex_data(days=90):
        """Generates synthetic 15M forex data with session-based characteristics."""
        n_periods = days * 24 * 4  # 96 periods per day
        index = pd.date_range(start='2023-01-01', periods=n_periods, freq='15min', tz='UTC')

        # Base price movement with some noise
        base_price = 1.1000
        price = base_price + np.random.randn(n_periods).cumsum() * 0.0001

        # Introduce session volatility
        for i in range(len(index)):
            hour = index[i].hour
            # Asia session: low volatility
            if 0 <= hour < 8:
                price[i] += np.sin(i / 10) * 0.0005
            # UK session: higher volatility
            elif 7 <= hour < 16:
                price[i] += np.random.randn() * 0.0003

        # Create OHLC data
        df = pd.DataFrame(index=index)
        df['Close'] = price
        df['Open'] = df['Close'].shift(1).fillna(method='bfill')
        df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.rand(n_periods) * 0.0005
        df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.rand(n_periods) * 0.0005

        return df

    # 1. Load or generate data
    data = generate_forex_data(days=180) # Use 180 days for a more robust backtest
    # data = generate_perfect_test_data() # Use perfect data for logic verification

    # 2. Preprocess data
    processed_data = preprocess_data(data)

    if processed_data.empty:
        print("Data for backtest is empty after preprocessing. Check session times and data range.")
    else:
        # 3. Run backtest and optimization
        bt = Backtest(processed_data, AsiaLiquidityGrabUkSessionStrategy, cash=10000, commission=.0002, finalize_trades=True)

        # stats = bt.run() # Use run() for single backtest verification
        stats = bt.optimize(
            asia_range_max_perc=np.arange(0.5, 3.5, 0.5).tolist(),
            sl_buffer_pips=range(5, 25, 5),
            maximize='Sharpe Ratio'
        )

        print(stats)

        # 4. Save results
        import os
        os.makedirs('results', exist_ok=True)

        # Handle cases with zero trades
        if stats['# Trades'] > 0:
            result_data = {
                'strategy_name': 'asia_liquidity_grab_uk_session',
                'return': float(stats['Return [%]']),
                'sharpe': float(stats['Sharpe Ratio']),
                'max_drawdown': float(stats['Max. Drawdown [%]']),
                'win_rate': float(stats['Win Rate [%]']),
                'total_trades': int(stats['# Trades'])
            }
        else:
            result_data = {
                'strategy_name': 'asia_liquidity_grab_uk_session',
                'return': 0.0,
                'sharpe': None,
                'max_drawdown': float(stats['Max. Drawdown [%]']),
                'win_rate': 0.0,
                'total_trades': 0
            }

        with open('results/temp_result.json', 'w') as f:
            json.dump(result_data, f, indent=2)

        print("Backtest results saved to results/temp_result.json")

        # 5. Generate plot
        bt.plot()
