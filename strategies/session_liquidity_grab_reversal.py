from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json

# Pass-through function to use DataFrame columns as indicators
def pass_through(series):
    return series

class SessionLiquidityGrabReversalStrategy(Strategy):
    # Optimizable parameter for stop-loss buffer
    sl_buffer_pct = 0.01
    uk_session_code = 1 # Default for 'UK', can be overridden in optimize()

    def init(self):
        # State variables to track the two-bar pattern
        self.liquidity_grab_bar_high = None
        self.liquidity_grab_bar_low = None

        # Map pre-processed data columns to indicators
        self.session = self.I(pass_through, self.data.df['session'].values, plot=False)
        self.asia_high = self.I(pass_through, self.data.df['asia_high'].values)
        self.asia_low = self.I(pass_through, self.data.df['asia_low'].values)
        self.asia_range_pct = self.I(pass_through, self.data.df['asia_range_pct'].values, plot=False)

    def next(self):
        # Ensure we have valid Asia range data for the current bar
        if np.isnan(self.asia_high[-1]) or np.isnan(self.asia_low[-1]):
            return

        # Only trade during the UK session and if a position is not already open
        if self.session[-1] == self.uk_session_code and not self.position:

            # --- STATE 1: WAITING FOR SHORT CONFIRMATION ---
            if self.liquidity_grab_bar_high is not None:
                is_bearish_engulfing = self.data.Close[-1] < self.data.Open[-2] and \
                                       self.data.Open[-1] > self.data.Close[-2] and \
                                       self.data.Close[-1] < self.asia_high[-1]

                if is_bearish_engulfing:
                    sl = self.liquidity_grab_bar_high * (1 + self.sl_buffer_pct / 100)
                    tp = self.asia_low[-1]
                    self.sell(sl=sl, tp=tp)

                # Reset state regardless of entry; we only get one chance at confirmation
                self.liquidity_grab_bar_high = None

            # --- STATE 2: WAITING FOR LONG CONFIRMATION ---
            elif self.liquidity_grab_bar_low is not None:
                is_bullish_engulfing = self.data.Open[-1] < self.data.Close[-2] and \
                                       self.data.Close[-1] > self.data.Open[-2] and \
                                       self.data.Close[-1] > self.asia_low[-1]

                if is_bullish_engulfing:
                    sl = self.liquidity_grab_bar_low * (1 - self.sl_buffer_pct / 100)
                    tp = self.asia_high[-1]
                    self.buy(sl=sl, tp=tp)

                # Reset state regardless of entry
                self.liquidity_grab_bar_low = None

            # --- STATE 3: NO ACTIVE SETUP, LOOKING FOR A NEW ONE ---
            else:
                is_asia_range_valid = self.asia_range_pct[-1] < 2.0
                if is_asia_range_valid:
                    # Check for short setup first
                    if self.data.High[-1] > self.asia_high[-1]:
                        self.liquidity_grab_bar_high = self.data.High[-1]
                    # Else, check for long setup
                    elif self.data.Low[-1] < self.asia_low[-1]:
                        self.liquidity_grab_bar_low = self.data.Low[-1]

def generate_forex_data():
    """
    Generates synthetic 15-minute forex data with specific patterns for the
    Session Liquidity Grab Reversal strategy.
    """
    # Create a date range for a few days
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='15min')
    n = len(dates)

    # Base price movement with some noise
    price = 1.1000 + np.random.randn(n).cumsum() * 0.0001

    # Create DataFrame
    data = pd.DataFrame({
        'Open': price,
        'High': price,
        'Low': price,
        'Close': price
    }, index=dates)

    # Generate more realistic candles
    data['Open'] = data['Close'].shift(1).fillna(method='bfill')
    data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 0.0005, n)
    data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 0.0005, n)

    # --- Inject a specific SHORT pattern ---
    # Asia session on Jan 4th, 00:00 to 08:00
    asia_session_mask = (data.index.date == pd.to_datetime('2023-01-04').date()) & \
                        (data.index.hour >= 0) & (data.index.hour < 8)
    asia_high = data.loc[asia_session_mask, 'High'].max()

    # UK session open on Jan 4th, starting 08:00
    uk_session_start_time = pd.to_datetime('2023-01-04 08:00')
    grab_candle_time = uk_session_start_time
    engulfing_candle_time = grab_candle_time + pd.Timedelta(minutes=15)

    # 1. Liquidity grab candle (spike above Asia high)
    data.loc[grab_candle_time, 'High'] = asia_high + 0.0002
    data.loc[grab_candle_time, 'Open'] = asia_high - 0.0001
    data.loc[grab_candle_time, 'Close'] = asia_high + 0.0001
    data.loc[grab_candle_time, 'Low'] = data.loc[grab_candle_time, 'Open'] - 0.0001


    # 2. Bearish engulfing candle (closes back below Asia high)
    previous_close = data.loc[grab_candle_time, 'Close']
    previous_open = data.loc[grab_candle_time, 'Open']
    data.loc[engulfing_candle_time, 'Open'] = previous_close + 0.0001
    data.loc[engulfing_candle_time, 'Close'] = previous_open - 0.0002
    data.loc[engulfing_candle_time, 'High'] = data.loc[engulfing_candle_time, 'Open'] + 0.0001
    data.loc[engulfing_candle_time, 'Low'] = data.loc[engulfing_candle_time, 'Close'] - 0.0001


    # --- Inject a specific LONG pattern ---
    # Asia session on Jan 6th, 00:00 to 08:00
    asia_session_mask_long = (data.index.date == pd.to_datetime('2023-01-06').date()) & \
                             (data.index.hour >= 0) & (data.index.hour < 8)
    asia_low = data.loc[asia_session_mask_long, 'Low'].min()

    # UK session open on Jan 6th, starting 08:00
    uk_session_start_time_long = pd.to_datetime('2023-01-06 08:00')
    grab_candle_time_long = uk_session_start_time_long
    engulfing_candle_time_long = grab_candle_time_long + pd.Timedelta(minutes=15)

    # 1. Liquidity grab candle (spike below Asia low)
    data.loc[grab_candle_time_long, 'Low'] = asia_low - 0.0002
    data.loc[grab_candle_time_long, 'Open'] = asia_low + 0.0001
    data.loc[grab_candle_time_long, 'Close'] = asia_low - 0.0001
    data.loc[grab_candle_time_long, 'High'] = data.loc[grab_candle_time_long, 'Open'] + 0.0001

    # 2. Bullish engulfing candle (closes back above Asia low)
    previous_close_long = data.loc[grab_candle_time_long, 'Close']
    previous_open_long = data.loc[grab_candle_time_long, 'Open']
    data.loc[engulfing_candle_time_long, 'Open'] = previous_close_long - 0.0001
    data.loc[engulfing_candle_time_long, 'Close'] = previous_open_long + 0.0002
    data.loc[engulfing_candle_time_long, 'Low'] = data.loc[engulfing_candle_time_long, 'Open'] - 0.0001
    data.loc[engulfing_candle_time_long, 'High'] = data.loc[engulfing_candle_time_long, 'Close'] + 0.0001

    return data

def preprocess_data(df):
    """
    Preprocesses the data to add session information and Asia session highs/lows.
    """
    # Define session times (UTC)
    asia_session_end = 8
    uk_session_end = 16

    # Determine session for each candle
    df['session'] = 'US'
    df.loc[df.index.hour < uk_session_end, 'session'] = 'UK'
    df.loc[df.index.hour < asia_session_end, 'session'] = 'Asia'

    # Calculate daily Asia session high and low
    asia_session_data = df[df['session'] == 'Asia']
    daily_asia_high = asia_session_data.groupby(asia_session_data.index.date)['High'].max()
    daily_asia_low = asia_session_data.groupby(asia_session_data.index.date)['Low'].min()

    # Map daily values to the entire day
    df['asia_high'] = pd.Series(df.index.date, index=df.index).map(daily_asia_high)
    df['asia_low'] = pd.Series(df.index.date, index=df.index).map(daily_asia_low)

    # Forward-fill the values to apply them to subsequent sessions
    df['asia_high'] = df['asia_high'].ffill()
    df['asia_low'] = df['asia_low'].ffill()

    # Calculate Asia range percentage
    df['asia_range_pct'] = (df['asia_high'] - df['asia_low']) / df['asia_low'] * 100

    # Drop rows with NaN values that were created during processing
    df.dropna(inplace=True)

    return df

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats to handle missing keys and non-serializable types.
    """
    # Use .get() to provide a default value (0) if a key is missing, e.g., if no trades occur
    return {
        'strategy_name': 'session_liquidity_grab_reversal',
        'return': float(stats.get('Return [%]', 0.0)),
        'sharpe': float(stats.get('Sharpe Ratio', 0.0)),
        'max_drawdown': float(stats.get('Max. Drawdown [%]', 0.0)),
        'win_rate': float(stats.get('Win Rate [%]', 0.0)),
        'total_trades': int(stats.get('# Trades', 0))
    }


if __name__ == '__main__':
    # Load or generate data
    data = generate_forex_data()

    # Preprocess data
    data = preprocess_data(data)

    # Define session categories and factorize to get consistent integer codes
    session_categories = ['Asia', 'UK', 'US']
    # Use pd.Categorical to ensure the order and get codes
    data['session'] = pd.Categorical(data['session'], categories=session_categories, ordered=True)
    session_codes = data['session'].cat.codes
    data['session'] = session_codes

    # Get the integer code for the UK session to pass to the strategy
    uk_session_code = session_categories.index('UK')

    # Run backtest, passing the UK session code to the strategy
    bt = Backtest(data, SessionLiquidityGrabReversalStrategy, cash=10000, commission=.002)

    # Optimize
    stats = bt.optimize(
        sl_buffer_pct=np.arange(0.01, 0.1, 0.01).tolist(),
        uk_session_code=uk_session_code, # Pass as a single value
        maximize='Sharpe Ratio'
    )

    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    sanitized_results = sanitize_stats(stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_results, f, indent=2)

    # Generate plot, handling potential errors
    try:
        bt.plot()
    except TypeError as e:
        print(f"Could not generate plot due to a TypeError (often a pandas/plotting library incompatibility): {e}")
