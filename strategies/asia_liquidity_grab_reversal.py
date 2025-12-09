
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data to add session information and Asia session highs/lows.
    Asia Session: 00:00 - 08:00 UTC
    London Session: 08:00 - 16:00 UTC
    New York Session: 13:00 - 21:00 UTC (overlaps with London)
    """
    df['hour'] = df.index.hour

    conditions = [
        (df['hour'] >= 0) & (df['hour'] < 8),
        (df['hour'] >= 8) & (df['hour'] < 16),
        (df['hour'] >= 13) & (df['hour'] < 21)
    ]
    choices = ['Asia', 'London', 'New York']
    df['session'] = np.select(conditions, choices, default='Other')

    # Calculate daily Asia high and low
    asia_session_data = df[df['session'] == 'Asia'].copy()
    daily_asia_high = asia_session_data.groupby(asia_session_data.index.date)['High'].max()
    daily_asia_low = asia_session_data.groupby(asia_session_data.index.date)['Low'].min()

    # Map the daily values to each row
    df['asia_high'] = df.index.map(lambda x: daily_asia_high.get(x.date()))
    df['asia_low'] = df.index.map(lambda x: daily_asia_low.get(x.date()))

    # Forward-fill the Asia high/low for the rest of the day
    df['asia_high'] = df['asia_high'].ffill()
    df['asia_low'] = df['asia_low'].ffill()

    df = df.dropna()

    return df

class AsiaLiquidityGrabReversalStrategy(Strategy):
    asia_range_filter_pct = 0.02
    sl_buffer_pct = 0.001

    def init(self):
        self.grabbed_high = False
        self.grabbed_low = False

        # Use a simple pass-through function to make pre-calculated columns available
        def pass_through(series):
            return series

        self.asia_high = self.I(pass_through, self.data.df['asia_high'])
        self.asia_low = self.I(pass_through, self.data.df['asia_low'])
        self.session = self.I(pass_through, self.data.df['session'].astype('category').cat.codes)
        self.session_map = dict(enumerate(self.data.df['session'].astype('category').cat.categories))

        self.current_day = -1


    def next(self):
        # Reset state at the start of a new day
        current_date = self.data.index[-1].date()
        if self.current_day != current_date:
            self.current_day = current_date
            self.grabbed_high = False
            self.grabbed_low = False

        # Get current session name
        current_session_code = self.session[-1]
        current_session = self.session_map.get(current_session_code, 'Other')

        if len(self.data.Close) < 2:
            return

        # Only trade during the London session
        if current_session != 'London':
            return

        # Check if a position is already open
        if self.position:
            return

        # Check Asia range filter
        asia_high = self.asia_high[-1]
        asia_low = self.asia_low[-1]
        if (asia_high - asia_low) / asia_low > self.asia_range_filter_pct:
            return

        # --- SHORT ENTRY LOGIC ---
        # 1. Price grabs liquidity above Asia High
        if self.data.High[-1] > asia_high:
            self.grabbed_high = True

        # 2. Bearish engulfing candle forms after grab
        if self.grabbed_high:
            is_prev_bullish = self.data.Close[-2] > self.data.Open[-2]
            is_curr_bearish = self.data.Close[-1] < self.data.Open[-1]
            is_engulfing = self.data.Open[-1] >= self.data.Close[-2] and self.data.Close[-1] < self.data.Open[-2]

            if is_prev_bullish and is_curr_bearish and is_engulfing:
                # 3. Enter short
                sl = self.data.High[-1] * (1 + self.sl_buffer_pct)
                tp = asia_low
                self.sell(sl=sl, tp=tp)
                self.grabbed_high = False # Reset after entry

        # --- LONG ENTRY LOGIC ---
        # 1. Price grabs liquidity below Asia Low
        if self.data.Low[-1] < asia_low:
            self.grabbed_low = True

        # 2. Bullish engulfing candle forms after grab
        if self.grabbed_low:
            is_prev_bearish = self.data.Close[-2] < self.data.Open[-2]
            is_curr_bullish = self.data.Close[-1] > self.data.Open[-1]
            is_engulfing = self.data.Close[-1] >= self.data.Open[-2] and self.data.Open[-1] < self.data.Close[-2]

            if is_prev_bearish and is_curr_bullish and is_engulfing:
                # 3. Enter long
                sl = self.data.Low[-1] * (1 - self.sl_buffer_pct)
                tp = asia_high
                self.buy(sl=sl, tp=tp)
                self.grabbed_low = False # Reset after entry

def generate_synthetic_data():
    """Generates synthetic 24-hour data for testing."""
    n_days = 100
    freq = '15min'
    date_rng = pd.date_range(start='2023-01-01', periods=n_days * 24 * 4, freq=freq)

    price = 100
    prices = []

    # Simple sine wave for daily seasonality
    day_wave = np.sin(np.linspace(0, 2 * np.pi, 96)) * 2 # 96 periods of 15 min in a day

    for i in range(len(date_rng)):
        price += (np.random.randn() * 0.1) + (day_wave[i % 96] * 0.1)
        prices.append(price)

    df = pd.DataFrame(prices, index=date_rng, columns=['Close'])
    df['Open'] = df['Close'].shift(1).fillna(df['Close'])
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 0.1, len(df))
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 0.1, len(df))
    df['Volume'] = np.random.randint(100, 1000, len(df))

    return df

if __name__ == '__main__':
    data = generate_synthetic_data()
    data = preprocess_data(data)

    # Ensure the results directory exists
    import os
    os.makedirs('results', exist_ok=True)

    # Run backtest
    bt = Backtest(data, AsiaLiquidityGrabReversalStrategy, cash=10000, commission=.002)

    # Optimize
    stats = bt.optimize(
        asia_range_filter_pct=list(np.arange(0.01, 0.05, 0.01)),
        sl_buffer_pct=list(np.arange(0.001, 0.005, 0.001)),
        maximize='Sharpe Ratio'
    )

    print(stats)

    # Save results to a temporary file
    with open('results/temp_result.json', 'w') as f:
        # Sanitize the stats object for JSON serialization
        sanitized_stats = {key: (float(value) if isinstance(value, (int, float, np.number)) else str(value)) for key, value in stats.items() if key != '_strategy' and key != '_equity_curve' and key != '_trades'}

        json_output = {
            'strategy_name': 'asia_liquidity_grab_reversal',
            'return': sanitized_stats.get('Return [%]', 0.0),
            'sharpe': sanitized_stats.get('Sharpe Ratio', 0.0),
            'max_drawdown': sanitized_stats.get('Max. Drawdown [%]', 0.0),
            'win_rate': sanitized_stats.get('Win Rate [%]', 0.0),
            'total_trades': sanitized_stats.get('# Trades', 0)
        }
        json.dump(json_output, f, indent=2)

    # Generate plot
    bt.plot(filename='results/asia_liquidity_grab_reversal.html', open_browser=False)
