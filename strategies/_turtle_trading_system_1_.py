
from backtesting import Strategy, Backtest
import pandas as pd
import pandas_ta as ta

def preprocess_data(df, **params):
    """
    Adds the necessary indicators to the DataFrame.
    """
    # The Turtle system uses daily data for its rules, so we must resample
    daily_df = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # Calculate ATR (N)
    daily_df['N'] = ta.atr(daily_df['High'], daily_df['Low'], daily_df['Close'], length=20)

    # Calculate Donchian Channels (rolling highs/lows)
    daily_df['donchian_upper_10'] = daily_df['High'].rolling(10).max().shift(1)
    daily_df['donchian_lower_10'] = daily_df['Low'].rolling(10).min().shift(1)
    daily_df['donchian_upper_20'] = daily_df['High'].rolling(20).max().shift(1)
    daily_df['donchian_lower_20'] = daily_df['Low'].rolling(20).min().shift(1)
    daily_df['donchian_upper_55'] = daily_df['High'].rolling(55).max().shift(1)
    daily_df['donchian_lower_55'] = daily_df['Low'].rolling(55).min().shift(1)

    # Map daily indicators back to the original intraday dataframe
    daily_indicators = ['N', 'donchian_upper_10', 'donchian_lower_10',
                        'donchian_upper_20', 'donchian_lower_20',
                        'donchian_upper_55', 'donchian_lower_55']

    normalized_index = df.index.normalize()
    for col in daily_indicators:
        df[col] = normalized_index.map(daily_df[col])

    df.ffill(inplace=True)
    # Drop rows where the longest lookback indicators are NaN to prevent empty DataFrame
    df.dropna(subset=['donchian_upper_55', 'donchian_lower_55', 'N'], inplace=True)

    return df


class TurtleTradingSystem1(Strategy):
    """
    Implements the Turtle Trading System 1, a classic trend-following strategy.
    System 1 uses a 20-day breakout for entries and a 10-day breakout for exits.
    It includes a filter to skip signals if the last trade was a winner,
    and a 55-day failsafe breakout.
    """
    # Optimizable parameters
    max_units = 4

    def init(self):
        """Initialize the state variables for the strategy."""
        # State variables
        self.units = 0
        self.last_entry_price = 0.0
        self.position_stop_price = 0.0

        # Filter flag: if True, the last trade was a loser (stopped out),
        # so the next 20-day breakout signal should be taken.
        self.take_next_20day_breakout = True

        # To prevent taking multiple signals on the same day's breakout
        self.last_breakout_day = None

    def calculate_unit_size(self, N):
        """
        Calculates the position size for one Unit based on the Turtle rules.
        Unit Size = (1% of Account Equity) / (N * Dollars per Point)
        """
        # Assuming "Dollars per Point" is 1 for simplicity with BTC-USD
        if N == 0:
            return 0
        # Using FractionalBacktest, size is a fraction of equity.
        # But Turtle unit is a specific number of contracts/shares.
        # We calculate the number of shares and then convert it to an equity fraction.
        unit_size_in_contracts = (0.01 * self.equity) / N
        return int(unit_size_in_contracts) # Truncate to nearest whole contract

    def next(self):
        """The main trading logic loop."""
        price = self.data.Close[-1]
        N = self.data.N[-1]

        # --- State Reset on Position Close ---
        # If we had units but the position is now closed, reset state.
        if self.units > 0 and not self.position:
            self.units = 0
            self.last_entry_price = 0.0
            self.position_stop_price = 0.0

        # --- Active Position Management ---
        if self.position:
            # 1. Check for Stop-Loss Hit
            if (self.position.is_long and price <= self.position_stop_price) or \
               (self.position.is_short and price >= self.position_stop_price):
                self.take_next_20day_breakout = True  # A stopped trade is a "loser"
                self.position.close()
                return

            # 2. Check for Take-Profit Exit (10-day breakout)
            if (self.position.is_long and price < self.data.donchian_lower_10[-1]) or \
               (self.position.is_short and price > self.data.donchian_upper_10[-1]):
                self.take_next_20day_breakout = False # A 10-day exit makes the last trade a "winner"
                self.position.close()
                return

            # 3. Check for Adding Units (Pyramiding)
            if self.units > 0 and self.units < self.max_units:
                add_unit = False
                if self.position.is_long and price >= self.last_entry_price + 0.5 * N:
                    add_unit = True
                elif self.position.is_short and price <= self.last_entry_price - 0.5 * N:
                    add_unit = True

                if add_unit:
                    size = self.calculate_unit_size(N)
                    # Use signal bar price as proxy for entry price
                    entry_price = self.data.Close[-1]
                    if self.position.is_long:
                        self.buy(size=size)
                        self.position_stop_price = entry_price - 2 * N
                    else:
                        self.sell(size=size)
                        self.position_stop_price = entry_price + 2 * N

                    self.units += 1
                    self.last_entry_price = entry_price
            return

        # --- New Entry Logic ---
        # Only check for entries if there's no open position
        if not self.position:
            current_day = self.data.index[-1].date()
            if self.last_breakout_day == current_day:
                return  # Avoid multiple signals on the same day

            # Long Entry Conditions
            is_20day_breakout = price > self.data.donchian_upper_20[-1]
            is_55day_breakout = price > self.data.donchian_upper_55[-1]

            if is_20day_breakout:
                self.last_breakout_day = current_day
                if self.take_next_20day_breakout:
                    size = self.calculate_unit_size(N)
                    self.buy(size=size)
                    self.units = 1
                    self.last_entry_price = self.data.Close[-1]
                    self.position_stop_price = self.last_entry_price - 2 * N
                    return
                elif not self.take_next_20day_breakout and is_55day_breakout:
                    # Failsafe entry
                    size = self.calculate_unit_size(N)
                    self.buy(size=size)
                    self.units = 1
                    self.last_entry_price = self.data.Close[-1]
                    self.position_stop_price = self.last_entry_price - 2 * N
                    return

            # Short Entry Conditions
            is_20day_breakdown = price < self.data.donchian_lower_20[-1]
            is_55day_breakdown = price < self.data.donchian_lower_55[-1]

            if is_20day_breakdown:
                self.last_breakout_day = current_day
                if self.take_next_20day_breakout:
                    size = self.calculate_unit_size(N)
                    self.sell(size=size)
                    self.units = 1
                    self.last_entry_price = self.data.Close[-1]
                    self.position_stop_price = self.last_entry_price + 2 * N
                    return
                elif not self.take_next_20day_breakout and is_55day_breakdown:
                    # Failsafe entry
                    size = self.calculate_unit_size(N)
                    self.sell(size=size)
                    self.units = 1
                    self.last_entry_price = self.data.Close[-1]
                    self.position_stop_price = self.last_entry_price + 2 * N
                    return

if __name__ == '__main__':
    df = None
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
        df.columns = [col.strip().capitalize() for col in df.columns]
        # Check if the loaded data is long enough for the 55-day lookback
        required_days = 60  # 55 days for lookback + buffer
        if (df.index.max() - df.index.min()).days < required_days:
            print(f"Historical data is shorter than the required {required_days} days. Generating synthetic data.")
            df = None  # Force synthetic data generation
    except FileNotFoundError:
        print("Data file not found. Generating synthetic data for backtest...")

    if df is None:
        import numpy as np
        # Generate enough data for a 55-day lookback + trading
        periods = 200 * 24 * 4 # 200 days of 15-min data
        dates = pd.date_range('2023-01-01', periods=periods, freq='15min')
        price = 1000 + np.cumsum(np.random.randn(periods) * 2)
        df = pd.DataFrame({
            'Open': price,
            'High': price + np.random.rand(periods) * 2,
            'Low': price - np.random.rand(periods) * 2,
            'Close': price + np.random.randn(periods),
            'Volume': np.random.rand(periods) * 1000
        }, index=dates)
        df.index.name = 'datetime'

    # Preprocess the data
    processed_df = preprocess_data(df)

    if processed_df.empty:
        raise ValueError("Preprocessing returned an empty DataFrame. "
                         "This can happen if the dataset is shorter than the longest lookback period (55 days).")

    # Backtest
    bt = Backtest(processed_df, TurtleTradingSystem1, cash=1_000_000, commission=.002, finalize_trades=True)
    stats = bt.run()

    print(stats)

    # Save plot
    bt.plot(filename='results/turtle_trading_system_1.html', open_browser=False)

    # Save results
    # Sanitize stats for JSON serialization
    sanitized_stats = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized_stats[key] = str(value)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            sanitized_stats[key] = value
        else:
            sanitized_stats[key] = str(value) # Convert other types to string

    with open('results/temp_result.json', 'w') as f:
        import json
        json.dump(sanitized_stats, f, indent=4)
