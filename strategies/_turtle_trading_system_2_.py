"""
Turtle Trading System 2
========================

This strategy implements the Turtle Trading System 2, a trend-following
breakout system.

Entry Rules:
- Long: Price exceeds the high of the preceding 55 days.
- Short: Price drops below the low of the preceding 55 days.
- Add Units: Add to the position at ½ N intervals.

Exit Rules:
- Long Exit: Price trades below the 20-day low.
- Short Exit: Price trades above the 20-day high.
- Stop Loss: 2N trailing stop from the last entry price.
"""
import pandas as pd
import pandas_ta as ta
from backtesting import Strategy, Backtest
import json
import os

def preprocess_data(df, **params):
    """
    Calculates indicators on a daily timeframe and merges them back
    into the original dataframe.
    """
    df = df.copy()

    # Resample to daily timeframe to calculate indicators
    daily_df = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()

    # Calculate indicators on daily data
    daily_df['ATR_20'] = ta.atr(daily_df['High'], daily_df['Low'], daily_df['Close'], length=20)
    daily_df['entry_high_55'] = daily_df['High'].rolling(55).max().shift(1)
    daily_df['entry_low_55'] = daily_df['Low'].rolling(55).min().shift(1)
    daily_df['exit_high_20'] = daily_df['High'].rolling(20).max().shift(1)
    daily_df['exit_low_20'] = daily_df['Low'].rolling(20).min().shift(1)

    # Map daily indicators back to the original timeframe
    df['N'] = daily_df['ATR_20'].reindex(df.index, method='ffill')
    df['entry_high'] = daily_df['entry_high_55'].reindex(df.index, method='ffill')
    df['entry_low'] = daily_df['entry_low_55'].reindex(df.index, method='ffill')
    df['exit_high'] = daily_df['exit_high_20'].reindex(df.index, method='ffill')
    df['exit_low'] = daily_df['exit_low_20'].reindex(df.index, method='ffill')

    return df


class TurtleTradingSystem2(Strategy):
    """
    Implements the Turtle Trading System 2.
    """

    entry_period = 55
    exit_period = 20
    atr_period = 20
    stop_n = 2.0
    add_unit_n = 0.5
    max_units = 4

    def init(self):
        self.n = self.I(lambda: self.data.N, name='N_ATR')
        self.entry_high = self.I(lambda: self.data.entry_high, name='Entry_High')
        self.entry_low = self.I(lambda: self.data.entry_low, name='Entry_Low')
        self.exit_high = self.I(lambda: self.data.exit_high, name='Exit_High')
        self.exit_low = self.I(lambda: self.data.exit_low, name='Exit_Low')

        self.units = 0
        self.last_entry_price = 0
        self.stop_loss_price = 0

    def next(self):
        price = self.data.Close[-1]
        n_value = self.n[-1]

        if pd.isna(n_value) or pd.isna(self.entry_high[-1]):
            return

        if self.position:
            if (self.position.is_long and price <= self.stop_loss_price) or \
               (self.position.is_short and price >= self.stop_loss_price):
                self.position.close()
                self.units = 0
                return

            if (self.position.is_long and price < self.exit_low[-1]) or \
               (self.position.is_short and price > self.exit_high[-1]):
                self.position.close()
                self.units = 0
                return

            if self.units > 0 and self.units < self.max_units:
                add_unit_size = (0.01 * self.equity) / n_value
                if self.position.is_long and price > self.last_entry_price + (self.add_unit_n * n_value):
                    self.buy(size=add_unit_size)
                    self.units += 1
                    self.last_entry_price = price
                    self.stop_loss_price = price - (self.stop_n * n_value)
                elif self.position.is_short and price < self.last_entry_price - (self.add_unit_n * n_value):
                    self.sell(size=add_unit_size)
                    self.units += 1
                    self.last_entry_price = price
                    self.stop_loss_price = price + (self.stop_n * n_value)
            return

        if not self.position:
            initial_unit_size = (0.01 * self.equity) / n_value

            if price > self.entry_high[-1]:
                self.buy(size=initial_unit_size)
                self.units = 1
                self.last_entry_price = price
                self.stop_loss_price = price - (self.stop_n * n_value)

            elif price < self.entry_low[-1]:
                self.sell(size=initial_unit_size)
                self.units = 1
                self.last_entry_price = price
                self.stop_loss_price = price + (self.stop_n * n_value)


if __name__ == '__main__':
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
        # Sanitize column names
        df.columns = [col.strip().capitalize() for col in df.columns]
        # The CSV has a trailing comma, which creates an unnamed column. Remove it.
        if 'Unnamed: 6' in df.columns:
            df.drop(columns=['Unnamed: 6'], inplace=True)

    except FileNotFoundError:
        print("Data file not found. Please place 'BTC-USD-15m.csv' in the 'data' directory.")
        exit(1)

    df = preprocess_data(df)
    df.dropna(inplace=True)

    bt = Backtest(df, TurtleTradingSystem2, cash=1_000_000, commission=.002)
    stats = bt.run()

    print(stats)

    os.makedirs('results', exist_ok=True)

    sanitized_stats = {key: str(value) if isinstance(value, (pd.Timestamp, pd.Timedelta)) else value
                       for key, value in stats.items() if not isinstance(value, (pd.Series, pd.DataFrame, type(None)))}
    sanitized_stats = {k: v for k, v in sanitized_stats.items() if pd.notna(v)}

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    bt.plot(filename='results/turtle_trading_system_2.html', open_browser=False)
