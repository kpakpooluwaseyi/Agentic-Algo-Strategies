"""
Volatility Smirk / Skew Mean Reversion Strategy (Proxy)
=========================================================

This strategy implements a proxy for the "volatility smirk" mean-reversion strategy
described in Ernest P. Chan's "Machine Trading".

The original strategy is a cross-sectional stock strategy that uses options-implied
volatility data (skew) to identify assets with overpriced fear (high skew) or
complacency (low skew) and bets on their reversion to the mean.

Since options data is not available for BTC-USD, this implementation uses a
proxy for skew: the rolling skewness of historical returns.

Proxy Logic:
- High positive skew (long tail of positive returns) is a proxy for market
  complacency/greed. The strategy shorts these periods, betting on a downturn.
- High negative skew (long tail of negative returns) is a proxy for market
  fear. The strategy longs these periods, betting on a relief rally.

Entry Rules:
- Short when rolling skew is in the top quintile (most complacent).
- Long when rolling skew is in the bottom quintile (most fearful).

Exit Rules:
- Hold positions for a fixed period (e.g., one week).
"""

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

def preprocess_data(df, skew_period=252, **params):
    """
    Calculates the rolling skewness of returns and identifies the top and bottom
    quintiles to generate trading signals.
    """
    # Calculate daily returns
    df['returns'] = df['Close'].pct_change()

    # Calculate rolling skewness
    df['skew'] = df['returns'].rolling(window=skew_period).skew()

    # Determine quintiles
    df['skew_quintile'] = pd.qcut(df['skew'], 5, labels=False, duplicates='drop')

    # Generate signals
    # Bottom quintile (fear) -> Long signal
    df['go_long'] = (df['skew_quintile'] == 0).astype(int)
    # Top quintile (complacency) -> Short signal
    df['go_short'] = (df['skew_quintile'] == 4).astype(int)

    return df


class VolatilitySmirkSkewMeanReversion(Strategy):
    """
    Strategy class for the Volatility Smirk Skew Mean Reversion (Proxy).
    """
    # Optimizable parameters
    skew_period = 252  # Lookback period for skew calculation
    hold_period = 7    # How many bars to hold the position

    def init(self):
        self.go_long = self.I(lambda: self.data.go_long, name='go_long')
        self.go_short = self.I(lambda: self.data.go_short, name='go_short')
        self.entry_bar = -1
        self.entry_price_proxy = 0.0

    def next(self):
        current_bar = len(self.data) - 1

        # Exit logic: Close position after hold_period
        if self.position:
            if current_bar - self.entry_bar >= self.hold_period:
                self.position.close()
                self.entry_bar = -1

        # Entry logic
        if not self.position and self.entry_bar == -1:
            if self.go_long[-1]:
                self.buy()
                self.entry_bar = current_bar
                self.entry_price_proxy = self.data.Close[-1]
            elif self.go_short[-1]:
                self.sell()
                self.entry_bar = current_bar
                self.entry_price_proxy = self.data.Close[-1]


def run_backtest(data_path='data/BTC-USD-15m.csv'):
    """
    Loads data, preprocesses it, and runs the backtest.
    """
    try:
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        # Sanitize column names (e.g., 'open' -> 'Open')
        df.columns = [c.strip().title() for c in df.columns]
    except FileNotFoundError:
        print(f"Data file not found at {data_path}. Generating synthetic data.")
        # Fallback to synthetic data if the file is missing
        dates = pd.date_range('2020-01-01', periods=5000, freq='D')
        price = 10000 + np.cumsum(np.random.randn(5000) * 100)
        df = pd.DataFrame({
            'Open': price,
            'High': price + np.random.rand(5000) * 50,
            'Low': price - np.random.rand(5000) * 50,
            'Close': price + np.random.randn(5000) * 10,
            'Volume': np.random.rand(5000) * 100
        }, index=dates)
        df.index.name = 'datetime'


    # The strategy is designed for a weekly timeframe. Let's resample the 15m data.
    df = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()


    # Preprocess the data
    df_processed = preprocess_data(df.copy(), skew_period=252)
    df_processed = df_processed.dropna()

    # Run the backtest
    bt = Backtest(df_processed, VolatilitySmirkSkewMeanReversion, cash=100_000, commission=.002, finalize_trades=True)
    stats = bt.run()

    print("=== Volatility Smirk Skew Mean Reversion (Proxy) ===")
    print(stats)

    # Save results and plot
    import json
    import os

    if not os.path.exists('results'):
        os.makedirs('results')

    # Sanitize stats for JSON serialization
    sanitized_stats = {}
    for key, value in stats.items():
        # Exclude non-serializable objects
        if key in ['_strategy', '_equity_curve', '_trades']:
            continue
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized_stats[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized_stats[key] = float(value)
        elif isinstance(value, pd.DataFrame):
            # Don't include DataFrames in the JSON output
            pass
        else:
            sanitized_stats[key] = value


    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    plot_filename = 'results/volatility_smirk_skew_mean_reversion.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Plot saved to {plot_filename}")


if __name__ == '__main__':
    run_backtest()
