
import pandas as pd
from backtesting import Backtest, Strategy

class RegressionTreeSpyDailyReturnsExtremeLeaves(Strategy):
    """
    Implements a mean-reversion strategy based on the extreme leaf nodes of a
    regression tree model trained on SPY daily returns.

    Entry Rules:
    - Long: Enter a long position if ret2 < 1.53% AND ret1 < -1.39%.
    - Short: Enter a short position if ret2 >= 1.53%.

    Exit Rules:
    - All positions are held for one trading day.
    """

    def init(self):
        # Pre-calculate indicators here if necessary
        self.ret1 = self.I(lambda x: pd.Series(x).pct_change(1), self.data.Close, name="ret1")
        self.ret2 = self.I(lambda x: pd.Series(x).pct_change(2), self.data.Close, name="ret2")

    def next(self):
        # The strategy holds for one day. A position is opened on the current bar's close
        # and closed on the next bar's close.
        # `self.position` is True if a position was opened on the *previous* bar.
        if self.position:
            self.position.close()
            return # Exit after closing, don't enter a new position on the same bar

        # Only check for entries if we don't have a position
        ret1_val = self.ret1[-1] * 100
        ret2_val = self.ret2[-1] * 100

        # Long Entry: ret2 < 1.53% AND ret1 < -1.39%
        if ret2_val < 1.53 and ret1_val < -1.39:
            self.buy()

        # Short Entry: ret2 >= 1.53%
        elif ret2_val >= 1.53:
            self.sell()

if __name__ == '__main__':
    import os
    import json
    import numpy as np

    # Use synthetic data for demonstration purposes as SPY data is not available
    def generate_synthetic_data(days=5000):
        """Generates synthetic daily data that resembles stock price movements."""
        rng = np.random.default_rng(42)
        dates = pd.date_range(start='2010-01-01', periods=days, freq='D')
        price = 100
        prices = [price]
        # Simulate returns with some autocorrelation to make ret1/ret2 meaningful
        returns = rng.normal(0, 0.015, size=days-1)
        for r in returns:
            price *= (1 + r)
            prices.append(price)

        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = df['Close'].shift(1).fillna(df['Close'])
        high_noise = rng.uniform(0, 0.01, size=days)
        low_noise = rng.uniform(0, 0.01, size=days)
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + high_noise)
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - low_noise)
        df['Volume'] = rng.integers(1_000_000, 10_000_000, size=days)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True, skipinitialspace=True)
        # More robust column cleaning
        data.columns = [c.strip().title() for c in data.columns]
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')] # Drop Unnamed columns

        # Resample 15-minute BTC data to daily timeframe to match the strategy's logic
        data = data.resample('D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        print("Successfully loaded and resampled data/BTC-USD-15m.csv")
    except FileNotFoundError:
        print("data/BTC-USD-15m.csv not found. Generating synthetic data instead.")
        data = generate_synthetic_data(days=5000)


    bt = Backtest(data, RegressionTreeSpyDailyReturnsExtremeLeaves, cash=100_000, commission=.002)

    print("Running backtest...")
    stats = bt.run()

    # --- Result Saving ---
    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats_obj):
        """
        Cleans the backtesting stats object by removing non-serializable items
        and converting numpy types to native Python types.
        """
        sanitized = {}
        for key, value in stats_obj.items():
            if key == '_strategy' or key == '_equity_curve' or key == '_trades':
                continue
            if isinstance(value, (np.integer, np.int64)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                sanitized[key] = float(value)
            elif isinstance(value, pd.Timestamp):
                sanitized[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                 sanitized[key] = str(value)
            elif pd.isna(value):
                sanitized[key] = None
            else:
                sanitized[key] = value
        return sanitized

    clean_stats = sanitize_stats(stats)

    # Save to JSON
    with open('results/temp_result.json', 'w') as f:
        json.dump(clean_stats, f, indent=4)

    print("Backtest statistics saved to results/temp_result.json")
    print(stats)

    # Generate plot
    try:
        plot_filename = 'results/regression_tree_spy_daily_returns_extreme_leaves.html'
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
