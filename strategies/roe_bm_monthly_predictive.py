import json
import os
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# It's not possible to implement the requested ROE/BM factor-based strategy
# on single-instrument BTC price data. This script implements a proxy strategy
# using moving averages to represent the concepts of 'momentum' (like ROE) and
# 'value' (like BM).

def sanitize_stats(stats):
    """Sanitizes the stats dictionary for JSON serialization."""
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (np.integer, np.int64)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            sanitized[key] = float(value) if not np.isnan(value) else None
        elif isinstance(value, pd.Timestamp):
            sanitized[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            sanitized[key] = str(value)
        elif isinstance(value, pd.Series):
            sanitized[key] = value.to_dict()
        elif key == '_strategy':
            continue  # Exclude non-serializable strategy object
        else:
            sanitized[key] = value
    return sanitized

def SMA(array: pd.Series, n: int) -> pd.Series:
    """Returns the simple moving average of an array."""
    return pd.Series(array).rolling(n).mean()


class ROEBMProxyStrategy(Strategy):
    """
    A proxy strategy for the ROE/BM concept using moving averages.
    - Long-term SMA proxies momentum (like ROE).
    - Short-term SMA proxies value (like BM).
    - Aims to enter on pullbacks in an established trend.
    - Exits after a fixed holding period.
    """
    # --- Strategy Parameters ---
    long_sma_period = 200  # Proxy for long-term trend/momentum (ROE)
    short_sma_period = 50  # Proxy for short-term value/pullbacks (BM)
    hold_period = 28 * 24 * 4  # Approx. 1 month in 15-min bars (28 days)

    def init(self):
        """Initialize indicators."""
        self.long_sma = self.I(SMA, self.data.Close, self.long_sma_period)
        self.short_sma = self.I(SMA, self.data.Close, self.short_sma_period)
        self.entry_bar = None

    def next(self):
        """Define the strategy logic."""
        # --- Time-based Exit Logic ---
        if self.position:
            if len(self.data) - 1 - self.entry_bar >= self.hold_period:
                self.position.close()
                self.entry_bar = None
            return

        # --- Entry Logic ---
        price = self.data.Close[-1]

        # Long Entry: Price is in an uptrend (above long SMA) and crosses
        # above the short-term SMA (pullback entry).
        is_uptrend = price > self.long_sma[-1]
        long_signal = crossover(self.data.Close, self.short_sma)
        if is_uptrend and long_signal:
            self.buy()
            self.entry_bar = len(self.data) - 1

        # Short Entry: Price is in a downtrend (below long SMA) and crosses
        # below the short-term SMA (pullback entry).
        is_downtrend = price < self.long_sma[-1]
        short_signal = crossover(self.short_sma, self.data.Close)
        if is_downtrend and short_signal:
            self.sell()
            self.entry_bar = len(self.data) - 1


if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Clean data columns
    data.columns = [col.strip().capitalize() for col in data.columns]

    bt = Backtest(data, ROEBMProxyStrategy, cash=100_000, commission=.002)
    stats = bt.run()

    print("Backtest Results:")
    print(stats)

    os.makedirs('results', exist_ok=True)

    # Sanitize and save results
    results_dict = sanitize_stats(stats.to_dict())
    strategy_name = "roe_bm_monthly_predictive_proxy"

    final_results = {
        'strategy_name': strategy_name,
        'return': results_dict.get('Return [%]'),
        'sharpe': results_dict.get('Sharpe Ratio'),
        'max_drawdown': results_dict.get('Max. Drawdown [%]'),
        'win_rate': results_dict.get('Win Rate [%]'),
        'total_trades': results_dict.get('# Trades')
    }

    results_filename = 'results/temp_result.json'
    with open(results_filename, 'w') as f:
        json.dump(final_results, f, indent=4)
        f.write('\n')

    print(f"\nResults saved to {results_filename}")

    # Generate and save plot
    plot_filename = f'results/{strategy_name}.html'
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")
