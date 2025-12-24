from backtesting import Backtest, Strategy
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to ensure it's JSON-serializable.
    Converts specific numpy types and pandas objects to native Python types,
    and removes non-serializable objects like the strategy instance.
    """
    sanitized = {}
    for key, value in stats.items():
        # Skip internal objects that are not JSON serializable
        if key.startswith('_'):
            continue

        if isinstance(value, (pd.DataFrame, pd.Series)):
            sanitized[key] = None
        elif isinstance(value, (np.integer, np.int_)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            sanitized[key] = float(value)
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    return sanitized


class WyckoffTrendFollowing(Strategy):
    """
    A trend-following strategy based on the Wyckoff method, using EMAs for trend
    identification and entering on pullbacks.
    """
    # Optimizable parameters
    fast_ema = 50
    slow_ema = 200
    atr_period = 14
    atr_multiplier = 2.0
    rr_ratio = 1.5

    def init(self):
        """
        Initialize indicators and strategy variables.
        """
        self.fast_ema_val = self.I(ta.ema, pd.Series(self.data.Close), length=self.fast_ema)
        self.slow_ema_val = self.I(ta.ema, pd.Series(self.data.Close), length=self.slow_ema)
        self.atr_val = self.I(ta.atr, pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), length=self.atr_period)


    def next(self):
        """
        Defines the trading logic for each bar.
        """
        # Wait for enough data to compute indicators
        if len(self.data.Close) < self.slow_ema:
            return

        # --- Trend Identification ---
        is_uptrend = self.fast_ema_val[-1] > self.slow_ema_val[-1]
        is_downtrend = self.fast_ema_val[-1] < self.slow_ema_val[-1]

        # --- Entry Conditions ---
        price = self.data.Close[-1]
        low_price = self.data.Low[-1]
        high_price = self.data.High[-1]
        atr = self.atr_val[-1]

        # --- LONG ENTRY ---
        if is_uptrend and not self.position:
            # Pullback to the slow EMA
            if low_price <= self.slow_ema_val[-1]:
                sl = price - atr * self.atr_multiplier
                tp = price + (price - sl) * self.rr_ratio

                # Make sure stop loss is valid
                if price > sl:
                    self.buy(sl=sl, tp=tp)

        # --- SHORT ENTRY ---
        elif is_downtrend and not self.position:
            # Pullback to the slow EMA
            if high_price >= self.slow_ema_val[-1]:
                sl = price + atr * self.atr_multiplier
                tp = price - (sl - price) * self.rr_ratio

                # Make sure stop loss is valid
                if price < sl:
                    self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)

    # Clean up column names for backtesting.py
    # Remove unnamed columns from trailing commas in header
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    # Strip whitespace, convert to lowercase, and then capitalize
    data.columns = [col.strip().lower() for col in data.columns]
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    bt = Backtest(data, WyckoffTrendFollowing, cash=10000, commission=.002)

    stats = bt.run()

    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)

    # Sanitize the stats for JSON serialization
    # NOTE: stats from bt.run() is a Series, not a dict like in the original template.
    # We need to handle this appropriately.
    sanitized_stats = sanitize_stats(stats)

    # Save the results to a JSON file
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")

    # Generate and save the plot
    try:
        plot_filename = 'results/strategy_c150aaf9610b.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
