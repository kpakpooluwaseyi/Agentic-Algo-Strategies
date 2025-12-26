
import pandas as pd
from backtesting import Backtest, Strategy
import numpy as np
import json
import os
import pandas_ta as ta

# --- Helper function to adapt pandas-ta to backtesting.py ---
def pta(series, func, **kwargs):
    """A wrapper to apply any pandas-ta indicator that uses a single series."""
    # Convert numpy array to pandas Series
    series = pd.Series(series)
    # Dynamically call the function from the ta module
    indicator_func = getattr(ta, func)
    indicator = indicator_func(series, **kwargs)
    return indicator.values

def pta_atr(high, low, close, **kwargs):
    """A specific wrapper for ATR which requires multiple series."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    indicator = ta.atr(high=high_s, low=low_s, close=close_s, **kwargs)
    return indicator.values

# --- Strategy Definition ---

class ImpliedMomentsOptionPredictionStrategy(Strategy):
    """
    This strategy is an adaptation of the 'Implied Moments Option Prediction'
    from Ernest P. Chan's "Machine Trading". The original strategy uses
    options-derived data (Implied Volatility, Skewness, Kurtosis) to create a
    long-short portfolio of stocks.

    Since we are working with single-asset OHLCV data (BTC-USD), we cannot
    calculate the true implied moments. Instead, we create mathematical proxies
    to model the core logic:

    - Implied Volatility (IV) is proxied by Average True Range (ATR).
    - Implied Skewness (IS) is proxied by the rolling skewness of returns.
    - Implied Kurtosis (IK) is proxied by the rolling kurtosis of returns.

    The trading logic is adapted for a single asset:
    - A 'buy' signal is generated if all three proxy indicators are in the top
      30% of their historical distribution, suggesting a high-return regime.
    - A 'sell' signal is generated if they are all in the bottom 30%,
      suggesting a low-return or reversal regime.
    - Positions are rebalanced monthly, as in the original strategy.
    """

    # --- Strategy Parameters ---
    iv_lookback = 30
    is_lookback = 30
    ik_lookback = 30
    percentile_threshold = 0.7 # Top/Bottom 30%
    rebalance_period_days = 30

    def init(self):
        # Initialize proxy indicators using pandas-ta
        self.iv = self.I(pta_atr, self.data.High, self.data.Low, self.data.Close, length=self.iv_lookback)
        self.is_ = self.I(lambda x, n: pta(x, 'skew', length=n), self.data.Close, self.is_lookback)
        self.ik = self.I(lambda x, n: pta(x, 'kurtosis', length=n), self.data.Close, self.ik_lookback)

        # State tracking
        self.last_rebalance_day = -1

    def next(self):
        # Monthly rebalancing logic
        current_month = self.data.index[-1].month
        if self.last_rebalance_day != current_month:
            if self.position:
                self.position.close()
            self.last_rebalance_day = current_month

            # --- Calculate Percentiles ---
            iv_percentile = np.nan_to_num(np.percentile(self.iv, np.arange(101)))
            is_percentile = np.nan_to_num(np.percentile(self.is_, np.arange(101)))
            ik_percentile = np.nan_to_num(np.percentile(self.ik, np.arange(101)))

            current_iv_percentile = np.searchsorted(iv_percentile, self.iv[-1]) / 100
            current_is_percentile = np.searchsorted(is_percentile, self.is_[-1]) / 100
            current_ik_percentile = np.searchsorted(ik_percentile, self.ik[-1]) / 100

            # --- Entry Logic ---
            is_buy_signal = (
                current_iv_percentile >= self.percentile_threshold and
                current_is_percentile >= self.percentile_threshold and
                current_ik_percentile >= self.percentile_threshold
            )

            is_sell_signal = (
                current_iv_percentile <= (1 - self.percentile_threshold) and
                current_is_percentile <= (1 - self.percentile_threshold) and
                current_ik_percentile <= (1 - self.percentile_threshold)
            )

            if is_buy_signal:
                self.buy()
            elif is_sell_signal:
                self.sell()

# --- Backtesting Runner ---
if __name__ == '__main__':
    # --- Load Data ---
    # Make sure the data path is correct
    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
        # Sanitize column names (e.g., ' open ' -> 'Open')
        data.columns = [col.strip().capitalize() for col in data.columns]
    except FileNotFoundError:
        print("Error: Data file not found. Make sure 'data/BTC-USD-15m.csv' exists.")
        # As a fallback, use synthetic data for demonstration
        from backtesting.test import GOOG
        data = GOOG.copy()
        data = data.iloc[-2000:] # Use a subset for speed

    # --- Run Backtest ---
    bt = Backtest(data, ImpliedMomentsOptionPredictionStrategy, cash=100_000, commission=.002)
    stats = bt.run()

    print(stats)

    # --- Save Results ---
    def sanitize_stats(stats):
        # Remove non-serializable objects
        sanitized = stats.to_dict()
        for key in ['_strategy', '_equity_curve', '_trades']:
            if key in sanitized:
                del sanitized[key]

        # Convert pandas types to native Python types
        for key, value in sanitized.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                sanitized[key] = str(value)
            elif pd.isna(value):
                sanitized[key] = None
        return sanitized

    results_dict = sanitize_stats(stats)

    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)

    # Save to JSON
    with open('results/temp_result.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

    # --- Generate Plot ---
    try:
        plot_filename = 'results/implied_moments_option_prediction.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
