
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

def atr(high, low, close, timeperiod):
    """ Custom ATR indicator function for backtesting.py """
    return talib.ATR(high, low, close, timeperiod=timeperiod)

def rolling_skew(data, window):
    """ Custom rolling skew indicator function """
    series = pd.Series(data)
    return series.rolling(window=window).skew().values

def rolling_kurt(data, window):
    """ Custom rolling kurtosis indicator function """
    series = pd.Series(data)
    return series.rolling(window=window).kurt().values


class ImpliedMomentsProxyStrategy(Strategy):
    """
    A proxy implementation of the "Implied Moments Stock Selection" strategy.

    This strategy adapts the core concepts of using volatility, skewness, and
    kurtosis for a single time-series instrument (e.g., BTC-USD) instead of
    a cross-section of stocks with options data.

    Proxies Used:
    - Implied Volatility -> Average True Range (ATR)
    - Implied Skewness   -> Rolling Skewness of Returns
    - Implied Kurtosis   -> Rolling Kurtosis of Returns
    """
    # --- Strategy Parameters ---
    lookback_period = 30  # Lookback for indicators, proxy for 30-day tenor
    hold_period = 2880    # Bars to hold position, proxy for monthly rebalance (30 days * 24h * 4 bars/h)

    # --- Thresholds for Entry ---
    # Note: In a real scenario, these might be dynamic (e.g., percentiles)
    atr_quantile_threshold = 0.7  # Use top 70% of ATR values as high vol
    skew_long_threshold = 0.2     # Positive skew for long
    kurt_long_threshold = 1.5     # High kurtosis (fat tails) for long

    skew_short_threshold = -0.2   # Negative skew for short
    kurt_short_threshold = 1.5    # High kurtosis for short (high risk both ways)

    # --- Risk Management ---
    sl_pct = 2.0  # Stop Loss percentage
    tp_pct = 4.0  # Take Profit percentage

    def init(self):
        # --- Pre-calculate Indicators ---
        self.daily_returns = self.I(lambda x: pd.Series(x).pct_change().values, self.data.Close)

        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, timeperiod=self.lookback_period)
        self.skew = self.I(rolling_skew, self.daily_returns, window=self.lookback_period)
        self.kurt = self.I(rolling_kurt, self.daily_returns, window=self.lookback_period)

        # For dynamic ATR threshold, we need to compute it over a rolling window
        # This is complex with self.I, so we'll use a simplified approach in next()
        # For a more robust implementation, this should be pre-processed.
        self.atr_history = []

        self.entry_bar = None

    def next(self):
        # --- Store ATR history for dynamic threshold ---
        self.atr_history.append(self.atr[-1])
        if len(self.atr_history) > self.lookback_period * 10: # Keep history manageable
             self.atr_history.pop(0)

        # --- Exit Logic ---
        if self.position:
            # Time-based exit
            current_bar = len(self.data) - 1
            if current_bar - self.entry_bar >= self.hold_period:
                self.position.close()
                self.entry_bar = None
            return # Don't check for new entries if a position is open

        # --- Entry Logic ---
        if len(self.atr_history) < self.lookback_period * 2: # Wait for enough data
            return

        # Get current indicator values
        current_atr = self.atr[-1]
        current_skew = self.skew[-1]
        current_kurt = self.kurt[-1]

        # Calculate dynamic ATR threshold
        atr_threshold_value = np.quantile(self.atr_history, self.atr_quantile_threshold)

        # --- Long Entry Condition ---
        # High Volatility (ATR) AND Positive Skewness AND High Kurtosis
        is_long_signal = (current_atr > atr_threshold_value and
                          current_skew > self.skew_long_threshold and
                          current_kurt > self.kurt_long_threshold)

        # --- Short Entry Condition ---
        # High Volatility (ATR) AND Negative Skewness AND High Kurtosis
        # Note: We still look for high vol for shorts, as the original strategy
        # sorts by IV ascending but selects from the BOTTOM 30%, which could still
        # be high-vol relative to the entire market.
        is_short_signal = (current_atr > atr_threshold_value and
                           current_skew < self.skew_short_threshold and
                           current_kurt > self.kurt_short_threshold)

        if is_long_signal:
            sl = self.data.Close[-1] * (1 - self.sl_pct / 100)
            tp = self.data.Close[-1] * (1 + self.tp_pct / 100)
            self.buy(sl=sl, tp=tp)
            self.entry_bar = len(self.data) - 1

        elif is_short_signal:
            sl = self.data.Close[-1] * (1 + self.sl_pct / 100)
            tp = self.data.Close[-1] * (1 - self.tp_pct / 100)
            self.sell(sl=sl, tp=tp)
            self.entry_bar = len(self.data) - 1


if __name__ == '__main__':
    import os
    import json
    import backtesting

    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}. Please ensure the data is available.")

    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # Ensure standard column names
    data.columns = [c.strip().title() for c in data.columns]


    bt = Backtest(data, ImpliedMomentsProxyStrategy, cash=100_000, commission=.002, finalize_trades=True)

    print("Running single backtest with default parameters...")
    stats = bt.run()
    print(stats)

    # --- Save results ---
    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats_obj):
        """Prepares the backtesting stats object for JSON serialization."""
        sanitized = {}
        for key, value in stats_obj.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                sanitized[key] = str(value)
            elif isinstance(value, (np.int64, np.integer)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.float64, np.floating)):
                sanitized[key] = float(value)
            elif isinstance(value, (backtesting.Strategy, pd.Series, pd.DataFrame)):
                continue # Skip non-serializable objects
            else:
                sanitized[key] = value
        return sanitized

    # Sanitize the main stats object
    clean_stats = sanitize_stats(stats)

    # Also include parameters in the output
    params = stats._strategy._params
    clean_stats['parameters'] = {p: getattr(stats._strategy, p) for p in params}


    result_file = 'results/temp_result.json'
    with open(result_file, 'w') as f:
        json.dump(clean_stats, f, indent=4)

    print(f"Backtest stats saved to {result_file}")

    # --- Generate plot ---
    plot_file = 'results/implied_moments_stock_selection.html'
    print(f"Generating plot... saved to {plot_file}")
    bt.plot(filename=plot_file)
