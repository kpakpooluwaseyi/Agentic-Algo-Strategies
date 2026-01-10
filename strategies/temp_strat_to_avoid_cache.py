"""
Strategy: Options Implied Moments Factor Strategy (Proxy Implementation)
Source: Machine Trading by Ernest P. Chan (Strategy 5)
Original Concept: A cross-sectional factor model where stocks with high implied
                  volatility, skewness, and kurtosis are bought, and those with
                  low values are sold.

Proxy Implementation for Single-Asset Time-Series (BTC-USD):
-------------------------------------------------------------
This script adapts the original multi-asset, options-based strategy for a
single-asset (Bitcoin) time-series dataset, as required by the project constraints.
Since options-implied data is unavailable, we create mathematical proxies for the
"implied moments" using standard OHLCV data.

- Implied Volatility Proxy: Average True Range (ATR)
- Implied Skewness Proxy: Rolling Skewness of historical returns
- Implied Kurtosis Proxy: Rolling Kurtosis of historical returns

The strategy enters a long position when all three proxies are simultaneously in
a high quantile (e.g., top 30%), indicating a market state analogous to the
original strategy's "buy" signal. It enters a short position when they are in a
low quantile.

This implementation adheres to the repository's guidelines, using backtesting.py,
a standalone runnable script, and ATR-based risk management.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting import Strategy, Backtest
import json
import os

def preprocess_data(df, moment_window=100, atr_period=14):
    """
    Calculates and appends proxy indicators for implied moments to the DataFrame.

    Args:
        df (pd.DataFrame): Input OHLCV DataFrame.
        moment_window (int): The lookback period for calculating rolling moments.
        atr_period (int): The lookback period for the ATR calculation.

    Returns:
        pd.DataFrame: DataFrame with added indicator columns.
    """
    # Calculate daily returns for moment calculations
    df['returns'] = df['Close'].pct_change()

    # Volatility Proxy: Average True Range (ATR)
    # Ensure the column name is predictable for the strategy
    df[f'ATR_{atr_period}'] = df.ta.atr(length=atr_period)

    # Skewness Proxy: Rolling Skewness of returns
    df['skewness'] = df['returns'].rolling(window=moment_window).skew()

    # Kurtosis Proxy: Rolling Kurtosis of returns
    df['kurtosis'] = df['returns'].rolling(window=moment_window).kurt()

    return df


class OptionsImpliedMomentsFactorStrategy(Strategy):
    """
    Proxy implementation of the Options Implied Moments Factor Strategy.
    """
    # --- Optimizable Parameters ---
    quantile_lookback = 252  # Lookback period for calculating quantiles
    buy_quantile = 0.7       # Quantile threshold for long entries (top 30%)
    sell_quantile = 0.3      # Quantile threshold for short entries (bottom 30%)
    moment_window = 100      # Lookback for skew/kurtosis calculation
    atr_period = 14          # Period for ATR calculation
    sl_atr_multiplier = 2    # ATR multiplier for stop-loss
    tp_atr_multiplier = 4    # ATR multiplier for take-profit

    def init(self):
        """
        Initialize the strategy and indicators.
        """
        # Create aliases for the pre-calculated indicator columns
        atr_col_name = f'ATR_{self.atr_period}'
        self.atr = self.I(lambda: self.data.df[atr_col_name], name="ATR")
        self.skew = self.I(lambda: self.data.df['skewness'], name="Skewness")
        self.kurt = self.I(lambda: self.data.df['kurtosis'], name="Kurtosis")

    def next(self):
        """
        Define the trading logic for the next bar.
        """
        # Ensure we have enough data to calculate quantiles
        if len(self.data) < self.quantile_lookback:
            return

        # Get current values of the indicators
        current_atr = self.atr[-1]
        current_skew = self.skew[-1]
        current_kurt = self.kurt[-1]

        # Calculate historical quantiles for each indicator on the lookback window
        # Using .iloc to avoid potential issues with non-unique indices
        atr_series = pd.Series(self.atr).iloc[-self.quantile_lookback:]
        skew_series = pd.Series(self.skew).iloc[-self.quantile_lookback:]
        kurt_series = pd.Series(self.kurt).iloc[-self.quantile_lookback:]

        atr_buy_threshold = atr_series.quantile(self.buy_quantile)
        skew_buy_threshold = skew_series.quantile(self.buy_quantile)
        kurt_buy_threshold = kurt_series.quantile(self.buy_quantile)

        atr_sell_threshold = atr_series.quantile(self.sell_quantile)
        skew_sell_threshold = skew_series.quantile(self.sell_quantile)
        kurt_sell_threshold = kurt_series.quantile(self.sell_quantile)

        # --- Entry Logic ---
        if not self.position:
            # Long entry condition: All proxies are in the top quantile
            if (current_atr > atr_buy_threshold and
                current_skew > skew_buy_threshold and
                current_kurt > kurt_buy_threshold):

                sl = self.data.Close[-1] - (current_atr * self.sl_atr_multiplier)
                tp = self.data.Close[-1] + (current_atr * self.tp_atr_multiplier)
                if tp > 0 and sl > 0: self.buy(sl=sl, tp=tp)

            # Short entry condition: All proxies are in the bottom quantile
            elif (current_atr < atr_sell_threshold and
                  current_skew < skew_sell_threshold and
                  current_kurt < kurt_sell_threshold):

                sl = self.data.Close[-1] + (current_atr * self.sl_atr_multiplier)
                tp = self.data.Close[-1] - (current_atr * self.tp_atr_multiplier)
                if tp > 0 and sl > 0: self.sell(sl=sl, tp=tp)


if __name__ == '__main__':
    # --- Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    results_dir = 'results'
    plot_filename = os.path.join(results_dir, 'options_implied_moments_factor_strategy.html')
    json_filename = os.path.join(results_dir, 'temp_result.json')

    # --- Data Loading and Preprocessing ---
    try:
        # Load data, ensuring datetime index and handling tricky CSV headers
        df = pd.read_csv(data_path,
                         index_col='datetime',
                         parse_dates=True)
        # Sanitize column names: remove leading/trailing spaces and trailing commas
        df.columns = [col.strip().rstrip(',') for col in df.columns]
        # Select only the necessary columns after sanitizing
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = [col.capitalize() for col in df.columns]
    except FileNotFoundError:
        print(f"Data file not found at {data_path}. Generating synthetic data.")
        # Generate synthetic data if the file is missing
        dates = pd.date_range('2022-01-01', periods=20000, freq='15min')
        np.random.seed(42)
        price = 20000 + np.cumsum(np.random.randn(20000) * 20)
        df = pd.DataFrame({
            'Open': price,
            'High': price + np.random.rand(20000) * 10,
            'Low': price - np.random.rand(20000) * 10,
            'Close': price + np.random.randn(20000) * 5,
            'Volume': np.random.rand(20000) * 100
        }, index=dates)

    # Preprocess data to add indicators
    # Use the same parameters as in the strategy class for consistency
    df = preprocess_data(df,
                         moment_window=OptionsImpliedMomentsFactorStrategy.moment_window,
                         atr_period=OptionsImpliedMomentsFactorStrategy.atr_period)

    # Remove rows with NaN values resulting from indicator calculations
    df.dropna(inplace=True)

    # --- Backtesting ---
    bt = Backtest(df,
                  OptionsImpliedMomentsFactorStrategy,
                  cash=100_000,
                  commission=.002)

    stats = bt.run()
    print("--- Strategy Stats ---")
    print(stats)

    # --- Output Results ---
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Save plot
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not save plot: {e}")

    # Save stats to JSON
    # Sanitize stats for JSON serialization
    stats_dict = dict(stats)
    for key, value in stats_dict.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_dict[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            stats_dict[key] = float(value)
        elif hasattr(value, 'to_dict'): # Handle pandas Series/DataFrames
             stats_dict[key] = value.to_dict()

    # Remove non-serializable items
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    with open(json_filename, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print(f"Stats saved to {json_filename}")
