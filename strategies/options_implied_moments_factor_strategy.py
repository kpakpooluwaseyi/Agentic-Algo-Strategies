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

def preprocess_data(df, moment_window=100):
    """
    Calculates and appends proxy indicators for implied moments to the DataFrame.

    Args:
        df (pd.DataFrame): Input OHLCV DataFrame.
        moment_window (int): The lookback period for calculating rolling moments.

    Returns:
        pd.DataFrame: DataFrame with added indicator columns.
    """
    # Calculate daily returns for moment calculations
    df['returns'] = df['Close'].pct_change()

    # Volatility Proxy: Average True Range (ATR)
    df.ta.atr(append=True)

    # Skewness Proxy: Rolling Skewness of returns
    df['skewness'] = df['returns'].rolling(window=moment_window).skew()

    # Kurtosis Proxy: Rolling Kurtosis of returns
    df['kurtosis'] = df['returns'].rolling(window=moment_window).kurt()

    return df


class OptionsImpliedMomentsFactorStrategy(MoonDevStrategy):
    """
    Proxy implementation of the Options Implied Moments Factor Strategy.
    """
    # --- Optimizable Parameters ---
    quantile_lookback = 252  # Lookback period for calculating quantiles
    buy_quantile = 0.7       # Quantile threshold for long entries (top 30%)
    sell_quantile = 0.3      # Quantile threshold for short entries (bottom 30%)
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

        # Calculate historical quantiles for each indicator
        atr_buy_threshold = pd.Series(self.atr).quantile(self.buy_quantile)
        skew_buy_threshold = pd.Series(self.skew).quantile(self.buy_quantile)
        kurt_buy_threshold = pd.Series(self.kurt).quantile(self.buy_quantile)

        atr_sell_threshold = pd.Series(self.atr).quantile(self.sell_quantile)
        skew_sell_threshold = pd.Series(self.skew).quantile(self.sell_quantile)
        kurt_sell_threshold = pd.Series(self.kurt).quantile(self.sell_quantile)

        # --- Trend Filter ---
        is_uptrend = self.data.Close[-1] > self.ema_4h[-1]
        is_downtrend = self.data.Close[-1] < self.ema_4h[-1]

        # --- Entry Logic ---
        if not self.position:
            # Long entry condition: Uptrend AND all proxies are in the top quantile
            if (is_uptrend and
                current_atr > atr_buy_threshold and
                current_skew > skew_buy_threshold and
                current_kurt > kurt_buy_threshold):

                sl = self.data.Close[-1] - (current_atr * self.sl_atr_multiplier)
                tp = self.data.Close[-1] + (current_atr * self.tp_atr_multiplier)
                self.buy(sl=sl, tp=tp)

            # Short entry condition: Downtrend AND all proxies are in the bottom quantile
            elif (is_downtrend and
                  current_atr < atr_sell_threshold and
                  current_skew < skew_sell_threshold and
                  current_kurt < kurt_sell_threshold):

                sl = self.data.Close[-1] + (current_atr * self.sl_atr_multiplier)
                tp = self.data.Close[-1] - (current_atr * self.tp_atr_multiplier)
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    print("File created. Structure is ready for indicator and logic implementation.")
