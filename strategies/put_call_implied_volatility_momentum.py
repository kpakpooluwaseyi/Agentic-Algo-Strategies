"""
Strategy: Put-Call Implied Volatility Momentum (Proxy)
------------------------------------------------------

This strategy is a proxy implementation of the "Put-Call Implied Volatility Momentum"
concept described in "Machine Trading" by Ernest P. Chan.

The original strategy is a cross-sectional, weekly stock selection model based on
options data (put-call implied volatility difference), which is not available in
this backtesting environment.

This proxy adapts the core idea of "momentum" to a single-instrument, time-series
context using the `BTC-USD` dataset. It adheres to the project's mandatory
development guidelines.

Proxy Logic:
- **Momentum Indicator:** Uses the `rsimfi` (RSI + Money Flow Index) from the
  `vumanchu` library as a proxy for momentum. Positive `rsimfi` suggests bullish
  momentum, while negative suggests bearish momentum.
- **Entry Rules:**
    - Long: Enters when `rsimfi` is positive, confirming bullish momentum.
    - Short: Enters when `rsimfi` is negative, confirming bearish momentum.
- **Mandatory Filters:**
    - **Multi-Timeframe:** A 4-hour EMA is used as a trend filter. Longs are
      only taken above the 4H EMA, and shorts are only taken below it.
    - **Volume Confirmation:** Entry requires the current bar's volume to be
      above its 20-period moving average.
    - **ATR-Based Risk Management:** Stop loss and take profit levels are
      calculated dynamically using the Average True Range (ATR).
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import talib
from backtesting import Strategy, Backtest

# Add parent directory to path to allow imports from `src`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

def sanitize_stats(stats):
    """
    Sanitizes the backtesting stats object to be JSON serializable.
    Removes non-serializable types like DataFrames and converts numpy/pandas
    types to standard Python types.
    """
    if stats is None:
        return {}

    # If stats is a Series, convert to dict
    if isinstance(stats, pd.Series):
        stats = stats.to_dict()

    # Create a copy to avoid modifying the original
    sanitized = {}

    # List of keys to remove that are typically DataFrames or complex objects
    keys_to_remove = ['_equity_curve', '_trades', '_strategy']

    for key, value in stats.items():
        if key in keys_to_remove:
            continue

        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.int64)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            sanitized[key] = float(value)
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value

    return sanitized

def preprocess_data(df, **params):
    """
    Adds all required indicators and filters to the DataFrame.
    """
    df = df.copy()

    # 1. Add VuManchu Cipher B indicators (provides rsimfi for momentum)
    df = cipher_b(df)

    # 2. Add Multi-Timeframe Trend Filter (4H EMA 200)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)

    # Map trend direction (1 for up, -1 for down)
    df_4h['htf_trend'] = np.where(df_4h['Close'] > df_4h['ema_200'], 1, -1)

    # Forward-fill the 4H trend onto the original 15m timeframe
    df['htf_trend'] = df_4h['htf_trend'].reindex(df.index, method='ffill')

    # 3. Add Volume Confirmation Filter (20-period SMA)
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=20)

    # 4. Add ATR for risk management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    return df


class PutCallMomentumProxy(Strategy):
    """
    Proxy strategy for Put-Call Implied Volatility Momentum.
    Uses RSIMFI as the core momentum signal, filtered by higher-timeframe
    trend and volume, with ATR-based risk management.
    """
    # Optimizable parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_ma_multiplier = 1.0

    def init(self):
        # Initialize indicators using self.I() for backtesting.py
        self.momentum = self.I(lambda: self.data.rsimfi, name='momentum_rsimfi')
        self.htf_trend = self.I(lambda: self.data.htf_trend, name='htf_trend')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')
        self.atr = self.I(lambda: self.data.atr, name='atr')

    def next(self):
        # Wait for all indicators to have enough data
        if pd.isna(self.htf_trend[-1]) or pd.isna(self.volume_ma[-1]) or pd.isna(self.atr[-1]):
            return

        price = self.data.Close[-1]

        # --- Filters ---
        is_uptrend = self.htf_trend[-1] == 1
        is_downtrend = self.htf_trend[-1] == -1
        has_volume = self.data.Volume[-1] > (self.volume_ma[-1] * self.volume_ma_multiplier)

        # --- Momentum Signal ---
        is_bullish_momentum = self.momentum[-1] > 0
        is_bearish_momentum = self.momentum[-1] < 0

        # --- Entry Logic ---
        if not self.position:
            # Long Entry: Uptrend + Volume + Bullish Momentum
            if is_uptrend and has_volume and is_bullish_momentum:
                sl = price - (self.atr[-1] * self.atr_sl_multiplier)
                tp = price + (self.atr[-1] * self.atr_tp_multiplier)
                self.buy(sl=sl, tp=tp)

            # Short Entry: Downtrend + Volume + Bearish Momentum
            elif is_downtrend and has_volume and is_bearish_momentum:
                sl = price + (self.atr[-1] * self.atr_sl_multiplier)
                tp = price - (self.atr[-1] * self.atr_tp_multiplier)
                self.sell(sl=sl, tp=tp)


if __name__ == '__main__':
    # --- Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    results_dir = 'results'
    strategy_name = 'put_call_implied_volatility_momentum'
    json_path = os.path.join(results_dir, 'temp_result.json')
    plot_path = os.path.join(results_dir, f"{strategy_name}.html")

    # --- Data Loading ---
    try:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Sanitize column names (e.g., lowercase -> Titlecase)
        df.columns = [col.strip().title() for col in df.columns]
        print(f"Loaded data from {data_path}")
    except FileNotFoundError:
        print(f"Data file not found at {data_path}. Generating synthetic data for testing.")
        dates = pd.date_range('2022-01-01', periods=20000, freq='15min')
        np.random.seed(42)
        price_change = np.random.randn(len(dates)).cumsum()
        price = 40000 + price_change * 10
        df = pd.DataFrame({
            'Open': price,
            'High': price + np.random.uniform(0, 50, len(dates)),
            'Low': price - np.random.uniform(0, 50, len(dates)),
            'Close': price + np.random.normal(0, 5, len(dates)),
            'Volume': np.random.uniform(10, 500, len(dates))
        }, index=dates)
        df.index.name = 'datetime'

    # --- Backtesting ---
    print("Preprocessing data...")
    df_processed = preprocess_data(df)

    print("Running backtest...")
    bt = Backtest(df_processed, PutCallMomentumProxy, cash=100_000, commission=.002)
    stats = bt.run()

    print("\n--- Backtest Results ---")
    print(stats)

    # --- Save Results ---
    os.makedirs(results_dir, exist_ok=True)

    # Save sanitized stats to JSON
    sanitized_stats = sanitize_stats(stats)
    with open(json_path, 'w') as f:
        json.dump(sanitized_stats, f, indent=4)
    print(f"\nSaved backtest stats to {json_path}")

    # Save plot
    bt.plot(filename=plot_path, open_browser=False)
    print(f"Saved plot to {plot_path}")
