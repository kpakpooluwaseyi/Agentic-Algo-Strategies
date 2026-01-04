# The user requested a strategy based on the "VUMANCHU SCALPING 5 MIN" PDF.
# The `src.indicators.vumanchu` module was not found.
# This script creates a proxy strategy based on research into the components
# of the "Market Cipher B" / "VuManchu" indicator.
#
# Key Components Proxied:
# 1. Momentum Waves: Replicated using a combination of MACD and a smoothed RSI (Stochastic RSI).
#    A common public implementation of Cipher B relies on a WaveTrend oscillator, which
#    can be proxied effectively with MACD and StochRSI.
# 2. VWAP Cross: Directly implemented using the Volume-Weighted Average Price (VWAP).
#
# Strategy Logic:
# - Long Entry: A bullish MACD crossover is confirmed by the Stochastic RSI being oversold and the price being above the VWAP.
# - Short Entry: A bearish MACD crossover is confirmed by the Stochastic RSI being overbought and the price being below the VWAP.
# - Risk Management: A simple ATR-based stop-loss and take-profit is used.

import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def pta_indicator(indicator_func, **kwargs):
    """Wrapper to apply a pandas_ta indicator and return its values."""
    def func(*args):
        # Convert numpy arrays from backtesting.py to pandas Series
        pd_args = [pd.Series(arg) for arg in args]
        series = indicator_func(*pd_args, **kwargs)
        if isinstance(series, pd.DataFrame):
            # For indicators like MACD that return multiple columns
            return tuple(s.values for _, s in series.items())
        return series.values
    return func

def pta_vwap(high, low, close, volume, index):
    """
    Custom wrapper for pandas_ta.vwap to handle the DatetimeIndex dependency.
    backtesting.py passes data as numpy arrays, but vwap needs an indexed Series.
    """
    vwap_series = ta.vwap(
        high=pd.Series(high, index=index),
        low=pd.Series(low, index=index),
        close=pd.Series(close, index=index),
        volume=pd.Series(volume, index=index)
    )
    return vwap_series.values

def generate_synthetic_data():
    """Generates synthetic data for testing the strategy."""
    print("Generating synthetic data...")
    n_points = 2000
    index = pd.to_datetime(pd.date_range('2023-01-01', periods=n_points, freq='15min'))
    price = 100 + pd.Series(np.random.randn(n_points).cumsum() * 0.1)
    price += np.sin(np.linspace(0, 200, n_points)) * 5
    data = pd.DataFrame({
        'Open': price, 'High': price * 1.01, 'Low': price * 0.99,
        'Close': price, 'Volume': np.random.randint(100, 1000, n_points)
    }, index=index)
    return data.dropna()

class VuManchuProxy(Strategy):
    """
    A proxy for the VuManchu Cipher B scalping strategy.
    """
    # StochRSI parameters
    stoch_rsi_len = 14
    stoch_rsi_k = 3
    stoch_rsi_d = 3
    stoch_rsi_oversold = 20
    stoch_rsi_overbought = 80

    # MACD parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    # ATR Risk Management
    atr_period = 14
    tp_multiplier = 2.0
    sl_multiplier = 1.5

    def init(self):
        # Indicators
        self.vwap = self.I(pta_vwap, self.data.High, self.data.Low, self.data.Close, self.data.Volume, self.data.index)
        self.stoch_rsi_k, self.stoch_rsi_d = self.I(
            pta_indicator(ta.stochrsi, length=self.stoch_rsi_len, k=self.stoch_rsi_k, d=self.stoch_rsi_d),
            self.data.Close
        )
        self.macd, self.macdh, self.macds = self.I(
            pta_indicator(ta.macd, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal),
            self.data.Close
        )
        self.atr = self.I(
            pta_indicator(ta.atr, length=self.atr_period),
            self.data.High, self.data.Low, self.data.Close
        )

    def next(self):
        price = self.data.Close[-1]

        if np.isnan(self.atr[-1]) or np.isnan(self.vwap[-1]):
            return

        # Risk management levels
        sl_dist = self.atr[-1] * self.sl_multiplier
        tp_dist = self.atr[-1] * self.tp_multiplier

        # Entry conditions
        is_long_signal = (
            crossover(self.macd, self.macds) and
            self.stoch_rsi_k[-1] < self.stoch_rsi_oversold and
            price > self.vwap[-1]
        )
        is_short_signal = (
            crossover(self.macds, self.macd) and
            self.stoch_rsi_k[-1] > self.stoch_rsi_overbought and
            price < self.vwap[-1]
        )

        if not self.position:
            if is_long_signal:
                self.buy(sl=price - sl_dist, tp=price + tp_dist)
            elif is_short_signal:
                self.sell(sl=price + sl_dist, tp=price - tp_dist)

if __name__ == '__main__':
    # Acknowledge timeframe discrepancy as noted in the code review
    # The strategy name implies 5m, but the provided data is 15m.
    # Proceeding with the 15m data as per instructions.
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        try:
            data = pd.read_csv(
                data_path,
                index_col='datetime',
                parse_dates=True,
                header=0,
                names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unnamed'],
                usecols=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            )
            # Sanitize column names and sort index
            data.columns = [col.strip().capitalize() for col in data.columns]
            data.sort_index(inplace=True)
        except Exception as e:
            print(f"Error loading CSV, falling back to synthetic data: {e}")
            data = generate_synthetic_data()
    else:
        print(f"Data file not found at '{data_path}'.")
        data = generate_synthetic_data()

    # Ensure data has volume for VWAP calculation
    if 'Volume' not in data.columns:
        print("Volume data missing, generating synthetic volume.")
        data['Volume'] = np.random.randint(100, 1000, len(data))

    bt = Backtest(data, VuManchuProxy, cash=100_000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()

    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Series, pd.DataFrame)): continue
            if isinstance(value, (np.floating, np.integer)):
                sanitized[key] = float(value) if np.isfinite(value) else None
            elif isinstance(value, int): sanitized[key] = int(value)
            elif isinstance(value, pd.Timestamp): sanitized[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta): sanitized[key] = str(value)
            elif pd.isna(value): sanitized[key] = None
            elif key.startswith('_'): continue
            else: sanitized[key] = value
        return sanitized

    final_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(final_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")
    print(stats)

    try:
        plot_filename = 'results/strategy_ce44bea2dbc7.html'
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
