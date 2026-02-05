"""
VuManchu B Divergence Trading Strategy
=======================================
A strategy that trades on bullish and bearish divergences between price and the VuManChu B Momentum Wave.
"""

import os
import sys
import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df, ema_period=21, atr_period=14, mtf_ema_period=50):
    """Applies indicators and MTF filter to the dataframe."""
    # Base indicators
    df = cipher_b(df)
    df.ta.ema(length=ema_period, append=True)
    df.ta.atr(length=atr_period, append=True)

    # Multi-Timeframe (MTF) Filter (4H)
    df_4h = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    df_4h[f'EMA_{mtf_ema_period}'] = df_4h['Close'].ewm(span=mtf_ema_period, adjust=False).mean()
    df_4h['trend_4h'] = np.where(df_4h['Close'] > df_4h[f'EMA_{mtf_ema_period}'], 1, -1)

    # Merge MTF trend back to the 15m dataframe
    df['trend_4h'] = df_4h['trend_4h'].reindex(df.index, method='ffill')
    df['trend_4h'].fillna(0, inplace=True) # Fill initial NaNs

    # Volume Confirmation
    df['Volume_MA'] = df['Volume'].rolling(window=50).mean()

    return df

class VuManchuBDivergence(Strategy):
    """
    Trades divergences between price and the VuManChu B Momentum Wave.
    """
    # Strategy parameters
    lookback_period = 30  # Lookback window for finding divergence
    ema_period = 21       # EMA for trend confirmation
    atr_period = 14       # ATR for risk management
    sl_atr_multiplier = 2.0  # ATR multiplier for stop loss
    tp_rr_ratio = 3.0        # Risk:Reward ratio for take profit
    peak_distance = 5        # Min distance between peaks for divergence detection

    def init(self):
        """Initialize indicators."""
        # Convenient access to price data
        self.close = self.data.Close
        self.low = self.data.Low
        self.high = self.data.High

        # Wrap pre-calculated indicators with self.I()
        self.momentum_wave = self.I(lambda: self.data.df['wt1'], name="MomentumWave")
        self.money_flow = self.I(lambda: self.data.df['rsimfi'], name="MoneyFlow")
        self.ema = self.I(lambda: self.data.df[f'EMA_{self.ema_period}'], name="EMA")
        self.atr = self.I(lambda: self.data.df[f'ATRr_{self.atr_period}'], name="ATR")


    def next(self):
        """Main trading logic."""
        # Wait for indicator warmup
        if len(self.data.Close) < self.lookback_period:
            return

        # === Divergence Detection ===
        # === Divergence Detection ===
        is_bullish_divergence = self._find_bullish_divergence()
        is_bearish_divergence = self._find_bearish_divergence()

        # === Entry and Exit Logic ===
        if not self.position:
            # Check for long entry
            if is_bullish_divergence and self.money_flow[-1] > self.money_flow[-2] and self.close[-1] > self.ema[-1]:
                sl = self.low[-1] - self.atr[-1] * self.sl_atr_multiplier
                tp = self.close[-1] + (self.close[-1] - sl) * self.tp_rr_ratio
                self.buy(sl=sl, tp=tp)

            # Check for short entry
            elif is_bearish_divergence and self.money_flow[-1] < self.money_flow[-2] and self.close[-1] < self.ema[-1]:
                sl = self.high[-1] + self.atr[-1] * self.sl_atr_multiplier
                tp = self.close[-1] - (sl - self.close[-1]) * self.tp_rr_ratio
                self.sell(sl=sl, tp=tp)

    def _find_bullish_divergence(self):
        """Checks for bullish divergence."""
        # Look for a lower low in price
        price_lookback = self.low[-self.lookback_period:]
        last_low_idx = np.argmin(price_lookback)

        # Ensure the low is recent
        if last_low_idx < self.lookback_period - 5:
            return False

        # Find a higher low in the momentum wave
        momentum_lookback = self.momentum_wave[-self.lookback_period:]
        corresponding_momentum_low = momentum_lookback[last_low_idx]

        # Check for a higher low in momentum in the period *before* the price low
        prior_momentum = momentum_lookback[:last_low_idx]
        if len(prior_momentum) > 0 and any(m < corresponding_momentum_low for m in prior_momentum):
            # Find the index of the true momentum low
            momentum_low_idx = np.argmin(momentum_lookback[:last_low_idx+1])
            # Check if price at momentum low is higher than price at last low
            if price_lookback[momentum_low_idx] > price_lookback[last_low_idx]:
                return True
        return False

    def _find_bearish_divergence(self):
        """Checks for bearish divergence."""
        # Look for a higher high in price
        price_lookback = self.high[-self.lookback_period:]
        last_high_idx = np.argmax(price_lookback)

        # Ensure the high is recent
        if last_high_idx < self.lookback_period - 5:
            return False

        # Find a lower high in the momentum wave
        momentum_lookback = self.momentum_wave[-self.lookback_period:]
        corresponding_momentum_high = momentum_lookback[last_high_idx]

        # Check for a lower high in momentum in the period *before* the price high
        prior_momentum = momentum_lookback[:last_high_idx]
        if len(prior_momentum) > 0 and any(m > corresponding_momentum_high for m in prior_momentum):
            # Find the index of the true momentum high
            momentum_high_idx = np.argmax(momentum_lookback[:last_high_idx+1])
            # Check if price at momentum high is lower than price at last high
            if price_lookback[momentum_high_idx] < price_lookback[last_high_idx]:
                return True
        return False


if __name__ == '__main__':
    # Load data
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Data file not found. Please place 'BTC-USD-15m.csv' in the 'data' directory.")
        sys.exit(1)

    # Sanitize column names (e.g., ' open' -> 'Open')
    df.columns = [col.strip().title() for col in df.columns]

    # Preprocess data
    strategy_params = {
        'ema_period': VuManchuBDivergence.ema_period,
        'atr_period': VuManchuBDivergence.atr_period,
    }
    df_processed = preprocess_data(df.copy(), **strategy_params)
    # df_processed.dropna(inplace=True) # This is too aggressive and can empty the dataframe

    # Backtest
    bt = Backtest(df_processed, VuManchuBDivergence, cash=100_000, commission=.002)
    stats = bt.run()
    print(stats)

    # Save results and plot
    if not os.path.exists('results'):
        os.makedirs('results')

    # Sanitize stats for JSON serialization
    stats_dict = dict(stats)
    stats_dict.pop('_strategy', None)  # Remove non-serializable strategy object
    stats_dict.pop('_equity_curve', None) # Remove bulky data series
    stats_dict.pop('_trades', None) # Remove bulky data series

    # Convert remaining special types
    for key, value in stats_dict.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_dict[key] = str(value)
        elif pd.isna(value):
            stats_dict[key] = None

    with open('results/temp_result.json', 'w') as f:
        import json
        json.dump(stats_dict, f, indent=4)

    bt.plot(filename='results/vumanchu_b_divergence_trading.html', open_browser=False)
    print("\nBacktest complete. Results saved and plot generated.")
