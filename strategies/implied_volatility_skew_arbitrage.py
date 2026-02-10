# coding: utf-8
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

import pandas as pd
import pandas_ta as ta

"""
This strategy is a proxy for "Implied Volatility Skew Arbitrage".
The original strategy is for options and relies on data not available here.

Proxy Logic:
- "Implied Volatility" is proxied by the Average True Range (ATR).
- A "fitted IV curve" (fair value) is proxied by a Simple Moving Average of ATR.
- "Overpriced" (high IV) means ATR > SMA(ATR) -> Short signal.
- "Underpriced" (low IV) means ATR < SMA(ATR) -> Long signal.

It combines this volatility proxy with a multi-timeframe trend filter (4H EMA),
volume confirmation, and entry signals from a proxy of the 'Cipher B' indicator.
"""


# Proxy for the vumanchu/cipher_b indicator as it does not exist in the repository.
# This uses standard pandas_ta indicators to create a similar set of signals.
def cipher_b_proxy(df, length=14):
    """
    Calculates WaveTrend, RSI, MFI, and Stochastic RSI to generate buy/sell signals.
    Args:
        df: DataFrame with OHLCV data
        length: Lookback period for indicators

    Returns:
        DataFrame with added indicator and signal columns.
    """
    # WaveTrend using crossover of two moving averages
    esa = ta.ema(df['close'], length=length)
    de = ta.ema(abs(df['close'] - esa), length=length)
    ci = (df['close'] - esa) / (0.015 * de)
    wt1 = ta.ema(ci, length=21)
    wt2 = ta.sma(wt1, length=4)
    df['wt1'] = wt1
    df['wt2'] = wt2

    # RSI + MFI Composite
    rsi = ta.rsi(df['close'], length=length)
    mfi = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=length)
    df['rsimfi'] = (rsi + mfi) / 2

    # Stochastic RSI
    stoch_rsi = ta.stochrsi(df['close'], length=length)
    df['stoch_rsi_k'] = stoch_rsi[f'STOCHRSIk_{length}_{length}_3_3']
    df['stoch_rsi_d'] = stoch_rsi[f'STOCHRSId_{length}_{length}_3_3']

    # Signals
    df['buy_signal'] = (
        (df['wt1'] > df['wt2']) &
        (df['rsimfi'] < 40) &
        (df['stoch_rsi_k'] > df['stoch_rsi_d'])
    )
    df['sell_signal'] = (
        (df['wt1'] < df['wt2']) &
        (df['rsimfi'] > 60) &
        (df['stoch_rsi_k'] < df['stoch_rsi_d'])
    )
    return df


def preprocess_data(df):
    """Apply all preprocessing steps to the data."""
    # Run the proxy first, as it expects lowercase columns
    df = cipher_b_proxy(df)

    # --- Add other indicators using lowercase column names ---

    # Volume Confirmation
    df['volume_sma'] = ta.sma(df['volume'], length=20)

    # Volatility Proxy
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_sma'] = ta.sma(df['atr'], length=50)

    # Multi-Timeframe Trend Filter (4H)
    # Resample to 4H, calculate EMA, and merge back to the original timeframe
    ema_period = 50
    df_4h = df.resample('4H').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_4h['ema_4h'] = ta.ema(df_4h['close'], length=ema_period)

    # Merge 4H EMA back to the 15m dataframe
    df['ema_4h'] = df_4h['ema_4h'].reindex(df.index, method='ffill')

    # Capitalize all columns for backtesting.py compatibility as the final step
    df.columns = [column.capitalize() for column in df.columns]

    df.dropna(inplace=True)
    return df


class ImpliedVolatilitySkewArbitrage(Strategy):
    # Default parameters
    atr_period = 14
    atr_multiplier_tp = 3.0
    atr_multiplier_sl = 2.0

    def init(self):
        # backtesting.py automatically makes pre-calculated columns available
        # on self.data. So, no need to declare them with self.I
        pass

    def next(self):
        price = self.data.Close[-1]
        atr_val = self.data.Atr[-1]

        # --- Entry Conditions ---
        is_long_trend = price > self.data.Ema_4h[-1]
        is_short_trend = price < self.data.Ema_4h[-1]
        is_volume_confirmed = self.data.Volume[-1] > self.data.Volume_sma[-1]
        is_vol_underpriced = self.data.Atr[-1] < self.data.Atr_sma[-1]
        is_vol_overpriced = self.data.Atr[-1] > self.data.Atr_sma[-1]

        # Long Entry Logic
        if not self.position and is_long_trend and is_volume_confirmed and is_vol_underpriced and self.data.Buy_signal[-1]:
            sl = price - atr_val * self.atr_multiplier_sl
            tp = price + atr_val * self.atr_multiplier_tp
            self.buy(sl=sl, tp=tp)

        # Short Entry Logic
        elif not self.position and is_short_trend and is_volume_confirmed and is_vol_overpriced and self.data.Sell_signal[-1]:
            sl = price + atr_val * self.atr_multiplier_sl
            tp = price - atr_val * self.atr_multiplier_tp
            self.sell(sl=sl, tp=tp)


if __name__ == '__main__':
    import json
    import os

    # Ensure results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    # Load and preprocess data
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    # Load data, ignoring the malformed header and providing clean column names.
    # This ensures consistency regardless of spaces or trailing commas in the file.
    column_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df = pd.read_csv(
        data_path,
        header=0,
        names=column_names,
        index_col='datetime',
        parse_dates=True,
        usecols=column_names # Explicitly use these columns, ignoring the trailing empty one
    )
    df = preprocess_data(df)

    # Run the backtest
    bt = Backtest(df, ImpliedVolatilitySkewArbitrage, cash=100000, commission=.002)
    stats = bt.run()
    print(stats)

    # Save stats to JSON
    stats_dict = stats.to_dict()

    # First, remove non-serializable items
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    # Then, sanitize the remaining values for JSON serialization
    for key, value in stats_dict.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_dict[key] = str(value)
        elif pd.isna(value):
            stats_dict[key] = None
        elif isinstance(value, (int, float, str, bool)) or value is None:
            continue # Already serializable
        else:
            # Fallback for other potential non-serializable types
            stats_dict[key] = str(value)

    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)

    # Plot the backtest
    plot_filename = 'results/implied_volatility_skew_arbitrage.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Plot saved to {plot_filename}")
