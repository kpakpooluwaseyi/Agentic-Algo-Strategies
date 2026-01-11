"""
VuManchu Cipher B Strategy
==========================
A backtesting strategy using the VuManchu Cipher B indicator.

Buy Signal: WaveTrend cross up while oversold (wt1 & wt2 <= -53)
Sell Signal: WaveTrend cross down while overbought (wt1 & wt2 >= 53)
"""

from backtesting import Strategy
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b
import talib


def preprocess_data(df, **params):
    """
    Adds all indicators to the DataFrame.
    Includes multi-timeframe features.
    """
    # Add VuManchu Cipher B indicators
    df = cipher_b(df)
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    # ATR for risk management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Higher timeframe trend filter (4H EMA 200)
    df_4h = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)

    # Create a boolean trend filter
    df_4h['htf_trend_up'] = df_4h['Close'] > df_4h['ema_200']

    # Map 4H trend to original timeframe
    df['htf_trend_up'] = df_4h['htf_trend_up'].reindex(df.index, method='ffill')

    # Volume Confirmation
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=20)

    return df


class Strategy5476c1a49f1b(Strategy):
    """
    Strategy based on VuManchu Cipher B with mandatory risk management features.
    """
    # Optimizable parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_ma_multiplier = 1.0

    def init(self):
        # Indicators
        self.buy_sig = self.I(lambda: self.data.buy_signal, name='buy_signal')
        self.sell_sig = self.I(lambda: self.data.sell_signal, name='sell_signal')
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.htf_trend_up = self.I(lambda: self.data.htf_trend_up, name='htf_trend_up')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')
        self.money_flow = self.I(lambda: self.data.rsimfi, name='money_flow')

    def next(self):
        price = self.data.Close[-1]

        # Exit logic for existing positions
        if self.position:
            # Simple exit on opposite signal
            if self.position.is_long and self.sell_sig[-1]:
                self.position.close()
            elif self.position.is_short and self.buy_sig[-1]:
                self.position.close()

        # Entry logic
        if not self.position:
            # Volume filter
            volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_ma_multiplier

            # Long entry conditions
            if self.htf_trend_up[-1] and self.buy_sig[-1] and volume_confirmed:
                sl = price - self.atr[-1] * self.atr_sl_multiplier
                tp = price + self.atr[-1] * self.atr_tp_multiplier
                self.buy(sl=sl, tp=tp)

            # Short entry conditions
            elif not self.htf_trend_up[-1] and self.sell_sig[-1] and volume_confirmed:
                sl = price + self.atr[-1] * self.atr_sl_multiplier
                tp = price - self.atr[-1] * self.atr_tp_multiplier
                self.sell(sl=sl, tp=tp)


# Standalone testing
if __name__ == '__main__':
    from backtesting import Backtest
    import json

    # Load data
    data_path = 'data/BTC-USD-15m.csv'
    try:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Data file not found at {data_path}. Please check the path.")
        # As a fallback for CI/CD, generate sample data
        dates = pd.date_range('2023-01-01', periods=4000, freq='15min')
        np.random.seed(42)
        price = 16500 + np.cumsum(np.random.randn(4000) * 2)
        df = pd.DataFrame({
            'Open': price,
            'High': price + np.random.rand(4000) * 5,
            'Low': price - np.random.rand(4000) * 5,
            'Close': price + np.random.randn(4000),
            'Volume': np.random.rand(4000) * 100
        }, index=dates)

    # Preprocess
    df = preprocess_data(df)

    # Drop NaN rows which are created by indicators with lookback periods
    df = df.dropna()

    # Ensure there's data left to test
    if df.empty:
        raise ValueError("DataFrame is empty after preprocessing and dropping NaNs. Check indicator periods and data length.")

    # Run backtest
    bt = Backtest(df, Strategy5476c1a49f1b, cash=100_000, commission=.002)
    stats = bt.run()

    print("\n=== Strategy5476c1a49f1b Backtest Results ===")
    print(stats)

    # Save plot
    plot_filename = 'results/strategy_5476c1a49f1b.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"\nPlot saved to {plot_filename}")

    # Save results to JSON
    results_path = 'results/temp_result.json'
    stats_dict = dict(stats)
    # Convert non-serializable types
    for key, value in stats_dict.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_dict[key] = str(value)

    # Remove non-serializable objects
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    with open(results_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print(f"Results saved to {results_path}")
