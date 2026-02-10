# coding: utf-8
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtesting import Backtest, Strategy
import pandas as pd
from src.indicators.vumanchu import cipher_b

"""
This strategy trades based on the mean-reversion of volatility, proxied by the
Average True Range (ATR). It identifies periods where volatility is unusually
high or low compared to its recent average and looks for confirming entry
signals from the VuManchu Cipher B indicator.

Proxy Logic:
- "Implied Volatility" is proxied by the Average True Range (ATR).
- A "fitted IV curve" (fair value) is proxied by a Simple Moving Average of ATR.
- "Overpriced" (high IV) means ATR > SMA(ATR) -> Short signal.
- "Underpriced" (low IV) means ATR < SMA(ATR) -> Long signal.

It combines this volatility proxy with a multi-timeframe trend filter (4H EMA)
and volume confirmation.
"""

def preprocess_data(df):
    """Apply all preprocessing steps to the data."""
    # Capitalize columns first for vumanchu compatibility
    df.columns = [column.capitalize() for column in df.columns]

    # VuManchu Cipher B indicator
    df = cipher_b(df)

    # --- Add other indicators using capitalized column names ---
    import pandas_ta as ta

    # Volume Confirmation
    df['Volume_sma'] = ta.sma(df['Volume'], length=20)

    # Volatility Proxy
    df['Atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Atr_sma'] = ta.sma(df['Atr'], length=50)

    # Multi-Timeframe Trend Filter (4H)
    ema_period = 50
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_4h['Ema_4h'] = ta.ema(df_4h['Close'], length=ema_period)

    df['Ema_4h'] = df_4h['Ema_4h'].reindex(df.index, method='ffill')

    # One final capitalization to ensure all columns (including from vumanchu) are correct
    df.columns = [column.capitalize() for column in df.columns]

    df.dropna(inplace=True)
    return df


class AtrMeanReversion(Strategy):
    # Default parameters
    atr_multiplier_tp = 3.0
    atr_multiplier_sl = 2.0

    def init(self):
        pass

    def next(self):
        price = self.data.Close[-1]
        atr_val = self.data.Atr[-1]

        is_long_trend = price > self.data.Ema_4h[-1]
        is_short_trend = price < self.data.Ema_4h[-1]
        is_volume_confirmed = self.data.Volume[-1] > self.data.Volume_sma[-1]
        is_vol_underpriced = self.data.Atr[-1] < self.data.Atr_sma[-1]
        is_vol_overpriced = self.data.Atr[-1] > self.data.Atr_sma[-1]

        if not self.position and is_long_trend and is_volume_confirmed and is_vol_underpriced and self.data.Buy_signal[-1]:
            sl = price - atr_val * self.atr_multiplier_sl
            tp = price + atr_val * self.atr_multiplier_tp
            self.buy(sl=sl, tp=tp)

        elif not self.position and is_short_trend and is_volume_confirmed and is_vol_overpriced and self.data.Sell_signal[-1]:
            sl = price + atr_val * self.atr_multiplier_sl
            tp = price - atr_val * self.atr_multiplier_tp
            self.sell(sl=sl, tp=tp)


if __name__ == '__main__':
    import json
    import os

    if not os.path.exists('results'):
        os.makedirs('results')

    column_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df = pd.read_csv(
        'data/BTC-USD-15m.csv',
        header=0,
        names=column_names,
        index_col='datetime',
        parse_dates=True,
        usecols=column_names
    )
    df = preprocess_data(df)

    bt = Backtest(df, AtrMeanReversion, cash=100000, commission=.002)
    stats = bt.run()
    print(stats)

    stats_dict = stats.to_dict()
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    for key, value in stats_dict.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_dict[key] = str(value)
        elif pd.isna(value):
            stats_dict[key] = None
        elif isinstance(value, (int, float, str, bool)) or value is None:
            continue
        else:
            stats_dict[key] = str(value)

    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)

    plot_filename = 'results/atr_mean_reversion.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Plot saved to {plot_filename}")
