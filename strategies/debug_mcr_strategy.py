"""
Market Cipher Reversal Strategy
================================
A backtesting strategy that uses the VuManchu Cipher B indicator to find reversals
at overbought/oversold levels, with additional confluence from high volume.

Buy Signal: WaveTrend cross up while oversold (wt1 & wt2 <= -53)
            AND Volume is above its moving average.
Sell Signal: WaveTrend cross down while overbought (wt1 & wt2 >= 53)
             AND Volume is above its moving average.
"""

from backtesting import Strategy
import numpy as np
import pandas as pd
import pandas_ta as ta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.indicators.vumanchu import cipher_b


def preprocess_data(df, atr_period=14, volume_ma_period=20, **params):
    df = df.copy()
    df.ta.atr(length=atr_period, append=True)
    df.ta.sma(close=df['Volume'], length=volume_ma_period, append=True)
    df = cipher_b(df)
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)
    return df


class MarketCipherReversal(Strategy):
    stop_loss_atr_multiplier = 2.0
    take_profit_atr_multiplier = 3.0
    volume_ma_period = 20
    atr_period = 14

    def init(self):
        atr_col = f'ATRr_{self.atr_period}'
        vol_ma_col = f'SMA_{self.volume_ma_period}'

        self.buy_sig = self.I(lambda: self.data.df['buy_signal'], name='buy_signal')
        self.sell_sig = self.I(lambda: self.data.df['sell_signal'], name='sell_signal')
        self.atr = self.I(lambda: self.data.df[atr_col], name='ATR')
        self.volume_ma = self.I(lambda: self.data.df[vol_ma_col], name='Volume_MA')

    def next(self):
        if len(self.data) < max(65, self.volume_ma_period):
            return

        current_price = self.data.Close[-1]
        current_volume = self.data.Volume[-1]

        if not self.position:
            if self.buy_sig[-1] == 1 and current_volume > self.volume_ma[-1]:
                sl = current_price - self.atr[-1] * self.stop_loss_atr_multiplier
                tp = current_price + self.atr[-1] * self.take_profit_atr_multiplier
                if tp > sl and sl > 0: self.buy(sl=sl, tp=tp)

            elif self.sell_sig[-1] == 1 and current_volume > self.volume_ma[-1]:
                sl = current_price + self.atr[-1] * self.stop_loss_atr_multiplier
                tp = current_price - self.atr[-1] * self.take_profit_atr_multiplier
                if tp < sl and tp > 0: self.sell(sl=sl, tp=tp)
        else:
            if self.position.is_long and self.sell_sig[-1] == 1:
                self.position.close()
            elif self.position.is_short and self.buy_sig[-1] == 1:
                self.position.close()


if __name__ == '__main__':
    from backtesting import Backtest
    import json

    os.makedirs('results', exist_ok=True)

    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
        df.columns = [col.strip().capitalize() for col in df.columns]
    except FileNotFoundError:
        print("data/BTC-USD-15m.csv not found. Using sample data generation...")
        # Increased periods to 5000 to ensure enough data for indicator warmup
        dates = pd.date_range('2023-01-01', periods=5000, freq='15min')
        np.random.seed(42)
        price = 40000 + np.cumsum(np.random.randn(5000) * 20)
        df = pd.DataFrame({
            'Open': price, 'High': price + np.random.rand(5000) * 50,
            'Low': price - np.random.rand(5000) * 50, 'Close': price + np.random.randn(5000) * 10,
            'Volume': np.random.rand(5000) * 100
        }, index=dates)

    print("--- Before Preprocessing ---")
    df.info()

    df = preprocess_data(df, atr_period=14, volume_ma_period=20)

    print("\n--- After Preprocessing ---")
    df.info()
    print("\n--- NaN Count ---")
    print(df.isna().sum())

    df = df.dropna()

    print("\n--- After dropna() ---")
    df.info()

    if df.empty:
        print("\nDataFrame is empty after dropna(). Exiting.")
        sys.exit(1)

    bt = Backtest(df, MarketCipherReversal, cash=100_000, commission=.002)
    stats = bt.run()

    print("\n=== Market Cipher Reversal Strategy Results ===")
    print(stats)

    def sanitize_stats(stats_series):
        sanitized = {}
        for key, value in stats_series.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                sanitized[key] = str(value)
            elif isinstance(value, (np.integer, np.floating)):
                sanitized[key] = float(value)
            elif isinstance(value, type) or hasattr(value, 'to_json'):
                continue
            else:
                sanitized[key] = value
        return sanitized

    json_filename = 'results/market_cipher_reversal.json'
    with open(json_filename, 'w') as f:
        json.dump(sanitize_stats(stats), f, indent=4)
    print(f"\nStats saved to {json_filename}")

    plot_filename = 'results/market_cipher_reversal.html'
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"\nPlot saved to {plot_filename}")
    except Exception as e:
        print(f"\nCould not save plot: {e}")
