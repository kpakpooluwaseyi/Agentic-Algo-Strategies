"""
A technical trading strategy based on the VuManchu Cipher B indicator,
filtered by higher-timeframe trend and volume confirmation.
"""
import sys
import os
sys.path.append(os.getcwd()) # Add the project root to the Python path

from backtesting import Strategy
import numpy as np
import talib
import pandas as pd
from src.indicators.vumanchu import cipher_b

def preprocess_data(df, **params):
    """
    Adds necessary indicators to the dataframe for the strategy.
    """
    df = df.copy()

    # Add VuManchu Cipher B indicators
    df = cipher_b(df)

    # Add ATR for risk management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Add Volume Moving Average for confirmation
    df['volume_ma'] = df['Volume'].rolling(20).mean()

    # Add 4H Trend Filter
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)
    df_4h['htf_trend_up'] = df_4h['Close'] > df_4h['ema_200']

    # Map the 4H trend back to the main timeframe
    df['htf_trend_up'] = df_4h['htf_trend_up'].reindex(df.index, method='ffill')

    # Drop rows with NaN values resulting from indicator calculations
    return df.dropna(subset=['htf_trend_up', 'volume_ma', 'atr'])


class DividendSpreadArbitrage(Strategy):
    """
    Trades on VuManchu Cipher B signals, confirming with a 4H trend
    and volume analysis.
    """

    # ===== OPTIMIZABLE PARAMETERS =====
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_ma_multiplier = 1.0

    def init(self):
        """
        Initialize indicators here.
        """
        self.htf_trend_up = self.I(lambda: self.data.htf_trend_up, name='htf_trend_up')
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')
        self.buy_signal = self.I(lambda: self.data.buy_signal, name='buy_signal')
        self.sell_signal = self.I(lambda: self.data.sell_signal, name='sell_signal')

    def next(self):
        """
        Main trading logic.
        """
        if self.position:
            return

        volume_check = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_ma_multiplier

        # Long Entry
        if self.buy_signal[-1] and self.htf_trend_up[-1] and volume_check:
            sl = self.data.Close[-1] - (self.atr[-1] * self.atr_sl_multiplier)
            tp = self.data.Close[-1] + (self.atr[-1] * self.atr_tp_multiplier)
            self.buy(sl=sl, tp=tp)

        # Short Entry
        elif self.sell_signal[-1] and not self.htf_trend_up[-1] and volume_check:
            sl = self.data.Close[-1] + (self.atr[-1] * self.atr_sl_multiplier)
            tp = self.data.Close[-1] - (self.atr[-1] * self.atr_tp_multiplier)
            self.sell(sl=sl, tp=tp)


# ===== STANDALONE MODE =====
if __name__ == '__main__':
    import pandas as pd
    from backtesting import Backtest

    # Load sample data
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: Could not find data file at 'data/BTC-USD-15m.csv'")
        sys.exit(1)

    # Sanitize column names
    df.columns = [col.strip().capitalize() for col in df.columns]

    # Preprocess data
    df = preprocess_data(df)

    # Run backtest
    bt = Backtest(df, DividendSpreadArbitrage, cash=100_000, commission=.002)
    stats = bt.run()
    print(stats)

    # --- Save results and plot ---
    import json

    def sanitize_stats(stats_obj):
        stats_dict = stats_obj.to_dict()
        stats_dict.pop('_strategy', None)
        stats_dict.pop('_equity_curve', None)
        stats_dict.pop('_trades', None)

        for key, value in stats_dict.items():
            if pd.isna(value):
                stats_dict[key] = None
            elif isinstance(value, (np.integer, np.floating)):
                stats_dict[key] = value.item()
            elif isinstance(value, pd.Timestamp):
                stats_dict[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                stats_dict[key] = str(value)
        return stats_dict

    sanitized_stats = sanitize_stats(stats)
    os.makedirs('results', exist_ok=True)

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    bt.plot(filename='results/plot.html')
