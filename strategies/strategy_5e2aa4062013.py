"""
Strategy 5e2aa4062013
==========================
A backtesting strategy using the VuManchu Cipher B indicator, incorporating mandatory
development guidelines such as ATR-based risk management, a multi-timeframe trend
filter, and volume confirmation.
"""

from backtesting import Strategy, Backtest
import talib
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b


def preprocess_data(df, **params):
    """
    Adds all indicators and filters to the dataframe.
    """
    # VuManchu Cipher B
    df = cipher_b(df)
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    # ATR for risk management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Higher timeframe trend filter (4H)
    df_4h = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)
    df['htf_uptrend'] = (df_4h['Close'] > df_4h['ema_200']).reindex(df.index, method='ffill')
    df['htf_uptrend'] = df['htf_uptrend'].fillna(False).astype(int)

    # Volume confirmation
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=20)

    return df


class Strategy5e2aa4062013(Strategy):
    """
    Implements the VuManchu Cipher B strategy with required filters.
    """
    # Optimizable parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    def init(self):
        # Indicators
        self.buy_sig = self.I(lambda: self.data.buy_signal, name='buy_signal')
        self.sell_sig = self.I(lambda: self.data.sell_signal, name='sell_signal')
        self.mf = self.I(lambda: self.data.rsimfi, name='money_flow')
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.htf_uptrend = self.I(lambda: self.data.htf_uptrend, name='htf_uptrend')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')

    def next(self):
        # Correct warmup period: 200 periods on 4H timeframe = 200 * (4 * 60 / 15) = 3200 15m bars
        if len(self.data) < 3200:
            return

        price = self.data.Close[-1]

        # ==> FILTERS
        # 1. Higher Timeframe Trend Filter
        is_htf_uptrend = self.htf_uptrend[-1] == 1
        is_htf_downtrend = not is_htf_uptrend

        # 2. Volume Filter
        is_volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1]

        # ==> ENTRY LOGIC
        if not self.position:
            # Long Entry
            # Use Money Flow (rsimfi) as a final confluence filter
            if self.buy_sig[-1] and is_htf_uptrend and is_volume_confirmed and self.mf[-1] > 0:
                sl = price - (self.atr[-1] * self.atr_sl_multiplier)
                tp = price + (self.atr[-1] * self.atr_tp_multiplier)
                self.buy(sl=sl, tp=tp)

            # Short Entry
            # Use Money Flow (rsimfi) as a final confluence filter
            elif self.sell_sig[-1] and is_htf_downtrend and is_volume_confirmed and self.mf[-1] < 0:
                sl = price + (self.atr[-1] * self.atr_sl_multiplier)
                tp = price - (self.atr[-1] * self.atr_tp_multiplier)
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    # Load data
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Data file not found. Please place 'BTC-USD-15m.csv' in the 'data' directory.")
        sys.exit(1)

    # Sanitize column names
    df.columns = [col.strip().capitalize() for col in df.columns]

    # Preprocess data
    df = preprocess_data(df)

    # Run backtest
    bt = Backtest(df, Strategy5e2aa4062013, cash=100_000, commission=.002)
    stats = bt.run()

    print("\n" + "="*50)
    print("Strategy: VuManchu Cipher B with Filters")
    print("-" * 50)
    print(stats)
    print("="*50 + "\n")

    # Save plot
    if not os.path.exists('results'):
        os.makedirs('results')
    plot_filename = 'results/strategy_5e2aa4062013.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Plot saved to {plot_filename}")

    # Save results to json
    import json

    # Create a serializable dictionary from the stats object
    results_dict = {key: value for key, value in stats.items() if not key.startswith('_')}

    for key, value in results_dict.items():
        if isinstance(value, (np.integer, np.floating)):
            results_dict[key] = float(value)
        elif isinstance(value, pd.Timestamp):
            results_dict[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            results_dict[key] = str(value)

    results_filename = 'results/temp_result.json'
    with open(results_filename, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {results_filename}")
