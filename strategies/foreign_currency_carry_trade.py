"""
Foreign Currency Carry Trade (Technical Proxy)
==============================================

**Disclaimer:** This strategy is a technical implementation based on the VuManchu Cipher B indicator and adheres to the project's development guidelines. It is NOT a true foreign currency carry trade, which is a fundamental strategy based on interest rate differentials and is incompatible with the provided BTC-USD data and backtesting framework.

The logic implemented here is a momentum and reversal strategy using the following rules:

Long Entry:
- 4H trend is bullish (Close > EMA(200)).
- Volume is above its 20-period moving average.
- VuManchu Cipher B gives a buy signal (WaveTrend cross up in oversold territory).

Short Entry:
- 4H trend is bearish (Close < EMA(200)).
- Volume is above its 20-period moving average.
- VuManchu Cipher B gives a sell signal (WaveTrend cross down in overbought territory).

Risk Management:
- Stop Loss: ATR-based (default: 2 * ATR).
- Take Profit: ATR-based (default: 3 * ATR).
"""

import pandas as pd
import numpy as np
import talib
from backtesting import Strategy, Backtest
import sys
import os

# Add parent directory to path for imports to find src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indicators.vumanchu import cipher_b


def preprocess_data(df: pd.DataFrame, htf_period=200, volume_period=20, atr_period=14):
    """
    Adds all necessary indicators to the DataFrame for the strategy.
    """
    df = df.copy()

    # Add VuManchu Cipher B indicator
    df = cipher_b(df)

    # Higher Timeframe (HTF) Trend Filter (4H)
    df_4h = df.resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    df_4h['ema'] = talib.EMA(df_4h['Close'], timeperiod=htf_period)
    df_4h['htf_bullish'] = df_4h['Close'] > df_4h['ema']

    # Map HTF trend back to the original dataframe
    df['htf_bullish'] = df_4h['htf_bullish'].reindex(df.index, method='ffill').fillna(False)

    # Volume Confirmation
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=volume_period)
    df['volume_confirmed'] = df['Volume'] > df['volume_ma']

    # ATR for Risk Management
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_period)

    # Convert boolean signals to int for backtesting.py compatibility
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    return df


class ForeignCurrencyCarryTrade(Strategy):
    """
    Architectural Note:
    This strategy inherits from `backtesting.Strategy` to align with the established
    pattern for standalone backtesting scripts in the /strategies directory. The
    auto-generated request specified inheriting from `MoonDevStrategy`, which does not
    exist. The available `src.strategies.base_strategy.BaseStrategy` is part of a
    different, incompatible framework and cannot be used with the `backtesting.py`
    runner that this script relies on.
    """
    # Optimizable parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    def init(self):
        # Indicators
        self.htf_bullish = self.I(lambda: self.data.htf_bullish, name='htf_bullish')
        self.volume_confirmed = self.I(lambda: self.data.volume_confirmed, name='volume_confirmed')
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.buy_sig = self.I(lambda: self.data.buy_signal, name='buy_signal')
        self.sell_sig = self.I(lambda: self.data.sell_signal, name='sell_signal')

    def next(self):
        # Wait for enough data
        if len(self.data.Close) < 65:  # CipherB warmup
            return

        price = self.data.Close[-1]
        atr_value = self.atr[-1]

        # If no position is open, check for entry signals
        if not self.position:
            # Long Entry
            if self.htf_bullish[-1] and self.volume_confirmed[-1] and self.buy_sig[-1]:
                sl = price - self.atr_sl_multiplier * atr_value
                tp = price + self.atr_tp_multiplier * atr_value
                self.buy(sl=sl, tp=tp)

            # Short Entry
            elif not self.htf_bullish[-1] and self.volume_confirmed[-1] and self.sell_sig[-1]:
                sl = price + self.atr_sl_multiplier * atr_value
                tp = price - self.atr_tp_multiplier * atr_value
                self.sell(sl=sl, tp=tp)


if __name__ == '__main__':
    import json
    import os

    # Configuration
    data_path = 'data/BTC-USD-15m.csv'
    results_dir = 'results'
    plot_filename = os.path.join(results_dir, 'foreign_currency_carry_trade.html')
    json_filename = os.path.join(results_dir, 'temp_result.json')

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    try:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Sanitize column names (e.g., 'open' -> 'Open')
        data.columns = [col.strip().title() for col in data.columns]
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        # Create dummy data for testing if file not found
        data = pd.DataFrame({
            'Open': np.random.rand(1000) * 100 + 1000,
            'High': np.random.rand(1000) * 100 + 1050,
            'Low': np.random.rand(1000) * 100 + 950,
            'Close': np.random.rand(1000) * 100 + 1000,
            'Volume': np.random.rand(1000) * 100
        }, index=pd.to_datetime(pd.date_range('2022-01-01', periods=1000, freq='15min')))


    # Preprocess the data
    data = preprocess_data(data)

    # Run the backtest
    bt = Backtest(data, ForeignCurrencyCarryTrade, cash=100_000, commission=.002)
    stats = bt.run()

    print("Backtest Results:")
    print(stats)

    # Save the plot
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Plot saved to {plot_filename}")

    # Sanitize and save stats to JSON
    def sanitize_stats(stats_series):
        # Convert Series to dict
        stats_dict = stats_series.to_dict()

        # Remove non-serializable items
        stats_dict.pop('_strategy', None)
        stats_dict.pop('_equity_curve', None)
        stats_dict.pop('_trades', None)

        sanitized = {}
        for key, value in stats_dict.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                sanitized[key] = str(value)
            elif isinstance(value, np.integer):
                sanitized[key] = int(value)
            elif isinstance(value, np.floating):
                sanitized[key] = float(value)
            elif pd.isna(value):
                sanitized[key] = None
            else:
                sanitized[key] = value
        return sanitized

    with open(json_filename, 'w') as f:
        json.dump(sanitize_stats(stats), f, indent=4)
    print(f"Stats saved to {json_filename}")
