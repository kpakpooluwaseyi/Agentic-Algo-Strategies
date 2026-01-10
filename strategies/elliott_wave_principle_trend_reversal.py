"""
Elliott Wave Principle Trend Reversal Strategy
===============================================
This strategy serves as a quantitative proxy for the Elliott Wave Principle,
which posits that markets move in repetitive patterns of impulse (trend) and
corrective (counter-trend) waves. Instead of attempting complex algorithmic
wave counting, this implementation uses a combination of modern indicators
to capture the spirit of the theory: identifying the end of a correction
to enter a new impulse.

Strategy Logic:
- Trend Filter: A higher timeframe (4-hour) EMA is used to define the dominant
  trend direction (proxy for the main impulse wave direction).
- Entry Signal: The VuManchu Cipher B indicator's buy/sell signals are used to
  identify potential reversal points, acting as a proxy for the completion of
  a corrective wave (e.g., Wave 2 or 4).
- Volume Confirmation: Entry requires volume to be above its moving average,
  confirming conviction in the new impulse, a key concept in EWP.
- Risk Management: Stop loss and take profit levels are dynamically set based
  on the Average True Range (ATR), adhering to modern risk management best
  practices.

Entry Conditions:
- Long: Price is above the 4H EMA, a Cipher B buy signal occurs, and volume is
  above its moving average.
- Short: Price is below the 4H EMA, a Cipher B sell signal occurs, and volume
  is above its moving average.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Add parent directory to path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df, **params):
    """
    Applies all necessary indicators to the raw OHLCV dataframe.
    """
    df = df.copy()

    # 1. Higher Timeframe Trend (4H EMA)
    ema_4h = ta.ema(df.resample('4H').agg({'Close': 'last'}).Close, length=50)
    df['ema_4h'] = ema_4h.reindex(df.index, method='ffill')

    # 2. Entry Signals (VuManchu Cipher B)
    df = cipher_b(df)
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    # 3. Volume Confirmation (20-period SMA)
    df['volume_ma'] = ta.sma(df['Volume'], length=20)

    # 4. Risk Management (14-period ATR)
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # Drop rows with NaN values resulting from indicator calculations
    df.dropna(inplace=True)
    return df

class ElliottWavePrincipleTrendReversal(Strategy):
    """
    Strategy class that implements the Elliott Wave proxy logic.
    """
    # Optimizable parameters for risk management
    sl_multiplier = 2.0  # ATR multiplier for stop loss
    tp_multiplier = 3.5  # ATR multiplier for take profit

    def init(self):
        # Wrap indicators with self.I() for backtesting.py compatibility
        self.ema_4h = self.I(lambda: self.data.ema_4h, name='ema_4h')
        self.buy_signal = self.I(lambda: self.data.buy_signal, name='buy_signal')
        self.sell_signal = self.I(lambda: self.data.sell_signal, name='sell_signal')
        self.volume = self.I(lambda: self.data.Volume, name='volume')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')
        self.atr = self.I(lambda: self.data.atr, name='atr')

    def next(self):
        price = self.data.Close[-1]

        # Check if we are already in a position
        if self.position:
            return

        # Long Entry Conditions
        if self.buy_signal[-1] == 1 and price > self.ema_4h[-1] and self.volume[-1] > self.volume_ma[-1]:
            sl = price - self.atr[-1] * self.sl_multiplier
            tp = price + self.atr[-1] * self.tp_multiplier
            if tp > price and sl < price: # Basic validation
                self.buy(sl=sl, tp=tp)

        # Short Entry Conditions
        elif self.sell_signal[-1] == 1 and price < self.ema_4h[-1] and self.volume[-1] > self.volume_ma[-1]:
            sl = price + self.atr[-1] * self.sl_multiplier
            tp = price - self.atr[-1] * self.tp_multiplier
            if tp < price and sl > price: # Basic validation
                self.sell(sl=sl, tp=tp)

def sanitize_stats(stats):
    """
    Cleans the backtesting stats object to make it JSON serializable.
    Removes non-serializable types like dataframes and timestamps.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta, pd.DataFrame)):
            continue
        if isinstance(value, (np.integer, np.int64)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            sanitized[key] = float(value)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            sanitized[key] = value
    return sanitized

if __name__ == '__main__':
    # Configuration
    data_path = 'data/BTC-USD-15m.csv'
    output_plot_path = 'results/elliott_wave_principle_trend_reversal.html'
    output_json_path = 'results/temp_result.json'

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Load data
    try:
        # Robustly load data, specifying columns to avoid issues with trailing commas
        df = pd.read_csv(
            data_path,
            index_col='datetime',
            parse_dates=True,
            usecols=['datetime', 'open', 'high', 'low', 'close', 'volume'],
            skipinitialspace=True  # Handle potential whitespace in header
        )
        # Sanitize column names (e.g., 'open' -> 'Open')
        df.columns = [col.strip().title() for col in df.columns]
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        # As a fallback, create synthetic data for demonstration
        print("Generating synthetic data...")
        date_rng = pd.date_range(start='2020-01-01', end='2023-01-01', freq='15min')
        ohlc = {
            'Open': np.random.uniform(20000, 21000, size=len(date_rng)),
            'High': np.random.uniform(20100, 21100, size=len(date_rng)),
            'Low': np.random.uniform(19900, 20900, size=len(date_rng)),
            'Close': np.random.uniform(20000, 21000, size=len(date_rng)),
            'Volume': np.random.uniform(100, 1000, size=len(date_rng))
        }
        df = pd.DataFrame(ohlc, index=date_rng)
        df.index.name = 'datetime'

    # Preprocess the data
    print("Preprocessing data...")
    df_processed = preprocess_data(df)

    # Run backtest
    print("Running backtest...")
    bt = Backtest(df_processed, ElliottWavePrincipleTrendReversal, cash=100_000, commission=.002)
    stats = bt.run()

    print("\n" + "="*50)
    print("Elliott Wave Principle Trend Reversal Strategy Results")
    print("="*50)
    print(stats)

    # Save plot
    print(f"\nSaving plot to {output_plot_path}...")
    try:
        bt.plot(filename=output_plot_path, open_browser=False)
        print("Plot saved successfully.")
    except Exception as e:
        print(f"Could not save plot: {e}")

    # Save results to JSON
    print(f"Saving results to {output_json_path}...")
    cleaned_stats = sanitize_stats(stats)
    with open(output_json_path, 'w') as f:
        json.dump(cleaned_stats, f, indent=4)
    print("Results saved successfully.")
