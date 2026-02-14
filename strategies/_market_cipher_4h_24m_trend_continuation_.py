"""
Market Cipher 4h/24m Trend Continuation Strategy
=================================================
This strategy identifies a trend on a 4-hour timeframe using Market Cipher B
indicators and then looks for a precise entry on the 24-minute timeframe.

Long Entry Logic:
1.  **4H Trend**: Money Flow, Momentum, and Price are all trending upwards.
2.  **4H Alert**: A "Green Dot" signal appears on the 4H chart, with momentum
    not in the overbought zone (>60).
3.  **24M Entry**: After the 4H conditions are met, enter on either a "Green Dot"
    or a Money Flow cross to the upside on the 24m chart. Entry is filtered
    by volume being above its moving average.

Short Entry Logic:
1.  **4H Trend**: Money Flow, Momentum, and Price are all trending downwards.
2.  **4H Alert**: A "Red Dot" signal appears on the 4H chart, with momentum
    not in the oversold zone (<-60).
3.  **24M Entry**: After the 4H conditions are met, enter on either a "Red Dot"
    or a Money Flow cross to the downside on the 24m chart. Entry is filtered
    by volume being above its moving average.

Risk Management:
- Stop Loss: ATR-based (2 * ATR)
- Take Profit: ATR-based (3 * ATR)
"""
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import sys
import os
import pandas_ta as ta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b


def preprocess_data(data_path):
    """
    Loads and prepares multi-timeframe data for the strategy.
    - 4h for trend direction
    - 24m for entries
    """
    # Load the base data (15m) and standardize column names
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.columns = [c.strip().capitalize() for c in df.columns]

    # --- 1. Prepare 4H Data ---
    df_4h = df.resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # Calculate 4H Cipher B indicators
    df_4h = cipher_b(df_4h)

    # Identify 4H signals based on strategy rules
    # Long Trend: MF, Momentum, and Price are getting higher
    df_4h['price_rising'] = df_4h['Close'] > df_4h['Close'].shift(1)
    df_4h['mf_rising'] = df_4h['rsimfi'] > df_4h['rsimfi'].shift(1)
    df_4h['mom_rising'] = df_4h['wt1'] > df_4h['wt1'].shift(1)
    df_4h['long_trend_4h'] = (df_4h['price_rising'] & df_4h['mf_rising'] & df_4h['mom_rising'])

    # Short Trend: MF, Momentum, and Price are getting lower
    df_4h['price_falling'] = df_4h['Close'] < df_4h['Close'].shift(1)
    df_4h['mf_falling'] = df_4h['rsimfi'] < df_4h['rsimfi'].shift(1)
    df_4h['mom_falling'] = df_4h['wt1'] < df_4h['wt1'].shift(1)
    df_4h['short_trend_4h'] = (df_4h['price_falling'] & df_4h['mf_falling'] & df_4h['mom_falling'])

    # Filter for the alert condition (dot printing within os/ob levels)
    # Using 'buy_signal'/'sell_signal' from vumanchu as a proxy for dots
    df_4h['long_alert_4h'] = df_4h['buy_signal'] & (df_4h['wt1'] < 60)
    df_4h['short_alert_4h'] = df_4h['sell_signal'] & (df_4h['wt1'] > -60)

    # Select and rename 4H columns to avoid clashes during merge
    df_4h_signals = df_4h[[
        'long_trend_4h', 'short_trend_4h', 'long_alert_4h', 'short_alert_4h'
    ]]

    # --- 2. Prepare 24M Data ---
    df_24m = df.resample('24min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # Calculate 24M indicators
    df_24m = cipher_b(df_24m)
    df_24m.ta.atr(length=14, append=True)
    df_24m.rename(columns={'ATRr_14': 'ATR'}, inplace=True)
    df_24m['volume_ma'] = df_24m['Volume'].rolling(30).mean()

    # Define 24M entry signals
    df_24m['mf_cross_up_24m'] = (df_24m['rsimfi'].shift(1) <= 0) & (df_24m['rsimfi'] > 0)
    df_24m['mf_cross_down_24m'] = (df_24m['rsimfi'].shift(1) >= 0) & (df_24m['rsimfi'] < 0)
    df_24m['green_dot_24m'] = df_24m['buy_signal']
    df_24m['red_dot_24m'] = df_24m['sell_signal']

    # --- 3. Merge Timeframes ---
    # Merge 4H signals into 24M dataframe, forward-filling the 4H signal
    df_merged = pd.merge_asof(df_24m, df_4h_signals, left_index=True, right_index=True, direction='backward')

    return df_merged.dropna()


class MarketCipher4h24mTrendContinuation(Strategy):
    # Default parameters, can be optimized
    atr_stop_multiplier = 2.0
    atr_tp_multiplier = 3.0

    def init(self):
        # Nothing to initialize here, as signals are pre-calculated
        pass

    def next(self):
        # Get the most recent data point
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]
        volume_ma = self.data.volume_ma[-1]

        # === Position Management ===
        # If a position is already open, do nothing until SL/TP is hit.
        if self.position:
            return

        # === Entry Logic ===
        # Check for long entry conditions
        is_long_trend = self.data.long_trend_4h[-1] and self.data.long_alert_4h[-1]
        is_long_entry_24m = self.data.green_dot_24m[-1] or self.data.mf_cross_up_24m[-1]

        if is_long_trend and is_long_entry_24m and volume > volume_ma:
            sl = price - self.data.ATR[-1] * self.atr_stop_multiplier
            tp = price + self.data.ATR[-1] * self.atr_tp_multiplier
            self.buy(sl=sl, tp=tp)
            return

        # Check for short entry conditions
        is_short_trend = self.data.short_trend_4h[-1] and self.data.short_alert_4h[-1]
        is_short_entry_24m = self.data.red_dot_24m[-1] or self.data.mf_cross_down_24m[-1]

        if is_short_trend and is_short_entry_24m and volume > volume_ma:
            sl = price + self.data.ATR[-1] * self.atr_stop_multiplier
            tp = price - self.data.ATR[-1] * self.atr_tp_multiplier
            self.sell(sl=sl, tp=tp)
            return

# Standalone execution
if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
    else:
        print("Preprocessing data...")
        data = preprocess_data(data_path)

        print("Data loaded and preprocessed:")
        print(data.head())
        print("\nColumns available:")
        print(data.columns)

        print("\nRunning backtest...")
        bt = Backtest(data, MarketCipher4h24mTrendContinuation, cash=100000, commission=.001)
        stats = bt.run()

        print("\nBacktest Results:")
        print(stats)

        # Save stats to a file
        stats_df = pd.DataFrame(stats).transpose()
        stats_df.to_json("results/temp_result.json")
        print("\nStats saved to results/temp_result.json")

        # Save plot
        plot_filename = 'results/market_cipher_4h_24m_trend_continuation.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
