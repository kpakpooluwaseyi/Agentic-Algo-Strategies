
import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import find_peaks
from backtesting import Strategy
from backtesting.lib import FractionalBacktest, crossover

def find_divergence(price, indicator, lookback=60, peak_distance=5):
    """
    Finds bullish and bearish divergences between price and an indicator.
    Returns two boolean series: bullish_divergence and bearish_divergence.
    """
    # Find peaks (highs) and troughs (lows)
    price_highs, _ = find_peaks(price, distance=peak_distance)
    price_lows, _ = find_peaks(-price, distance=peak_distance)
    indicator_highs, _ = find_peaks(indicator, distance=peak_distance)
    indicator_lows, _ = find_peaks(-indicator, distance=peak_distance)

    bullish_divergence = pd.Series(False, index=price.index)
    bearish_divergence = pd.Series(False, index=price.index)

    # Bullish Divergence: Lower low in price, higher low in indicator
    for pl2_idx in range(1, len(price_lows)):
        pl1_idx = price_lows[pl2_idx - 1]
        pl2 = price_lows[pl2_idx]

        if price.iloc[pl2] < price.iloc[pl1_idx]:
            # Find corresponding indicator lows within the same period
            indicator_lows_in_range = indicator_lows[(indicator_lows >= pl1_idx) & (indicator_lows <= pl2)]
            if len(indicator_lows_in_range) >= 2:
                il1 = indicator_lows_in_range[0]
                il2 = indicator_lows_in_range[-1]
                if indicator.iloc[il2] > indicator.iloc[il1]:
                    bullish_divergence.iloc[pl2] = True

    # Bearish Divergence: Higher high in price, lower high in indicator
    for ph2_idx in range(1, len(price_highs)):
        ph1_idx = price_highs[ph2_idx - 1]
        ph2 = price_highs[ph2_idx]

        if price.iloc[ph2] > price.iloc[ph1_idx]:
            # Find corresponding indicator highs within the same period
            indicator_highs_in_range = indicator_highs[(indicator_highs >= ph1_idx) & (indicator_highs <= ph2)]
            if len(indicator_highs_in_range) >= 2:
                ih1 = indicator_highs_in_range[0]
                ih2 = indicator_highs_in_range[-1]
                if indicator.iloc[ih2] < indicator.iloc[ih1]:
                    bearish_divergence.iloc[ph2] = True

    return bullish_divergence, bearish_divergence

def preprocess_data(df):
    """Calculate indicators and divergence signals."""
    # Calculate MACD
    macd_df = df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=False)
    df['MACD_histogram'] = macd_df['MACDh_12_26_9']
    df['MACD_12_26_9'] = macd_df['MACD_12_26_9']
    df['MACDs_12_26_9'] = macd_df['MACDs_12_26_9']

    # Calculate MFI
    df.ta.mfi(close='Close', high='High', low='Low', volume='Volume', length=14, append=True)

    # Find divergences
    bullish_div, bearish_div = find_divergence(df['Close'], df['MACD_histogram'])
    df['bullish_divergence'] = bullish_div
    df['bearish_divergence'] = bearish_div

    return df

class MarketCipherBMACDDivergence(Strategy):
    """
    Strategy based on MACD divergence with Money Flow confirmation.

    NOTE: The requested 'Market Cipher B' indicator (`src.indicators.vumanchu`) was
    not found in the repository. This implementation uses standard `pandas_ta` MACD
    and MFI as proxies, which is an approximation of the original strategy.
    """
    risk_reward_ratio = 2.0
    position_size = 0.1 # Trade 10% of equity

    def init(self):
        # Make signals available to the strategy
        self.bullish_divergence = self.I(lambda x: x, self.data.df['bullish_divergence'])
        self.bearish_divergence = self.I(lambda x: x, self.data.df['bearish_divergence'])
        self.money_flow = self.I(lambda x: x, self.data.df['MFI_14'])
        # Pre-calculating MACD line and signal line for crossover exit logic
        self.macd_line = self.I(lambda x: x, self.data.df['MACD_12_26_9'])
        self.macd_signal_line = self.I(lambda x: x, self.data.df['MACDs_12_26_9'])

    def next(self):
        price = self.data.Close[-1]

        # --- Exit Logic ---
        if self.position:
            # Bullish exit: MACD crosses below signal
            if self.position.is_long and crossover(self.macd_signal_line, self.macd_line):
                self.position.close()
            # Bearish exit: MACD crosses above signal
            elif self.position.is_short and crossover(self.macd_line, self.macd_signal_line):
                self.position.close()

        # --- Long Entry ---
        # A bullish divergence was detected on the bar at t-2
        # Money flow must be "curving up" (i.e., increasing)
        if (self.bullish_divergence[-2] and
            self.money_flow[-2] > self.money_flow[-3] and
            not self.position):
            # The bar at t-1 is the confirmation candle.
            # It must close above the high of the divergence candle (t-2)
            if self.data.Close[-1] > self.data.High[-2]:
                # Stop loss is placed below the low of the divergence candle (t-2)
                sl = self.data.Low[-2]

                # Pre-trade validation
                if sl < price:
                    tp = price + (price - sl) * self.risk_reward_ratio
                    self.buy(sl=sl, tp=tp, size=self.position_size)

        # --- Short Entry ---
        # A bearish divergence was detected on the bar at t-2
        # Money flow must be "curving down" (i.e., decreasing)
        elif (self.bearish_divergence[-2] and
              self.money_flow[-2] < self.money_flow[-3] and
              not self.position):
            # The bar at t-1 is the confirmation candle.
            # It must close below the low of the divergence candle (t-2)
            if self.data.Close[-1] < self.data.Low[-2]:
                # Stop loss is placed above the high of the divergence candle (t-2)
                sl = self.data.High[-2]

                # Pre-trade validation
                if sl > price:
                    tp = price - (sl - price) * self.risk_reward_ratio
                    self.sell(sl=sl, tp=tp, size=self.position_size)

if __name__ == '__main__':
    import json
    import os

    # --- Backtesting Setup ---
    data_path = 'data/BTC-USD-15m.csv'

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Please ensure the file is in the correct location.")
    else:
        # Load and preprocess data
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)

        # Sanitize column names (strip whitespace, capitalize)
        df.columns = [c.strip().capitalize() for c in df.columns]

        # Keep only the essential OHLCV columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Preprocess data to add indicators and signals
        df = preprocess_data(df)
        df = df.dropna()

        # Run the backtest using FractionalBacktest for more realistic position sizing
        bt = FractionalBacktest(df, MarketCipherBMACDDivergence, cash=10000, commission=.002)
        stats = bt.run()
        print(stats)

        # --- Result Handling ---
        # Create results directory if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')

        # Save stats to a JSON file
        stats_dict = dict(stats)

        # Sanitize the stats dictionary for JSON serialization
        for key in list(stats_dict.keys()):
            if isinstance(stats_dict[key], pd.Timestamp):
                stats_dict[key] = stats_dict[key].isoformat()
            elif isinstance(stats_dict[key], pd.Timedelta):
                stats_dict[key] = str(stats_dict[key])
            elif isinstance(stats_dict[key], (pd.NA.__class__, type(np.nan))):
                 stats_dict[key] = None
            elif isinstance(stats_dict[key], np.integer):
                stats_dict[key] = int(stats_dict[key])
            elif isinstance(stats_dict[key], np.floating):
                stats_dict[key] = float(stats_dict[key])

        # Remove non-serializable items
        stats_dict.pop('_strategy', None)
        stats_dict.pop('_equity_curve', None)
        stats_dict.pop('_trades', None)

        with open('results/temp_result.json', 'w') as f:
            json.dump(stats_dict, f, indent=4)

        # Plot the results
        bt.plot(filename='results/market_cipher_b_macd_divergence.html')
