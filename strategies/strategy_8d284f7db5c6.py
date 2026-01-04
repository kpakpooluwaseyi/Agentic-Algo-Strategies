
import json
import os
import sys
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

def find_divergence(price: pd.Series, indicator: pd.Series, lookback: int, is_bullish: bool = True) -> pd.Series:
    """
    Finds bullish or bearish divergences between price and an indicator using an efficient single-pass algorithm.
    """
    divergence_signals = pd.Series(0, index=price.index)

    if is_bullish:
        # Find troughs (lows) in price and indicator
        price_extrema, _ = find_peaks(-price, distance=5)
        indicator_extrema, _ = find_peaks(-indicator, distance=5)
    else:
        # Find peaks (highs) in price and indicator
        price_extrema, _ = find_peaks(price, distance=5)
        indicator_extrema, _ = find_peaks(indicator, distance=5)

    if len(price_extrema) < 2:
        return divergence_signals

    # Create a series of indicator extrema points for efficient lookup
    indicator_extrema_series = pd.Series(indicator.iloc[indicator_extrema].values, index=indicator_extrema)

    for i in range(1, len(price_extrema)):
        p2_idx = price_extrema[i]
        p1_idx = price_extrema[i-1]

        # Ensure the lookback period is respected
        if p2_idx - p1_idx > lookback:
            continue

        p2_price = price.iloc[p2_idx]
        p1_price = price.iloc[p1_idx]

        # Find corresponding indicator extrema within the price extrema window
        corresponding_indicator_extrema = indicator_extrema_series.loc[p1_idx:p2_idx]

        if len(corresponding_indicator_extrema) < 2:
            continue

        i2_idx = corresponding_indicator_extrema.index[-1]
        i1_idx = corresponding_indicator_extrema.index[-2]

        i2_indicator = corresponding_indicator_extrema.iloc[-1]
        i1_indicator = corresponding_indicator_extrema.iloc[-2]

        if is_bullish:
            # Bullish Divergence: Price makes a lower low, indicator makes a higher low
            if p2_price < p1_price and i2_indicator > i1_indicator:
                divergence_signals.iloc[p2_idx] = 1
        else:
            # Bearish Divergence: Price makes a higher high, indicator makes a lower high
            if p2_price > p1_price and i2_indicator < i1_indicator:
                divergence_signals.iloc[p2_idx] = 1

    return divergence_signals

def preprocess_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Applies multi-timeframe analysis and divergence detection to the data.
    """
    divergence_lookback = kwargs.get('divergence_lookback', 50)

    # Ensure column names are in the required format for indicators
    df.columns = [col.capitalize() for col in df.columns]

    # -- Environmental Timeframe (4h) --
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_4h = cipher_b(df_4h)
    df_4h['env_overbought'] = df_4h['wt_overbought']
    df_4h['env_oversold'] = df_4h['wt_oversold']
    df = pd.merge(df, df_4h[['env_overbought', 'env_oversold']],
                  left_index=True, right_index=True, how='left')
    df['env_overbought'] = df['env_overbought'].ffill().fillna(False)
    df['env_oversold'] = df['env_oversold'].ffill().fillna(False)

    # -- Execution Timeframe (15m) --
    # Calculate indicators needed for divergence
    df = cipher_b(df) # wt1 (Momentum), rsimfi (Money Flow)
    df['vwap_diff'] = df['wt_vwap'] # VWAP difference from Cipher B

    # Detect divergences
    df['mf_bull_div'] = find_divergence(df['Low'], df['rsimfi'], divergence_lookback, is_bullish=True)
    df['mf_bear_div'] = find_divergence(df['High'], df['rsimfi'], divergence_lookback, is_bullish=False)

    df['mom_bull_div'] = find_divergence(df['Low'], df['wt1'], divergence_lookback, is_bullish=True)
    df['mom_bear_div'] = find_divergence(df['High'], df['wt1'], divergence_lookback, is_bullish=False)

    df['vwap_bull_div'] = find_divergence(df['Low'], df['vwap_diff'], divergence_lookback, is_bullish=True)
    df['vwap_bear_div'] = find_divergence(df['High'], df['vwap_diff'], divergence_lookback, is_bullish=False)

    # Combine divergence signals
    df['bullish_divergence'] = (df['mf_bull_div'] | df['mom_bull_div'] | df['vwap_bull_div']).astype(int)
    df['bearish_divergence'] = (df['mf_bear_div'] | df['mom_bear_div'] | df['vwap_bear_div']).astype(int)

    # Add ATR for risk management
    df.ta.atr(append=True, length=14)

    return df

class MarketCipherBTwoTimeframeDivergence(Strategy):
    """
    Strategy based on Market Cipher B, using a 4-hour environmental timeframe to determine
    overbought/oversold conditions and a 15-minute execution timeframe to find divergences
    for trade entries.

    Entry Rules:
    - Long: 4h timeframe is oversold AND a bullish divergence is detected on 15m.
    - Short: 4h timeframe is overbought AND a bearish divergence is detected on 15m.

    Exit Rules:
    - Stop Loss: ATR-based (2 * ATR by default)
    - Take Profit: ATR-based (3 * ATR by default)
    """
    # Optimizable parameters
    divergence_lookback = 50
    atr_period = 14
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0

    def init(self):
        """
        Initialize indicators and signals.
        """
        # Environmental signals (from 4h)
        self.env_overbought = self.I(lambda: self.data.env_overbought, name="env_overbought")
        self.env_oversold = self.I(lambda: self.data.env_oversold, name="env_oversold")

        # Execution signals (from 15m)
        self.bullish_divergence = self.I(lambda: self.data.bullish_divergence, name="bullish_divergence")
        self.bearish_divergence = self.I(lambda: self.data.bearish_divergence, name="bearish_divergence")

        # ATR for risk management
        self.atr = self.I(lambda: self.data.ATRr_14, name="ATR")

    def next(self):
        """
        Define the strategy logic for each bar.
        """
        # Ensure enough data is available and ATR is calculated
        if len(self.data) < self.divergence_lookback or np.isnan(self.atr[-1]):
            return

        price = self.data.Close[-1]

        # --- Entry Logic ---
        if not self.position:
            # Long entry condition
            if self.env_oversold[-1] and self.bullish_divergence[-1] == 1:
                sl = price - self.atr[-1] * self.atr_sl_multiplier
                tp = price + self.atr[-1] * self.atr_tp_multiplier
                self.buy(sl=sl, tp=tp)

            # Short entry condition
            elif self.env_overbought[-1] and self.bearish_divergence[-1] == 1:
                sl = price + self.atr[-1] * self.atr_sl_multiplier
                tp = price - self.atr[-1] * self.atr_tp_multiplier
                self.sell(sl=sl, tp=tp)
def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to be JSON serializable.
    Removes non-serializable types like DataFrames, Series, and Strategy objects.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = value.item()
        elif isinstance(value, (pd.DataFrame, pd.Series, Strategy)):
            # Skip non-serializable objects
            continue
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    return sanitized

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Attempting to generate synthetic data.")
        # Generate synthetic data if the file doesn't exist
        from backtesting.test import GOOG
        data = GOOG.copy()
        data.index = pd.to_datetime(data.index)
        # Ensure it has the same OHLCV column names as the requested file
        data = data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})

    else:
        # Load data and set index
        data = pd.read_csv(data_path)
        # Sanitize column names
        data.columns = [x.strip().lower() for x in data.columns]
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime')
        data = data.sort_index()

    # Preprocess data
    data = preprocess_data(data)

    # Instantiate and run backtest
    bt = Backtest(data, MarketCipherBTwoTimeframeDivergence, cash=100_000, commission=.002)
    stats = bt.run()

    print("Backtest Stats:")
    print(stats)

    # Sanitize and save results
    sanitized_stats = sanitize_stats(stats)
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, 'temp_result.json')
    with open(results_path, 'w') as f:
        json.dump(sanitized_stats, f, indent=4)
    print(f"Results saved to {results_path}")

    # Generate and save plot
    plot_filename = os.path.join(results_dir, f'strategy_8d284f7db5c6_{datetime.now().strftime("%Y%m%d%H%M%S")}.html')
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not save plot: {e}")
