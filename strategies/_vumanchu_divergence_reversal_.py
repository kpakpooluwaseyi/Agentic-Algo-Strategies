"""
Vumanchu Divergence Reversal Strategy
=====================================
A counter-trend strategy that enters on price/RSI divergence, confirmed by VuManchu signals.
This implementation adheres to the mandatory MoonDev strategy development guidelines.
"""

import pandas as pd
import numpy as np
import talib
from scipy.signal import find_peaks

# Add parent directory to path to allow imports from src
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indicators.vumanchu import cipher_b

def find_divergence(price, indicator, lookback=60, order=5):
    """
    Finds bullish and bearish divergence between price and an indicator.

    A robust method using scipy.signal.find_peaks to identify swing points.
    - Bearish Divergence: Higher High in price, Lower High in indicator.
    - Bullish Divergence: Lower Low in price, Higher Low in indicator.

    Args:
        price (pd.Series): The price series (e.g., Close).
        indicator (pd.Series): The indicator series (e.g., RSI).
        lookback (int): How far back to look for divergence confirmation.
        order (int): The prominence order for peak/trough detection.

    Returns:
        tuple[pd.Series, pd.Series]: A tuple of boolean Series: (bullish_divergence, bearish_divergence).
    """
    # Find peaks (for highs) and troughs (for lows)
    high_peaks, _ = find_peaks(price, distance=order, prominence=price.std() / 2)
    low_troughs, _ = find_peaks(-price, distance=order, prominence=price.std() / 2)

    indicator_high_peaks, _ = find_peaks(indicator, distance=order, prominence=indicator.std() / 2)
    indicator_low_troughs, _ = find_peaks(-indicator, distance=order, prominence=indicator.std() / 2)

    bullish_divergence = pd.Series(False, index=price.index)
    bearish_divergence = pd.Series(False, index=price.index)

    # Bearish Divergence (HH Price, LH Indicator)
    for i in range(1, len(high_peaks)):
        prev_peak_idx, current_peak_idx = high_peaks[i-1], high_peaks[i]

        # Check for Higher High in price
        if price.iloc[current_peak_idx] > price.iloc[prev_peak_idx]:
            # Find corresponding indicator peaks
            indicator_peaks_in_range = indicator_high_peaks[
                (indicator_high_peaks >= prev_peak_idx) &
                (indicator_high_peaks <= current_peak_idx)
            ]

            if len(indicator_peaks_in_range) >= 2:
                first_indicator_peak = indicator.iloc[indicator_peaks_in_range[0]]
                last_indicator_peak = indicator.iloc[indicator_peaks_in_range[-1]]

                # Check for Lower High in indicator
                if last_indicator_peak < first_indicator_peak:
                    bearish_divergence.iloc[prev_peak_idx:current_peak_idx+1] = True

    # Bullish Divergence (LL Price, HL Indicator)
    for i in range(1, len(low_troughs)):
        prev_trough_idx, current_trough_idx = low_troughs[i-1], low_troughs[i]

        # Check for Lower Low in price
        if price.iloc[current_trough_idx] < price.iloc[prev_trough_idx]:
            # Find corresponding indicator troughs
            indicator_troughs_in_range = indicator_low_troughs[
                (indicator_low_troughs >= prev_trough_idx) &
                (indicator_low_troughs <= current_trough_idx)
            ]

            if len(indicator_troughs_in_range) >= 2:
                first_indicator_trough = indicator.iloc[indicator_troughs_in_range[0]]
                last_indicator_trough = indicator.iloc[indicator_troughs_in_range[-1]]

                # Check for Higher Low in indicator
                if last_indicator_trough > first_indicator_trough:
                    bullish_divergence.iloc[prev_trough_idx:current_trough_idx+1] = True

    return bullish_divergence, bearish_divergence


def preprocess_data(df, **params):
    """
    Applies all necessary indicators and filters to the DataFrame.
    Adheres to the MoonDev strategy development guidelines.
    """
    df = df.copy()

    # 1. Add VuManChu Cipher B indicators
    df = cipher_b(df)
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    # 2. Add Standard Indicators (RSI, ATR)
    df['rsi'] = talib.RSI(df['Close'], timeperiod=14)
    df['atr'] = talib.ATR(df['High'], df.get('Low'), df.get('Close'), timeperiod=14)

    # 3. Add Volume Confirmation Filter
    df['volume_ma'] = df['Volume'].rolling(20).mean()

    # 4. Add Multi-Timeframe (4H) Trend Filter
    df_4h = df.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)

    # Trend is up if price is above the 4H 200 EMA
    df_4h['htf_trend_up'] = (df_4h['Close'] > df_4h['ema_200']).astype(int)

    # Map the 4H trend back to the 15m DataFrame
    df['htf_trend'] = df_4h['htf_trend_up'].reindex(df.index, method='ffill').fillna(0)

    # 5. Calculate Divergence (the core of the strategy)
    # Ensure RSI has no NaNs before calculating divergence
    rsi_series = df['rsi'].dropna()
    price_series = df['Close'].loc[rsi_series.index]

    bullish_div, bearish_div = find_divergence(price_series, rsi_series)

    df['bullish_divergence'] = bullish_div.reindex(df.index, fill_value=False)
    df['bearish_divergence'] = bearish_div.reindex(df.index, fill_value=False)

    return df

from backtesting import Strategy, Backtest

class VuManchuDivergenceReversal(Strategy):
    """
    Implements the VuManchu Divergence Reversal strategy with MoonDev guidelines.

    Entry Logic:
    - Long: Bullish divergence on RSI, confirmed by a VuManChu green dot and rising money flow.
            Must be in a 4H uptrend and have above-average volume.
    - Short: Bearish divergence on RSI, confirmed by a VuManChu red dot and falling money flow.
             Must be in a 4H downtrend and have above-average volume.

    Exit Logic:
    - Exits are handled exclusively by ATR-based Stop Loss and Take Profit levels
      as required by the development guidelines.
    """

    # Optimizable parameters for risk management and filters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.5

    def init(self):
        """Initialize all pre-calculated indicators and signals."""
        self.rsi = self.I(lambda: self.data.rsi, name='rsi')
        self.atr = self.I(lambda: self.data.atr, name='atr')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='volume_ma')
        self.htf_trend = self.I(lambda: self.data.htf_trend, name='htf_trend')

        # VuManChu signals
        self.buy_signal = self.I(lambda: self.data.buy_signal, name='buy_signal')
        self.sell_signal = self.I(lambda: self.data.sell_signal, name='sell_signal')
        self.rsimfi = self.I(lambda: self.data.rsimfi, name='rsimfi')

        # Divergence signals
        self.bullish_divergence = self.I(lambda: self.data.bullish_divergence, name='bullish_divergence')
        self.bearish_divergence = self.I(lambda: self.data.bearish_divergence, name='bearish_divergence')

    def next(self):
        """Main trading logic executed on each bar."""

        # --- Mandatory Guideline Filters ---

        # 1. Volume Confirmation
        if self.data.Volume[-1] < self.volume_ma[-1]:
            return # Skip if volume is below average

        # If a position is already open, manage it (or in this case, let SL/TP handle it)
        if self.position:
            return

        # --- Entry Logic ---

        current_price = self.data.Close[-1]

        # 2. Higher Timeframe Trend Filter & Entry Conditions

        # Check for LONG entry
        is_htf_uptrend = self.htf_trend[-1] == 1
        has_bullish_divergence = self.bullish_divergence[-1]
        has_vumanchu_buy = self.buy_signal[-1] == 1
        is_mf_rising = self.rsimfi[-1] > self.rsimfi[-2]

        if is_htf_uptrend and has_bullish_divergence and has_vumanchu_buy and is_mf_rising:
            # All conditions met, place a long trade with ATR-based risk management
            stop_loss = current_price - (self.atr_sl_multiplier * self.atr[-1])
            take_profit = current_price + (self.atr_tp_multiplier * self.atr[-1])
            self.buy(sl=stop_loss, tp=take_profit)
            return

        # Check for SHORT entry
        is_htf_downtrend = self.htf_trend[-1] == 0
        has_bearish_divergence = self.bearish_divergence[-1]
        has_vumanchu_sell = self.sell_signal[-1] == 1
        is_mf_falling = self.rsimfi[-1] < self.rsimfi[-2]

        if is_htf_downtrend and has_bearish_divergence and has_vumanchu_sell and is_mf_falling:
            # All conditions met, place a short trade with ATR-based risk management
            stop_loss = current_price + (self.atr_sl_multiplier * self.atr[-1])
            take_profit = current_price - (self.atr_tp_multiplier * self.atr[-1])
            self.sell(sl=stop_loss, tp=take_profit)
            return

if __name__ == '__main__':
    # Set up a simple JSON-friendly stats sanitization function
    def sanitize_stats(stats):
        # Remove non-serializable objects
        if '_strategy' in stats:
            del stats['_strategy']
        if '_equity_curve' in stats:
            del stats['_equity_curve']
        if '_trades' in stats:
            del stats['_trades']

        # Convert pandas/numpy types to native Python types
        for key, value in stats.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                stats[key] = str(value)
            elif isinstance(value, (np.integer, np.floating)):
                stats[key] = float(value) if pd.notna(value) else None
            elif pd.isna(value):
                stats[key] = None
        return stats

    # --- Backtest Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    output_json_path = 'results/temp_result.json'
    output_plot_path = 'results/vumanchu_divergence_reversal.html'

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    print(f"Loading data from {data_path}...")
    try:
        # Robustly load data, ignoring the malformed header and assigning correct names.
        # This prevents an extra NaN column from being created due to the trailing comma.
        df = pd.read_csv(
            data_path,
            header=0,
            names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume'],
            usecols=[0, 1, 2, 3, 4, 5], # Only use the first 6 columns
            index_col='datetime',
            parse_dates=True
        )
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    print("Preprocessing data and calculating indicators...")
    df_processed = preprocess_data(df)

    # Drop initial NaN rows created by indicators
    df_processed.dropna(inplace=True)

    print("Running backtest...")
    bt = Backtest(df_processed, VuManchuDivergenceReversal, cash=100_000, commission=.001)
    stats = bt.run()

    print("\n--- Backtest Results ---")
    print(stats)

    # Save sanitized stats to JSON
    try:
        sanitized = sanitize_stats(stats.to_dict())
        with open(output_json_path, 'w') as f:
            import json
            json.dump(sanitized, f, indent=4)
        print(f"\nSuccessfully saved results to {output_json_path}")
    except Exception as e:
        print(f"\nError saving JSON results: {e}")

    # Save plot
    try:
        bt.plot(filename=output_plot_path, open_browser=False)
        print(f"Successfully saved plot to {output_plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}. Plotting may be disabled in this environment.")
