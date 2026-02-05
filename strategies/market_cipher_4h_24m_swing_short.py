"""
Strategy: Market Cipher 4H/24M Swing Short
Author: Jules
Date: 2024-07-25

Description: This strategy implements the "Market Cipher 4H/24M Swing Short" trading system.
It uses a 4-hour timeframe to establish the environmental trend and a 24-minute timeframe for execution.
This implementation uses the required `src.indicators.vumanchu.cipher_b` function.

Note on Base Class: The user request specified inheriting from `MoonDevStrategy`, however, the
`src.strategies.base_strategy.BaseStrategy` class is incompatible with the `backtesting.py`
framework. To create a runnable backtest as requested, this strategy inherits from
`backtesting.Strategy`.

Indicator Proxies:
- Wolfpack ID: `pandas_ta.mom` (Momentum oscillator) - This is the only proxy used as it is not
  part of the provided vumanchu library.
- Market Cipher A (Ribbon 5): `pandas_ta.ema` with length 21
"""
import sys
import os
import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import find_peaks
from backtesting import Strategy, Backtest

# Add parent directory to path to allow import of src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.indicators.vumanchu import cipher_b

def indicator_wrapper(indicator_func, *args, **kwargs):
    """
    A generic wrapper for pandas-ta indicators to make them compatible with backtesting.py's self.I().
    It converts input arrays to pandas Series, calls the indicator, and returns the resulting values.
    """
    series_args = [pd.Series(arg) for arg in args]
    indicator_series = indicator_func(*series_args, **kwargs)
    if indicator_series is not None and not indicator_series.empty:
        # Some indicators return a DataFrame, so we select the first column by default.
        if isinstance(indicator_series, pd.DataFrame):
            return indicator_series.iloc[:, 0].values
        return indicator_series.values
    return np.full(len(args[0]), np.nan)

def heikin_ashi(df):
    """Converts OHLC data to Heikin Ashi candles."""
    df_ha = df.copy()
    df_ha['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df_ha['Open'] = ((df['Open'].shift(1) + df['Close'].shift(1)) / 2).fillna(df['Open'])
    df_ha['High'] = df[['High', 'Open', 'Close']].max(axis=1)
    df_ha['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
    return df_ha

def preprocess_data(df: pd.DataFrame, timeframe_24m: str = '24min', timeframe_4h: str = '4H'):
    """
    Prepares the data by calculating multi-timeframe indicators.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Resample to the required timeframes
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    df_24m = df.resample(timeframe_24m).agg(agg_dict).dropna()
    df_4h = df.resample(timeframe_4h).agg(agg_dict).dropna()

    # --- 4H ENVIRONMENTAL INDICATORS ---
    # 1. Apply Cipher B indicator suite
    df_4h = cipher_b(df_4h)
    df_4h.rename(columns={'rsimfi': '4h_rsimfi', 'wt1': '4h_wt1', 'wt2': '4h_wt2'}, inplace=True)

    # 2. Market Cipher A Ribbon 5 (Proxy: 21 EMA)
    df_4h['4h_ema21'] = ta.ema(df_4h['Close'], length=21)

    # 3. Wolfpack ID Crossover (Proxy: Momentum)
    df_4h['4h_mom'] = ta.mom(df_4h['Close'], length=10)

    # --- 24M EXECUTION INDICATORS ---
    # 1. Apply Cipher B indicator suite
    df_24m = cipher_b(df_24m)
    df_24m.rename(columns={'rsimfi': '24m_rsimfi', 'wt1': '24m_wt1', 'wt2': '24m_wt2'}, inplace=True)

    # 2. Wolfpack ID Crossover (Proxy: Momentum)
    df_24m['24m_mom'] = ta.mom(df_24m['Close'], length=10)

    # 3. Convert 24m chart to Heikin Ashi for trend analysis
    df_24m_ha = heikin_ashi(df_24m)

    # Add Heikin Ashi Doji signal
    ha_body = abs(df_24m_ha['Open'] - df_24m_ha['Close'])
    ha_range = df_24m_ha['High'] - df_24m_ha['Low']
    df_24m['ha_is_doji'] = (ha_body / ha_range < 0.1).astype(float) # Doji if body is <10% of range

    # --- MERGE DATA ---
    # Merge 4H indicators into the 24m dataframe
    df_merged = pd.merge(df_24m, df_4h, left_index=True, right_index=True, how='left', suffixes=('', '_4h_drop'))
    df_merged.drop([col for col in df_merged.columns if '_4h_drop' in col], axis=1, inplace=True)
    df_merged.ffill(inplace=True)
    df_merged.dropna(inplace=True)

    return df_merged

# --- Strategy Definition ---
class MarketCipher4h24mSwingShort(Strategy):
    """
    Implements the Market Cipher 4H/24M Swing Short strategy with proxy indicators.
    Uses a state machine to follow the complex sequence of entry conditions.
    """
    # --- Optimizable Parameters ---
    sl_atr_multiplier = 2.0
    tp_atr_multiplier = 4.0
    atr_period = 14
    divergence_lookback = 30 # Bars to look back for divergence
    volume_ma_period = 20
    anchor_wave_threshold = 60
    tsl_lookback = 10 # Bars to look back for swing high for trailing stop

    # State definitions
    STATE_SEARCHING = 0
    STATE_ANCHOR_WAVE_PENDING = 1
    STATE_TRIGGER_PENDING = 2

    def init(self):
        # --- State Machine ---
        self.state = self.STATE_SEARCHING
        self.last_4h_wt_high = None

        # --- Indicators ---
        self.atr = self.I(indicator_wrapper, ta.atr, self.data.High, self.data.Low, self.data.Close, length=self.atr_period)

        # 4H Environmental Indicators (from preprocessed data)
        self.four_h_rsimfi = self.I(lambda: self.data['4h_rsimfi'], name='4h_rsimfi')
        self.four_h_wt1 = self.I(lambda: self.data['4h_wt1'], name='4h_wt1')
        self.four_h_ema21 = self.I(lambda: self.data['4h_ema21'], name='4h_ema21')
        self.four_h_mom = self.I(lambda: self.data['4h_mom'], name='4h_mom')

        # 24m Execution Indicators (from preprocessed data)
        self.tf24m_wt1 = self.I(lambda: self.data['24m_wt1'], name='24m_wt1')
        self.tf24m_rsimfi = self.I(lambda: self.data['24m_rsimfi'], name='24m_rsimfi')
        self.tf24m_mom = self.I(lambda: self.data['24m_mom'], name='24m_mom')
        self.ha_is_doji = self.I(lambda: self.data['ha_is_doji'], name='ha_is_doji')
        self.volume_ma = self.I(indicator_wrapper, ta.sma, self.data.Volume, length=self.volume_ma_period)

    def _is_bullish_divergence(self):
        """
        Checks for bullish divergence by finding the two most recent troughs in price and indicator.
        Divergence is present if price has a lower low and the indicator has a higher low.
        """
        if len(self.data.Close) < self.divergence_lookback:
            return False

        price = self.data.Close[-self.divergence_lookback:]
        indicator = self.four_h_wt1[-self.divergence_lookback:]

        # Find troughs (peaks of the negative series)
        price_troughs, _ = find_peaks(-price)
        indicator_troughs, _ = find_peaks(-indicator)

        if len(price_troughs) < 2 or len(indicator_troughs) < 2:
            return False

        # Get the last two troughs for price
        p_low_1_idx, p_low_2_idx = price_troughs[-2], price_troughs[-1]
        p_low_1, p_low_2 = price[p_low_1_idx], price[p_low_2_idx]

        # Get the last two troughs for the indicator
        i_low_1_idx, i_low_2_idx = indicator_troughs[-2], indicator_troughs[-1]
        i_low_1, i_low_2 = indicator[i_low_1_idx], indicator[i_low_2_idx]

        # Check for bullish divergence condition
        is_lower_low_price = p_low_2 < p_low_1
        is_higher_low_indicator = i_low_2 > i_low_1

        return is_lower_low_price and is_higher_low_indicator

    def next(self):
        # --- Exit Logic ---
        if self.position:
            exit_signal = False
            exit_comment = ""

            # 1. Bullish Divergence
            if self._is_bullish_divergence():
                exit_signal = True
                exit_comment = "Bullish divergence detected"

            # 2. EMA Trend Reversal (Price crosses above 21 EMA)
            elif self.data.Close[-1] > self.four_h_ema21[-1]:
                exit_signal = True
                exit_comment = "Price crossed above 4H EMA21"

            # 3. Heikin Ashi Doji signal
            elif self.ha_is_doji[-1]:
                exit_signal = True
                exit_comment = "Heikin Ashi Doji signal"

            # Execute exit if a signal was triggered
            if exit_signal:
                self.position.close(comment=exit_comment)

            # 4. Trailing Stop-Loss (executes independently of other exit signals)
            if len(self.data.High) > self.tsl_lookback:
                new_sl = self.position.sl
                if self.data.Close[-1] < self.position.entry_price:
                    # Find the highest high in the lookback period
                    swing_high = self.data.High[-self.tsl_lookback:].max()
                    potential_new_sl = swing_high

                    # Adjust SL if the new swing high is lower than the current SL and below the EMA
                    if potential_new_sl < self.position.sl and potential_new_sl < self.four_h_ema21[-1]:
                        new_sl = potential_new_sl

                if new_sl != self.position.sl and self.position:
                    self.position.sl = new_sl

            return

        # --- Entry Logic State Machine ---
        if self.state == self.STATE_SEARCHING:
            # 1. 4H Money Flow is red (curving down)
            mfi_is_red = self.four_h_rsimfi[-1] < 0 and self.four_h_rsimfi[-1] < self.four_h_rsimfi[-2]
            # 2. 4H EMA is in a downtrend (price < EMA)
            price_below_ema = self.data.Close[-1] < self.four_h_ema21[-1]
            # 3. 4H Red dot prints (WaveTrend peak) forming a lower high
            is_wt_peak = self.four_h_wt1[-2] > self.four_h_wt1[-1] and self.four_h_wt1[-2] > self.four_h_wt1[-3]
            is_lower_high_wt = False
            if is_wt_peak:
                if self.last_4h_wt_high is None or self.four_h_wt1[-2] < self.last_4h_wt_high:
                    is_lower_high_wt = True
            # 4. 4H Wolfpack (momentum) crosses below zero
            mom_cross_down = self.four_h_mom[-2] > 0 and self.four_h_mom[-1] < 0

            if mfi_is_red and price_below_ema and is_lower_high_wt and mom_cross_down:
                self.last_4h_wt_high = self.four_h_wt1[-2]
                self.state = self.STATE_ANCHOR_WAVE_PENDING

        elif self.state == self.STATE_ANCHOR_WAVE_PENDING:
            # 5. 24m Anchor Wave: big red dots, green money flow, wave > anchor_wave_threshold
            # Interpretation: WaveTrend peaks above OB level, while Money Flow is positive
            wt_is_high = self.tf24m_wt1[-1] > self.anchor_wave_threshold
            mfi_is_green = self.tf24m_rsimfi[-1] > 0

            if wt_is_high and mfi_is_green:
                self.state = self.STATE_TRIGGER_PENDING
            elif self.data.Close[-1] > self.four_h_ema21[-1]:
                self.state = self.STATE_SEARCHING

        elif self.state == self.STATE_TRIGGER_PENDING:
            # 6. 24m Red dot prints (WaveTrend peak)
            tf24m_wt_peak = self.tf24m_wt1[-2] > self.tf24m_wt1[-1] and self.tf24m_wt1[-2] > self.tf24m_wt1[-3]
            # 7. 24m Money Flow crosses into red
            tf24m_mfi_cross_red = self.tf24m_rsimfi[-2] > 0 and self.tf24m_rsimfi[-1] < 0
            # 8. 24m Wolfpack (momentum) crosses below zero
            tf24m_mom_cross_down = self.tf24m_mom[-2] > 0 and self.tf24m_mom[-1] < 0
            # 9. Volume Confirmation
            volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1]

            if tf24m_wt_peak and tf24m_mfi_cross_red and tf24m_mom_cross_down and volume_confirmed:
                sl = self.data.Close[-1] + self.atr[-1] * self.sl_atr_multiplier
                tp = self.data.Close[-1] - self.atr[-1] * self.tp_atr_multiplier
                if self.data.Close[-1] > tp:
                    self.sell(sl=sl, tp=tp)
                self.state = self.STATE_SEARCHING
            elif self.data.Close[-1] > self.four_h_ema21[-1]:
                 self.state = self.STATE_SEARCHING

if __name__ == '__main__':
    # Load data
    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Data file not found. Please place 'BTC-USD-15m.csv' in the 'data/' directory.")
        exit()

    # --- Data Cleaning ---
    # Clean column names (strip whitespace, capitalize)
    data.columns = [col.strip().capitalize() for col in data.columns]
    # Ensure required columns are present
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Missing one of the required columns: {required_cols}")

    # Preprocess data
    processed_data = preprocess_data(data)

    if processed_data.empty:
        print("Data processing resulted in an empty DataFrame. Check resampling and indicator logic.")
        exit()

    print("--- Preprocessed Data Sample ---")
    print(processed_data.head())
    print(f"\nData shape: {processed_data.shape}")

    # --- Backtesting ---
    print("\n--- Running Backtest ---")
    bt = Backtest(processed_data, MarketCipher4h24mSwingShort, cash=100_000, commission=.002)
    stats = bt.run()

    print("\n--- Backtest Results ---")
    print(stats)

    # Save plot
    plot_filename = 'results/market_cipher_4h_24m_swing_short.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"\nPlot saved to {plot_filename}")

    # Save results to JSON
    stats_df = pd.DataFrame([stats]).T
    stats_df.to_json("results/temp_result.json")
    print("\nResults saved to results/temp_result.json")
