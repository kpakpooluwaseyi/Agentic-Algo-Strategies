
import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks
import os
import sys

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df: pd.DataFrame, **params):
    """
    Applies preprocessing and indicators to the input DataFrame.
    """
    # Sanitize column names
    df.columns = [c.strip().title() for c in df.columns]

    # Drop the unused column if it exists
    if 'Unnamed: 6' in df.columns:
        df.drop(columns=['Unnamed: 6'], inplace=True)

    # Add VuManchu Cipher B indicator
    df = cipher_b(df)
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    # Add standard indicators
    df.ta.ema(length=5, append=True)
    df.ta.ema(length=13, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.rsi(length=14, append=True)

    # Add multi-timeframe filter (4H)
    df_4h = df.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_4h['ema_200_4h'] = ta.ema(df_4h.Close, length=50) # Using 50 on 4H as a proxy for 200 on 1H
    df['ema_200_4h'] = df_4h['ema_200_4h'].reindex(df.index, method='ffill')

    # Add volume confirmation
    df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()

    # --- Daily Level Count ---
    # Resample to daily timeframe to find major swing points
    daily_df = df.resample('D').agg({
        'High': 'max',
        'Low': 'min',
    }).dropna()

    # Find peaks (swing highs) and troughs (swing lows) on the daily chart
    high_peaks_indices, _ = find_peaks(daily_df['High'], distance=3) # distance=3 ensures a multi-day cycle
    low_troughs_indices, _ = find_peaks(-daily_df['Low'], distance=3)

    daily_df['swing_high'] = False
    daily_df.iloc[high_peaks_indices, daily_df.columns.get_loc('swing_high')] = True
    daily_df['swing_low'] = False
    daily_df.iloc[low_troughs_indices, daily_df.columns.get_loc('swing_low')] = True

    # Calculate level counts based on consecutive lower highs (for sells) or higher lows (for buys)
    daily_df['sell_level_count'] = 0
    daily_df['buy_level_count'] = 0

    current_sell_level = 0
    for i in range(1, len(daily_df)):
        if daily_df['High'].iloc[i] > daily_df['High'].iloc[i-1]: # Corrected: count higher highs for a rise
            current_sell_level += 1
        else:
            current_sell_level = 0
        if daily_df['swing_high'].iloc[i]:
            daily_df.iloc[i, daily_df.columns.get_loc('sell_level_count')] = current_sell_level

    current_buy_level = 0
    for i in range(1, len(daily_df)):
        if daily_df['Low'].iloc[i] < daily_df['Low'].iloc[i-1]: # Corrected: count lower lows for a drop
            current_buy_level += 1
        else:
            current_buy_level = 0
        if daily_df['swing_low'].iloc[i]:
            daily_df.iloc[i, daily_df.columns.get_loc('buy_level_count')] = current_buy_level

    # Forward-fill the levels to apply them to the intraday timeframe
    daily_df['sell_level_count'] = daily_df['sell_level_count'].replace(0, pd.NA).ffill().fillna(0).astype(int)
    daily_df['buy_level_count'] = daily_df['buy_level_count'].replace(0, pd.NA).ffill().fillna(0).astype(int)

    # Merge daily levels back into the 15m dataframe
    df = pd.merge(df, daily_df[['sell_level_count', 'buy_level_count']], left_index=True, right_index=True, how='left')
    df['sell_level_count'] = df['sell_level_count'].ffill()
    df['buy_level_count'] = df['buy_level_count'].ffill()

    return df

class ThreeDayThreeLevelReversal(Strategy):
    """
    Implements the Three-Day, Three-Level Reversal strategy.
    """
    # --- Strategy Parameters ---
    atr_multiplier_sl = 2.0
    atr_multiplier_tp = 3.0

    def init(self):
        """
        Initialize indicators and strategy state.
        """
        # State tracking for the 3-day setup
        self.sell_setup_active = False
        self.buy_setup_active = False
        self.setup_bar_index = 0


    def next(self):
        """
        Main strategy logic executed on each bar.
        """
        # --- Warmup Period ---
        if len(self.data.Close) < 850:
            return

        # --- Invalidation Logic ---
        # Invalidate setup after N bars if no trade is triggered
        BARS_TO_INVALIDATE = 96 # 1 day on 15m chart
        if (self.sell_setup_active or self.buy_setup_active) and (len(self.data.Close) > self.setup_bar_index + BARS_TO_INVALIDATE):
             self.sell_setup_active = False
             self.buy_setup_active = False

        # --- Entry Conditions ---
        if not self.position:
            # STAGE 1: Detect 3-Day Setup
            if not self.sell_setup_active and not self.buy_setup_active:
                if self.data.sell_level_count[-1] >= 3:
                    self.sell_setup_active = True
                    self.setup_bar_index = len(self.data.Close)
                elif self.data.buy_level_count[-1] >= 3:
                    self.buy_setup_active = True
                    self.setup_bar_index = len(self.data.Close)

            # STAGE 2: Wait for Trigger
            if self.sell_setup_active:
                # Confluence Checks for entry
                if (self.data.Close[-1] < self.data.ema_200_4h[-1] and
                    self.data.sell_signal[-1] and
                    self.data.Volume[-1] > self.data.volume_ma_20[-1]):

                    sl = self.data.Close[-1] + self.data.ATRr_14[-1] * self.atr_multiplier_sl
                    tp = self.data.Close[-1] - self.data.ATRr_14[-1] * self.atr_multiplier_tp
                    if tp < sl:
                        self.sell(sl=sl, tp=tp)
                        self.sell_setup_active = False # Reset state

            elif self.buy_setup_active:
                # Confluence Checks for entry
                if (self.data.Close[-1] > self.data.ema_200_4h[-1] and
                    self.data.buy_signal[-1] and
                    self.data.Volume[-1] > self.data.volume_ma_20[-1]):

                    sl = self.data.Close[-1] - self.data.ATRr_14[-1] * self.atr_multiplier_sl
                    tp = self.data.Close[-1] + self.data.ATRr_14[-1] * self.atr_multiplier_tp
                    if tp > sl:
                        self.buy(sl=sl, tp=tp)
                        self.buy_setup_active = False # Reset state


def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to be JSON serializable.
    """
    if stats is None:
        return {}

    # Extract the necessary data, handling potential missing keys
    sanitized = {
        'Start': str(stats.get('Start', pd.NaT)),
        'End': str(stats.get('End', pd.NaT)),
        'Duration': str(stats.get('Duration', pd.NaT)),
        'Exposure Time [%]': stats.get('Exposure Time [%]', 0.0),
        'Equity Final [$]': stats.get('Equity Final [$]', 0.0),
        'Equity Peak [$]': stats.get('Equity Peak [$]', 0.0),
        'Return [%]': stats.get('Return [%]', 0.0),
        'Buy & Hold Return [%]': stats.get('Buy & Hold Return [%]', 0.0),
        'Return (Ann.) [%]': stats.get('Return (Ann.) [%]', 0.0),
        'Volatility (Ann.) [%]': stats.get('Volatility (Ann.) [%]', 0.0),
        'Sharpe Ratio': stats.get('Sharpe Ratio', 0.0),
        'Sortino Ratio': stats.get('Sortino Ratio', 0.0),
        'Calmar Ratio': stats.get('Calmar Ratio', 0.0),
        'Max. Drawdown [%]': stats.get('Max. Drawdown [%]', 0.0),
        'Avg. Drawdown [%]': stats.get('Avg. Drawdown [%]', 0.0),
        'Max. Drawdown Duration': str(stats.get('Max. Drawdown Duration', pd.NaT)),
        'Avg. Drawdown Duration': str(stats.get('Avg. Drawdown Duration', pd.NaT)),
        '# Trades': stats.get('# Trades', 0),
        'Win Rate [%]': stats.get('Win Rate [%]', 0.0),
        'Best Trade [%]': stats.get('Best Trade [%]', 0.0),
        'Worst Trade [%]': stats.get('Worst Trade [%]', 0.0),
        'Avg. Trade [%]': stats.get('Avg. Trade [%]', 0.0),
        'Max. Trade Duration': str(stats.get('Max. Trade Duration', pd.NaT)),
        'Avg. Trade Duration': str(stats.get('Avg. Trade Duration', pd.NaT)),
        'Profit Factor': stats.get('Profit Factor', 0.0),
        'Expectancy [%]': stats.get('Expectancy [%]', 0.0),
        'SQN': stats.get('SQN', 0.0)
    }

    # Clean up any remaining non-serializable types
    for key, value in sanitized.items():
        if isinstance(value, (np.integer, np.floating)):
            sanitized[key] = float(value)
        elif pd.isna(value):
            sanitized[key] = None

    return sanitized


if __name__ == '__main__':
    import json

    # --- Configuration ---
    DATA_PATH = 'data/BTC-USD-15m.csv'
    CASH = 100_000
    COMMISSION = 0.002

    # --- Data Loading ---
    try:
        df = pd.read_csv(DATA_PATH, index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        # As a fallback, create synthetic data for demonstration
        print("Generating synthetic data...")
        from backtesting.test import GOOG
        df = GOOG.copy()
        df = df.iloc[-2000:] # Use a subset for speed


    # --- Preprocessing ---
    df_processed = preprocess_data(df)
    df_processed.dropna(inplace=True)

    # --- Backtesting ---
    bt = Backtest(df_processed, ThreeDayThreeLevelReversal, cash=CASH, commission=COMMISSION)
    stats = bt.run()

    # --- Results ---
    print("--- Backtest Results ---")
    print(stats)

    # --- Save Results ---
    os.makedirs('results', exist_ok=True)

    # Save stats to JSON
    stats_dict = sanitize_stats(stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print("\nSaved backtest stats to results/temp_result.json")

    # Save plot
    plot_filename = 'results/three_day_three_level_reversal.html'
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Saved plot to {plot_filename}")
    except Exception as e:
        print(f"\nCould not save plot: {e}")
