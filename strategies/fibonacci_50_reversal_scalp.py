from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import json
import os

def generate_synthetic_data(n_points=5000):
    """
    Generates synthetic price data with M and W patterns.
    """
    np.random.seed(42)
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_points, freq='1min'))
    price = 100 * (1 + np.random.randn(n_points).cumsum() * 0.0001)

    df = pd.DataFrame({
        'Open': price,
        'High': price,
        'Low': price,
        'Close': price
    }, index=dates)

    # --- Inject M-Pattern ---
    # A (High) -> B (Low) -> C (Retracement High) -> D (Confirmation Low)
    start_m = 500
    df.iloc[start_m+10:start_m+60, 1] += np.linspace(0, 10, 50) # A
    df.iloc[start_m+60:start_m+110, 2] -= np.linspace(0, 10, 50) # B
    df.iloc[start_m+110:start_m+160, 1] += np.linspace(0, 5, 50) # C (50% retracement)
    df.iloc[start_m+160:start_m+210, 2] -= np.linspace(0, 12, 50) # D

    # --- Inject W-Pattern ---
    # A (Low) -> B (High) -> C (Retracement Low) -> D (Confirmation High)
    start_w = 1500
    df.iloc[start_w+10:start_w+60, 2] -= np.linspace(0, 10, 50) # A
    df.iloc[start_w+60:start_w+110, 1] += np.linspace(0, 10, 50) # B
    df.iloc[start_w+110:start_w+160, 2] -= np.linspace(0, 5, 50) # C (50% retracement)
    df.iloc[start_w+160:start_w+210, 1] += np.linspace(0, 12, 50) # D

    # Finalize OHLC
    df['Open'] = df['Close'].shift(1)
    df = df.iloc[1:] # Drop first row with NaN open
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.rand(len(df)) * 0.1
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.rand(len(df)) * 0.1

    return df

from scipy.signal import find_peaks

def preprocess_data(df_1m):
    """
    Pre-processes 1M data to add signals and indicators from a 15M perspective.
    """
    # --- Resample to 15M to find major swing points ---
    df_15m = df_1m['Close'].resample('15min').ohlc()

    # Find peaks (swing highs) and troughs (swing lows) on the 15M chart
    # Note: inverting the series to find troughs with find_peaks
    peak_indices, _ = find_peaks(df_15m['high'], prominence=1, width=3)
    trough_indices, _ = find_peaks(-df_15m['low'], prominence=1, width=3)

    df_15m['swing_high'] = np.nan
    df_15m['swing_low'] = np.nan
    df_15m.iloc[peak_indices, df_15m.columns.get_loc('swing_high')] = df_15m['high'].iloc[peak_indices]
    df_15m.iloc[trough_indices, df_15m.columns.get_loc('swing_low')] = df_15m['low'].iloc[trough_indices]

    df_15m['swing_high'] = df_15m['swing_high'].ffill()
    df_15m['swing_low'] = df_15m['swing_low'].ffill()

    # --- Identify M and W pattern setups ---
    df_15m['signal'] = 0
    df_15m['fib_level'] = np.nan

    # M-Pattern: a swing high followed by a lower swing low
    m_pattern = (df_15m['swing_high'].shift(1) > df_15m['swing_low']) & (df_15m['swing_high'].shift(1) == df_15m['swing_high'])

    # W-Pattern: a swing low followed by a higher swing high
    w_pattern = (df_15m['swing_low'].shift(1) < df_15m['swing_high']) & (df_15m['swing_low'].shift(1) == df_15m['swing_low'])

    # Calculate Fib 50% level for valid setups
    df_15m.loc[m_pattern, 'fib_level'] = df_15m['swing_low'] + (df_15m['swing_high'].shift(1) - df_15m['swing_low']) * 0.5
    df_15m.loc[m_pattern, 'signal'] = -1

    df_15m.loc[w_pattern, 'fib_level'] = df_15m['swing_high'] - (df_15m['swing_high'] - df_15m['swing_low'].shift(1)) * 0.5
    df_15m.loc[w_pattern, 'signal'] = 1

    # --- Merge 15M signals back into 1M dataframe ---
    df_merged = pd.merge_asof(df_1m, df_15m[['signal', 'fib_level', 'swing_high', 'swing_low']],
                              left_index=True, right_index=True, direction='forward')

    # --- Calculate Confluence Indicators ---
    # Asia Session (e.g., 00:00 - 08:00 UTC)
    df_merged['is_asia'] = (df_merged.index.hour >= 0) & (df_merged.index.hour < 8)
    asia_high = df_merged[df_merged['is_asia']]['High'].resample('D').max()
    asia_low = df_merged[df_merged['is_asia']]['Low'].resample('D').min()
    df_merged['asia_high'] = df_merged.index.normalize().map(asia_high)
    df_merged['asia_low'] = df_merged.index.normalize().map(asia_low)

    # High/Low of Day
    df_merged['hod'] = df_merged['High'].resample('D').cummax().ffill()
    df_merged['lod'] = df_merged['Low'].resample('D').cummin().ffill()

    df_merged.fillna(method='ffill', inplace=True)
    df_merged.dropna(inplace=True)

    return df_merged


def EMA(array, n):
    """Exponential moving average"""
    return pd.Series(array).ewm(span=n, adjust=False).mean().values

class Fibonacci50ReversalScalpStrategy(Strategy):
    # --- Strategy Parameters ---
    ema_period = 50
    min_rr = 3.0

    def init(self):
        # --- Instance variables for state tracking ---
        # These are used to store key price levels of a pattern
        # once an entry has been made, for TP calculation.
        self.m_peak = None
        self.m_trough = None
        self.w_peak = None
        self.w_trough = None

        # --- Indicators ---
        self.ema = self.I(EMA, self.data.Close, self.ema_period)

        # --- Pre-processed Data Columns ---
        # These columns are expected to be present in the input DataFrame
        # and are calculated in the pre-processing step
        self.signal = self.data.df['signal']
        self.swing_high = self.data.df['swing_high']
        self.swing_low = self.data.df['swing_low']
        self.fib_level = self.data.df['fib_level']
        self.is_asia = self.data.df['is_asia']
        self.asia_high = self.data.df['asia_high']
        self.asia_low = self.data.df['asia_low']
        self.lod = self.data.df['lod']
        self.hod = self.data.df['hod']


    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # --- Confluence Check Helper ---
        def is_in_confluence_zone(level, price, ema, hod, lod, asia_high, asia_low):
            # Check if the level is within a small percentage of key areas
            tolerance = 0.005 # 0.5% tolerance
            if abs(price / level - 1) > tolerance:
                return False # Price must be near the Fib level to consider entry

            # Check alignment with other key levels
            if abs(ema / level - 1) < tolerance: return True
            if hod is not np.nan and abs(hod / level - 1) < tolerance: return True
            if lod is not np.nan and abs(lod / level - 1) < tolerance: return True
            if asia_high is not np.nan and abs(asia_high / level - 1) < tolerance: return True
            if asia_low is not np.nan and abs(asia_low / level - 1) < tolerance: return True

            return False

        # --- Reversal Pattern Check Helper ---
        def is_bearish_reversal():
            # Bearish Engulfing
            if (self.data.Close[-1] < self.data.Open[-1] and # Current is bearish
                self.data.Open[-1] > self.data.Close[-2] and
                self.data.Close[-1] < self.data.Open[-2]):
                return True
            # Add other reversal patterns here if needed (e.g., Pin Bar)
            return False

        def is_bullish_reversal():
            # Bullish Engulfing
            if (self.data.Close[-1] > self.data.Open[-1] and # Current is bullish
                self.data.Open[-1] < self.data.Close[-2] and
                self.data.Close[-1] > self.data.Open[-2]):
                return True
            return False

        # --- Trade Management ---
        # If a trade is already open, manage the exit
        if self.position:
            trade = self.trades[0]
            # Exit logic for 'bag flip' based on Fib 2
            if trade.is_long:
                # Correct TP logic for W-pattern (Long)
                # Fib 2 is from the retracement low (C) to the retracement high (B)
                retracement_low = self.w_trough # Point C
                retracement_high = self.w_peak   # Point B
                if retracement_high and retracement_low and retracement_high > retracement_low:
                    fib2_tp_level = retracement_low + (retracement_high - retracement_low) * 0.5
                    if not trade.tp: # Set TP if not already set
                         self.trades[0].tp = fib2_tp_level

            else: # Short trade
                # Correct TP logic for M-pattern (Short)
                # Fib 2 is from the retracement high (C) to the retracement low (B)
                retracement_high = self.m_peak   # Point C
                retracement_low = self.m_trough  # Point B
                if retracement_high and retracement_low and retracement_high > retracement_low:
                    fib2_tp_level = retracement_high - (retracement_high - retracement_low) * 0.5
                    if not trade.tp: # Set TP if not already set
                        self.trades[0].tp = fib2_tp_level

            # Simple exit - let SL/TP handle it. Bag-flip is complex for a state machine.
            return


        # --- Entry Logic ---
        # No new trades if one is already open
        if self.position:
            return

        # --- Short Entry (M-Pattern) ---
        if self.signal[-1] == -1: # M-setup identified in pre-processing
            fib_level = self.fib_level[-1]

            # Check for confluence
            confluence = is_in_confluence_zone(fib_level, price, self.ema[-1], self.hod[-1], self.lod[-1], self.asia_high[-1], self.asia_low[-1])

            if confluence and is_bearish_reversal():
                # Define SL and TP
                stop_loss = high + (high * 0.001) # SL above the reversal candle high

                # The TP is dynamic (Fib 2), so we place the trade and set TP in the next bars
                # For R:R check, we can estimate a TP.
                # Let's assume the M-pattern completes, so TP is near the initial trough.
                estimated_tp = self.swing_low[-1]

                if estimated_tp and (price - estimated_tp) / (stop_loss - price) >= self.min_rr:
                    self.sell(sl=stop_loss)
                    # Store key levels for TP calculation later
                    self.m_peak = high
                    self.m_trough = self.swing_low[-1]


        # --- Long Entry (W-Pattern) ---
        elif self.signal[-1] == 1: # W-setup identified in pre-processing
            fib_level = self.fib_level[-1]

            confluence = is_in_confluence_zone(fib_level, price, self.ema[-1], self.hod[-1], self.lod[-1], self.asia_high[-1], self.asia_low[-1])

            if confluence and is_bullish_reversal():
                stop_loss = low - (low * 0.001) # SL below the reversal candle low

                estimated_tp = self.swing_high[-1]

                if estimated_tp and (estimated_tp - price) / (price - stop_loss) >= self.min_rr:
                    self.buy(sl=stop_loss)
                    # Store key levels for TP calculation later
                    self.w_trough = low
                    self.w_peak = self.swing_high[-1]


if __name__ == '__main__':
    # --- 1. Data Generation & Pre-processing ---
    data_1m = generate_synthetic_data(n_points=10000)
    data_processed = preprocess_data(data_1m)

    # --- 2. Backtest Initialization ---
    bt = Backtest(data_processed, Fibonacci50ReversalScalpStrategy, cash=100_000, commission=.002)

    # --- 3. Optimization ---
    stats = bt.optimize(
        ema_period=range(20, 101, 10),
        min_rr=np.arange(2.0, 5.1, 0.5).tolist(),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.ema_period > 10 # Example constraint
    )

    print("Best Run Stats:")
    print(stats)
    print("\nBest Parameters:")
    print(stats._strategy)

    # --- 4. Save Results ---
    os.makedirs('results', exist_ok=True)
    results_path = 'results/temp_result.json'

    # Handle potential NaN values for JSON compatibility
    sharpe = stats.get('Sharpe Ratio', 0.0)

    with open(results_path, 'w') as f:
        json.dump({
            'strategy_name': 'fibonacci_50_reversal_scalp',
            'return': stats.get('Return [%]', 0.0),
            'sharpe': 0.0 if np.isnan(sharpe) else sharpe,
            'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
            'win_rate': stats.get('Win Rate [%]', 0.0),
            'total_trades': stats.get('# Trades', 0)
        }, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # --- 5. Generate Plot ---
    plot_path = 'results/fibonacci_50_reversal_scalp_plot.html'
    try:
        bt.plot(filename=plot_path, open_browser=False)
        print(f"Plot saved to {plot_path}")
    except TypeError as e:
        print(f"\nCould not generate plot due to a known issue with the plotting library: {e}")
        print("This does not affect the JSON results.")
