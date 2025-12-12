from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
from scipy.signal import find_peaks

def generate_synthetic_data(days=15, freq='1min'):
    """Generates synthetic price data with M/W patterns."""
    n_periods = days * 24 * 60
    index = pd.date_range(start='2023-01-01', periods=n_periods, freq=freq)

    base_price = 100
    wave1 = np.sin(np.linspace(0, 80 * np.pi, n_periods)) * 2 # Main swings
    wave2 = np.sin(np.linspace(0, 160 * np.pi, n_periods)) * 0.5 # Smaller volatility

    trend = np.linspace(0, 20, n_periods)
    noise = np.random.randn(n_periods) * 0.3

    close = base_price + wave1 + wave2 + trend + noise

    open_price = close - np.random.rand(n_periods) * 0.2
    high = np.maximum(close, open_price) + np.random.rand(n_periods) * 0.3
    low = np.minimum(close, open_price) - np.random.rand(n_periods) * 0.3

    df = pd.DataFrame({'Open': open_price, 'High': high, 'Low': low, 'Close': close}, index=index)
    return df

def preprocess_data(df):
    """Calculates HOD/LOD and session data, merging it back to the main timeframe."""
    df_copy = df.copy()

    # Define sessions
    asia_session_start = '00:00'
    asia_session_end = '08:00'

    # Create a daily dataframe with HOD/LOD
    daily_df = df_copy.resample('D').agg({'High':'max', 'Low':'min'}).rename(columns={'High':'HOD', 'Low':'LOD'})

    # Calculate Asia session range and merge
    asia_range_df = df_copy.between_time(asia_session_start, asia_session_end).resample('D').agg({'High':'max', 'Low':'min'})
    asia_range_df.rename(columns={'High':'Asia_High', 'Low':'Asia_Low'}, inplace=True)
    daily_df = daily_df.merge(asia_range_df, left_index=True, right_index=True, how='left')

    # Forward fill session data for weekends/holidays
    daily_df['Asia_High'] = daily_df['Asia_High'].ffill()
    daily_df['Asia_Low'] = daily_df['Asia_Low'].ffill()
    daily_df['Asia_50_Range'] = daily_df['Asia_Low'] + (daily_df['Asia_High'] - daily_df['Asia_Low']) * 0.5

    # Merge daily data back to the minute data using the date as a key
    df_copy['Date'] = pd.to_datetime(df_copy.index.date)
    daily_df.index = pd.to_datetime(daily_df.index.date)

    final_df = pd.merge(df_copy, daily_df, left_on='Date', right_index=True, how='left')
    final_df.index = df_copy.index # Restore original datetime index

    final_df.drop(columns=['Date'], inplace=True)

    # The first day might have NaNs for daily/session data, ffill them
    final_df['HOD'] = final_df['HOD'].ffill()
    final_df['LOD'] = final_df['LOD'].ffill()
    final_df.bfill(inplace=True) # Backfill any remaining at the start

    return final_df

def detect_mw_patterns(highs, lows, prominence, **kwargs):
    """
    Custom indicator to detect M and W patterns using scipy.find_peaks.
    Returns multiple signal arrays.
    """
    highs = pd.Series(highs)
    lows = pd.Series(lows)

    # Find peaks (swing highs) and troughs (swing lows)
    peak_indices, _ = find_peaks(highs, prominence=prominence)
    trough_indices, _ = find_peaks(-lows, prominence=prominence)

    swings = sorted(list(set(peak_indices) | set(trough_indices)))

    # Initialize output arrays
    n = len(highs)
    pattern_type = np.full(n, 0)      # 0: None, 1: M, 2: W
    aoi_level = np.full(n, np.nan)    # 50% Fib level
    p1_price = np.full(n, np.nan)
    t1_price = np.full(n, np.nan)
    p2_price = np.full(n, np.nan)
    t2_price = np.full(n, np.nan)

    for i in range(len(swings) - 2):
        s1_idx, s2_idx, s3_idx = swings[i], swings[i+1], swings[i+2]

        # M-Pattern (Peak-Trough-Peak)
        if s1_idx in peak_indices and s2_idx in trough_indices and s3_idx in peak_indices:
            p1_val, t1_val = highs[s1_idx], lows[s2_idx]

            # The pattern is confirmed at the third swing point (s3_idx)
            pattern_type[s3_idx] = 1 # M-Pattern
            aoi_level[s3_idx] = t1_val + (p1_val - t1_val) * 0.5
            p1_price[s3_idx] = p1_val
            t1_price[s3_idx] = t1_val
            p2_price[s3_idx] = highs[s3_idx]

        # W-Pattern (Trough-Peak-Trough)
        elif s1_idx in trough_indices and s2_idx in peak_indices and s3_idx in trough_indices:
            t1_val, p1_val = lows[s1_idx], highs[s2_idx]

            pattern_type[s3_idx] = 2 # W-Pattern
            aoi_level[s3_idx] = p1_val - (p1_val - t1_val) * 0.5
            t1_price[s3_idx] = t1_val
            p1_price[s3_idx] = p1_val
            t2_price[s3_idx] = lows[s3_idx]

    return pattern_type, aoi_level, p1_price, t1_price, p2_price, t2_price

class MeasuredMWCenterPeakScalpStrategy(Strategy):
    # Parameters will be related to pattern detection
    peak_prominence = 1.5 # Simulates HTF swing point identification
    ema_period = 50

    def init(self):
        # 1. EMA for trend/confluence
        self.ema = self.I(lambda x, n: pd.Series(x).ewm(span=n, adjust=False).mean(), self.data.Close, self.ema_period)

        # 2. Detect M/W patterns and their AOIs
        self.patterns = self.I(
            detect_mw_patterns,
            self.data.High,
            self.data.Low,
            self.peak_prominence,
            plot=False # Do not plot the indicator itself, just the trades
        )
        self.pattern_type = self.patterns[0]
        self.aoi_level    = self.patterns[1]
        self.p1_price     = self.patterns[2]
        self.t1_price     = self.patterns[3]

        # 3. State machine variables
        self.trade_setup = 0 # 0: None, 1: M-wait, 1.5: M-in-AOI, 2: W-wait, 2.5: W-in-AOI
        self.current_aoi = np.nan
        self.reversal_high = np.nan
        self.reversal_low = np.nan
        self.setup_bar_index = 0
        self.pattern_p1 = np.nan
        self.pattern_t1 = np.nan

    def next(self):
        # --- Invalidation: If a setup is active for too long, cancel it ---
        if self.trade_setup != 0 and (len(self.data) - self.setup_bar_index) > 240: # 4-hour timeout
            self.trade_setup = 0

        # If a position is already open, do nothing
        if self.position:
            return

        # --- State 0: Look for a new pattern ---
        if self.trade_setup == 0:
            if self.pattern_type[-1] == 1: # M-Pattern
                self.trade_setup = 1
                self.current_aoi = self.aoi_level[-1]
                self.pattern_t1 = self.t1_price[-1]
                self.setup_bar_index = len(self.data) - 1
            elif self.pattern_type[-1] == 2: # W-Pattern
                self.trade_setup = 2
                self.current_aoi = self.aoi_level[-1]
                self.pattern_p1 = self.p1_price[-1]
                self.setup_bar_index = len(self.data) - 1

        # --- State 1: M-Pattern Detected, Awaiting Price Retracement to AOI ---
        if self.trade_setup == 1 and self.data.High[-1] >= self.current_aoi:
            self.trade_setup = 1.5
            self.reversal_high = self.data.High[-1] # Initial high of the move into the AOI

        # --- State 1.5: Price in AOI, Tracking Reversal Peak for Stop-Loss ---
        elif self.trade_setup == 1.5:
            self.reversal_high = max(self.reversal_high, self.data.High[-1])
            # Condition: price closes back below the AOI
            if self.data.Close[-1] < self.current_aoi:
                # Confluence: check if we are in a downtrend (below EMA)
                if self.data.Close[-1] < self.ema[-1]:
                    sl = self.reversal_high
                    # TP is a 50% measured move of the center peak structure
                    tp = self.reversal_high - (self.reversal_high - self.pattern_t1) * 0.5
                    if tp < self.data.Close[-1]: # Ensure TP is valid
                        self.sell(sl=sl, tp=tp)
                self.trade_setup = 0 # Reset setup

        # --- State 2: W-Pattern Detected, Awaiting Price Retracement to AOI ---
        if self.trade_setup == 2 and self.data.Low[-1] <= self.current_aoi:
            self.trade_setup = 2.5
            self.reversal_low = self.data.Low[-1] # Initial low of the move into the AOI

        # --- State 2.5: Price in AOI, Tracking Reversal Trough for Stop-Loss ---
        elif self.trade_setup == 2.5:
            self.reversal_low = min(self.reversal_low, self.data.Low[-1])
            # Condition: price closes back above the AOI
            if self.data.Close[-1] > self.current_aoi:
                # Confluence: check if we are in an uptrend (above EMA)
                if self.data.Close[-1] > self.ema[-1]:
                    sl = self.reversal_low
                    # TP is a 50% measured move of the center peak structure
                    tp = self.reversal_low + (self.pattern_p1 - self.reversal_low) * 0.5
                    if tp > self.data.Close[-1]: # Ensure TP is valid
                        self.buy(sl=sl, tp=tp)
                self.trade_setup = 0 # Reset setup

if __name__ == '__main__':
    # 1. Generate and preprocess data
    data = generate_synthetic_data(days=30)
    data = preprocess_data(data)

    # 2. Run backtest
    bt = Backtest(data, MeasuredMWCenterPeakScalpStrategy, cash=100_000, commission=.002)

    # 3. Optimize
    print("Optimizing strategy...")
    stats = bt.optimize(
        peak_prominence=list(np.arange(0.5, 3.0, 0.5)),
        ema_period=range(20, 101, 10),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.ema_period > 10 # Example constraint
    )

    print("Best stats:", stats)

    # Re-run with best params for plotting
    # Note: Optimization returns the stats series, the strategy with best params is in _strategy
    best_params = stats._strategy
    bt.run(peak_prominence=best_params.peak_prominence, ema_period=best_params.ema_period)

    # 4. Save results
    import os
    os.makedirs('results', exist_ok=True)

    # Handle cases with no trades
    if stats['# Trades'] > 0:
        win_rate = float(stats['Win Rate [%]'])
        sharpe = float(stats['Sharpe Ratio'])
    else:
        win_rate = 0.0
        sharpe = 0.0

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'measured_m_w_center_peak_scalp',
            'return': float(stats['Return [%]']),
            'sharpe': sharpe,
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': win_rate,
            'total_trades': int(stats['# Trades'])
        }, f, indent=2)

    # 5. Generate plot
    print("Generating plot...")
    bt.plot(filename='results/measured_m_w_center_peak_scalp.html')
    print(stats)
