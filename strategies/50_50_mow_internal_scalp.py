import json
import os
from datetime import time

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks


class FiftyFiftyMowInternalScalpStrategy(Strategy):
    """
    Implements the 50/50 Mow Internal Scalp strategy, a multi-timeframe
    Fibonacci-based reversal strategy.
    """
    # Strategy parameters for optimization
    ema_period = 20
    rr_ratio = 5
    peak_prominence = 0.02
    ema_confluence_threshold = 0.001 # 0.1%

    def init(self):
        """Initialize the strategy."""
        # State machine for tracking trade setup
        self.trade_setup_state = "SEARCHING"
        self.current_setup_high = None
        self.current_setup_low = None
        self.entry_peak = None

        # Expose pre-calculated data to the strategy
        self.aoi_level = self.data.df['aoi_level']
        self.setup_high = self.data.df['setup_high']
        self.setup_low = self.data.df['setup_low']
        self.ema_15m = self.data.df['ema_15m']
        self.prev_low = self.data.df['prev_low']
        self.asia_50_pct = self.data.df['asia_50_pct']


    def next(self):
        """Main strategy logic executed on each data bar."""
        price = self.data.Close[-1]

        # --- State: SEARCHING for a new setup ---
        if self.trade_setup_state == "SEARCHING":
            # A new setup is identified if the setup_high has changed from the previous bar
            if self.setup_high[-1] != self.setup_high[-2]:
                self.current_setup_high = self.setup_high[-1]
                self.current_setup_low = self.setup_low[-1]
                self.trade_setup_state = "IN_AOI_WAIT"
                # print(f"{self.data.index[-1]}: New setup detected. High: {self.current_setup_high}, Low: {self.current_setup_low}, AOI: {self.aoi_level[-1]}")

        # --- State: IN_AOI_WAIT - Waiting for price to enter the AOI ---
        elif self.trade_setup_state == "IN_AOI_WAIT":
            # Invalidation: If a new lower low forms, the setup is invalid
            if self.data.Low[-1] < self.current_setup_low:
                self.trade_setup_state = "SEARCHING"
                # print(f"{self.data.index[-1]}: Setup invalidated. Price made a new low.")
                return

            # Check if price has entered the AOI
            if self.data.High[-1] >= self.aoi_level[-1]:
                # Check for confluence
                ema_close = abs(price - self.ema_15m[-1]) / price < self.ema_confluence_threshold
                lod_close = abs(price - self.prev_low[-1]) / price < self.ema_confluence_threshold
                asia_close = abs(price - self.asia_50_pct[-1]) / price < self.ema_confluence_threshold

                # We need at least one confluence factor to be true
                if ema_close or lod_close or asia_close:
                    self.trade_setup_state = "ENTRY_TRIGGER"
                    self.entry_peak = self.data.High[-1] # Initial peak
                    # print(f"{self.data.index[-1]}: Price entered AOI with confluence. Waiting for entry trigger.")

        # --- State: ENTRY_TRIGGER - Waiting for a reversal candle ---
        elif self.trade_setup_state == "ENTRY_TRIGGER":
            # Update the peak of the entry structure
            self.entry_peak = max(self.entry_peak, self.data.High[-1])

            # Invalidation: If price breaks significantly above setup high, invalidate
            if price > self.current_setup_high * 1.002:
                 self.trade_setup_state = "SEARCHING"
                 # print(f"{self.data.index[-1]}: Setup invalidated. Price broke structure high.")
                 return

            # Entry Trigger: Bearish reversal candle (Close < Open) after touching AOI
            is_bearish_candle = self.data.Close[-1] < self.data.Open[-1]
            if is_bearish_candle and not self.position:

                # --- Risk Management ---
                stop_loss = self.entry_peak * 1.001 # SL just above the entry structure

                # TP is 50% of the move from the initial low to the entry peak
                take_profit = self.entry_peak - (self.entry_peak - self.current_setup_low) * 0.5

                risk = stop_loss - price
                reward = price - take_profit

                if risk <= 0 or reward <= 0: return # Invalid R:R

                calculated_rr = reward / risk

                if calculated_rr >= self.rr_ratio:
                    self.sell(sl=stop_loss, tp=take_profit, size=1)
                    # print(f"{self.data.index[-1]}: SELL ORDER PLACED | Price: {price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f} | R:R: {calculated_rr:.2f}")

        # --- Position Management ---
        if self.position:
            # If the trade is closed (either by SL or TP), reset state
            pass # backtesting.py handles closure, we just need to reset state
        elif self.trade_setup_state == "ENTRY_TRIGGER" and not self.position:
             # This handles case where order was placed but not filled, then setup invalidates etc.
             # Or more importantly, when a trade is closed, we need to reset.
             self.trade_setup_state = "SEARCHING"


def generate_synthetic_data(days=90):
    """
    Generates synthetic 1-minute OHLCV data with embedded 'M' patterns
    for testing the 50/50 Mow strategy.
    """
    n_minutes = days * 24 * 60
    index = pd.date_range(start="2023-01-01", periods=n_minutes, freq='1min')
    base_price = 100
    volatility = 0.0005

    # Generate random walk for base price
    rng = np.random.default_rng(seed=42)
    returns = rng.normal(loc=0, scale=volatility, size=n_minutes)
    price = base_price * np.exp(np.cumsum(returns))

    # Inject 'M' patterns
    num_patterns = int(days / 3) # One pattern every 3 days
    for i in range(num_patterns):
        try:
            start_idx = rng.integers(i * (n_minutes // num_patterns), (i + 1) * (n_minutes // num_patterns) - 1000)

            # 1. Initial drop (leg 1)
            p1_len = rng.integers(120, 240) # 2-4 hours
            drop_amount = price[start_idx] * 0.02 # 2% drop
            leg1 = np.linspace(0, -drop_amount, p1_len)
            price[start_idx : start_idx + p1_len] += leg1

            # 2. Retracement (center of M)
            p2_len = rng.integers(120, 240)
            retrace_amount = drop_amount * 0.5 # 50% retrace
            leg2 = np.linspace(0, retrace_amount, p2_len)
            price[start_idx + p1_len : start_idx + p1_len + p2_len] += leg2

            # 3. Final drop
            p3_len = rng.integers(120, 240)
            final_drop = drop_amount * 1.2 # Lower low
            leg3 = np.linspace(0, -final_drop, p3_len)
            price[start_idx + p1_len + p2_len : start_idx + p1_len + p2_len + p3_len] += leg3
        except IndexError:
            continue # Skip if pattern would go out of bounds

    # Create DataFrame
    df = pd.DataFrame(price, index=index, columns=['Close'])
    df['Open'] = df['Close'].shift(1)
    df['High'] = df[['Open', 'Close']].max(axis=1) + rng.uniform(0, volatility, size=n_minutes)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - rng.uniform(0, volatility, size=n_minutes)
    df['Volume'] = rng.integers(100, 1000, size=n_minutes)

    return df.dropna()

def preprocess_data(df_1m, peak_prominence_val, ema_period_val):
    """
    Pre-processes 1-minute data to include higher timeframe (15M) indicators
    and session data required for the strategy.
    """
    # --- Higher Timeframe (15M) Analysis ---
    df_15m = df_1m['Close'].resample('15min').ohlc()

    # Find swing highs and lows on 15M
    high_peaks_idx, _ = find_peaks(df_15m['high'], prominence=df_15m['high'].mean() * peak_prominence_val)
    low_troughs_idx, _ = find_peaks(-df_15m['low'], prominence=df_15m['low'].mean() * peak_prominence_val)

    df_15m['swing_high'] = np.nan
    df_15m['swing_low'] = np.nan
    df_15m.iloc[high_peaks_idx, df_15m.columns.get_loc('swing_high')] = df_15m.iloc[high_peaks_idx]['high']
    df_15m.iloc[low_troughs_idx, df_15m.columns.get_loc('swing_low')] = df_15m.iloc[low_troughs_idx]['low']

    df_15m['swing_high'] = df_15m['swing_high'].ffill()
    df_15m['swing_low'] = df_15m['swing_low'].ffill()

    # Identify M-pattern setup: a swing high followed by a new swing low
    df_15m['setup_high'] = df_15m['swing_high'].where(df_15m['swing_high'] != df_15m['swing_high'].shift(1))
    df_15m['setup_low'] = df_15m['swing_low'].where(df_15m['swing_low'] != df_15m['swing_low'].shift(1))

    df_15m['setup_high'] = df_15m['setup_high'].ffill()
    df_15m['setup_low'] = df_15m['setup_low'].where(df_15m['setup_low'] < df_15m['setup_low'].shift(1).fillna(np.inf)).ffill()

    # Calculate AOI (50% Fib)
    df_15m['aoi_level'] = (df_15m['setup_high'] + df_15m['setup_low']) / 2.0

    # --- Confluence Indicators ---
    # 15M EMA
    df_15m['ema_15m'] = pd.Series(df_15m['close']).ewm(span=ema_period_val, adjust=False).mean()

    # Daily High/Low (LOD/HOD)
    df_daily = df_1m['Close'].resample('D').ohlc()
    df_daily['prev_low'] = df_daily['low'].shift(1)
    df_daily['prev_high'] = df_daily['high'].shift(1)

    # Asia Session Range (00:00-08:00 UTC)
    asia_session = df_1m.between_time('00:00', '08:00').resample('D')
    asia_high = asia_session['high'].max()
    asia_low = asia_session['low'].min()
    asia_50_pct = (asia_high + asia_low) / 2
    df_daily['asia_50_pct'] = asia_50_pct.shift(1) # Use previous day's session

    # --- Merge back to 1M DataFrame ---
    df_merged = pd.merge(df_1m, df_15m[['aoi_level', 'setup_high', 'setup_low', 'ema_15m']], left_index=True, right_index=True, how='left')
    df_merged = pd.merge(df_merged, df_daily[['prev_low', 'prev_high', 'asia_50_pct']], left_on=df_merged.index.date, right_index=True, how='left')

    df_merged.set_index(df_1m.index, inplace=True)

    # Forward fill the merged data to apply HTF context to every 1M bar
    cols_to_fill = ['aoi_level', 'setup_high', 'setup_low', 'ema_15m', 'prev_low', 'prev_high', 'asia_50_pct']
    df_merged[cols_to_fill] = df_merged[cols_to_fill].ffill()

    return df_merged.dropna()


if __name__ == '__main__':
    # --- 1. Data Generation ---
    raw_data = generate_synthetic_data(days=60)

    # --- 2. Backtest Execution & Optimization ---
    # The optimization loop requires re-running pre-processing for parameters
    # that affect the data itself (peak_prominence, ema_period).

    def run_optimization(peak_prominence_val, ema_period_val):
        """Function to run a backtest for a given set of data-level parameters."""
        try:
            # Re-process data with the new parameter values
            processed_data = preprocess_data(raw_data.copy(), peak_prominence_val, ema_period_val)
            if processed_data.empty:
                return None

            bt = Backtest(processed_data, FiftyFiftyMowInternalScalpStrategy, cash=100_000, commission=.002)

            stats = bt.optimize(
                ema_confluence_threshold=[0.001, 0.002],
                rr_ratio=range(5, 8, 1),
                maximize='Sharpe Ratio',
                constraint=lambda p: p.rr_ratio > 0
            )
            # Attach the data-level params to the results for later reference
            stats._strategy.peak_prominence = peak_prominence_val
            stats._strategy.ema_period = ema_period_val
            return stats
        except Exception as e:
            print(f"Error during optimization run: {e}")
            return None

    # Manually iterate over data-level parameters
    prominence_range = [0.005, 0.015]
    ema_period_range = [20, 50]
    all_stats = []

    print("Starting optimization over data-level parameters...")
    for p_val in prominence_range:
        for ema_val in ema_period_range:
            print(f"  Testing Prominence: {p_val}, EMA Period: {ema_val}")
            run_stats = run_optimization(p_val, ema_val)
            if run_stats:
                all_stats.append(run_stats)

    # Filter out None results and find the best run
    valid_stats = [s for s in all_stats if s is not None and s.get('# Trades', 0) > 0]

    if not valid_stats:
        print("Optimization did not yield any valid results with trades. Exiting.")
    else:
        # Find the best stats based on the Sharpe Ratio
        stats = max(valid_stats, key=lambda s: s['Sharpe Ratio'])
        best_params = stats._strategy
        print("\nBest run found:")
        print(stats)
        print("\nWith parameters:")
        print(f"  - R:R Ratio: {best_params.rr_ratio}")
        print(f"  - EMA Threshold: {best_params.ema_confluence_threshold}")
        print(f"  - Peak Prominence: {best_params.peak_prominence}")
        print(f"  - EMA Period: {best_params.ema_period}")

        # --- 3. Final Run with Best Parameters & Plotting ---
        final_data = preprocess_data(raw_data.copy(), best_params.peak_prominence, best_params.ema_period)
        bt = Backtest(final_data, FiftyFiftyMowInternalScalpStrategy, cash=100_000, commission=.002)

        # Run the final backtest with the single best set of parameters
        final_stats = bt.run(
            rr_ratio=best_params.rr_ratio,
            ema_confluence_threshold=best_params.ema_confluence_threshold,
            ema_period=best_params.ema_period
        )

        # --- 4. Result Output ---
        os.makedirs('results', exist_ok=True)

        results = {
            'strategy_name': '50_50_mow_internal_scalp',
            'return': final_stats.get('Return [%]', 0.0),
            'sharpe': final_stats.get('Sharpe Ratio', None),
            'max_drawdown': final_stats.get('Max. Drawdown [%]', 0.0),
            'win_rate': final_stats.get('Win Rate [%]', 0.0),
            'total_trades': final_stats.get('# Trades', 0)
        }

        # Ensure numeric types are native Python types for JSON serialization
        for key in ['return', 'sharpe', 'max_drawdown', 'win_rate']:
            if pd.isna(results[key]):
                 results[key] = None
            elif results[key] is not None:
                results[key] = float(results[key])
        results['total_trades'] = int(results['total_trades'])

        # Write results to JSON
        with open('results/temp_result.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved to results/temp_result.json")

        # --- 5. Plotting ---
        plot_filename = 'results/50_50_mow_internal_scalp.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
