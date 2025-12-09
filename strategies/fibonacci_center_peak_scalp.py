
import json
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks

def generate_synthetic_data(periods=5000):
    """
    Generates synthetic 1-minute OHLC data with specific "Level Drop" patterns.
    """
    rng = np.random.default_rng(seed=42)
    dt_index = pd.date_range(start='2023-01-01', periods=periods, freq='1min')
    price = np.zeros(periods)
    price[0] = 100

    # Baseline random walk
    shocks = rng.normal(0, 0.05, periods)
    price = np.add.accumulate(shocks) + 100

    # Inject "Level Drop" patterns
    for _ in range(5): # Inject 5 patterns
        drop_start = rng.integers(100, periods - 200)

        # High point before the drop
        high_point_val = price[drop_start-50:drop_start].max() + 5
        price[drop_start-10:drop_start] = np.linspace(price[drop_start-10], high_point_val, 10)

        # The Drop
        drop_end = drop_start + 50
        drop_low = high_point_val - rng.uniform(10, 20)
        price[drop_start:drop_end] = np.linspace(high_point_val, drop_low, 50)

        # Retracement (center peak) to ~50% Fib
        retrace_end = drop_end + 30
        retrace_high = drop_low + (high_point_val - drop_low) * rng.uniform(0.48, 0.52)
        price[drop_end:retrace_end] = np.linspace(drop_low, retrace_high, 30)

        # Continuation
        continuation_end = retrace_end + 100
        price[retrace_end:continuation_end] = np.linspace(retrace_high, drop_low - 5, 70)
        price[continuation_end-30:continuation_end] = price[continuation_end-30:continuation_end] + rng.normal(0, 0.1, 30)


    # Create DataFrame
    df = pd.DataFrame(price, index=dt_index, columns=['Close'])
    df['Open'] = df['Close'].shift(1).fillna(df['Close'])
    df['High'] = df[['Open', 'Close']].max(axis=1) + rng.uniform(0, 0.1, periods)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - rng.uniform(0, 0.1, periods)

    return df

def preprocess_data(df_1m, prominence=1.5, ema_period=200):
    """
    Performs 15M analysis and adds other required indicators (EMA, LOD, Sessions)
    before merging results back into the 1M dataframe.
    """
    # --- Add Confluence Indicators ---
    # 1. EMA
    df_1m['ema'] = df_1m['Close'].ewm(span=ema_period, adjust=False).mean()

    # 2. Previous Day Low (LOD)
    daily_low = df_1m['Low'].resample('D').min()
    df_1m['lod'] = daily_low.shift(1).reindex(df_1m.index, method='ffill')

    # 3. Asia Session 50% Range (00:00 - 08:00)
    is_asia = (df_1m.index.hour >= 0) & (df_1m.index.hour < 8)
    asia_data = df_1m[is_asia]
    daily_asia_high = asia_data['High'].resample('D').max()
    daily_asia_low = asia_data['Low'].resample('D').min()
    asia_50_pct = daily_asia_low + (daily_asia_high - daily_asia_low) * 0.5
    df_1m['asia_50_pct'] = asia_50_pct.reindex(df_1m.index, method='ffill')

    # --- 15M Multi-Timeframe Analysis ---
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()

    # Find significant peaks (swing highs) and troughs (swing lows) on 15M
    peaks_idx, _ = find_peaks(df_15m['High'], prominence=prominence, width=3)
    troughs_idx, _ = find_peaks(-df_15m['Low'], prominence=prominence, width=3)

    setups = []

    # Identify Level Drops (peak followed by a trough)
    for p_idx in peaks_idx:
        next_troughs = troughs_idx[troughs_idx > p_idx]
        if len(next_troughs) > 0:
            t_idx = next_troughs[0]
            peak_time, trough_time = df_15m.index[p_idx], df_15m.index[t_idx]
            peak_price, trough_price = df_15m['High'].iloc[p_idx], df_15m['Low'].iloc[t_idx]

            if peak_price > trough_price:
                price_range = peak_price - trough_price
                setups.append({
                    'start_time': peak_time,
                    'end_time': trough_time + pd.Timedelta(minutes=14),
                    'aoi_level': trough_price + price_range * 0.5,
                    'fib_382': trough_price + price_range * 0.382,
                    'fib_618': trough_price + price_range * 0.618,
                    'fib_786': trough_price + price_range * 0.786,
                    'setup_high': peak_price,
                    'setup_low': trough_price
                })

    # Initialize columns for propagation
    setup_cols = ['aoi_level', 'fib_382', 'fib_618', 'fib_786', 'setup_high', 'setup_low']
    for col in setup_cols:
        df_1m[col] = np.nan

    # Propagate the 15M context down to the 1M data
    for setup in setups:
        mask = (df_1m.index > setup['end_time'])
        for col in setup_cols:
            df_1m.loc[mask, col] = setup[col]

    # A setup is valid until a new one begins
    for col in setup_cols:
        df_1m[col] = df_1m[col].ffill()

    df_1m.dropna(inplace=True) # Drop rows before the first setup and where indicators are NaN

    return df_1m

class FibonacciCenterPeakScalpStrategy(Strategy):
    # --- Optimization Parameters ---
    ema_period = 200
    reversal_confirmation_candles = 2

    # --- State Tracking Variables ---
    setup_is_valid = True
    in_aoi = False
    bearish_candle_count = 0
    entry_peak_high = 0

    def init(self):
        # --- Pre-calculated Data ---
        self.aoi_level = self.I(lambda x: x, self.data.df['aoi_level'], name='aoi')
        self.setup_high = self.I(lambda x: x, self.data.df['setup_high'], name='setup_high')
        self.setup_low = self.I(lambda x: x, self.data.df['setup_low'], name='setup_low')
        self.fib_382 = self.I(lambda x: x, self.data.df['fib_382'], name='fib_382')
        self.fib_618 = self.I(lambda x: x, self.data.df['fib_618'], name='fib_618')
        self.fib_786 = self.I(lambda x: x, self.data.df['fib_786'], name='fib_786')

        # --- Confluence Indicators ---
        self.ema = self.I(lambda x: x, self.data.df['ema'], name='ema')
        self.lod = self.I(lambda x: x, self.data.df['lod'], name='lod')
        self.asia_50 = self.I(lambda x: x, self.data.df['asia_50_pct'], name='asia_50')

    def next(self):
        if self.position:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # --- Setup Invalidation Logic ---
        # If a new setup begins (identified by change in setup_high), reset validity
        if self.setup_high[-1] != self.setup_high[-2]:
            self.setup_is_valid = True
            self.in_aoi = False # Reset AOI state on new setup

        if not self.setup_is_valid:
            return

        # Check for invalidation by touching other Fib levels before the AOI
        if not self.in_aoi:
            if (low <= self.fib_382[-1] and high >= self.fib_382[-1]) or \
               (low <= self.fib_618[-1] and high >= self.fib_618[-1]) or \
               (low <= self.fib_786[-1] and high >= self.fib_786[-1]):
                self.setup_is_valid = False
                return

        # --- Entry State Machine ---
        if not self.in_aoi and low <= self.aoi_level[-1] and high >= self.aoi_level[-1]:
            self.in_aoi = True
            self.entry_peak_high = high
            self.bearish_candle_count = 0

        if self.in_aoi:
            self.entry_peak_high = max(self.entry_peak_high, high)

            if high > self.setup_high[-1]:
                self.in_aoi = False
                self.setup_is_valid = False # Invalidate until next setup
                return

            if price < self.data.Open[-1]:
                self.bearish_candle_count += 1
            else:
                self.bearish_candle_count = 0

            if self.bearish_candle_count >= self.reversal_confirmation_candles:

                # --- Confluence Checks ---
                is_below_ema = price < self.ema[-1]
                is_above_lod = price > self.lod[-1]
                # is_above_asia_50 = price > self.asia_50[-1] # Example, can be added

                if not (is_below_ema and is_above_lod):
                    return

                # --- Risk & Entry ---
                stop_loss = self.entry_peak_high * 1.001
                take_profit = self.entry_peak_high - (self.entry_peak_high - self.setup_low[-1]) * 0.5

                risk = stop_loss - price
                reward = price - take_profit
                if reward > 2 * risk and risk > 0:
                    self.sell(sl=stop_loss, tp=take_profit)

                self.in_aoi = False
                self.setup_is_valid = False # Invalidate until next setup

if __name__ == '__main__':
    # --- 1. Data Generation & Preprocessing ---
    raw_data = generate_synthetic_data(periods=10000)

    # The optimization loop for the pre-processing parameter `prominence`
    # cannot be done directly in bt.optimize(). We must loop through it manually.
    best_prominence = None
    best_sharpe = -1
    best_stats = None

    # Determine the range for prominence based on data price range
    price_range = raw_data['High'].max() - raw_data['Low'].min()

    # This strategy has two types of parameters:
    # 1. Pre-processing parameters (prominence, ema_period)
    # 2. Strategy parameters (reversal_confirmation_candles)
    # We need a nested loop to optimize them all.

    best_overall_stats = None
    best_sharpe = -1
    best_preprocessing_params = {}

    price_range = raw_data['High'].max() - raw_data['Low'].min()

    # Outer loop: Optimize pre-processing parameters
    for p_pct in np.arange(0.02, 0.11, 0.04):
        for ema_p in [100, 200]:
            print(f"\n--- Testing Preprocessing Params: Prominence Pct={p_pct}, EMA Period={ema_p} ---")
            prominence_val = price_range * p_pct
            data = preprocess_data(raw_data.copy(), prominence=prominence_val, ema_period=ema_p)

            if data.empty:
                print("Warning: Preprocessing resulted in empty data. Skipping.")
                continue

            bt = Backtest(data, FibonacciCenterPeakScalpStrategy, cash=100_000, commission=.002)

            # Inner loop: Optimize strategy parameters
            stats = bt.optimize(
                reversal_confirmation_candles=range(2, 5),
                maximize='Sharpe Ratio'
            )

            if stats['Sharpe Ratio'] > best_sharpe:
                best_sharpe = stats['Sharpe Ratio']
                best_overall_stats = stats
                best_preprocessing_params = {'prominence_pct': p_pct, 'ema_period': ema_p}

    print("\n--- Best Overall Run Stats ---")
    print(best_overall_stats)
    print(f"Best Pre-processing Params: {best_preprocessing_params}")

    # --- 4. Final Run with Best Parameters ---
    print("\nRunning final backtest with all best parameters...")
    final_prominence_val = price_range * best_preprocessing_params['prominence_pct']
    final_data = preprocess_data(raw_data.copy(),
                                 prominence=final_prominence_val,
                                 ema_period=best_preprocessing_params['ema_period'])

    best_strategy_params = best_overall_stats._strategy

    final_bt = Backtest(final_data, type(best_strategy_params), cash=100_000, commission=.002)

    final_stats = final_bt.run(
        reversal_confirmation_candles=best_strategy_params.reversal_confirmation_candles
    )

    # --- 5. Output Results ---
    import os
    os.makedirs('results', exist_ok=True)

    # Ensure all values are native Python types for JSON serialization
    output_stats = {
        'strategy_name': 'fibonacci_center_peak_scalp',
        'return': float(final_stats.get('Return [%]', 0.0)),
        'sharpe': float(final_stats.get('Sharpe Ratio', 0.0)),
        'max_drawdown': float(final_stats.get('Max. Drawdown [%]', 0.0)),
        'win_rate': float(final_stats.get('Win Rate [%]', 0.0)),
        'total_trades': int(final_stats.get('# Trades', 0))
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(output_stats, f, indent=2)

    print("\nSaved best run stats to results/temp_result.json")

    # --- 6. Generate Plot ---
    plot_filename = 'results/fibonacci_center_peak_scalp.html'
    final_bt.plot(filename=plot_filename)
    print(f"Saved plot to {plot_filename}")
