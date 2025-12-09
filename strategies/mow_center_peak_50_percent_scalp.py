import json
import os
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

def find_swing_highs(data: pd.Series, n: int) -> pd.Series:
    """Finds swing highs. A swing high is a peak higher than the n periods before and after it."""
    series = data.rolling(2 * n + 1, center=True, min_periods=1).max()
    return (data == series) & data.notna()

def find_swing_lows(data: pd.Series, n: int) -> pd.Series:
    """Finds swing lows. A swing low is a trough lower than the n periods before and after it."""
    series = data.rolling(2 * n + 1, center=True, min_periods=1).min()
    return (data == series) & data.notna()


def preprocess_data(df: pd.DataFrame, params: dict):
    """
    Preprocesses the 1M data to identify 15M setup zones.
    """
    # Resample to 15M. Use .asfreq() to ensure the index is preserved for merging
    df_15m = df.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()

    # --- 15M Swing Points ---
    swing_window = params.get('swing_window', 5) # Use a smaller, more sensitive default
    is_swing_high = find_swing_highs(df_15m['High'], swing_window)
    is_swing_low = find_swing_lows(df_15m['Low'], swing_window)

    df_15m['swing_high_price'] = np.nan
    df_15m.loc[is_swing_high, 'swing_high_price'] = df_15m['High'][is_swing_high]
    df_15m['swing_high_price'] = df_15m['swing_high_price'].ffill()

    df_15m['swing_low_price'] = np.nan
    df_15m.loc[is_swing_low, 'swing_low_price'] = df_15m['Low'][is_swing_low]
    df_15m['swing_low_price'] = df_15m['swing_low_price'].ffill()

    # --- Identify Leg 1 and AOI ---
    df_15m['leg1_high'] = np.where(
        (df_15m['swing_low_price'].notna()) & (df_15m['swing_low_price'] != df_15m['swing_low_price'].shift(1)),
        df_15m['swing_high_price'].shift(1),
        np.nan
    )
    df_15m['leg1_high'] = df_15m['leg1_high'].ffill()
    df_15m['leg1_low'] = df_15m['swing_low_price']

    # Calculate 50% retracement AOI
    df_15m['aoi_50_level'] = df_15m['leg1_low'] + (df_15m['leg1_high'] - df_15m['leg1_low']) * 0.5

    # --- Merge back to 1M data ---
    # Use merge_asof for clean merging of time-series data
    df_with_signals = pd.merge_asof(
        df,
        df_15m[['leg1_high', 'leg1_low', 'aoi_50_level']],
        left_index=True,
        right_index=True,
        direction='backward'
    )

    # Drop rows at the beginning where we don't have HTF context yet
    df_with_signals.dropna(inplace=True)

    return df_with_signals


class MowCenterPeak50PercentScalpStrategy(Strategy):
    swing_window = 10 # Default value, will be optimized

    def init(self):
        # Make preprocessed data available as indicators
        self.leg1_high = self.I(lambda x: x, self.data.df['leg1_high'], name="Leg1_High")
        self.leg1_low = self.I(lambda x: x, self.data.df['leg1_low'], name="Leg1_Low")
        self.aoi_50_level = self.I(lambda x: x, self.data.df['aoi_50_level'], name="AOI_50")

        self.in_aoi = False
        self.retracement_peak = None


    def next(self):
        # --- State Management & Setup Detection ---

        # Check if we are in the Area of Interest
        is_in_aoi = self.data.High[-1] > self.aoi_50_level[-1] and self.data.Low[-1] < self.aoi_50_level[-1]

        if not self.in_aoi and is_in_aoi:
            self.in_aoi = True
            self.retracement_peak = self.data.High[-1]

        # Update retracement peak if we are still in the AOI
        if self.in_aoi:
            self.retracement_peak = max(self.retracement_peak, self.data.High[-1])

        # Invalidation: If price moves far away, reset
        if self.in_aoi and self.data.Close[-1] < self.leg1_low[-1]:
             self.in_aoi = False
             self.retracement_peak = None

        # --- Entry Logic ---
        if self.in_aoi and not self.position:
            # Bearish Engulfing Candle check
            is_bearish_engulfing = (
                self.data.Close[-1] < self.data.Open[-1] and # Current is bearish
                self.data.Close[-2] > self.data.Open[-2] and # Previous was bullish
                self.data.Open[-1] >= self.data.Close[-2] and
                self.data.Close[-1] <= self.data.Open[-2]
            )

            if is_bearish_engulfing:
                # --- Risk Management & Target Calculation ---
                stop_loss = self.data.High[-1] * 1.001 # SL above the reversal candle high

                # TP is 50% of the move from leg1_low to the retracement peak
                take_profit = self.retracement_peak - (self.retracement_peak - self.leg1_low[-1]) * 0.5

                # Ensure TP is valid
                if take_profit >= self.data.Close[-1]:
                    return # Invalid TP, skip trade

                self.sell(sl=stop_loss, tp=take_profit, size=0.99)
                self.in_aoi = False # Reset state after entry
                self.retracement_peak = None

        # Exit logic is handled by SL/TP


def generate_synthetic_data():
    """
    Generates synthetic data with a very clear 'W' pattern for testing.
    This function creates a textbook pattern to ensure the strategy logic can be validated.
    """
    n_total = 800
    time = pd.date_range('2023-01-01', periods=n_total, freq='min')
    price = np.ones(n_total) * 100

    # --- Start of Pattern ---
    start_idx = 100

    # 1. Pre-trend (stable price)
    # (price is already 100)

    # 2. Leg 1: Sharp Drop (this is our major swing high to swing low)
    leg1_start_idx = start_idx
    leg1_end_idx = leg1_start_idx + 60
    leg1_high = 110
    leg1_low = 90
    price[leg1_start_idx] = leg1_high
    price[leg1_start_idx+1:leg1_end_idx] = np.linspace(leg1_high, leg1_low, leg1_end_idx - (leg1_start_idx+1))

    # 3. Retracement: Move up to the 50% level
    retracement_start_idx = leg1_end_idx
    retracement_end_idx = retracement_start_idx + 60
    aoi_50_level = leg1_low + (leg1_high - leg1_low) * 0.5 # This is 100
    price[retracement_start_idx:retracement_end_idx] = np.linspace(leg1_low, aoi_50_level, retracement_end_idx - retracement_start_idx)

    # 4. Reversal Signal at AOI: Create a clear bearish engulfing candle
    reversal_idx = retracement_end_idx
    price[reversal_idx-1] = aoi_50_level - 0.5 # Previous candle bullish

    # Engulfing candle
    open_price = aoi_50_level + 0.6
    close_price = aoi_50_level - 1.0
    high_price = open_price + 0.2
    low_price = close_price - 0.2

    # 5. Post-reversal: Price drops towards the target
    target_start_idx = reversal_idx + 1
    target_end_idx = target_start_idx + 60
    retracement_peak = high_price
    target_price = retracement_peak - (retracement_peak - leg1_low) * 0.5
    price[target_start_idx:target_end_idx] = np.linspace(close_price, target_price, target_end_idx - target_start_idx)

    # 6. Fill rest of data with some noise
    price[target_end_idx:] += np.random.randn(n_total - target_end_idx).cumsum() * 0.05

    # --- Construct DataFrame ---
    df = pd.DataFrame(index=time)
    df['Open'] = price
    # Add minor noise and create OHLC
    df['High'] = df['Open'] + np.random.uniform(0, 0.1, n_total)
    df['Low'] = df['Open'] - np.random.uniform(0, 0.1, n_total)
    df['Close'] = df['Open'] + np.random.uniform(-0.05, 0.05, n_total)

    # --- Plant the specific candles ---
    # Bullish candle before reversal
    df.loc[df.index[reversal_idx-2], 'Open'] = aoi_50_level - 0.5
    df.loc[df.index[reversal_idx-2], 'Close'] = aoi_50_level

    # Bearish engulfing reversal candle
    df.loc[df.index[reversal_idx-1], 'Open'] = open_price
    df.loc[df.index[reversal_idx-1], 'Close'] = close_price
    df.loc[df.index[reversal_idx-1], 'High'] = high_price
    df.loc[df.index[reversal_idx-1], 'Low'] = low_price

    df['Volume'] = np.random.randint(100, 1000, n_total)
    return df


if __name__ == '__main__':
    data = generate_synthetic_data()

    # The optimization function needs a way to pass params to preprocessing
    # We will create a custom objective function to handle this.
    def run_backtest(swing_window):
        params = {'swing_window': int(swing_window)}
        processed_data = preprocess_data(data.copy(), params)
        if processed_data.empty:
            return -1 # Return a poor value if data is empty

        bt = Backtest(processed_data, MowCenterPeak50PercentScalpStrategy, cash=100000, commission=.002)
        stats = bt.run(swing_window=int(swing_window))
        return stats['Sharpe Ratio']

    # Since bt.optimize doesn't directly support preprocessing, we have to do it manually.
    # We will find the best swing_window first.
    swing_windows = range(5, 20, 2)
    best_sharpe = -1
    best_swing_window = 10

    for sw in swing_windows:
        sharpe = run_backtest(sw)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_swing_window = sw

    print(f"Best Swing Window: {best_swing_window} with Sharpe Ratio: {best_sharpe}")

    # Now run the final backtest with the best parameter to get full stats and plot
    final_params = {'swing_window': best_swing_window}
    final_data = preprocess_data(data.copy(), final_params)

    bt = Backtest(final_data, MowCenterPeak50PercentScalpStrategy, cash=100000, commission=.002)
    stats = bt.run(swing_window=best_swing_window)

    print(stats)

    # --- Save results ---
    os.makedirs('results', exist_ok=True)

    # Sanitize stats for JSON output
    win_rate = stats.get('Win Rate [%]', 0.0)

    # Handle potential NaN values from backtesting
    result_dict = {
        'strategy_name': 'mow_center_peak_50_percent_scalp',
        'return': float(stats.get('Return [%]', 0.0)),
        'sharpe': float(stats.get('Sharpe Ratio', 0.0)),
        'max_drawdown': float(stats.get('Max. Drawdown [%]', 0.0)),
        'win_rate': float(win_rate if np.isfinite(win_rate) else 0.0),
        'total_trades': int(stats.get('# Trades', 0))
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(result_dict, f, indent=2)

    print("Results saved to results/temp_result.json")

    # --- Generate plot ---
    try:
        bt.plot(filename='results/mow_center_peak_50_percent_scalp_plot.html', open_browser=False)
        print("Plot saved to results/mow_center_peak_50_percent_scalp_plot.html")
    except Exception as e:
        print(f"Could not generate plot: {e}")
