from backtesting import Strategy
from backtesting.lib import resample_apply
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import json
import os

def ema(arr: np.ndarray, n: int) -> np.ndarray:
    """Computes the Exponential Moving Average (EMA)."""
    return pd.Series(arr).ewm(span=n, adjust=False).mean().values

def find_swing_points(data: pd.DataFrame, distance: int):
    """
    Identifies swing highs and lows in the price data using scipy.signal.find_peaks.

    Args:
        data (pd.DataFrame): The OHLC data.
        distance (int): The minimum horizontal distance between peaks.

    Returns:
        Tuple[pd.Series, pd.Series]: Two series containing the prices of swing highs and lows at their respective indices, otherwise NaN.
    """
    # Find swing highs (resistance)
    high_peaks_indices, _ = find_peaks(data['High'], distance=distance)
    swing_highs = pd.Series(np.nan, index=data.index)
    swing_highs.iloc[high_peaks_indices] = data['High'].iloc[high_peaks_indices]

    # Find swing lows (support)
    low_peaks_indices, _ = find_peaks(-data['Low'], distance=distance)
    swing_lows = pd.Series(np.nan, index=data.index)
    swing_lows.iloc[low_peaks_indices] = data['Low'].iloc[low_peaks_indices]

    return swing_highs, swing_lows

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to ensure it's JSON-serializable.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            sanitized[key] = None  # Or a more suitable representation
        elif isinstance(value, (np.integer, np.int64)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            sanitized[key] = float(value) if not np.isnan(value) else None
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif pd.isna(value) or value is pd.NA:
            sanitized[key] = None
        else:
            sanitized[key] = value
    # Remove non-serializable objects if they haven't been caught
    if '_strategy' in sanitized:
        del sanitized['_strategy']
    if '_equity_curve' in sanitized:
        del sanitized['_equity_curve']
    if '_trades' in sanitized:
        del sanitized['_trades']
    return sanitized

class TrendlineSRBreakoutTrading(Strategy):
    """
    Trades the breakout of a horizontal support or resistance level that forms
    near a larger timeframe trendline.
    """
    # --- Optimizable Parameters ---
    trend_ma_period = 200     # Period for the EMA to determine the main trend
    peak_distance = 30        # How far apart swing points must be to be considered distinct
    proximity_pct = 0.02      # How close the S/R level must be to the trendline (e.g., 0.02 = 2%)
    sl_buffer_pct = 0.01      # Buffer for stop-loss placement (e.g., 0.01 = 1%)
    rr_ratio = 2.0            # Risk:Reward ratio for take-profit

    def init(self):
        """
        Initialize indicators and strategy variables.
        """
        # --- Indicators ---
        self.trend_ma = self.I(ema, self.data.Close, self.trend_ma_period)
        # Pass pre-calculated swing points as indicators
        self.swing_highs = self.I(lambda x: x, self.data.swing_highs, plot=False, name="Swing Highs")
        self.swing_lows = self.I(lambda x: x, self.data.swing_lows, plot=False, name="Swing Lows")

        # --- State Variables ---
        self.last_valid_resistance = None
        self.last_valid_support = None

    def next(self):
        """
        Defines the trading logic for each bar.
        """
        current_price = self.data.Close[-1]

        # --- S/R Level Detection & Validation ---
        # Use [-2] to check the completed candle for a confirmed swing point
        if not pd.isna(self.swing_highs[-2]):
            resistance_level = self.swing_highs[-2]
            ma_at_resistance = self.trend_ma[-2]
            # Check if the swing high was near the trend MA
            if abs(resistance_level - ma_at_resistance) / ma_at_resistance <= self.proximity_pct:
                self.last_valid_resistance = resistance_level

        if not pd.isna(self.swing_lows[-2]):
            support_level = self.swing_lows[-2]
            ma_at_support = self.trend_ma[-2]
            # Check if the swing low was near the trend MA
            if abs(support_level - ma_at_support) / ma_at_support <= self.proximity_pct:
                self.last_valid_support = support_level

        # --- Entry Logic ---
        if self.position:
            return

        # --- LONG ENTRY ---
        is_uptrend = current_price > self.trend_ma[-1]
        if is_uptrend and self.last_valid_resistance is not None:
            # Check for breakout above the validated resistance level
            if current_price > self.last_valid_resistance:
                entry_price = current_price
                stop_loss = self.last_valid_resistance * (1 - self.sl_buffer_pct)
                take_profit = entry_price + (entry_price - stop_loss) * self.rr_ratio

                if entry_price > stop_loss: # Final validation
                    self.buy(sl=stop_loss, tp=take_profit)
                    self.last_valid_resistance = None # Reset after use

        # --- SHORT ENTRY ---
        is_downtrend = current_price < self.trend_ma[-1]
        if is_downtrend and self.last_valid_support is not None:
            # Check for breakout below the validated support level
            if current_price < self.last_valid_support:
                entry_price = current_price
                stop_loss = self.last_valid_support * (1 + self.sl_buffer_pct)
                take_profit = entry_price - (stop_loss - entry_price) * self.rr_ratio

                if entry_price < stop_loss: # Final validation
                    self.sell(sl=stop_loss, tp=take_profit)
                    self.last_valid_support = None # Reset after use

if __name__ == '__main__':
    from backtesting import Backtest
    from backtesting.lib import FractionalBacktest

    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        print(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Ensure correct column names
        data.columns = [c.strip().title() for c in data.columns]

        # --- Pre-computation Step ---
        # Find swing points and add them to the DataFrame
        swing_highs, swing_lows = find_swing_points(data, distance=TrendlineSRBreakoutTrading.peak_distance)
        data['swing_highs'] = swing_highs
        data['swing_lows'] = swing_lows

        # Use a slice of data for faster testing/optimization if needed
        # data = data.iloc[-5000:]

        # --- Backtest Execution ---
        bt = Backtest(data, TrendlineSRBreakoutTrading, cash=100_000, commission=.002, finalize_trades=True)

        print("Running single backtest with default parameters...")
        stats = bt.run()
        print(stats)

        # --- Save Results ---
        os.makedirs('results', exist_ok=True)

        # Sanitize and save stats to JSON
        sanitized_stats = sanitize_stats(stats)
        with open('results/temp_result.json', 'w') as f:
            json.dump(sanitized_stats, f, indent=4)
        print("\nBacktest stats saved to results/temp_result.json")

        # Generate and save the plot
        plot_filename = 'results/trendline_sr_breakout_trading.html'
        try:
            bt.plot(filename=plot_filename, open_browser=False)
            print(f"Backtest plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot: {e}")
