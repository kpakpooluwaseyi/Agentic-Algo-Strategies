import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import json
from scipy.signal import find_peaks
import os


# Helper function to calculate Volume Profile
def calculate_volume_profile(prices, volumes, bins):
    """Calculates a simple volume profile based on closing prices."""
    if len(prices) == 0:
        return np.array([]), np.array([])
    price_min, price_max = prices.min(), prices.max()
    if price_min == price_max:
        return np.array([]), np.array([])

    price_range = np.linspace(price_min, price_max, bins + 1)
    volume_by_price = np.zeros(bins)

    # Vectorized approach to binning prices
    price_indices = np.searchsorted(price_range, prices, side='right') - 1
    # Clip indices to be within the bounds of volume_by_price array
    price_indices = np.clip(price_indices, 0, bins - 1)
    np.add.at(volume_by_price, price_indices, volumes)

    return volume_by_price, price_range


class VolumeProfileHvnLvnLiquidityFlow(Strategy):
    # Strategy parameters
    ema_period_fast = 50
    ema_period_slow = 200
    volume_profile_lookback = 1000
    volume_profile_bins = 200
    peak_prominence_multiplier = 1.5  # Multiplier for std dev to determine prominence
    swing_lookback = 50  # Lookback period for market structure

    def init(self):
        # Higher Timeframe Trend
        # Resample to 1-hour and calculate EMA using pandas
        close_1h = self.data.df['Close'].resample('1h').last()
        ema_slow_1h = close_1h.ewm(span=self.ema_period_slow, adjust=False).mean()

        # Map 1H trend back to 15M timeframe
        df = pd.DataFrame({'ema_slow_1h': ema_slow_1h}, index=close_1h.index)
        if self.data.index.tz:
             df.index = df.index.tz_localize(self.data.index.tz)

        # Use merge_asof to align the 1H data with the 15M index
        merged_df = pd.merge_asof(
            self.data.df,
            df,
            left_index=True,
            right_index=True,
            direction='backward'
        )

        # Create the final, correctly-sized indicator
        self.htf_bullish = self.I(lambda: merged_df['ema_slow_1h'].values < self.data.Close, name="htf_bullish")


        # Local Trend
        self.ema_fast = self.I(lambda x: pd.Series(x).ewm(span=self.ema_period_fast).mean(), self.data.Close, name="ema_fast")

        # Initialize state variables
        self.hvns = []
        self.lvns = []
        self.poc = None
        self.is_bullish_structure = False
        self.is_bearish_structure = False

    def next(self):
        # Wait for enough data
        if len(self.data.Close) < max(self.volume_profile_lookback, self.swing_lookback):
            return

        # --- Market Structure Analysis ---
        highs = self.data.High[-self.swing_lookback:]
        lows = self.data.Low[-self.swing_lookback:]

        swing_high_indices, _ = find_peaks(highs, distance=5)
        swing_low_indices, _ = find_peaks(-lows, distance=5)

        if len(swing_high_indices) >= 2 and len(swing_low_indices) >= 2:
            last_high = highs[swing_high_indices[-1]]
            prev_high = highs[swing_high_indices[-2]]
            last_low = lows[swing_low_indices[-1]]
            prev_low = lows[swing_low_indices[-2]]

            # Bullish structure: Higher Highs and Higher Lows
            if last_high > prev_high and last_low > prev_low:
                self.is_bullish_structure = True
                self.is_bearish_structure = False
            # Bearish structure: Lower Highs and Lower Lows
            elif last_high < prev_high and last_low < prev_low:
                self.is_bearish_structure = True
                self.is_bullish_structure = False
            else:
                # Reset if structure is unclear
                self.is_bullish_structure = False
                self.is_bearish_structure = False

        # --- Volume Profile Analysis (from previous step) ---
        prices = self.data.Close[-self.volume_profile_lookback:]
        volumes = self.data.Volume[-self.volume_profile_lookback:]
        volume_profile, price_bins = calculate_volume_profile(prices, volumes, self.volume_profile_bins)

        if len(volume_profile) == 0:
            return

        prominence = np.std(volume_profile) * self.peak_prominence_multiplier
        hvn_indices, _ = find_peaks(volume_profile, prominence=prominence)
        lvn_indices, _ = find_peaks(-volume_profile, prominence=prominence)

        self.hvns = price_bins[hvn_indices]
        self.lvns = price_bins[lvn_indices]
        poc_index = np.argmax(volume_profile)
        self.poc = price_bins[poc_index]

        # --- Core Trading Logic ---
        if self.position:
            return # Don't enter new trades if already in a position

        current_price = self.data.Close[-1]
        is_htf_bullish = self.htf_bullish[-1] > 0
        is_local_bullish = current_price > self.ema_fast[-1]

        # --- Long Entry Logic ---
        if self.is_bullish_structure and is_htf_bullish and is_local_bullish:
            # Find the HVN that is acting as support
            support_hvns = self.hvns[self.hvns < current_price]
            if len(support_hvns) == 0:
                return

            current_hvn_support = support_hvns[-1]

            # Price must be close to the support HVN (the actual support, not the overall POC)
            if not (abs(current_price - current_hvn_support) / current_price < 0.005): # within 0.5%
                 return

            # Find the next HVN to target
            target_hvns = self.hvns[self.hvns > current_price]
            if len(target_hvns) == 0:
                return

            next_hvn_target = target_hvns[0]

            # Check for a clear path (LVN) between current price and target
            path_lvns = self.lvns[(self.lvns > current_hvn_support) & (self.lvns < next_hvn_target)]
            if len(path_lvns) > 0:
                stop_loss = current_hvn_support * 0.99
                take_profit = next_hvn_target

                # Ensure SL and TP are valid
                if take_profit > current_price and stop_loss < current_price:
                    self.buy(sl=stop_loss, tp=take_profit)

        # --- Short Entry Logic ---
        elif self.is_bearish_structure and not is_htf_bullish and not is_local_bullish:
            # Find the HVN that is acting as resistance
            resistance_hvns = self.hvns[self.hvns > current_price]
            if len(resistance_hvns) == 0:
                return

            current_hvn_resistance = resistance_hvns[0]

            # Price must be close to the resistance HVN (the actual resistance, not the overall POC)
            if not (abs(current_price - current_hvn_resistance) / current_price < 0.005): # within 0.5%
                return

            # Find the next HVN to target
            target_hvns = self.hvns[self.hvns < current_price]
            if len(target_hvns) == 0:
                return

            next_hvn_target = target_hvns[-1]

            # Check for a clear path (LVN) between current price and target
            path_lvns = self.lvns[(self.lvns < current_hvn_resistance) & (self.lvns > next_hvn_target)]
            if len(path_lvns) > 0:
                stop_loss = current_hvn_resistance * 1.01
                take_profit = next_hvn_target

                # Ensure SL and TP are valid
                if take_profit < current_price and stop_loss > current_price:
                    self.sell(sl=stop_loss, tp=take_profit)


if __name__ == '__main__':
    # Ensure results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
        # Clean and sanitize column names
        data.columns = [col.strip() for col in data.columns]
        if 'Unnamed: 6' in data.columns:
            data = data.drop(columns=['Unnamed: 6'])
        data.columns = [col.capitalize() for col in data.columns]
    except FileNotFoundError:
        print("Data file not found. Please ensure 'data/BTC-USD-15m.csv' exists.")
        # As a fallback, create some synthetic data to allow the script to run
        from backtesting.test import GOOG
        data = GOOG.copy()
        data = data.resample('15min').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

    bt = Backtest(data, VolumeProfileHvnLvnLiquidityFlow, cash=100000, commission=.002)

    stats = bt.run()

    print(stats)

    # Sanitize stats for JSON serialization
    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                sanitized[key] = str(value)
            elif isinstance(value, (np.integer, np.floating)):
                sanitized[key] = float(value)
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                continue # Skip pandas objects
            elif pd.isna(value):
                sanitized[key] = None
            else:
                sanitized[key] = value
        return sanitized

    sanitized_results = sanitize_stats(stats)
    sanitized_results['_strategy'] = str(sanitized_results.get('_strategy')) # Convert strategy object to string

    # Save results to JSON
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_results, f, indent=4)

    # Generate plot
    try:
        bt.plot(filename='results/_volume_profile_hvn_lvn_liquidity_flow_.html', open_browser=False)
    except Exception as e:
        print(f"Error generating plot: {e}")
