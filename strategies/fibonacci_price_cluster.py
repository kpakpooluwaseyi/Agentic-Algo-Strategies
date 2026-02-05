
"""
Fibonacci Price Cluster Strategy
================================
This strategy identifies price zones where multiple Fibonacci levels (retracements,
extensions, projections) converge, creating a "cluster." It assumes these
clusters represent strong support or resistance levels and enters trades when
price enters a cluster that aligns with the higher-timeframe trend.

This implementation adheres to the mandatory agent development guidelines:
- ATR-based risk management (2x SL, 3x TP)
- 4H EMA-based multi-timeframe trend filter
- Volume confirmation for entries
"""

from backtesting import Strategy, Backtest
import numpy as np
import pandas as pd
import talib
from scipy.signal import find_peaks

def preprocess_data(df, **params):
    """
    Pre-calculates indicators and features for the strategy.
    - ATR for risk management
    - 4H EMA for trend filter
    - Volume MA for confirmation
    - Swing points for Fibonacci calculations
    """
    # Robustly clean and format column names
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    # ATR
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Higher timeframe trend (4H)
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    df_4h['ema_200'] = talib.EMA(df_4h['Close'], timeperiod=200)
    df_4h['htf_uptrend'] = (df_4h['Close'] > df_4h['ema_200']).astype(int)

    # Map 4H trend back to the original timeframe
    df['htf_uptrend'] = df_4h['htf_uptrend'].reindex(df.index, method='ffill').fillna(0)

    # Volume moving average
    df['volume_ma'] = talib.SMA(df['Volume'], timeperiod=20)

    # Find swing points (peaks and troughs)
    # The 'distance' parameter is crucial for finding meaningful swings
    peak_indices, _ = find_peaks(df['High'], distance=params.get('swing_distance', 10))
    trough_indices, _ = find_peaks(-df['Low'], distance=params.get('swing_distance', 10))
    df['swing_high'] = np.nan
    df['swing_low'] = np.nan
    df.loc[df.index[peak_indices], 'swing_high'] = df['High'].iloc[peak_indices]
    df.loc[df.index[trough_indices], 'swing_low'] = df['Low'].iloc[trough_indices]

    return df


class FibonacciPriceCluster(Strategy):
    """
    Trades on Fibonacci cluster zones aligned with the HTF trend.
    """
    # Optimizable parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    cluster_threshold_pct = 0.1 # Pct of ATR for grouping fib levels
    min_cluster_fibs = 3       # Minimum number of fibs to form a cluster
    swing_distance = 10        # Distance parameter for find_peaks

    def init(self):
        # Wrap pre-calculated data for easy access
        self.atr = self.I(lambda: self.data.atr, name='ATR')
        self.htf_uptrend = self.I(lambda: self.data.htf_uptrend, name='HTF_Uptrend')
        self.volume_ma = self.I(lambda: self.data.volume_ma, name='Volume_MA')
        self.swing_highs = self.I(lambda: self.data.swing_high, name='Swing_Highs')
        self.swing_lows = self.I(lambda: self.data.swing_low, name='Swing_Lows')

        # State variable to efficiently track swings
        self.alternating_swings = []

    def _get_fib_levels(self, recent_swings):
        """
        Calculates Fibonacci retracements, extensions, and projections
        based on a list of recent swing points.
        """
        if len(recent_swings) < 3:
            return []

        levels = []

        # Unpack last 3 swings for clarity (A, B, C)
        c_price, c_idx, c_type = recent_swings[-1]
        b_price, b_idx, b_type = recent_swings[-2]
        a_price, a_idx, a_type = recent_swings[-3]

        # 1. Retracements (from B to C)
        swing_range = abs(c_price - b_price)
        for ratio in [0.236, 0.382, 0.50, 0.618, 0.786]:
            if c_price > b_price: # Uptrend swing
                levels.append(c_price - ratio * swing_range)
            else: # Downtrend swing
                levels.append(c_price + ratio * swing_range)

        # 2. Extensions (from B to C, extended from C)
        for ratio in [1.272, 1.618, 2.618]:
            if c_price > b_price: # Uptrend swing
                levels.append(c_price + ratio * swing_range)
            else: # Downtrend swing
                levels.append(c_price - ratio * swing_range)

        # 3. Projections (from A to B, projected from C)
        projection_range = abs(b_price - a_price)
        for ratio in [1.0, 1.618]:
            if c_price > b_price: # C is a trough after peak B
                 levels.append(c_price + ratio * projection_range)
            else: # C is a peak after trough B
                 levels.append(c_price - ratio * projection_range)

        return [level for level in levels if level > 0]

    def _find_clusters(self, levels, current_atr):
        """
        Groups a list of price levels into clusters based on proximity.
        """
        if not levels:
            return []

        # Sort levels to make clustering easier
        levels.sort()

        clusters = []
        current_cluster = [levels[0]]

        # Define the proximity threshold based on ATR
        threshold = current_atr * self.cluster_threshold_pct

        for i in range(1, len(levels)):
            if levels[i] - current_cluster[-1] <= threshold:
                current_cluster.append(levels[i])
            else:
                # Save the old cluster if it's large enough
                if len(current_cluster) >= self.min_cluster_fibs:
                    clusters.append(current_cluster)
                # Start a new cluster
                current_cluster = [levels[i]]

        # Check the last cluster
        if len(current_cluster) >= self.min_cluster_fibs:
            clusters.append(current_cluster)

        return clusters

    def next(self):
        # --- FILTERS ---
        # Wait for enough data
        if len(self.data) < 200:
            return

        # Check for open position
        if self.position:
            return

        # --- Efficient Swing Point Identification ---
        current_index = len(self.data) - 1

        # Check for a new swing high on the current bar
        if not np.isnan(self.swing_highs[-1]):
            new_swing = (self.swing_highs[-1], current_index, 'high')
            # Add if it's the first swing or if it alternates with the last one
            if not self.alternating_swings or self.alternating_swings[-1][2] != 'high':
                self.alternating_swings.append(new_swing)

        # Check for a new swing low on the current bar
        if not np.isnan(self.swing_lows[-1]):
            new_swing = (self.swing_lows[-1], current_index, 'low')
            # Add if it's the first swing or if it alternates with the last one
            if not self.alternating_swings or self.alternating_swings[-1][2] != 'low':
                self.alternating_swings.append(new_swing)

        # --- Fibonacci Calculation ---
        fib_levels = []
        if len(self.alternating_swings) >= 3:
            # Use the last 3 swings for calculations
            recent_swings = self.alternating_swings[-3:]
            fib_levels = self._get_fib_levels(recent_swings)

        # --- Cluster Detection & Entry Logic ---
        clusters = self._find_clusters(fib_levels, self.atr[-1])

        if not clusters:
            return

        current_price = self.data.Close[-1]

        # Check each cluster for a potential trade
        for cluster in clusters:
            cluster_min = min(cluster)
            cluster_max = max(cluster)

            # --- LONG ENTRY ---
            # Condition 1: Price is within a support cluster
            # Condition 2: HTF trend is up
            # Condition 3: Volume is above average
            if (cluster_min <= current_price <= cluster_max and
                self.htf_uptrend[-1] == 1 and
                self.data.Volume[-1] > self.volume_ma[-1]):

                sl = current_price - (self.atr[-1] * self.atr_sl_multiplier)
                tp = current_price + (self.atr[-1] * self.atr_tp_multiplier)
                self.buy(sl=sl, tp=tp)
                return # Exit after placing a trade

            # --- SHORT ENTRY ---
            # Condition 1: Price is within a resistance cluster
            # Condition 2: HTF trend is down
            # Condition 3: Volume is above average
            elif (cluster_min <= current_price <= cluster_max and
                  self.htf_uptrend[-1] == 0 and
                  self.data.Volume[-1] > self.volume_ma[-1]):

                  sl = current_price + (self.atr[-1] * self.atr_sl_multiplier)
                  tp = current_price - (self.atr[-1] * self.atr_tp_multiplier)
                  self.sell(sl=sl, tp=tp)
                  return # Exit after placing a trade


if __name__ == '__main__':
    # Load data
    try:
        df = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print("Data file not found. Generating synthetic data...")
        dates = pd.date_range('2023-01-01', periods=2000, freq='15min')
        price = 20000 + np.cumsum(np.random.randn(2000) * 15)
        df = pd.DataFrame({
            'Open': price,
            'High': price + np.random.rand(2000) * 10,
            'Low': price - np.random.rand(2000) * 10,
            'Close': price + np.random.randn(2000) * 5,
            'Volume': np.random.rand(2000) * 1000
        }, index=dates)
        df.index.name = 'datetime'

    # Preprocess the data with default params
    data = preprocess_data(df.copy(), swing_distance=10)

    # Run backtest
    bt = Backtest(data, FibonacciPriceCluster, cash=100000, commission=.001)
    stats = bt.run()

    print("\n--- Fibonacci Price Cluster Strategy ---")
    print(stats)

    # Save results and plot
    import json
    import os

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    def sanitize_stats(stats):
        """
        Sanitizes the stats object for JSON serialization by converting non-serializable
        types to serializable formats.
        """
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Series, pd.DataFrame)):
                continue  # Skip dataframes and series
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                sanitized[key] = str(value)
            elif pd.isna(value):
                sanitized[key] = None
            elif isinstance(value, (np.integer, np.floating)):
                sanitized[key] = value.item()
            elif isinstance(value, Strategy):
                sanitized[key] = value.__class__.__name__
            else:
                sanitized[key] = value
        return sanitized

    sanitized_stats = sanitize_stats(stats)

    with open(f'{results_dir}/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    bt.plot(filename=f'{results_dir}/fibonacci_price_cluster.html', open_browser=False)

    print(f"\nResults saved to {results_dir}/temp_result.json")
    print(f"Plot saved to {results_dir}/fibonacci_price_cluster.html")
