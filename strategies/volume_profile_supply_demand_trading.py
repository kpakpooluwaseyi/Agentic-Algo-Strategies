import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def calculate_poc(df, bins=100):
    """
    Calculates the Point of Control (POC) for a given DataFrame by
    distributing each candle's volume across the price bins it spans.
    """
    if df.empty:
        return np.nan

    price_low = df['Low'].min()
    price_high = df['High'].max()
    price_range = price_high - price_low

    if price_range <= 0:
        return df['Close'].mean() if not df.empty else np.nan

    bin_size = price_range / bins
    volume_per_bin = np.zeros(bins)

    # Vectorized calculation of start and end bins for each candle
    start_bins = ((df['Low'] - price_low) / bin_size).astype(int).clip(0, bins - 1)
    end_bins = ((df['High'] - price_low) / bin_size).astype(int).clip(0, bins - 1)

    # Calculate volume per bin for each candle
    bins_spanned = (end_bins - start_bins + 1).replace(0, 1) # Avoid division by zero
    volume_per_bin_candle = df['Volume'] / bins_spanned

    # This part is tricky to vectorize directly in pandas/numpy.
    # We iterate to distribute the volume correctly.
    for i in range(len(df)):
        vol = volume_per_bin_candle.iloc[i]
        start_bin = start_bins.iloc[i]
        end_bin = end_bins.iloc[i]
        # Add volume to all bins the candle touches
        volume_per_bin[start_bin : end_bin + 1] += vol

    if np.sum(volume_per_bin) == 0:
        return np.nan

    # Find the bin with the maximum volume
    poc_bin_index = np.argmax(volume_per_bin)

    # Calculate the price at the center of the POC bin
    poc_price = price_low + (poc_bin_index * bin_size) + (bin_size / 2)
    return poc_price

def is_bullish_engulfing(df, i):
    if i < 1: return False
    return df['Close'][i] > df['Open'][i] and \
           df['Close'][i-1] < df['Open'][i-1] and \
           df['Close'][i] > df['Open'][i-1] and \
           df['Open'][i] < df['Close'][i-1]

def is_bearish_engulfing(df, i):
    if i < 1: return False
    return df['Close'][i] < df['Open'][i] and \
           df['Close'][i-1] > df['Open'][i-1] and \
           df['Open'][i] > df['Close'][i-1] and \
           df['Close'][i] < df['Open'][i-1]

def is_hammer(df, i):
    body = abs(df['Close'][i] - df['Open'][i])
    lower_wick = df['Open'][i] - df['Low'][i] if df['Close'][i] > df['Open'][i] else df['Close'][i] - df['Low'][i]
    upper_wick = df['High'][i] - df['Close'][i] if df['Close'][i] > df['Open'][i] else df['High'][i] - df['Open'][i]
    return lower_wick > 2 * body and upper_wick < body

def is_shooting_star(df, i):
    body = abs(df['Close'][i] - df['Open'][i])
    upper_wick = df['High'][i] - df['Open'][i] if df['Close'][i] < df['Open'][i] else df['High'][i] - df['Close'][i]
    lower_wick = df['Close'][i] - df['Low'][i] if df['Close'][i] < df['Open'][i] else df['Open'][i] - df['Low'][i]
    return upper_wick > 2 * body and lower_wick < body

class VolumeProfileSupplyDemandStrategy(Strategy):
    # --- Optimization Parameters ---
    min_risk_reward_ratio = 1.5
    poc_cluster_lookback = 30  # Days for clustering recent daily POCs
    cluster_pct = 0.02         # Percentage to group POCs into a cluster
    min_pocs_in_cluster = 3    # Minimum POCs to be considered a valid cluster
    sl_buffer_pct = 0.001      # Percentage buffer for stop-loss

    def init(self):
        # The pre-computed zones are passed via the data object
        self.supply_zone_top = self.I(lambda x: x, self.data.df['supply_zone_top'], name="Supply Top")
        self.supply_zone_bottom = self.I(lambda x: x, self.data.df['supply_zone_bottom'], name="Supply Bottom")
        self.demand_zone_top = self.I(lambda x: x, self.data.df['demand_zone_top'], name="Demand Top")
        self.demand_zone_bottom = self.I(lambda x: x, self.data.df['demand_zone_bottom'], name="Demand Bottom")

    def next(self):
        price = self.data.Close[-1]
        i = len(self.data.Close) - 1

        # --- Long Entry ---
        if not self.position and \
           not pd.isna(self.demand_zone_bottom[-1]) and \
           self.demand_zone_bottom[-1] <= price <= self.demand_zone_top[-1]:

            if is_bullish_engulfing(self.data.df, i) or is_hammer(self.data.df, i):
                sl = self.demand_zone_bottom[-1] * (1 - self.sl_buffer_pct)

                # Dynamic TP: Target the next supply zone
                if not pd.isna(self.supply_zone_bottom[-1]):
                    tp = self.supply_zone_bottom[-1]

                    # R:R Check
                    risk = price - sl
                    reward = tp - price
                    if risk > 0 and reward / risk >= self.min_risk_reward_ratio:
                        self.buy(sl=sl, tp=tp)

        # --- Short Entry ---
        elif not self.position and \
             not pd.isna(self.supply_zone_bottom[-1]) and \
             self.supply_zone_bottom[-1] <= price <= self.supply_zone_top[-1]:

            if is_bearish_engulfing(self.data.df, i) or is_shooting_star(self.data.df, i):
                sl = self.supply_zone_top[-1] * (1 + self.sl_buffer_pct)

                # Dynamic TP: Target the next demand zone
                if not pd.isna(self.demand_zone_top[-1]):
                    tp = self.demand_zone_top[-1]

                    # R:R Check
                    risk = sl - price
                    reward = price - tp
                    if risk > 0 and reward / risk >= self.min_risk_reward_ratio:
                        self.sell(sl=sl, tp=tp)

def preprocess_data_causal(df, poc_cluster_lookback, cluster_pct, min_pocs_in_cluster):
    """
    Pre-computes supply/demand zones without lookahead bias.
    For each day, it looks back to identify zones based on past POC clusters.
    This simplified version omits the VRVP alignment to ensure functionality.
    """
    # Calculate all daily POCs once. This is not lookahead bias.
    df['date'] = df.index.date
    daily_pocs = df.groupby('date').apply(calculate_poc).rename('poc').dropna()

    all_supply_tops, all_supply_bottoms = [], []
    all_demand_tops, all_demand_bottoms = [], []

    unique_dates = df.index.normalize().unique()
    valid_zones = []

    for current_date in unique_dates:
        # --- SVP Clustering: Find recent POC clusters using past data ---
        cluster_start_date = (current_date - pd.Timedelta(days=poc_cluster_lookback)).date()
        current_date_only = current_date.date()
        recent_pocs = daily_pocs.loc[cluster_start_date:current_date_only].sort_values()

        if not recent_pocs.empty:
            clusters = []
            current_cluster = [recent_pocs.iloc[0]]
            for poc in recent_pocs.iloc[1:]:
                if poc <= current_cluster[-1] * (1 + cluster_pct):
                    current_cluster.append(poc)
                else:
                    if len(current_cluster) >= min_pocs_in_cluster: clusters.append(current_cluster)
                    current_cluster = [poc]
            if len(current_cluster) >= min_pocs_in_cluster: clusters.append(current_cluster)

            # --- Simplified Zone Validation (No VRVP) ---
            for cluster in clusters:
                cluster_mean = np.mean(cluster)
                zone_std = np.std(cluster)
                new_zone = (cluster_mean - zone_std, cluster_mean + zone_std)
                # Add new zones as they are discovered
                if new_zone not in valid_zones:
                    valid_zones.append(new_zone)

        # --- Assign zones to the bars of the current day ---
        day_bars = df[df.index.normalize() == current_date]
        for _, row in day_bars.iterrows():
            price = row['Close']
            zones_above = [z for z in valid_zones if z[0] > price]
            zones_below = [z for z in valid_zones if z[1] < price]

            if zones_above:
                closest_supply = min(zones_above, key=lambda z: z[0] - price)
                all_supply_bottoms.append(closest_supply[0])
                all_supply_tops.append(closest_supply[1])
            else:
                all_supply_bottoms.append(np.nan)
                all_supply_tops.append(np.nan)

            if zones_below:
                closest_demand = max(zones_below, key=lambda z: z[1] - price)
                all_demand_bottoms.append(closest_demand[0])
                all_demand_tops.append(closest_demand[1])
            else:
                all_demand_bottoms.append(np.nan)
                all_demand_tops.append(np.nan)

    df['supply_zone_top'] = all_supply_tops
    df['supply_zone_bottom'] = all_supply_bottoms
    df['demand_zone_top'] = all_demand_tops
    df['demand_zone_bottom'] = all_demand_bottoms

    return df.drop(columns=['date'])

import json

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to make it JSON serializable.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = float(value)
        elif isinstance(value, (pd.Series, pd.DataFrame)):
            continue # Skip dataframes and series
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    # Special handling for _strategy as it's an object
    if '_strategy' in stats:
        sanitized['_strategy'] = str(stats['_strategy'])
    return sanitized

if __name__ == '__main__':
    # NOTE: You can check "data/BTC-USD-15m.csv" for the data format
    data = pd.read_csv(
        "data/BTC-USD-15m.csv",
        header=0,
        names=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
        usecols=['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    data = data.set_index('Time')
    data.index = pd.to_datetime(data.index)

    # Remove all rows with NaN values
    data = data.dropna()

    # Preprocess data to find zones causally
    data = preprocess_data_causal(
        data,
        VolumeProfileSupplyDemandStrategy.poc_cluster_lookback,
        VolumeProfileSupplyDemandStrategy.cluster_pct,
        VolumeProfileSupplyDemandStrategy.min_pocs_in_cluster
    )

    bt = Backtest(data, VolumeProfileSupplyDemandStrategy, cash=100000)
    stats = bt.run()
    print(stats)

    # Sanitize and save stats to JSON
    sanitized_stats = sanitize_stats(stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    # Generate plot
    bt.plot(filename='results/volume_profile_supply_demand_trading.html')
