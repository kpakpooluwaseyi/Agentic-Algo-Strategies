
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import json
import os

def calculate_volume_profile(df, period='D'):
    """
    Calculates a simplified Volume Profile for a given period.
    Returns a DataFrame with POC, VAH, and VAL for each period.
    This is a simplified implementation for backtesting purposes.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    df_copy = df.copy()
    df_copy['price_volume'] = df_copy['Close'] * df_copy['Volume']

    # Define aggregation rules
    agg_rules = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'price_volume': 'sum'
    }

    # Resample to the specified period
    resampled_df = df_copy.resample(period).agg(agg_rules)
    resampled_df.dropna(inplace=True)

    if resampled_df.empty:
        return pd.DataFrame()

    # Calculate Volume Weighted Average Price (VWAP) as POC
    resampled_df[f'{period}_POC'] = resampled_df['price_volume'] / resampled_df['Volume']

    # For VAH/VAL, we'll use a proxy: the high/low of the period.
    # A more complex calculation is too slow for this environment.
    resampled_df[f'{period}_VAH'] = resampled_df['High']
    resampled_df[f'{period}_VAL'] = resampled_df['Low']

    return resampled_df[[f'{period}_POC', f'{period}_VAH', f'{period}_VAL']]


def add_volume_profile_levels(df, overlap_pct=0.01):
    """
    Pre-processes the data to add daily and weekly volume profile levels
    and identify overlap zones.
    """
    # Calculate Daily and Weekly profiles
    daily_profiles = calculate_volume_profile(df, 'D')
    weekly_profiles = calculate_volume_profile(df, 'W')

    # Map daily/weekly levels to the 15m dataframe
    df['date'] = df.index.date
    df['week_start'] = df.index.to_period('W').start_time.date

    # Ensure map keys match the lookup keys (datetime.date objects)
    daily_map = {ts.date(): vals for ts, vals in daily_profiles.to_dict('index').items()}
    weekly_map = {idx.to_period('W').start_time.date: vals for idx, vals in weekly_profiles.to_dict('index').items()}

    # Create profile dataframes separately and then join, providing explicit columns to prevent KeyErrors
    daily_cols = ['D_POC', 'D_VAH', 'D_VAL']
    weekly_cols = ['W_POC', 'W_VAH', 'W_VAL']
    daily_data = pd.DataFrame.from_records([daily_map.get(d, {}) for d in df['date']], index=df.index, columns=daily_cols)
    weekly_data = pd.DataFrame.from_records([weekly_map.get(w, {}) for w in df['week_start']], index=df.index, columns=weekly_cols)

    df = df.join(daily_data).join(weekly_data)

    # Shift the data to use the *previous* period's levels for signals
    df['D_POC'] = df.groupby('date')['D_POC'].transform(lambda x: x.shift(1))
    df['D_VAH'] = df.groupby('date')['D_VAH'].transform(lambda x: x.shift(1))
    df['D_VAL'] = df.groupby('date')['D_VAL'].transform(lambda x: x.shift(1))
    df['W_POC'] = df.groupby('week_start')['W_POC'].transform(lambda x: x.shift(1))
    df['W_VAH'] = df.groupby('week_start')['W_VAH'].transform(lambda x: x.shift(1))
    df['W_VAL'] = df.groupby('week_start')['W_VAL'].transform(lambda x: x.shift(1))

    # Identify Overlap Zones
    df['overlap_resistance'] = np.nan
    df['overlap_support'] = np.nan

    # Resistance: area where previous D_VAH and W_VAH are close
    res_cond = abs(df['D_VAH'] - df['W_VAH']) / df['W_VAH'] < overlap_pct
    df.loc[res_cond, 'overlap_resistance'] = df[['D_VAH', 'W_VAH']].mean(axis=1)

    # Support: area where previous D_VAL and W_VAL are close
    sup_cond = abs(df['D_VAL'] - df['W_VAL']) / df['W_VAL'] < overlap_pct
    df.loc[sup_cond, 'overlap_support'] = df[['D_VAL', 'W_VAL']].mean(axis=1)

    df.bfill(inplace=True)
    df.ffill(inplace=True)
    df.drop(columns=['date', 'week_start', 'price_volume'], inplace=True, errors='ignore')

    return df


class VolumeProfileOverlapStrategy(Strategy):

    overlap_proximity_pct = 0.005  # How close price needs to be to a level
    risk_reward_ratio = 2.0
    sl_buffer_pct = 0.002

    def init(self):
        # Make overlap levels accessible in `next`
        self.overlap_resistance = self.I(lambda x: x, self.data.overlap_resistance)
        self.overlap_support = self.I(lambda x: x, self.data.overlap_support)

    def next(self):
        if self.position:
            return

        price = self.data.Close[-1]

        # --- Check for Short Entry ---
        resistance_level = self.overlap_resistance[-1]
        if resistance_level and abs(price - resistance_level) / resistance_level < self.overlap_proximity_pct:
            # Check for bearish engulfing pattern as rejection signal
            is_bearish_engulfing = (self.data.Close[-1] < self.data.Open[-1] and
                                    self.data.Open[-1] >= self.data.Close[-2] and
                                    self.data.Close[-1] < self.data.Open[-2] and
                                    self.data.Close[-2] > self.data.Open[-2])

            if is_bearish_engulfing:
                sl = self.data.High[-1] * (1 + self.sl_buffer_pct)
                tp = price - (sl - price) * self.risk_reward_ratio
                if tp > 0:
                    self.sell(sl=sl, tp=tp)

        # --- Check for Long Entry ---
        support_level = self.overlap_support[-1]
        if support_level and abs(price - support_level) / support_level < self.overlap_proximity_pct:
            # Check for bullish engulfing pattern as rejection signal
            is_bullish_engulfing = (self.data.Close[-1] > self.data.Open[-1] and
                                    self.data.Open[-1] <= self.data.Close[-2] and
                                    self.data.Close[-1] > self.data.Open[-2] and
                                    self.data.Close[-2] < self.data.Open[-2])

            if is_bullish_engulfing:
                sl = self.data.Low[-1] * (1 - self.sl_buffer_pct)
                tp = price + (price - sl) * self.risk_reward_ratio
                if sl > 0:
                     self.buy(sl=sl, tp=tp)

def sanitize_stats(stats):
    """Sanitizes the backtest stats object to be JSON serializable."""
    if stats is None:
        return {}

    # Convert pandas Series to dict
    sanitized = stats.to_dict()

    # Remove non-serializable objects first to prevent errors in the loop
    sanitized.pop('_strategy', None)
    sanitized.pop('_equity_curve', None)
    sanitized.pop('_trades', None)

    # Handle non-serializable types in the remaining values
    for key, value in sanitized.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = value.item()
        elif pd.isna(value):
            sanitized[key] = None

    return sanitized

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(
        data_path,
        parse_dates=['datetime'],
        index_col='datetime'
    )

    print("Original data shape:", data.shape)

    # Sanitize column names (e.g., " open" -> "Open")
    data.columns = [c.strip().title() for c in data.columns]

    # Pre-process data to add volume profile levels
    data_processed = add_volume_profile_levels(data.copy())

    print("Processed data shape:", data_processed.shape)

    if data_processed.empty:
        print("Processed data is empty. Exiting.")
    else:
        bt = Backtest(data_processed, VolumeProfileOverlapStrategy, cash=100000, commission=.002)

        stats = bt.run()
        print(stats)

        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)

        # Save stats to JSON
        sanitized_stats = sanitize_stats(stats)
        with open('results/temp_result.json', 'w') as f:
            json.dump(sanitized_stats, f, indent=4)

        # Save plot
        plot_filename = 'results/volume_profile_overlap_level_rejection.html'
        try:
            bt.plot(filename=plot_filename, open_browser=False)
            print(f"Plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not save plot: {e}")
