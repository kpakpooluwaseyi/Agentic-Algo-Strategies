from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd

def calculate_daily_volume_profile(day_data, n_bins=100):
    """
    Calculates an approximate Volume Profile for a single day's data.
    """
    if day_data.empty:
        return None, None, None

    price_range = day_data['High'].max() - day_data['Low'].min()
    bin_size = price_range / n_bins

    if bin_size == 0: # Avoid division by zero if price didn't move
        poc = day_data['Close'].iloc[-1]
        return poc, poc, poc

    # Approximate volume at each price level within the day's range
    day_data['price_bin'] = ((day_data['High'] + day_data['Low']) / 2 / bin_size).round() * bin_size
    volume_profile = day_data.groupby('price_bin')['Volume'].sum().sort_index()

    if volume_profile.empty:
        poc = day_data['Close'].iloc[-1]
        return poc, poc, poc

    # Point of Control (POC)
    poc = volume_profile.idxmax()

    # Value Area (VA) - typically 70% of volume
    total_volume = volume_profile.sum()
    target_volume = total_volume * 0.7

    current_volume = volume_profile.loc[poc]
    vah, val = poc, poc

    # Expand from POC outwards to find Value Area
    above_poc = volume_profile[volume_profile.index > poc].index
    below_poc = volume_profile[volume_profile.index < poc].index[::-1]

    i_above, i_below = 0, 0
    while current_volume < target_volume and (i_above < len(above_poc) or i_below < len(below_poc)):
        vol_above = 0
        if i_above < len(above_poc):
            vol_above = volume_profile.loc[above_poc[i_above]]

        vol_below = 0
        if i_below < len(below_poc):
            vol_below = volume_profile.loc[below_poc[i_below]]

        if vol_above > vol_below:
            vah = above_poc[i_above]
            current_volume += vol_above
            i_above += 1
        else:
            val = below_poc[i_below]
            current_volume += vol_below
            i_below += 1

    return poc, vah, val

def preprocess_data(df):
    """
    Pre-processes the data to calculate the previous day's Volume Profile metrics
    (POC, VAH, VAL).
    """
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.set_index('Datetime')

    daily_grouped = df.groupby(df.index.date)
    daily_profiles = {}

    for date, day_data in daily_grouped:
        poc, vah, val = calculate_daily_volume_profile(day_data)
        daily_profiles[date] = {'poc': poc, 'vah': vah, 'val': val}

    profile_df = pd.DataFrame.from_dict(daily_profiles, orient='index')

    # Shift to get previous day's values
    profile_df['prev_day_poc'] = profile_df['poc'].shift(1)
    profile_df['prev_day_vah'] = profile_df['vah'].shift(1)
    profile_df['prev_day_val'] = profile_df['val'].shift(1)

    # Map back to the original dataframe
    date_map_poc = profile_df['prev_day_poc'].to_dict()
    date_map_vah = profile_df['prev_day_vah'].to_dict()
    date_map_val = profile_df['prev_day_val'].to_dict()

    df['prev_day_poc'] = pd.Series(df.index.date, index=df.index).map(date_map_poc)
    df['prev_day_vah'] = pd.Series(df.index.date, index=df.index).map(date_map_vah)
    df['prev_day_val'] = pd.Series(df.index.date, index=df.index).map(date_map_val)

    df.dropna(subset=['prev_day_poc', 'prev_day_vah', 'prev_day_val'], inplace=True)

    return df

class VolumeProfileValueAreaBreakout(Strategy):
    """
    A strategy that enters trades based on breakouts of the previous day's
    volume profile value area. The Value Area High (VAH), Value Area Low (VAL),
    and Point of Control (POC) are calculated daily based on volume distribution.
    """
    risk_reward_ratio = 1.5
    stop_loss_pct = 0.01

    def init(self):
        # Make the VAH, VAL, and POC from the previous day available as indicators
        self.prev_day_vah = self.I(lambda x: x, self.data.df['prev_day_vah'], name='PrevDayVAH')
        self.prev_day_val = self.I(lambda x: x, self.data.df['prev_day_val'], name='PrevDayVAL')
        self.prev_day_poc = self.I(lambda x: x, self.data.df['prev_day_poc'], name='PrevDayPOC')

    def next(self):
        # Scenario A & B: Breakout of Previous Day's VAH/VAL

        # Long Entry: Price breaks above previous day's VAH
        if not self.position and crossover(self.data.Close, self.prev_day_vah):
            sl = self.prev_day_vah * (1 - self.stop_loss_pct)
            tp = self.data.Close[-1] + (self.data.Close[-1] - sl) * self.risk_reward_ratio
            self.buy(sl=sl, tp=tp)

        # Short Entry: Price breaks below previous day's VAL
        elif not self.position and crossover(self.prev_day_val, self.data.Close):
            sl = self.prev_day_val * (1 + self.stop_loss_pct)
            tp = self.data.Close[-1] - (sl - self.data.Close[-1]) * self.risk_reward_ratio
            self.sell(sl=sl, tp=tp)

def sanitize_stats_for_json(stats):
    """Converts a backtesting.py stats object to a JSON-serializable dict."""
    import numpy as np
    stats_dict = stats.to_dict()
    # Remove non-serializable or bulky items
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    sanitized_dict = {}
    for key, value in stats_dict.items():
        if isinstance(value, pd.Timestamp):
            sanitized_dict[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            sanitized_dict[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized_dict[key] = float(value) if pd.notnull(value) else None
        elif key == '_strategy':
            sanitized_dict[key] = value.__class__.__name__
        elif hasattr(value, '__name__'):
            sanitized_dict[key] = value.__name__
        else:
            sanitized_dict[key] = value

    return sanitized_dict

if __name__ == '__main__':
    # Load data
    data = pd.read_csv('data/BTC-USD-15m.csv')

    # Sanitize column names (e.g., ' open' -> 'Open')
    data.columns = [col.strip().capitalize() for col in data.columns]

    # Preprocess data
    data = preprocess_data(data.copy())

    # Initialize and run the backtest, finalizing open trades
    bt = Backtest(data, VolumeProfileValueAreaBreakout, cash=100000, commission=.002, finalize_trades=True)
    stats = bt.run()

    # Print the stats
    print(stats)

    # Save the results
    import json
    sanitized_stats = sanitize_stats_for_json(stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    # Plot the results
    bt.plot(filename='results/volume_profile_value_area_breakout.html')
