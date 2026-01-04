
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import json
import os

# --- Data Preprocessing ---

def calculate_daily_volume_profile(daily_data, num_bins=100):
    """
    Calculates the POC, VAH, and VAL for a given day's data using a robust
    histogram-based method.
    """
    if daily_data.empty:
        return pd.Series({'poc': np.nan, 'vah': np.nan, 'val': np.nan})

    price_min = daily_data['Low'].min()
    price_max = daily_data['High'].max()

    if price_max == price_min:
        return pd.Series({'poc': price_min, 'vah': price_max, 'val': price_min})

    # Create a volume histogram weighted by the close price
    try:
        hist, bin_edges = np.histogram(
            a=daily_data['Close'],
            bins=num_bins,
            range=(price_min, price_max),
            weights=daily_data['Volume']
        )
    except ValueError:
        # Can happen if range is 0 or data is inconsistent
        return pd.Series({'poc': np.nan, 'vah': np.nan, 'val': np.nan})

    # Find POC: Price at the bin with the highest volume
    poc_index = np.argmax(hist)
    poc = (bin_edges[poc_index] + bin_edges[poc_index + 1]) / 2

    # Calculate Value Area (70% of volume)
    total_volume = daily_data['Volume'].sum()
    if total_volume == 0:
        return pd.Series({'poc': poc, 'vah': price_max, 'val': price_min})

    target_volume = total_volume * 0.7

    current_volume = hist[poc_index]
    vah_index, val_index = poc_index, poc_index

    # Expand from POC until 70% of volume is captured
    while current_volume < target_volume:
        next_up_index = vah_index + 1
        next_down_index = val_index - 1

        vol_up = hist[next_up_index] if next_up_index < len(hist) else 0
        vol_down = hist[next_down_index] if next_down_index >= 0 else 0

        if vol_up > vol_down:
            current_volume += vol_up
            vah_index = next_up_index
        else:
            current_volume += vol_down
            val_index = next_down_index

        if (next_up_index >= len(hist)) and (next_down_index < 0):
            break

    vah = bin_edges[min(vah_index + 1, len(bin_edges) - 1)]
    val = bin_edges[max(val_index, 0)]

    return pd.Series({'poc': poc, 'vah': vah, 'val': val})


def preprocess_data(df):
    """
    Calculates the previous day's VAH, VAL, and POC for each 15-minute bar.
    """
    # Ensure the dataframe has a date column for grouping
    df['date'] = df.index.date

    # Calculate volume profile for each day
    daily_profiles = df.groupby('date').apply(calculate_daily_volume_profile)

    # Shift the profiles by one day to avoid lookahead bias
    daily_profiles_shifted = daily_profiles.shift(1)
    daily_profiles_shifted.rename(columns={
        'vah': 'prev_vah',
        'val': 'prev_val',
        'poc': 'prev_poc'
    }, inplace=True)

    # Merge the shifted daily profiles back into the main dataframe
    df = pd.merge(df, daily_profiles_shifted, left_on='date', right_index=True, how='left')

    # Forward-fill the daily values to apply to all bars within the day
    df['prev_vah'] = df['prev_vah'].ffill()
    df['prev_val'] = df['prev_val'].ffill()
    df['prev_poc'] = df['prev_poc'].ffill()

    # Clean up and return
    return df.drop(columns=['date']).dropna()

# --- Strategy Definition ---

class VolumeProfileValueAreaReversionStrategy(Strategy):
    """
    A mean-reversion strategy that trades when the price reverts
    back into the previous day's value area.
    """

    # --- Strategy Parameters ---
    sl_buffer_pct = 0.01 # Percentage buffer for stop loss

    def init(self):
        """
        Initialize indicators and strategy state.
        """
        # Expose pre-calculated data columns to the strategy
        self.prev_vah = self.I(lambda x: x, self.data.df['prev_vah'].values, name="Prev_VAH")
        self.prev_val = self.I(lambda x: x, self.data.df['prev_val'].values, name="Prev_VAL")
        self.prev_poc = self.I(lambda x: x, self.data.df['prev_poc'].values, name="Prev_POC")

        # State variables to track if price has crossed outside the value area
        self.crossed_below_val = False
        self.crossed_above_vah = False


    def next(self):
        """
        Main strategy logic executed on each bar.
        """
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        prev_vah = self.prev_vah[-1]
        prev_val = self.prev_val[-1]
        prev_poc = self.prev_poc[-1]

        # Ignore bars with invalid profile data
        if pd.isna(prev_vah) or pd.isna(prev_val) or pd.isna(prev_poc):
            return

        # --- Position Management ---
        if self.position:
            # Simple time-based exit (e.g., end of day) could be added here
            return

        # --- Entry Logic ---

        # Conditions for a bullish reversal candle
        is_bullish_candle = self.data.Close[-1] > self.data.Open[-1]

        # Conditions for a bearish reversal candle
        is_bearish_candle = self.data.Close[-1] < self.data.Open[-1]

        # 1. Check for price crossing below VAL
        if price < prev_val:
            self.crossed_below_val = True
            self.crossed_above_vah = False # Reset other signal

        # 2. Check for price crossing above VAH
        if price > prev_vah:
            self.crossed_above_vah = True
            self.crossed_below_val = False # Reset other signal

        # 3. Long Entry: If price was below VAL and now re-enters with a bullish candle
        if self.crossed_below_val and price > prev_val and is_bullish_candle:
            sl = low * (1 - self.sl_buffer_pct)
            tp = prev_poc

            # Ensure TP and SL are valid
            if tp > price and sl < price:
                self.buy(sl=sl, tp=tp)
            self.crossed_below_val = False # Reset state

        # 4. Short Entry: If price was above VAH and now re-enters with a bearish candle
        if self.crossed_above_vah and price < prev_vah and is_bearish_candle:
            sl = high * (1 + self.sl_buffer_pct)
            tp = prev_poc

            # Ensure TP and SL are valid
            if tp < price and sl > price:
                self.sell(sl=sl, tp=tp)
            self.crossed_above_vah = False # Reset state

# --- Backtesting Execution ---

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        # In a real scenario, you might generate synthetic data here as a fallback
        # For this implementation, we will stop if data is not present.
    else:
        # Load data
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data.columns = [c.strip().title() for c in data.columns] # Sanitize column names

        # Preprocess data
        # NOTE: The volume profile calculation is computationally intensive and may cause
        # timeouts in environments with limited resources. If timeouts occur,
        # consider reducing the dataset size for testing purposes.
        print("Preprocessing data for volume profile calculation...")
        data = preprocess_data(data)

        if data.empty:
            print("Error: Preprocessing resulted in empty data. Halting backtest.")
        else:
            print("Data preprocessing complete.")
            # Configure and run the backtest
            bt = Backtest(data, VolumeProfileValueAreaReversionStrategy, cash=100_000, commission=.002)

            print("Running backtest...")
            stats = bt.run()

            print("\nBacktest Results:")
            print(stats)

            # --- Result Saving ---
            results_dir = 'results'
            os.makedirs(results_dir, exist_ok=True)

            # Sanitize stats for JSON serialization
            def sanitize_value(value):
                if pd.isna(value) or np.isnan(value):
                    return None
                if isinstance(value, (np.int64, np.int32)):
                    return int(value)
                if isinstance(value, (np.float64, np.float32)):
                    return float(value)
                if isinstance(value, pd.Timestamp):
                    return value.isoformat()
                if isinstance(value, pd.Timedelta):
                    return str(value)
                return value

            # Sanitize the entire stats series, handling nested structures if any
            clean_stats = {key: sanitize_value(value) for key, value in stats.items() if not isinstance(value, (pd.DataFrame, pd.Series))}

            # Save stats to JSON
            results_path = os.path.join(results_dir, 'temp_result.json')
            with open(results_path, 'w') as f:
                json.dump(clean_stats, f, indent=4)
            print(f"\nSaved backtest statistics to {results_path}")

            # Generate plot
            plot_path = os.path.join(results_dir, 'volume_profile_value_area_reversion.html')
            try:
                bt.plot(filename=plot_path, open_browser=False)
                print(f"Saved backtest plot to {plot_path}")
            except Exception as e:
                print(f"Could not generate plot due to an error: {e}")
