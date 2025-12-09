from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
import os

# --- Data Generation and Preprocessing ---

def generate_synthetic_data(days=10):
    """
    Generates synthetic 15-minute Forex data with a clear Asia session range
    followed by a liquidity grab and reversal pattern in the London session.
    """
    np.random.seed(42)
    periods_per_day = 96  # 24 hours * 4 periods per hour
    total_periods = periods_per_day * days

    # Create a 15-min frequency DatetimeIndex, excluding weekends
    index = pd.to_datetime(pd.bdate_range(end=pd.Timestamp.now(tz='UTC'), periods=total_periods, freq='15min'))

    data = pd.DataFrame(index=index, columns=['Open', 'High', 'Low', 'Close'])

    base_price = 1.1000

    # Iterate through each day to create patterns
    for day in range(days):
        day_start_index = day * periods_per_day
        day_end_index = (day + 1) * periods_per_day

        # --- Asia Session (00:00 - 08:00 UTC) ---
        asia_start_index = day_start_index
        asia_end_index = day_start_index + 32 # 8 hours * 4
        asia_range_low = base_price - 0.0010 # 10 pips
        asia_range_high = base_price + 0.0010 # 10 pips

        # Generate ranging prices for Asia session
        prices = np.random.uniform(asia_range_low, asia_range_high, size=asia_end_index - asia_start_index)
        data.iloc[asia_start_index:asia_end_index, data.columns.get_loc('Open')] = prices
        data.iloc[asia_start_index:asia_end_index, data.columns.get_loc('Close')] = prices + np.random.uniform(-0.0002, 0.0002, size=len(prices))
        data.iloc[asia_start_index:asia_end_index, data.columns.get_loc('High')] = data.iloc[asia_start_index:asia_end_index][['Open', 'Close']].max(axis=1) + 0.0001
        data.iloc[asia_start_index:asia_end_index, data.columns.get_loc('Low')] = data.iloc[asia_start_index:asia_end_index][['Open', 'Close']].min(axis=1) - 0.0001

        # --- London Session (08:00 - 16:00 UTC) ---
        london_start_index = asia_end_index

        # Create a textbook short setup pattern
        if day % 2 == 0:
            # 1. Grab candle: Spike above Asia high
            grab_candle_idx = london_start_index + 4 # 1 hour into London
            data.loc[data.index[grab_candle_idx], 'Open'] = asia_range_high - 0.0002
            data.loc[data.index[grab_candle_idx], 'High'] = asia_range_high + 0.0015 # Clear spike
            data.loc[data.index[grab_candle_idx], 'Low'] = asia_range_high - 0.0005
            data.loc[data.index[grab_candle_idx], 'Close'] = asia_range_high + 0.0003

            # 2. Reversal (Engulfing) candle: Opens above, closes deep inside the range
            reversal_candle_idx = grab_candle_idx + 1
            data.loc[data.index[reversal_candle_idx], 'Open'] = data.loc[data.index[grab_candle_idx], 'Close']
            data.loc[data.index[reversal_candle_idx], 'High'] = data.loc[data.index[reversal_candle_idx], 'Open'] + 0.0002
            data.loc[data.index[reversal_candle_idx], 'Close'] = asia_range_low + 0.0003 # Close deep inside
            data.loc[data.index[reversal_candle_idx], 'Low'] = data.loc[data.index[reversal_candle_idx], 'Close'] - 0.0002

        # Fill the rest of the day with some noise
        remaining_indices = data.iloc[day_start_index:day_end_index].index.difference(data.iloc[asia_start_index:reversal_candle_idx+1].index)
        if not remaining_indices.empty:
            noise = np.random.normal(0, 0.0003, size=len(remaining_indices)).cumsum()
            last_price = data.loc[data.index[reversal_candle_idx], 'Close']
            data.loc[remaining_indices, 'Close'] = last_price + noise
            data.loc[remaining_indices, 'Open'] = data.loc[remaining_indices, 'Close'].shift(1).fillna(last_price)
            data.loc[remaining_indices, 'High'] = data.loc[remaining_indices, ['Open', 'Close']].max(axis=1) + 0.0001
            data.loc[remaining_indices, 'Low'] = data.loc[remaining_indices, ['Open', 'Close']].min(axis=1) - 0.0001

        # Update base price for the next day
        base_price = data.iloc[day_end_index-1]['Close']
        if pd.isna(base_price): base_price = 1.1000 # Reset if NaN

    data.ffill(inplace=True)
    return data.astype(float)


def preprocess_data(df):
    """
    Calculates and appends session-based indicators to the DataFrame.
    - Session identification (Asia, UK)
    - Asia session high, low, and range percentage
    """
    df_copy = df.copy()

    # --- Session Identification ---
    df_copy['hour'] = df_copy.index.hour
    df_copy['is_asia_session'] = (df_copy['hour'] >= 0) & (df_copy['hour'] < 8)
    df_copy['is_uk_session'] = (df_copy['hour'] >= 8) & (df_copy['hour'] < 16)

    # --- Daily Asia Range Calculation ---
    # Create a daily resample to calculate Asia H/L
    daily_grouper = pd.Grouper(level=0, freq='D')

    # Function to get high/low of the Asia session for each day
    def get_asia_high(day_data):
        return day_data[day_data['is_asia_session']]['High'].max()

    def get_asia_low(day_data):
        return day_data[day_data['is_asia_session']]['Low'].min()

    daily_asia_high = df_copy.groupby(daily_grouper).apply(get_asia_high)
    daily_asia_low = df_copy.groupby(daily_grouper).apply(get_asia_low)

    # Map the daily values back to the 15-min timeframe
    df_copy['asia_high'] = daily_asia_high.reindex(df_copy.index, method='ffill')
    df_copy['asia_low'] = daily_asia_low.reindex(df_copy.index, method='ffill')

    # --- Asia Range Percentage ---
    df_copy['asia_range_perc'] = (df_copy['asia_high'] - df_copy['asia_low']) / df_copy['asia_low'] * 100

    # Clean up and return
    df_copy.drop(columns=['hour'], inplace=True)
    df_copy.dropna(inplace=True) # Drop rows where Asia H/L couldn't be calculated

    return df_copy

# --- Strategy Class ---

class AsiaLiquidityGrabReversalStrategy(Strategy):
    """
    A strategy that trades reversals after a liquidity grab of the Asian session high or low
    during the London session.
    """
    # Optimizable parameters
    asia_range_max_perc = 1.0  # Max Asia session range in percent
    sl_buffer_pips = 5         # Stop-loss buffer in pips

    def init(self):
        """Initialize the strategy and register indicators."""
        # Pass-through function for pre-calculated data
        def pass_through(series):
            return series.values

        # Register pre-calculated columns as indicators
        self.asia_high = self.I(pass_through, self.data.df['asia_high'])
        self.asia_low = self.I(pass_through, self.data.df['asia_low'])
        self.is_uk_session = self.I(pass_through, self.data.df['is_uk_session'])
        self.asia_range_perc = self.I(pass_through, self.data.df['asia_range_perc'])

    def next(self):
        """The main trading logic loop."""
        # Ensure we have enough data
        if len(self.data.Close) < 2:
            return

        # If a position is already open, do nothing.
        if self.position:
            return

        # General conditions for any trade
        is_london = self.is_uk_session[-1] == 1
        is_asia_range_valid = self.asia_range_perc[-1] < self.asia_range_max_perc

        if not is_london or not is_asia_range_valid:
            return

        # --- SHORT SETUP ---
        # 1. Liquidity Grab: Previous candle's high broke above Asia high
        grabbed_asia_high = self.data.High[-2] > self.asia_high[-2]

        # 2. Reversal Confirmation: Current candle is a bearish engulfing pattern
        #    that closes back inside the Asia range.
        prev_is_bullish = self.data.Close[-2] > self.data.Open[-2]
        curr_is_bearish = self.data.Close[-1] < self.data.Open[-1]
        is_bearish_engulfing = (curr_is_bearish and prev_is_bullish and
                                self.data.Open[-1] >= self.data.Close[-2] and
                                self.data.Close[-1] < self.data.Open[-2])

        closed_in_range = self.data.Close[-1] < self.asia_high[-1]

        if grabbed_asia_high and is_bearish_engulfing and closed_in_range:
            sl_price = self.data.High[-1] + (self.sl_buffer_pips * 0.0001)
            tp_price = self.asia_low[-1]

            # Ensure SL is above entry and TP is below
            if sl_price > self.data.Close[-1] and tp_price < self.data.Close[-1]:
                self.sell(sl=sl_price, tp=tp_price)

        # --- LONG SETUP ---
        # 1. Liquidity Grab: Previous candle's low broke below Asia low
        grabbed_asia_low = self.data.Low[-2] < self.asia_low[-2]

        # 2. Reversal Confirmation: Current candle is a bullish engulfing pattern
        #    that closes back inside the Asia range.
        prev_is_bearish = self.data.Close[-2] < self.data.Open[-2]
        curr_is_bullish = self.data.Close[-1] > self.data.Open[-1]
        is_bullish_engulfing = (curr_is_bullish and prev_is_bearish and
                                self.data.Open[-1] <= self.data.Close[-2] and
                                self.data.Close[-1] > self.data.Open[-2])

        closed_in_range_long = self.data.Close[-1] > self.asia_low[-1]

        if grabbed_asia_low and is_bullish_engulfing and closed_in_range_long:
            sl_price = self.data.Low[-1] - (self.sl_buffer_pips * 0.0001)
            tp_price = self.asia_high[-1]

            # Ensure SL is below entry and TP is above
            if sl_price < self.data.Close[-1] and tp_price > self.data.Close[-1]:
                self.buy(sl=sl_price, tp=tp_price)

# --- Utility Functions ---

def sanitize_for_json(obj):
    """
    Recursively sanitizes a data structure for JSON serialization.
    - Converts numpy numbers to native Python types.
    - Converts NaN/inf to None.
    - Converts pandas objects to None.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(i) for i in obj]
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (pd.Series, pd.DataFrame, pd.Timestamp)):
        return None
    return obj


# --- Main Execution Block ---

if __name__ == '__main__':
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # 1. Generate and Preprocess Data
    # Using a smaller dataset for faster optimization
    raw_data = generate_synthetic_data(days=50)
    data = preprocess_data(raw_data)

    # Check if data is empty after preprocessing
    if data.empty:
        print("Error: Data is empty after preprocessing. Cannot run backtest.")
    else:
        # 2. Initialize and Run Backtest
        bt = Backtest(data, AsiaLiquidityGrabReversalStrategy, cash=100_000, commission=.0002)

        # 3. Optimize the Strategy
        stats = bt.optimize(
            asia_range_max_perc=np.arange(0.5, 2.1, 0.25).tolist(),
            sl_buffer_pips=range(5, 16, 2),
            maximize='Sharpe Ratio',
            constraint=lambda p: p.asia_range_max_perc > 0
        )

        # 4. Save the Results
        results_dict = {
            'strategy_name': 'asia_liquidity_grab_reversal',
            'return': stats.get('Return [%]'),
            'sharpe': stats.get('Sharpe Ratio'),
            'max_drawdown': stats.get('Max. Drawdown [%]'),
            'win_rate': stats.get('Win Rate [%]'),
            'total_trades': stats.get('# Trades')
        }

        sanitized_results = sanitize_for_json(results_dict)

        with open('results/temp_result.json', 'w') as f:
            json.dump(sanitized_results, f, indent=2)

        print("Backtest results saved to results/temp_result.json")
        print(sanitized_results)

        # 5. Generate the Plot
        try:
            plot_filename = 'results/asia_liquidity_grab_reversal.html'
            bt.plot(filename=plot_filename, open_browser=False)
            print(f"Backtest plot saved to {plot_filename}")
        except TypeError as e:
            print(f"Could not generate plot due to a known issue with pandas version: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during plot generation: {e}")
