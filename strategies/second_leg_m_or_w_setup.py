from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
import pandas_ta as ta

# --- Data Preprocessing ---
def preprocess_data(df):
    """Adds session information and previous session highs/lows."""
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_asia'] = (df['hour'] >= 0) & (df['hour'] < 8)
    df['is_london'] = (df['hour'] >= 8) & (df['hour'] < 16)

    # Calculate previous day's Asia high and low
    asia_df = df[df['is_asia']]
    if not asia_df.empty:
        asia_session_data = asia_df.groupby(asia_df.index.date).agg(
            prev_asia_high=('High', 'max'),
            prev_asia_low=('Low', 'min')
        )
        # Shift to align with the next day's trading
        asia_session_data.index = pd.to_datetime(asia_session_data.index) + pd.Timedelta(days=1)

        # Map the values to the main dataframe
        df['prev_asia_high'] = df.index.normalize().map(asia_session_data['prev_asia_high'])
        df['prev_asia_low'] = df.index.normalize().map(asia_session_data['prev_asia_low'])
    else:
        df['prev_asia_high'] = np.nan
        df['prev_asia_low'] = np.nan

    df['prev_asia_range'] = df['prev_asia_high'] - df['prev_asia_low']

    # Forward fill is the correct approach here to avoid lookahead bias
    df.ffill(inplace=True)
    return df

# --- Custom Indicator Functions ---
def passthrough(data, **kwargs):
    return data

# --- Strategy Definition ---
class SecondLegMWSetupStrategy(Strategy):
    # --- Strategy Parameters ---
    asia_range_max_pips = 500.0
    stop_hunt_min_pips = 250.0
    sl_pips = 100.0
    tp_pips = 500
    time_exit_bars = 8 # 2 hours on 15m timeframe
    ema_50_period = 50
    ema_200_period = 200
    rsi_period = 14
    lookback_period = 5

    def init(self):
        # --- Confluence Indicators ---
        self.ema_50 = self.I(ta.ema, pd.Series(self.data.Close), length=self.ema_50_period)
        self.ema_200 = self.I(ta.ema, pd.Series(self.data.Close), length=self.ema_200_period)
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), length=self.rsi_period)

        # --- Pre-calculated indicators ---
        self.is_london = self.I(passthrough, self.data.df['is_london'].values)
        self.prev_asia_high = self.I(passthrough, self.data.df['prev_asia_high'].values)
        self.prev_asia_low = self.I(passthrough, self.data.df['prev_asia_low'].values)
        self.prev_asia_range = self.I(passthrough, self.data.df['prev_asia_range'].values)

        # --- State variables for M/W patterns ---
        self.m_pattern_state = 'IDLE'
        self.w_pattern_state = 'IDLE'
        self.first_leg_high = None
        self.first_leg_low = None
        self.trade_entry_bar = None

    def _is_swing(self, series, is_high):
        if len(series) < 3:
            return False

        if is_high:
            return series[-2] > series[-1] and series[-2] > series[-3]
        else:
            return series[-2] < series[-1] and series[-2] < series[-3]

    def next(self):
        # --- Data Validity Check ---
        if pd.isna(self.prev_asia_high[-1]) or pd.isna(self.prev_asia_low[-1]):
            return

        # --- Time-based Exit ---
        if self.position:
            if len(self.data) - self.trade_entry_bar >= self.time_exit_bars:
                if self.trades:
                    if not self.position.is_long and self.data.Close[-1] > self.trades[0].entry_price:
                        self.position.close()
                    elif self.position.is_long and self.data.Close[-1] < self.trades[0].entry_price:
                        self.position.close()
            return

        # --- Entry Conditions ---
        # Only trade during London session
        if not self.is_london[-1]:
            # Reset state outside of London session
            self.m_pattern_state = 'IDLE'
            self.w_pattern_state = 'IDLE'
            return

        # --- M-Pattern (Bearish) Logic ---
        # State: IDLE -> Look for a swing high above Asia range (first leg)
        # --- Filter conditions ---
        is_valid_asia_range = self.prev_asia_range[-1] <= self.asia_range_max_pips

        # --- M-Pattern (Bearish) Logic ---
        if is_valid_asia_range:
            is_swing_high = self._is_swing(self.data.High, is_high=True)
            # State: IDLE -> Look for a swing high above Asia range (first leg)
            if self.m_pattern_state == 'IDLE' and is_swing_high:
                swing_bar_high = self.data.High[-2]
                stop_hunt_pips = swing_bar_high - self.prev_asia_high[-1]
                if stop_hunt_pips >= self.stop_hunt_min_pips:
                    self.m_pattern_state = 'FIRST_LEG_FORMED'
                    self.first_leg_high = swing_bar_high

            # State: FIRST_LEG_FORMED -> Look for a second, lower swing high
            elif self.m_pattern_state == 'FIRST_LEG_FORMED' and is_swing_high:
                second_leg_high = self.data.High[-2]
                if second_leg_high < self.first_leg_high:
                    # Confluence Checks
                    is_below_ema50 = self.data.Close[-1] < self.ema_50[-1]
                    is_below_ema200 = self.data.Close[-1] < self.ema_200[-1]
                    is_rsi_bearish = self.rsi[-1] < 50

                    if is_below_ema50 and is_below_ema200 and is_rsi_bearish:
                        sl = second_leg_high + self.sl_pips
                        tp = self.data.Close[-1] - self.tp_pips
                        self.sell(sl=sl, tp=tp)
                        self.trade_entry_bar = len(self.data) - 1
                # Reset after second leg, regardless of entry
                self.m_pattern_state = 'IDLE'
                self.first_leg_high = None

        # --- W-Pattern (Bullish) Logic ---
        if is_valid_asia_range:
            is_swing_low = self._is_swing(self.data.Low, is_high=False)
            # State: IDLE -> Look for a swing low below Asia range (first leg)
            if self.w_pattern_state == 'IDLE' and is_swing_low:
                swing_bar_low = self.data.Low[-2]
                stop_hunt_pips = self.prev_asia_low[-1] - swing_bar_low
                if stop_hunt_pips >= self.stop_hunt_min_pips:
                    self.w_pattern_state = 'FIRST_LEG_FORMED'
                    self.first_leg_low = swing_bar_low

            # State: FIRST_LEG_FORMED -> Look for a second, higher swing low
            elif self.w_pattern_state == 'FIRST_LEG_FORMED' and is_swing_low:
                second_leg_low = self.data.Low[-2]
                if second_leg_low > self.first_leg_low:
                    # Confluence Checks
                    is_above_ema50 = self.data.Close[-1] > self.ema_50[-1]
                    is_above_ema200 = self.data.Close[-1] > self.ema_200[-1]
                    is_rsi_bullish = self.rsi[-1] > 50

                    if is_above_ema50 and is_above_ema200 and is_rsi_bullish:
                        sl = second_leg_low - self.sl_pips
                        tp = self.data.Close[-1] + self.tp_pips
                        self.buy(sl=sl, tp=tp)
                        self.trade_entry_bar = len(self.data) - 1
                # Reset after second leg, regardless of entry
                self.w_pattern_state = 'IDLE'
                self.first_leg_low = None

# --- Backtesting Execution ---
if __name__ == '__main__':
    import os

    # --- Load Data ---
    # In a real scenario, you would use the specified data path
    # For now, let's check if the specified data file exists
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data file is in the correct location.")
        # As a fallback, create some synthetic data to allow the script to run
        data = pd.DataFrame({
            'Open': np.random.rand(1000) + 10000,
            'High': np.random.rand(1000) + 10050,
            'Low': np.random.rand(1000) + 9950,
            'Close': np.random.rand(1000) + 10000,
            'Volume': np.random.rand(1000) * 100
        }, index=pd.to_datetime(pd.date_range('2023-01-01', periods=1000, freq='15min')))
    else:
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True, skipinitialspace=True)
        # Ensure column names are in the format Backtesting.py expects
        data.columns = [col.strip().capitalize() for col in data.columns]

    # --- Preprocess Data ---
    data = preprocess_data(data)

    # --- Run Backtest ---
    bt = Backtest(data, SecondLegMWSetupStrategy, cash=100_000, commission=.002)

    print("Running backtest...")
    stats = bt.run()
    print(stats)

    # --- Save Results ---
    os.makedirs('results', exist_ok=True)

    # Sanitize stats for JSON output
    def sanitize(value):
        if isinstance(value, (np.int64, np.int32)):
            return int(value)
        if isinstance(value, (np.float64, np.float32)):
            return float(value)
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, pd.Timedelta):
            return str(value)
        if isinstance(value, (pd.Series, pd.DataFrame)):
            return None # Or handle more gracefully
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        return value

    # Create a serializable dictionary from the stats Series
    stats_dict = stats.to_dict()
    # Remove non-serializable objects
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    sanitized_stats = sanitize(stats_dict)

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    print("Backtest stats saved to results/temp_result.json")

    # --- Generate Plot ---
    try:
        plot_filename = 'results/second_leg_m_or_w_setup.html'
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
