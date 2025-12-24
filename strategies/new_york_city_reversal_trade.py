from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os

def preprocess_data(df):
    """Calculates daily levels and maps them into the main dataframe, preserving the index."""
    # Ensure index is a DatetimeIndex and localize to UTC, then convert to NY time
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')

    # Calculate daily high, low, and ADR
    daily_df = df.resample('D').agg({
        'High': 'max',
        'Low': 'min',
        'Open': 'first',
        'Close': 'last'
    })
    daily_df['ADR'] = (daily_df['High'] - daily_df['Low']).rolling(window=14).mean()
    daily_df['ADR_High'] = daily_df['Open'] + daily_df['ADR']
    daily_df['ADR_Low'] = daily_df['Open'] - daily_df['ADR']

    # Shift to prevent lookahead bias
    daily_df['Prev_ADR_High'] = daily_df['ADR_High'].shift(1)
    daily_df['Prev_ADR_Low'] = daily_df['ADR_Low'].shift(1)
    daily_df['Prev_High'] = daily_df['High'].shift(1)
    daily_df['Prev_Low'] = daily_df['Low'].shift(1)

    # Map daily data to the 15m timeframe using the normalized index
    df['Prev_ADR_High'] = df.index.normalize().map(daily_df['Prev_ADR_High'])
    df['Prev_ADR_Low'] = df.index.normalize().map(daily_df['Prev_ADR_Low'])
    df['Prev_High'] = df.index.normalize().map(daily_df['Prev_High'])
    df['Prev_Low'] = df.index.normalize().map(daily_df['Prev_Low'])

    # Forward-fill the daily values to handle NaNs
    df[['Prev_ADR_High', 'Prev_ADR_Low', 'Prev_High', 'Prev_Low']] = df[['Prev_ADR_High', 'Prev_ADR_Low', 'Prev_High', 'Prev_Low']].ffill()

    return df

def is_bullish_engulfing(data):
    """Checks for a bullish engulfing pattern."""
    if len(data.Close) < 2:
        return False
    current_candle = data.df.iloc[-1]
    previous_candle = data.df.iloc[-2]
    # Previous candle is bearish, current is bullish
    if previous_candle['Close'] >= previous_candle['Open'] or current_candle['Close'] <= current_candle['Open']:
        return False
    # Current candle engulfs the previous one
    if current_candle['Close'] >= previous_candle['Open'] and current_candle['Open'] <= previous_candle['Close']:
        return True
    return False

def is_bearish_engulfing(data):
    """Checks for a bearish engulfing pattern."""
    if len(data.Close) < 2:
        return False
    current_candle = data.df.iloc[-1]
    previous_candle = data.df.iloc[-2]
    # Previous candle is bullish, current is bearish
    if previous_candle['Close'] <= previous_candle['Open'] or current_candle['Close'] >= current_candle['Open']:
        return False
    # Current candle engulfs the previous one
    if current_candle['Open'] >= previous_candle['Close'] and current_candle['Close'] <= previous_candle['Open']:
        return True
    return False

class NewYorkCityReversalTrade(Strategy):
    # Strategy parameters
    ema_fast_period = 5
    ema_slow_period = 13
    ema_target_period = 50
    rsi_period = 14
    rsi_upper = 70
    rsi_lower = 30
    ny_session_start = 8
    ny_session_end = 11
    sl_buffer_pct = 0.01 # 1% buffer
    level_one_range_pct = 0.1 # 0.1% opening range for Level I

    def init(self):
        # State machine for Level Count
        self.level = 1
        self.daily_open = None
        self.level_one_high = None
        self.level_one_low = None

        # Indicators
        self.ema_fast = self.I(ta.ema, pd.Series(self.data.Close), length=self.ema_fast_period)
        self.ema_slow = self.I(ta.ema, pd.Series(self.data.Close), length=self.ema_slow_period)
        self.ema_target = self.I(ta.ema, pd.Series(self.data.Close), length=self.ema_target_period)
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), length=self.rsi_period)

        # Daily data access
        self.prev_adr_high = self.I(lambda x: x, self.data.df['Prev_ADR_High'])
        self.prev_adr_low = self.I(lambda x: x, self.data.df['Prev_ADR_Low'])
        self.prev_high = self.I(lambda x: x, self.data.df['Prev_High'])
        self.prev_low = self.I(lambda x: x, self.data.df['Prev_Low'])

    def next(self):
        current_time = self.data.index[-1]
        price = self.data.Close[-1]

        # Reset level count at the start of a new day
        if self.daily_open is None or current_time.date() != pd.to_datetime(self.data.index[-2]).date():
            self.level = 1
            self.daily_open = self.data.Open[-1]
            self.level_one_high = self.daily_open * (1 + self.level_one_range_pct / 100)
            self.level_one_low = self.daily_open * (1 - self.level_one_range_pct / 100)

        # Level Count State Machine Proxy:
        # Level 1: Initial consolidation range after the daily open.
        # Level 2: Price breaks out of the initial range.
        # Level 3: Price reverses and returns to the initial range (reversal condition).
        if self.level == 1 and (price > self.level_one_high or price < self.level_one_low):
            self.level = 2
        elif self.level == 2 and (self.level_one_low <= price <= self.level_one_high):
            self.level = 3

        # Time-based condition (New York Session)
        if not (self.ny_session_start <= current_time.hour < self.ny_session_end):
            return

        # Only trade in Level III
        if self.level != 3:
            return

        # --- Entry Conditions ---

        # Long Entry
        adr_hit_long = self.data.Low[-1] < self.prev_adr_low[-1]
        price_away_from_emas_long = price < self.ema_fast[-1] and price < self.ema_slow[-1]
        reversal_pattern_long = is_bullish_engulfing(self.data)
        rsi_confirm_long = self.rsi[-1] < self.rsi_lower

        if adr_hit_long and price_away_from_emas_long and reversal_pattern_long and rsi_confirm_long:
            if not self.position:
                sl = self.data.Low[-2] * (1 - self.sl_buffer_pct)
                tp = self.ema_target[-1]
                if tp > price and sl < price:
                    self.buy(sl=sl, tp=tp)

        # Short Entry
        adr_hit_short = self.data.High[-1] > self.prev_adr_high[-1]
        price_away_from_emas_short = price > self.ema_fast[-1] and price > self.ema_slow[-1]
        reversal_pattern_short = is_bearish_engulfing(self.data)
        rsi_confirm_short = self.rsi[-1] > self.rsi_upper

        if adr_hit_short and price_away_from_emas_short and reversal_pattern_short and rsi_confirm_short:
            if not self.position:
                sl = self.data.High[-2] * (1 + self.sl_buffer_pct)
                tp = self.ema_target[-1]
                if tp < price and sl > price:
                    self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    import os
    import json
    import pandas as pd
    from backtesting import Backtest

    # --- Configuration ---
    data_path = 'data/BTC-USD-15m.csv'
    strategy_name = 'new_york_city_reversal_trade'
    output_json_path = f'results/temp_result.json'
    output_plot_path = f'results/{strategy_name}.html'

    # --- Data Loading and Preprocessing ---
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data.columns = [c.strip().title() for c in data.columns]

    # Preprocess the data to include daily levels
    data = preprocess_data(data)

    # --- Backtesting ---
    bt = Backtest(data, NewYorkCityReversalTrade, cash=100_000, commission=.002)
    stats = bt.run()

    # --- Results ---
    print(stats)

    # Save stats to JSON
    os.makedirs('results', exist_ok=True)

    # Save stats to JSON
    stats_dict = stats.to_dict()
    for key, value in stats_dict.items():
        if isinstance(value, pd.Timestamp):
            stats_dict[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            stats_dict[key] = str(value)

    # Remove non-serializable objects
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    with open(output_json_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)

    # Save plot
    bt.plot(filename=output_plot_path, open_browser=False)

    print(f"\nBacktest stats saved to: {output_json_path}")
    print(f"Backtest plot saved to: {output_plot_path}")