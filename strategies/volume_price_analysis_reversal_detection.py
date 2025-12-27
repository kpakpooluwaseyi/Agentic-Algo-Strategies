from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
import pandas_ta as ta
from enum import Enum

def generate_vpa_data(days=100):
    """
    Generates synthetic 15-minute data with a clear Volume Price Analysis (VPA)
    reversal pattern.
    - A sharp downtrend.
    - A high-volume buying climax candle.
    - A low-volume "mopping up" consolidation phase.
    - A high-volume bullish breakout confirmation candle.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range(start='2023-01-01', periods=days * 24 * 4, freq='15min')
    df = pd.DataFrame(index=dates)

    # Base price and volume
    price = 100
    volume = 100
    prices = [price]
    volumes = [volume]

    # Generate random walk for price and volume
    for _ in range(len(dates) - 1):
        price += rng.normal(0, 0.5)
        volume += rng.normal(0, 10)
        prices.append(max(50, price))
        volumes.append(max(10, volume))

    df['Close'] = pd.Series(prices, index=dates).rolling(window=5).mean()
    df['Volume'] = pd.Series(volumes, index=dates).rolling(window=5).mean()
    df.dropna(inplace=True)

    # --- Inject a textbook Bullish VPA Reversal Pattern ---
    pattern_day = days // 2
    start_time = pd.to_datetime(f'2023-01-01') + pd.Timedelta(days=pattern_day, hours=8)

    # 1. Downtrend
    for i in range(10):
        idx = start_time + pd.Timedelta(minutes=15 * i)
        df.loc[idx, 'Close'] = 100 - i * 1.0
        df.loc[idx, 'Volume'] = 100 + i * 5

    # 2. Buying Climax
    climax_idx = start_time + pd.Timedelta(minutes=15 * 10)
    df.loc[climax_idx, 'Close'] = 88 # Drops far but recovers slightly
    df.loc[climax_idx, 'Volume'] = 500 # Massive volume spike

    # 3. Mopping Up Phase (Congestion)
    for i in range(1, 6):
        idx = climax_idx + pd.Timedelta(minutes=15 * i)
        df.loc[idx, 'Close'] = 88.5 + rng.uniform(-0.5, 0.5)
        df.loc[idx, 'Volume'] = 40 # Very low volume

    # 4. Breakout Confirmation
    breakout_idx = climax_idx + pd.Timedelta(minutes=15 * 6)
    df.loc[breakout_idx, 'Close'] = 92 # Strong bullish candle
    df.loc[breakout_idx, 'Volume'] = 300 # High confirmation volume

    # Generate OHLC from Close and Volume
    df['Open'] = df['Close'].shift(1).fillna(df['Close'])
    spread = (df['Volume'] / df['Volume'].max()) * 0.01 * df['Close']
    df['High'] = df[['Open', 'Close']].max(axis=1) + spread
    df['Low'] = df[['Open', 'Close']].min(axis=1) - spread

    df.dropna(inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Define the states for the VPA pattern recognition
class VpaState(Enum):
    TREND_DETECTION = 0
    CLIMAX_WATCH = 1
    MOPPING_UP = 2
    BREAKOUT_WATCH = 3

class VolumePriceAnalysisReversalDetectionStrategy(Strategy):
    # --- Strategy Parameters ---
    trend_lookback = 20         # Number of bars to confirm a trend
    climax_vol_multiplier = 3.0 # Volume spike multiplier for climax detection
    mopping_up_period = 5       # Max bars for the consolidation phase
    mopping_up_vol_pct = 0.8    # Volume must be below this % of SMA during mopping up
    breakout_vol_multiplier = 2.0 # Volume spike multiplier for breakout confirmation
    risk_reward_ratio = 2.0     # Risk:Reward ratio for TP calculation
    sl_buffer_pct = 0.01        # Stop loss buffer percentage

    def init(self):
        # Calculate a Simple Moving Average of Volume to use as a baseline
        self.volume_sma = self.I(ta.sma, pd.Series(self.data.Volume), length=self.trend_lookback)

        # Initialize state machine and pattern variables
        self.state = VpaState.TREND_DETECTION
        self.trend_start_bar = 0
        self.climax_bar = None
        self.climax_price_extreme = None
        self.mopping_up_high = -np.inf
        self.mopping_up_low = np.inf
        self.trend_direction = 0 # 1 for uptrend, -1 for downtrend

    def _reset_state(self):
        """Resets the state machine to its initial state."""
        self.state = VpaState.TREND_DETECTION
        self.trend_start_bar = len(self.data.Close) -1
        self.climax_bar = None
        self.climax_price_extreme = None
        self.mopping_up_high = -np.inf
        self.mopping_up_low = np.inf
        self.trend_direction = 0

    def _is_trending(self):
        """Checks for a sustained trend (up or down)."""
        if len(self.data.Close) < self.trend_lookback:
            return 0

        current_slice = self.data.Close[-self.trend_lookback:]
        price_range = np.max(current_slice) - np.min(current_slice)

        # Simple trend detection: Is the price consistently moving?
        is_uptrend = self.data.Close[-1] > self.data.Close[-self.trend_lookback] and (self.data.Close[-1] - self.data.Close[-self.trend_lookback]) > price_range * 0.5
        is_downtrend = self.data.Close[-1] < self.data.Close[-self.trend_lookback] and (self.data.Close[-self.trend_lookback] - self.data.Close[-1]) > price_range * 0.5

        if is_uptrend:
            return 1
        if is_downtrend:
            return -1
        return 0

    def next(self):
        # Always manage open positions first
        if self.position:
            return

        # --- State Machine Logic ---
        if self.state == VpaState.TREND_DETECTION:
            self.trend_direction = self._is_trending()
            if self.trend_direction != 0:
                self.state = VpaState.CLIMAX_WATCH
                # print(f"{self.data.index[-1]}: Trend detected ({'Up' if self.trend_direction == 1 else 'Down'}). Watching for climax.")

        elif self.state == VpaState.CLIMAX_WATCH:
            is_high_volume = self.data.Volume[-1] > self.volume_sma[-1] * self.climax_vol_multiplier

            # Selling Climax (Top of Uptrend)
            if self.trend_direction == 1 and is_high_volume:
                self.climax_bar = len(self.data.Close) - 1
                self.climax_price_extreme = self.data.High[-1]
                self.state = VpaState.MOPPING_UP
                # print(f"{self.data.index[-1]}: Selling climax detected at {self.climax_price_extreme}. Entering mopping up phase.")

            # Buying Climax (Bottom of Downtrend)
            elif self.trend_direction == -1 and is_high_volume:
                self.climax_bar = len(self.data.Close) - 1
                self.climax_price_extreme = self.data.Low[-1]
                self.state = VpaState.MOPPING_UP
                # print(f"{self.data.index[-1]}: Buying climax detected at {self.climax_price_extreme}. Entering mopping up phase.")

            # If trend dissipates, reset
            elif self._is_trending() != self.trend_direction:
                self._reset_state()


        elif self.state == VpaState.MOPPING_UP:
            current_bar = len(self.data.Close) - 1
            bars_since_climax = current_bar - self.climax_bar

            # While in the mopping up period, monitor conditions and build the range
            if bars_since_climax <= self.mopping_up_period:
                is_low_volume = self.data.Volume[-1] < self.volume_sma[-1] * self.mopping_up_vol_pct

                # Invalidation conditions
                if not is_low_volume:
                    self._reset_state()
                    return
                if self.trend_direction == 1 and self.data.High[-1] > self.climax_price_extreme: # Short setup invalidation
                    self._reset_state()
                    return
                if self.trend_direction == -1 and self.data.Low[-1] < self.climax_price_extreme: # Long setup invalidation
                    self._reset_state()
                    return

                # Update the consolidation range
                if self.trend_direction == 1: # Short setup (uptrend climax)
                    self.mopping_up_low = min(self.mopping_up_low, self.data.Low[-1])
                    self.mopping_up_high = self.climax_price_extreme
                elif self.trend_direction == -1: # Long setup (downtrend climax)
                    self.mopping_up_high = max(self.mopping_up_high, self.data.High[-1])
                    self.mopping_up_low = self.climax_price_extreme

            # After the period, if not invalidated, move to breakout watch
            if bars_since_climax > self.mopping_up_period:
                self.state = VpaState.BREAKOUT_WATCH

        elif self.state == VpaState.BREAKOUT_WATCH:
            is_breakout_volume = self.data.Volume[-1] > self.volume_sma[-1] * self.breakout_vol_multiplier

            # Bearish Breakout (for Short Entry)
            if self.trend_direction == 1 and is_breakout_volume and self.data.Close[-1] < self.mopping_up_low:
                sl = self.climax_price_extreme * (1 + self.sl_buffer_pct)
                tp = self.data.Close[-1] - (sl - self.data.Close[-1]) * self.risk_reward_ratio
                if tp < self.data.Close[-1]: # Ensure TP is valid
                    self.sell(sl=sl, tp=tp)
                    # print(f"{self.data.index[-1]}: SELL ORDER PLACED. SL={sl}, TP={tp}")
                self._reset_state()

            # Bullish Breakout (for Long Entry)
            elif self.trend_direction == -1 and is_breakout_volume and self.data.Close[-1] > self.mopping_up_high:
                sl = self.climax_price_extreme * (1 - self.sl_buffer_pct)
                tp = self.data.Close[-1] + (self.data.Close[-1] - sl) * self.risk_reward_ratio
                if tp > self.data.Close[-1]: # Ensure TP is valid
                    self.buy(sl=sl, tp=tp)
                    # print(f"{self.data.index[-1]}: BUY ORDER PLACED. SL={sl}, TP={tp}")
                self._reset_state()

            # Invalidate if a breakout doesn't happen soon
            current_bar = len(self.data.Close) - 1
            if self.climax_bar is not None and (current_bar - self.climax_bar > self.mopping_up_period + 5): # 5 extra bars for breakout
                 self._reset_state()

if __name__ == '__main__':
    import os
    import json
    from backtesting import Backtest

    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        print(f"[Standardized Mode] Loading data from: {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Sanitize column names: remove leading/trailing spaces, trailing commas, and capitalize
        data.columns = [c.strip().replace(',', '').title() for c in data.columns]
    else:
        print("[Standalone Mode] Generating synthetic VPA data...")
        data = generate_vpa_data(days=200)

    # Ensure columns are in the correct format for the library
    if 'Volume' not in data.columns:
        # Find volume column ignoring case and handle potential extra chars
        vol_col = [c for c in data.columns if 'volume' in c.lower()]
        if vol_col:
            data.rename(columns={vol_col[0]: 'Volume'}, inplace=True)

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    bt = Backtest(data, VolumePriceAnalysisReversalDetectionStrategy, cash=10000, commission=.002)

    print("[Run Mode] Running single backtest with defaults...")
    stats = bt.run()

    # Save results
    os.makedirs('results', exist_ok=True)

    # Sanitize stats for JSON serialization
    def sanitize_stats(stats):
        sanitized = {}
        # Skip internal objects
        valid_keys = [k for k in stats.keys() if not k.startswith('_')]
        for key in valid_keys:
            value = stats[key]
            if isinstance(value, (pd.Series, pd.DataFrame)):
                sanitized[key] = None
            elif pd.isna(value):
                sanitized[key] = None
            elif isinstance(value, (np.int64, np.int32)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                sanitized[key] = float(value)
            elif isinstance(value, (pd.Timestamp)):
                 sanitized[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                sanitized[key] = str(value)
            else:
                sanitized[key] = value
        return sanitized

    clean_stats = sanitize_stats(stats)

    result_file = 'results/temp_result.json'
    with open(result_file, 'w') as f:
        json.dump(clean_stats, f, indent=2)

    print(f"Backtest stats saved to {result_file}")
    print(stats)

    try:
        plot_file = 'results/volume_price_analysis_reversal_detection.html'
        bt.plot(filename=plot_file)
        print(f"Plot saved to {plot_file}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
