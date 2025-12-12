from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
import os

from enum import Enum

# --- State Management ---
class TradeState(Enum):
    WAITING = "WAITING"
    MONITORING_SWEEP = "MONITORING_SWEEP"
    MSS_FVG_HUNT = "MSS_FVG_HUNT"

class IctAsianSweepLondonOpenStrategy(Strategy):
    # --- Strategy Parameters (can be optimized) ---
    fvg_sensitivity = 0.5
    mss_lookback = 10
    poi_proximity_pct = 0.1  # How close the sweep must be to the HTF POI (prev day H/L)

    def init(self):
        # --- Pre-calculated Indicators ---
        self.asia_high = self.I(lambda x: x, self.data.df['asia_high'].values)
        self.asia_low = self.I(lambda x: x, self.data.df['asia_low'].values)
        self.midnight_open = self.I(lambda x: x, self.data.df['midnight_open'].values)
        self.is_london = self.I(lambda x: x, self.data.df['is_london_session'].values)
        self.prev_day_high = self.I(lambda x: x, self.data.df['prev_day_high'].values)
        self.prev_day_low = self.I(lambda x: x, self.data.df['prev_day_low'].values)

        # --- State Machine ---
        self.trade_state = TradeState.WAITING
        self.manipulation_high = None
        self.manipulation_low = None
        self.mss_price = None

    def next(self):
        # --- Risk Management: End-of-Day Exit ---
        # Close any open positions at a specific time, e.g., 17:00 EST
        if self.position and self.data.index[-1].hour == 17:
            self.position.close()
            # Reset state after closing
            self.trade_state = TradeState.WAITING
            self.manipulation_high = None
            self.manipulation_low = None
            self.mss_price = None
            return # Exit for this bar

        # --- Daily State Reset ---
        # Reset at the beginning of each new day (e.g., at midnight)
        if len(self.data.index) > 1 and self.data.index[-1].date() != self.data.index[-2].date():
            self.trade_state = TradeState.WAITING
            self.manipulation_high = None
            self.manipulation_low = None
            self.mss_price = None

        # --- Core State Machine Logic ---

        # 1. WAITING State: Check for initial conditions during London session
        if self.trade_state == TradeState.WAITING and self.is_london[-1]:
            # Check proximity to HTF POI (Previous Day's High/Low)
            poi_high_proximity = abs(self.data.High[-1] - self.prev_day_high[-1]) / self.prev_day_high[-1] * 100
            poi_low_proximity = abs(self.data.Low[-1] - self.prev_day_low[-1]) / self.prev_day_low[-1] * 100

            # SHORT condition: Sweep AH, above MO, near PDH
            if self.data.High[-1] > self.asia_high[-1] and \
               self.data.Close[-1] > self.midnight_open[-1] and \
               poi_high_proximity <= self.poi_proximity_pct:
                self.trade_state = TradeState.MONITORING_SWEEP
                self.manipulation_high = self.data.High[-1]
                self.manipulation_low = None

            # LONG condition: Sweep AL, below MO, near PDL
            elif self.data.Low[-1] < self.asia_low[-1] and \
                 self.data.Close[-1] < self.midnight_open[-1] and \
                 poi_low_proximity <= self.poi_proximity_pct:
                self.trade_state = TradeState.MONITORING_SWEEP
                self.manipulation_low = self.data.Low[-1]
                self.manipulation_high = None

        # 2. MONITORING_SWEEP State: Find the peak/trough of the manipulation
        elif self.trade_state == TradeState.MONITORING_SWEEP:
            # If still in a short setup, keep updating the manipulation high
            if self.manipulation_high is not None:
                self.manipulation_high = max(self.manipulation_high, self.data.High[-1])
                # Check for MSS (Market Structure Shift)
                swing_low = self._find_recent_swing_low(self.mss_lookback)
                if swing_low is not None and self.data.Close[-1] < swing_low:
                    self.mss_price = swing_low
                    self.trade_state = TradeState.MSS_FVG_HUNT
                    # print(f"{self.data.index[-1]}: SHORT - MSS confirmed below {self.mss_price}. Hunting FVG.")

            # If in a long setup, keep updating the manipulation low
            elif self.manipulation_low is not None:
                self.manipulation_low = min(self.manipulation_low, self.data.Low[-1])
                # Check for MSS
                swing_high = self._find_recent_swing_high(self.mss_lookback)
                if swing_high is not None and self.data.Close[-1] > swing_high:
                    self.mss_price = swing_high
                    self.trade_state = TradeState.MSS_FVG_HUNT
                    # print(f"{self.data.index[-1]}: LONG - MSS confirmed above {self.mss_price}. Hunting FVG.")

        # 3. MSS_FVG_HUNT State: Look for FVG and place a trade
        elif self.trade_state == TradeState.MSS_FVG_HUNT:
            if self.position: return # Avoid placing multiple trades

            # SHORT entry: Look for a bearish FVG
            if self.manipulation_high is not None:
                fvg = self._find_fvg(-1)
                if fvg:
                    fvg_high, fvg_low = fvg
                    entry_price = fvg_high - (fvg_high - fvg_low) * self.fvg_sensitivity
                    sl = self.manipulation_high + 0.0002
                    tp = self.asia_low[-1]

                    if entry_price < sl and entry_price > tp and \
                       self._is_clear_path_to_tp(entry_price, tp, -1):
                        self.sell(limit=entry_price, sl=sl, tp=tp)
                        self.trade_state = TradeState.WAITING

            # LONG entry: Look for a bullish FVG
            elif self.manipulation_low is not None:
                fvg = self._find_fvg(1)
                if fvg:
                    fvg_high, fvg_low = fvg
                    entry_price = fvg_low + (fvg_high - fvg_low) * self.fvg_sensitivity
                    sl = self.manipulation_low - 0.0002
                    tp = self.asia_high[-1]

                    if entry_price > sl and entry_price < tp and \
                       self._is_clear_path_to_tp(entry_price, tp, 1):
                        self.buy(limit=entry_price, sl=sl, tp=tp)
                        self.trade_state = TradeState.WAITING

    # --- Helper Functions ---
    def _find_recent_swing_low(self, lookback):
        """Finds the most recent significant swing low."""
        # Simple implementation: lowest low in the lookback period BEFORE the peak
        # A more robust implementation would use peak/trough detection (e.g., scipy.signal)
        if len(self.data.Close) < lookback + 2: return None
        return np.min(self.data.Low[-lookback:-2])

    def _find_recent_swing_high(self, lookback):
        """Finds the most recent significant swing high."""
        if len(self.data.Close) < lookback + 2: return None
        return np.max(self.data.High[-lookback:-2])

    def _find_fvg(self, direction):
        """
        Identifies a Fair Value Gap (FVG) from the last 3 candles.
        direction: 1 for bullish (gap between candle 1 high and candle 3 low)
                  -1 for bearish (gap between candle 1 low and candle 3 high)
        Returns (fvg_high, fvg_low) tuple or None.
        """
        if len(self.data.Close) < 3: return None

        c1_high, c1_low = self.data.High[-3], self.data.Low[-3]
        c2_high, c2_low = self.data.High[-2], self.data.Low[-2]
        c3_high, c3_low = self.data.High[-1], self.data.Low[-1]

        # Bearish FVG: Gap between candle 1's low and candle 3's high
        if direction == -1 and c1_low > c3_high:
            # Also ensure the middle candle is strong (displacement)
            if (self.data.Open[-2] - self.data.Close[-2]) > 0: # Is a bearish candle
                return (c1_low, c3_high) # fvg_high, fvg_low

        # Bullish FVG: Gap between candle 1's high and candle 3's low
        if direction == 1 and c1_high < c3_low:
            if (self.data.Close[-2] - self.data.Open[-2]) > 0: # Is a bullish candle
                return (c3_low, c1_high) # fvg_high, fvg_low

        return None

    def _is_clear_path_to_tp(self, entry_price, tp, direction):
        """
        LRLC Proxy: Checks if there are significant swing points between entry and TP.
        Looks back over the last `mss_lookback` bars to find swings.
        """
        lookback_period = self.data.df.iloc[-self.mss_lookback:]

        if direction == -1: # Short
            # Find any swing lows between entry and TP
            resisting_lows = lookback_period[lookback_period['Low'] < entry_price]
            resisting_lows = resisting_lows[resisting_lows['Low'] > tp]
            return len(resisting_lows) == 0

        if direction == 1: # Long
            # Find any swing highs between entry and TP
            resisting_highs = lookback_period[lookback_period['High'] > entry_price]
            resisting_highs = resisting_highs[resisting_highs['High'] < tp]
            return len(resisting_highs) == 0

        return True

if __name__ == '__main__':
    def generate_forex_data(days=200, timeframe_minutes=15):
        """
        Generates synthetic 24-hour Forex data for backtesting ICT strategies.
        - Time is in EST, which is crucial for session definitions.
        - Simulates lower volatility during Asian session and higher during London.
        """
        rng = np.random.default_rng(42)

        # All times are in EST (UTC-5)
        start_date = '2023-01-01 00:00:00-05:00'

        periods_per_day = 24 * (60 // timeframe_minutes)
        total_periods = days * periods_per_day

        timestamps = pd.date_range(start_date, periods=total_periods, freq=f'{timeframe_minutes}min')

        price = 1.0500
        # Base random walk for price
        returns = rng.normal(loc=0, scale=0.0001, size=total_periods)
        prices = price * (1 + returns).cumprod()

        df = pd.DataFrame(index=timestamps, data={'Close': prices})
        df.index.name = 'Timestamp'

        # --- Simulate session-based characteristics ---
        # Asian Session (20:00 - 00:00 EST): Lower volatility, tends to be range-bound.
        asia_mask = (df.index.hour >= 20) | (df.index.hour == 0)

        # London Session (01:30 - 04:00 EST): Higher volatility, where sweeps often occur.
        london_mask = (df.index.hour >= 1) & (df.index.hour < 5)

        # Create price drifts to simulate session volatility
        # We use cumsum() on small random numbers to create gentle drifts
        df.loc[asia_mask, 'drift'] = rng.normal(loc=0, scale=0.00005, size=asia_mask.sum())
        df.loc[london_mask, 'drift'] = rng.normal(loc=0, scale=0.00020, size=london_mask.sum())
        df['drift'] = df['drift'].fillna(0)

        # Apply the drift cumulatively within each session block
        df['Close'] += df.groupby(df.index.date)['drift'].cumsum()

        # --- Generate OHLC data ---
        df['Open'] = df['Close'].shift(1).fillna(df['Close'])

        # Add random wicks based on session volatility
        wick_volatility = 0.0001  # Default
        wick_volatility_asia = 0.00015
        wick_volatility_london = 0.0004

        wicks = pd.Series(wick_volatility, index=df.index)
        wicks[asia_mask] = wick_volatility_asia
        wicks[london_mask] = wick_volatility_london

        high_wicks = rng.uniform(0, 1, size=total_periods) * wicks
        low_wicks = rng.uniform(0, 1, size=total_periods) * wicks

        df['High'] = df[['Open', 'Close']].max(axis=1) + high_wicks
        df['Low'] = df[['Open', 'Close']].min(axis=1) - low_wicks

        # Ensure OHLC consistency
        df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
        df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)

        df = df.drop(columns=['drift'])

        return df.dropna()

    def preprocess_data(df):
        """
        Calculates and adds session-based indicators to the DataFrame.
        - Asian Range (High/Low)
        - Midnight Open (00:00 EST)
        - London Session Flag
        """
        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # --- Calculate Daily Metrics ---
        # We need to find the previous day's 20:00 to the current day's 00:00
        # This is complex with standard pandas rolling windows. A robust
        # way is to iterate through each day and calculate the metrics.

        daily_metrics = []
        # Calculate previous day's high/low
        daily_high = df['High'].resample('D').max()
        daily_low = df['Low'].resample('D').min()
        prev_day_high = daily_high.shift(1)
        prev_day_low = daily_low.shift(1)

        for day in df.index.normalize().unique():
            # Define the Asian session for this specific day
            asia_start = day + pd.Timedelta(hours=20) - pd.Timedelta(days=1)
            asia_end = day

            midnight_open_time = day

            asia_session_df = df[(df.index >= asia_start) & (df.index < asia_end)]

            asia_high = asia_session_df['High'].max()
            asia_low = asia_session_df['Low'].min()

            try:
                midnight_open = df.loc[midnight_open_time, 'Open']
            except KeyError:
                midnight_open = np.nan

            daily_metrics.append({
                'date': day.date(),
                'asia_high': asia_high,
                'asia_low': asia_low,
                'midnight_open': midnight_open,
            })

        metrics_df = pd.DataFrame(daily_metrics).set_index('date')
        # metrics_df.index = pd.to_datetime(metrics_df.index)
        metrics_df['prev_day_high'] = prev_day_high.values
        metrics_df['prev_day_low'] = prev_day_low.values

        # --- Map Metrics to the Main DataFrame ---
        df['date'] = df.index.date
        df = df.join(metrics_df, on='date')

        # --- Define London Trading Window ---
        # 1:30 AM EST to 4:00 AM EST
        df['is_london_session'] = (
            (df.index.hour >= 1) & (df.index.minute >= 30) | (df.index.hour > 1)
        ) & (
            df.index.hour < 4
        )

        return df.drop(columns=['date']).dropna()


    # --- Main Execution ---

    # Load and preprocess data
    # Using 5-minute data for more granular entry signals as per strategy description
    data = generate_forex_data(days=200, timeframe_minutes=5)
    data = preprocess_data(data.copy())

    # Run backtest
    bt = Backtest(data, IctAsianSweepLondonOpenStrategy, cash=10000, commission=.002)

    # Optimize
    stats = bt.optimize(
        fvg_sensitivity=list(np.arange(0.1, 1.0, 0.2)),
        mss_lookback=range(5, 20, 5),
        poi_proximity_pct=list(np.arange(0.05, 0.5, 0.1)),
        maximize='Sharpe Ratio'
    )

    print("Optimization Results:")
    print(stats)

    # --- Result Sanitization and Saving ---
    def sanitize_stats(stats):
        """Prepares stats for JSON serialization."""
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (np.integer, np.int64)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                sanitized[key] = float(value)
            elif isinstance(value, pd.Series):
                # For _equity_curve and _trades, we might want specific handling
                # For now, let's just exclude them or provide a summary.
                sanitized[key] = None # Or some other JSON-friendly representation
            elif isinstance(value, pd.DataFrame):
                sanitized[key] = None
            elif isinstance(value, pd.Timestamp):
                 sanitized[key] = value.isoformat()
            else:
                sanitized[key] = value
        return sanitized

    # Save results
    os.makedirs('results', exist_ok=True)

    # We need to access the stats from the Series returned by optimize
    result_data = {
        'strategy_name': 'ict_asian_sweep_london_open',
        'return': stats.get('Return [%]', 0.0),
        'sharpe': stats.get('Sharpe Ratio', 0.0),
        'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
        'win_rate': stats.get('Win Rate [%]', 0.0),
        'total_trades': stats.get('# Trades', 0)
    }

    with open('results/temp_result.json', 'w') as f:
        # Use a recursive sanitizer in a real scenario, but this is fine for the required keys.
        json.dump(sanitize_stats(result_data), f, indent=2)

    print("\nResults saved to results/temp_result.json")

    # Generate plot
    try:
        bt.plot()
        print("Plot saved to current directory.")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")
