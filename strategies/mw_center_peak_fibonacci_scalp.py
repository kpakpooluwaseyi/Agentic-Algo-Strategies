from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json

def generate_synthetic_data(num_candles=1000):
    """Generates synthetic data with M and W patterns."""
    np.random.seed(42)
    data = np.random.randn(num_candles, 4).cumsum(axis=0) + 100
    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'])

    # Make OHLC data realistic
    df['Open'] = df['Close'].shift(1)
    df.iloc[0, 0] = df.iloc[0, 3] # Set first open to first close
    min_vals = df[['Open', 'Close']].min(axis=1)
    max_vals = df[['Open', 'Close']].max(axis=1)
    df['Low'] = min_vals - np.random.uniform(0, 2, size=len(df))
    df['High'] = max_vals + np.random.uniform(0, 2, size=len(df))

    # Inject M-pattern (for SHORT)
    # P1 (High) -> P2 (Low) -> P3 (Center Peak)
    df.loc[100:105, 'High'] = np.linspace(110, 105, 6)
    df.loc[100:105, 'Low'] = np.linspace(108, 100, 6)
    df.loc[100:105, 'Open'] = np.linspace(110, 102, 6)
    df.loc[100:105, 'Close'] = np.linspace(109, 101, 6)

    df.loc[106:110, 'High'] = np.linspace(102, 107, 5) # P3 Peak at 107
    df.loc[106:110, 'Low'] = np.linspace(100, 104, 5)
    df.loc[106:110, 'Open'] = np.linspace(101, 106, 5)
    df.loc[106:110, 'Close'] = np.linspace(102, 105, 5)

    df.loc[111:115, 'High'] = np.linspace(106, 100, 5)
    df.loc[111:115, 'Low'] = np.linspace(104, 98, 5)
    df.loc[111:115, 'Open'] = np.linspace(105, 99, 5)
    df.loc[111:115, 'Close'] = np.linspace(104, 99, 5)

    # Inject W-pattern (for LONG)
    # P1 (Low) -> P2 (High) -> P3 (Center Trough)
    df.loc[200:205, 'High'] = np.linspace(95, 90, 6)
    df.loc[200:205, 'Low'] = np.linspace(92, 85, 6) # P1 Trough at 85
    df.loc[200:205, 'Open'] = np.linspace(94, 88, 6)
    df.loc[200:205, 'Close'] = np.linspace(93, 86, 6)

    df.loc[206:210, 'High'] = np.linspace(88, 93, 5) # P2 Peak at 93
    df.loc[206:210, 'Low'] = np.linspace(86, 90, 5)
    df.loc[206:210, 'Open'] = np.linspace(86, 92, 5)
    df.loc[206:210, 'Close'] = np.linspace(87, 91, 5)

    df.loc[211:215, 'High'] = np.linspace(92, 88, 5)
    df.loc[211:215, 'Low'] = np.linspace(90, 86, 5) # P3 Trough at 86
    df.loc[211:215, 'Open'] = np.linspace(91, 87, 5)
    df.loc[211:215, 'Close'] = np.linspace(90, 88, 5)

    df.index = pd.to_datetime(pd.date_range('2020-01-01', periods=num_candles, freq='15min'))
    df = df.dropna()
    return df

def EMA(series, n):
    """Returns the EMA of a series."""
    return pd.Series(series).ewm(span=n, min_periods=n).mean()

def resample_apply(timeframe, func, series, *args, **kwargs):
    """
    Resamples the series to a different timeframe, applies a function,
    and then upsamples back to the original timeframe.
    """
    resampled = series.resample(timeframe).last()
    applied = func(resampled, *args, **kwargs)
    return applied.reindex(series.index, method='ffill')

def sanitize_stats_for_json(stats):
    """Recursively converts numpy number types to native Python types."""
    if isinstance(stats, dict):
        return {k: sanitize_stats_for_json(v) for k, v in stats.items()}
    elif isinstance(stats, list):
        return [sanitize_stats_for_json(v) for v in stats]
    elif isinstance(stats, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
        return int(stats)
    elif isinstance(stats, (np.float_, np.float16, np.float32, np.float64)):
        return float(stats)
    elif isinstance(stats, (np.ndarray,)):
        return stats.tolist()
    elif isinstance(stats, pd.Series):
        return sanitize_stats_for_json(stats.to_dict())
    return stats

class MwCenterPeakFibonacciScalpStrategy(Strategy):
    # Optimizable parameters
    ema_period = 20
    risk_reward_ratio = 5
    ema_confirmation_threshold = 0.005 # e.g., 0.5%
    htf_ema_period = 20 # Using a fixed period for the HTF EMA

    def init(self):
        # Indicators
        # Simulate HTF (1H) EMA on the current (15M) timeframe
        self.ema = self.I(resample_apply, '1H', EMA, self.data.Close.s, self.htf_ema_period)

        # State tracking for M-formation (short)
        self.m_point1_price = None
        self.m_point1_idx = None
        self.m_point2_price = None
        self.m_point2_idx = None
        self.m_point3_price = None
        self.m_point3_idx = None
        self.m_setup_active = False

        # State tracking for W-formation (long)
        self.w_point1_price = None
        self.w_point1_idx = None
        self.w_point2_price = None
        self.w_point2_idx = None
        self.w_point3_price = None
        self.w_point3_idx = None
        self.w_setup_active = False

    def _is_swing(self, lookback, high_or_low):
        """Checks if a point `lookback` bars ago is a swing high or low."""
        if len(self.data.Close) < lookback + 3:
            return False

        price_data = self.data.High if high_or_low == 'high' else self.data.Low

        # A swing point is a local max/min
        is_highest = price_data[-lookback-1] > price_data[-lookback-2] and \
                     price_data[-lookback-1] > price_data[-lookback]
        is_lowest = price_data[-lookback-1] < price_data[-lookback-2] and \
                    price_data[-lookback-1] < price_data[-lookback]

        if high_or_low == 'high' and is_highest:
            return True
        if high_or_low == 'low' and is_lowest:
            return True
        return False

    def next(self):
        current_price = self.data.Close[-1]

        # --- M-FORMATION LOGIC (SHORT) ---
        if not self.position and not self.m_setup_active:
            # Step 1: Find Point 1 (a swing high)
            if self.m_point1_price is None:
                if self._is_swing(1, 'high'):
                    self.m_point1_price = self.data.High[-2]
                    self.m_point1_idx = len(self.data.Close) - 2

            # Step 2: Find Point 2 (a swing low after Point 1)
            elif self.m_point2_price is None:
                if self._is_swing(1, 'low'):
                    if self.data.Low[-2] < self.m_point1_price:
                        self.m_point2_price = self.data.Low[-2]
                        self.m_point2_idx = len(self.data.Close) - 2
                    else: # Invalid pattern, reset
                        self._reset_m_state()

            # Step 3: Find Point 3 (Center Peak) and validate AOI
            elif self.m_point3_price is None:
                if self._is_swing(1, 'high'):
                    p3_candidate_price = self.data.High[-2]

                    # P3 must be lower than P1
                    if p3_candidate_price >= self.m_point1_price:
                        self._reset_m_state()
                        return

                    # Calculate Fib levels for AOI
                    fib_level_1 = self.m_point1_price - self.m_point2_price
                    fib_50 = self.m_point2_price + fib_level_1 * 0.5
                    fib_61_8 = self.m_point2_price + fib_level_1 * 0.618
                    fib_78_6 = self.m_point2_price + fib_level_1 * 0.786

                    # Validate retracement
                    if p3_candidate_price > fib_78_6:
                        self._reset_m_state() # Invalidated, retraced too far
                    elif p3_candidate_price >= fib_50:
                        # EMA Confirmation
                        ema_at_p3 = self.ema[-(len(self.data.Close) - (len(self.data.Close) - 2))]
                        price_distance = abs(p3_candidate_price - ema_at_p3)

                        if price_distance / p3_candidate_price <= self.ema_confirmation_threshold:
                            self.m_point3_price = p3_candidate_price
                            self.m_point3_idx = len(self.data.Close) - 2
                            self.m_setup_active = True # Setup is now active, wait for entry
                        else:
                            # EMA not close enough, could be a new P1
                            self._reset_m_state()
                            self.m_point1_price = p3_candidate_price
                            self.m_point1_idx = len(self.data.Close) - 2

        # Step 4: M-Formation Entry Logic
        if self.m_setup_active:
            # Entry confirmation: a bearish candle after P3
            is_bearish_candle = self.data.Close[-1] < self.data.Open[-1]

            if is_bearish_candle:
                entry_price = self.data.Close[-1]
                stop_loss = self.m_point3_price * 1.001 # SL just above center peak

                # Calculate TP based on the 50% Fib of the P2-P3 move
                fib_level_2 = self.m_point3_price - self.m_point2_price
                take_profit = self.m_point3_price - fib_level_2 * 0.5

                # Validate R:R and order levels
                risk = abs(stop_loss - entry_price)
                reward = abs(entry_price - take_profit)

                if (risk > 0 and reward / risk >= self.risk_reward_ratio and
                    entry_price < stop_loss and take_profit < entry_price):
                    self.sell(sl=stop_loss, tp=take_profit)

                # Reset whether trade was taken or not
                self._reset_m_state()

        # --- W-FORMATION LOGIC (LONG) ---
        if not self.position and not self.w_setup_active:
            # Step 1: Find Point 1 (a swing low)
            if self.w_point1_price is None:
                if self._is_swing(1, 'low'):
                    self.w_point1_price = self.data.Low[-2]
                    self.w_point1_idx = len(self.data.Close) - 2

            # Step 2: Find Point 2 (a swing high after Point 1)
            elif self.w_point2_price is None:
                if self._is_swing(1, 'high'):
                    if self.data.High[-2] > self.w_point1_price:
                        self.w_point2_price = self.data.High[-2]
                        self.w_point2_idx = len(self.data.Close) - 2
                    else: # Invalid pattern
                        self._reset_w_state()

            # Step 3: Find Point 3 (Center Trough) and validate AOI
            elif self.w_point3_price is None:
                if self._is_swing(1, 'low'):
                    p3_candidate_price = self.data.Low[-2]

                    if p3_candidate_price <= self.w_point1_price:
                        self._reset_w_state()
                        return

                    fib_level_1 = self.w_point2_price - self.w_point1_price
                    fib_50 = self.w_point1_price + fib_level_1 * (1 - 0.5)
                    fib_61_8 = self.w_point1_price + fib_level_1 * (1 - 0.618)
                    fib_78_6 = self.w_point1_price + fib_level_1 * (1 - 0.786)

                    if p3_candidate_price < fib_78_6:
                        self._reset_w_state()
                    elif p3_candidate_price <= fib_50:
                        # EMA Confirmation
                        ema_at_p3 = self.ema[-(len(self.data.Close) - (len(self.data.Close) - 2))]
                        price_distance = abs(p3_candidate_price - ema_at_p3)

                        if price_distance / p3_candidate_price <= self.ema_confirmation_threshold:
                            self.w_point3_price = p3_candidate_price
                            self.w_point3_idx = len(self.data.Close) - 2
                            self.w_setup_active = True
                        else:
                            self._reset_w_state()
                            self.w_point1_price = p3_candidate_price
                            self.w_point1_idx = len(self.data.Close) - 2

        # Step 4: W-Formation Entry Logic
        if self.w_setup_active:
            is_bullish_candle = self.data.Close[-1] > self.data.Open[-1]

            if is_bullish_candle:
                entry_price = self.data.Close[-1]
                stop_loss = self.w_point3_price * 0.999

                fib_level_2 = self.w_point2_price - self.w_point3_price
                take_profit = self.w_point3_price + fib_level_2 * 0.5

                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)

                if (risk > 0 and reward / risk >= self.risk_reward_ratio and
                    entry_price > stop_loss and take_profit > entry_price):
                    self.buy(sl=stop_loss, tp=take_profit)

                self._reset_w_state()

    def _reset_m_state(self):
        self.m_point1_price = None
        self.m_point1_idx = None
        self.m_point2_price = None
        self.m_point2_idx = None
        self.m_point3_price = None
        self.m_point3_idx = None
        self.m_setup_active = False

    def _reset_w_state(self):
        self.w_point1_price = None
        self.w_point1_idx = None
        self.w_point2_price = None
        self.w_point2_idx = None
        self.w_point3_price = None
        self.w_point3_idx = None
        self.w_setup_active = False

if __name__ == '__main__':
    # Load or generate data
    data = generate_synthetic_data(500)

    # Run backtest
    bt = Backtest(data, MwCenterPeakFibonacciScalpStrategy, cash=10000, commission=.002)

    # Optimize
    stats = bt.optimize(risk_reward_ratio=range(3, 8, 1),
                         ema_confirmation_threshold=list(np.arange(0.001, 0.01, 0.002)),
                         maximize='Sharpe Ratio')

    # Save results
    import os
    os.makedirs('results', exist_ok=True)

    # Sanitize stats for JSON serialization
    sanitized_stats = {
        'strategy_name': 'mw_center_peak_fibonacci_scalp',
        'return': float(stats.get('Return [%]', 0.0)),
        'sharpe': float(stats.get('Sharpe Ratio', 0.0)),
        'max_drawdown': float(stats.get('Max. Drawdown [%]', 0.0)),
        'win_rate': float(stats.get('Win Rate [%]', 0.0)),
        'total_trades': int(stats.get('# Trades', 0))
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=2)

    # Generate plot
    try:
        bt.plot()
    except TypeError as e:
        print(f"Could not generate plot due to a known issue with the library: {e}")
        print("Continuing without the plot...")
