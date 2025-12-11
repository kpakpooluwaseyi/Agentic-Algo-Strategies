from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import json

def EMA(array, n):
    """Custom EMA function to use with self.I"""
    return pd.Series(array).ewm(span=n, adjust=False).mean().values

class Fibonacci50ReversalScalpStrategy(Strategy):
    # --- Strategy Parameters ---
    ema_period = 50        # Period for the trend-following EMA
    peak_prominence = 5    # Prominence for detecting swing points (A, B, C)
    min_ab_range = 1.0     # Minimum price range between swing A and B
    max_c_retracement = 0.6  # Max retracement for C relative to A-B (e.g., 0.618)
    min_c_retracement = 0.4  # Min retracement for C relative to A-B (e.g., 0.5)
    lod_confluence = 0.01  # Proximity to LOD for confluence (as percentage)
    min_rr = 2.0           # Minimum Risk-to-Reward ratio

    def init(self):
        # --- Indicators ---
        self.ema = self.I(EMA, self.data.Close, self.ema_period)
        self.asia_high = self.data.Asia_High
        self.asia_low = self.data.Asia_Low

        # --- Daily State ---
        self.current_day = -1
        self.day_low = np.inf

        # --- State Machine ---
        self.state = 'SCANNING' # SCANNING, FOUND_A, FOUND_B, FOUND_C
        self.pattern_type = None # 'M' or 'W'

        # --- Swing Points ---
        self.A = None
        self.B = None
        self.C = None

        # Pre-calculate peaks and troughs for the entire dataset
        self.highs = self.data.High
        self.lows = self.data.Low

        self.peak_indices, self.trough_indices = self.find_swing_points(self.highs, self.lows, n=self.peak_prominence)

    def find_swing_points(self, highs, lows, n=5):
        """
        Identifies swing highs and lows using a simple rolling window.
        A peak is a value that's the highest in a window of size (2n+1).
        A trough is a value that's the lowest in a window of size (2n+1).
        Returns two sets of indices: peaks and troughs.
        """
        highs_series = pd.Series(highs)
        lows_series = pd.Series(lows)

        # Find peaks
        rolling_max = highs_series.rolling(window=2*n+1, center=True).max()
        peak_indices = np.where(highs_series == rolling_max)[0]

        # Find troughs
        rolling_min = lows_series.rolling(window=2*n+1, center=True).min()
        trough_indices = np.where(lows_series == rolling_min)[0]

        return set(peak_indices), set(trough_indices)

    def next(self):
        # --- Multi-Timeframe Simulation (Conceptual) ---
        # The logic here runs on the 15M data, but we'd conceptually
        # "drop to 1M" for entry confirmation. In this backtest, we'll
        # use a simple confirmation like a bearish/bullish candle.

        # --- Daily State Update ---
        today = self.data.index[-1].day
        if today != self.current_day:
            self.current_day = today
            self.day_low = self.data.Low[-1]
        else:
            self.day_low = min(self.day_low, self.data.Low[-1])

        current_index = len(self.data.Close) - 1

        # =====================================================================
        # STEP 1: SCANNING FOR PATTERNS (A and B points)
        # =====================================================================
        if self.state == 'SCANNING':
            # --- Look for M-Pattern Start (Peak A) ---
            if current_index in self.peak_indices:
                self.A = (current_index, self.highs[current_index])
                self.state = 'FOUND_A'
                self.pattern_type = 'M'

            # --- Look for W-Pattern Start (Trough A) ---
            elif current_index in self.trough_indices:
                self.A = (current_index, self.lows[current_index])
                self.state = 'FOUND_A'
                self.pattern_type = 'W'

        # =====================================================================
        # STEP 2: LOOKING FOR POINT B
        # =====================================================================
        elif self.state == 'FOUND_A':
            # --- M-Pattern: Look for Trough B ---
            if self.pattern_type == 'M' and current_index in self.trough_indices:
                # B must be after A
                if current_index > self.A[0]:
                    # Check if the A-B drop is significant enough
                    if self.A[1] - self.lows[current_index] >= self.min_ab_range:
                        self.B = (current_index, self.lows[current_index])
                        self.state = 'FOUND_B'
                    else: # Drop not significant, reset
                        self.state = 'SCANNING'

            # --- W-Pattern: Look for Peak B ---
            elif self.pattern_type == 'W' and current_index in self.peak_indices:
                 # B must be after A
                if current_index > self.A[0]:
                    # Check if the A-B rise is significant enough
                    if self.highs[current_index] - self.A[1] >= self.min_ab_range:
                        self.B = (current_index, self.highs[current_index])
                        self.state = 'FOUND_B'
                    else: # Rise not significant, reset
                        self.state = 'SCANNING'

            # Invalidation: Price makes a new high/low beyond A
            if self.pattern_type == 'M' and self.highs[current_index] > self.A[1]:
                self.state = 'SCANNING'
            elif self.pattern_type == 'W' and self.lows[current_index] < self.A[1]:
                self.state = 'SCANNING'

        # =====================================================================
        # STEP 3: LOOKING FOR POINT C & ENTRY
        # =====================================================================
        elif self.state == 'FOUND_B':
            ab_range = abs(self.A[1] - self.B[1])
            fib_50 = self.B[1] + ab_range * 0.5 if self.pattern_type == 'M' else self.B[1] - ab_range * 0.5
            fib_min = self.B[1] + ab_range * self.min_c_retracement if self.pattern_type == 'M' else self.B[1] - ab_range * self.min_c_retracement
            fib_max = self.B[1] + ab_range * self.max_c_retracement if self.pattern_type == 'M' else self.B[1] - ab_range * self.max_c_retracement

            # --- M-Pattern: Wait for price to retrace up to 50% level ---
            if self.pattern_type == 'M':
                # Check if current price is in the 50% zone
                if fib_min <= self.highs[current_index] <= fib_max:
                     # This is our potential C point, the peak of the retracement
                    self.C = (current_index, self.highs[current_index])

                    # --- ENTRY LOGIC ---
                    # Confluence Checks:
                    is_below_ema = self.data.Close[-1] < self.ema[-1]
                    is_near_asia_high = abs(self.C[1] - self.asia_high[-1]) / self.C[1] < self.lod_confluence

                    # Confirmation:
                    is_reversal_candle = self.data.Close[-1] < self.data.Open[-1]

                    if is_below_ema and is_near_asia_high and is_reversal_candle:

                        # Calculate TP and SL
                        bc_range = abs(self.C[1] - self.B[1])
                        take_profit = self.C[1] - bc_range * 0.5
                        stop_loss = self.C[1] + (self.C[1] * 0.005) # SL slightly above C

                        # --- Risk-to-Reward Check ---
                        risk = abs(self.data.Close[-1] - stop_loss)
                        reward = abs(self.data.Close[-1] - take_profit)
                        if risk > 0 and reward / risk >= self.min_rr:
                            self.sell(sl=stop_loss, tp=take_profit)

                        self.state = 'SCANNING' # Reset after trade

            # --- W-Pattern: Wait for price to retrace down to 50% level ---
            elif self.pattern_type == 'W':
                 # Check if current price is in the 50% zone
                if fib_min <= self.lows[current_index] <= fib_max:
                    self.C = (current_index, self.lows[current_index])

                    # --- ENTRY LOGIC ---
                    # Confluence Checks:
                    is_above_ema = self.data.Close[-1] > self.ema[-1]
                    is_near_lod = abs(self.C[1] - self.day_low) / self.C[1] < self.lod_confluence
                    is_near_asia_low = abs(self.C[1] - self.asia_low[-1]) / self.C[1] < self.lod_confluence

                    # Confirmation:
                    is_reversal_candle = self.data.Close[-1] > self.data.Open[-1]

                    if is_above_ema and (is_near_lod or is_near_asia_low) and is_reversal_candle:

                        # Calculate TP and SL
                        bc_range = abs(self.C[1] - self.B[1])
                        take_profit = self.C[1] + bc_range * 0.5
                        stop_loss = self.C[1] - (self.C[1] * 0.005) # SL slightly below C

                        # --- Risk-to-Reward Check ---
                        risk = abs(self.data.Close[-1] - stop_loss)
                        reward = abs(self.data.Close[-1] - take_profit)
                        if risk > 0 and reward / risk >= self.min_rr:
                            self.buy(sl=stop_loss, tp=take_profit)

                        self.state = 'SCANNING' # Reset after trade

            # Invalidation: Price breaks beyond point B before C is formed
            if self.pattern_type == 'M' and self.lows[current_index] < self.B[1]:
                self.state = 'SCANNING'
            elif self.pattern_type == 'W' and self.highs[current_index] > self.B[1]:
                self.state = 'SCANNING'

def preprocess_data(data, ema_period=50, asia_session_start_hour=20, asia_session_end_hour=5):
    """
    Adds EMA and Asia Session High/Low to the data.
    """
    # Calculate EMA
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()

    # Calculate Asia Session High/Low
    # Use UTC for session times
    data.index = data.index.tz_localize('UTC')

    # Identify session candles
    is_asia_session = (data.index.hour >= asia_session_start_hour) | (data.index.hour < asia_session_end_hour)

    # Create a daily session ID
    session_id = (data.index - pd.Timedelta(hours=asia_session_start_hour)).date

    # Calculate session highs and lows
    asia_session_high = data['High'][is_asia_session].groupby(session_id[is_asia_session]).max()
    asia_session_low = data['Low'][is_asia_session].groupby(session_id[is_asia_session]).min()

    # Map session levels to the entire day
    data['Asia_High'] = pd.Series(session_id).map(asia_session_high).ffill()
    data['Asia_Low'] = pd.Series(session_id).map(asia_session_low).ffill()

    data.index = data.index.tz_localize(None) # Remove timezone for backtesting.py compatibility

    return data

def generate_synthetic_data(num_candles=2000):
    """
    Generates synthetic OHLCV data with M and W patterns.
    """
    np.random.seed(42)
    # Use a 24h frequency to make session data meaningful
    time_index = pd.date_range(start='2023-01-01', periods=num_candles, freq='15min')
    price = np.zeros(num_candles)

    # Baseline random walk
    price[0] = 100
    price[1:] = 100 + np.random.randn(num_candles-1).cumsum() * 0.1

    # Inject M pattern (indices 200-280)
    # A (peak)
    price[200:220] = price[199] + np.linspace(0, 10, 20)
    # B (trough)
    price[220:240] = price[219] - np.linspace(0, 5, 20)
    # C (lower peak, corrected to be within 0.4-0.6 retracement)
    price[240:260] = price[239] + np.linspace(0, 3, 20) # Retraces to ~108
    # D (breakdown)
    price[260:280] = price[259] - np.linspace(0, 15, 20)

    # Inject W pattern (indices 500-580)
    # A (trough)
    price[500:520] = price[499] - np.linspace(0, 10, 20)
    # B (peak)
    price[520:540] = price[519] + np.linspace(0, 5, 20)
    # C (higher trough)
    price[540:560] = price[539] - np.linspace(0, 7, 20)
    # D (breakout)
    price[560:580] = price[559] + np.linspace(0, 15, 20)

    data = pd.DataFrame(index=time_index)
    data['Open'] = price
    data['High'] = data['Open'] + np.random.uniform(0, 1, num_candles)
    data['Low'] = data['Open'] - np.random.uniform(0, 1, num_candles)
    data['Close'] = data['Open'] + np.random.uniform(-0.5, 0.5, num_candles)
    data['Volume'] = np.random.randint(100, 1000, num_candles)

    # Ensure OHLC consistency
    data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 0.5, num_candles)
    data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 0.5, num_candles)

    return data

if __name__ == '__main__':
    # Load or generate data
    data = generate_synthetic_data(4000) # Increased size for more pattern opportunities
    data = preprocess_data(data)

    # Run backtest
    bt = Backtest(data, Fibonacci50ReversalScalpStrategy, cash=10000, commission=.002)

    # Optimize
    stats = bt.optimize(
        ema_period=range(20, 101, 10),
        peak_prominence=range(3, 16, 2),
        min_ab_range=list(np.arange(0.5, 3.1, 0.5)), # Convert np.arange to a list
        maximize='Sharpe Ratio'
    )

    # Save results
    import os
    os.makedirs('results', exist_ok=True)

    # Handle cases with no trades
    if stats['# Trades'] == 0:
        results = {
            'strategy_name': 'fibonacci_50_reversal_scalp',
            'return': 0.0,
            'sharpe': None,
            'max_drawdown': 0.0,
            'win_rate': None,
            'total_trades': 0
        }
    else:
        results = {
            'strategy_name': 'fibonacci_50_reversal_scalp',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }

    with open('results/temp_result.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate plot
    bt.plot()
