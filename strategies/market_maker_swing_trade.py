
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from scipy.signal import find_peaks
import json
import os

# --- Helper Functions & Pre-processing ---

def _find_recent_swings(highs, lows, distance):
    """
    Finds and returns alternating swing points from recent data.
    This is called causally from `next()` to avoid lookahead bias.
    """
    highs_s = pd.Series(highs)
    lows_s = pd.Series(lows)

    peak_indices, _ = find_peaks(highs_s, distance=distance)
    trough_indices, _ = find_peaks(-lows_s, distance=distance)

    swings = []
    for idx in peak_indices:
        swings.append({'index': idx, 'type': 'high', 'price': highs[idx]})
    for idx in trough_indices:
        swings.append({'index': idx, 'type': 'low', 'price': lows[idx]})

    swings.sort(key=lambda x: x['index'])

    alternating_swings = []
    if swings:
        last_swing_type = ''
        for swing in swings:
            if swing['type'] != last_swing_type:
                alternating_swings.append(swing)
                last_swing_type = swing['type']

    return alternating_swings

def preprocess_data(df: pd.DataFrame):
    """
    Adds session-based indicators and other necessary metrics to the DataFrame.
    """
    # Define session times (UTC)
    asia_session_start = 0
    asia_session_end = 8

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    # Calculate previous day's Asia session high/low
    is_asia = (df['hour'] >= asia_session_start) & (df['hour'] < asia_session_end)
    asia_highs = df['High'][is_asia].resample('D').max()
    asia_lows = df['Low'][is_asia].resample('D').min()

    # Shift to get previous day's values
    prev_asia_high = asia_highs.shift(1).rename('prev_asia_high')
    prev_asia_low = asia_lows.shift(1).rename('prev_asia_low')

    # Combine daily session data and merge it back in one go
    daily_data = pd.concat([prev_asia_high, prev_asia_low], axis=1)
    df = df.join(daily_data, on=df.index.normalize())

    df['prev_asia_high'] = df['prev_asia_high'].ffill()
    df['prev_asia_low'] = df['prev_asia_low'].ffill()

    df.drop(columns=['hour', 'day_of_week'], inplace=True)
    df.dropna(inplace=True)
    return df

def generate_synthetic_data(days=200):
    """
    Generates synthetic 15-minute data with clear M and W patterns for verification.
    """
    dates = pd.date_range(start='2023-01-01', periods=days * 24 * 4, freq='15min')
    price = 100
    prices = [price]

    # Simple sine wave for baseline movement
    for i in range(1, len(dates)):
        price += np.sin(i / 100) * 0.1 + np.random.uniform(-0.05, 0.05)
        prices.append(price)

    df = pd.DataFrame(index=dates, data={'Close': prices})

    # Inject a perfect W-Pattern (Low -> High -> Low)
    w_start_idx = 150
    df.iloc[w_start_idx - 20 : w_start_idx, df.columns.get_loc('Close')] = 105 # High lead-in
    df.iloc[w_start_idx : w_start_idx + 10, df.columns.get_loc('Close')] = np.linspace(105, 95, 10) # Down
    df.iloc[w_start_idx + 10 : w_start_idx + 20, df.columns.get_loc('Close')] = np.linspace(95, 100, 10) # Up
    df.iloc[w_start_idx + 20 : w_start_idx + 30, df.columns.get_loc('Close')] = np.linspace(100, 96, 10) # Down (2nd leg)
    df.iloc[w_start_idx + 30 : w_start_idx + 50, df.columns.get_loc('Close')] = np.linspace(96, 110, 20) # Strong recovery

    # Inject a perfect M-Pattern (High -> Low -> High)
    m_start_idx = 400
    df.iloc[m_start_idx - 20 : m_start_idx, df.columns.get_loc('Close')] = 95 # Low lead-in
    df.iloc[m_start_idx : m_start_idx + 10, df.columns.get_loc('Close')] = np.linspace(95, 115, 10) # Up
    df.iloc[m_start_idx + 10 : m_start_idx + 20, df.columns.get_loc('Close')] = np.linspace(115, 108, 10) # Down
    df.iloc[m_start_idx + 20 : m_start_idx + 30, df.columns.get_loc('Close')] = np.linspace(108, 114, 10) # Up (2nd leg)
    df.iloc[m_start_idx + 30 : m_start_idx + 50, df.columns.get_loc('Close')] = np.linspace(114, 90, 20) # Strong reversal

    df['Open'] = df['Close'].shift(1).fillna(method='bfill')
    df['High'] = df[['Open', 'Close']].max(axis=1) + 0.1
    df['Low'] = df[['Open', 'Close']].min(axis=1) - 0.1
    df['Volume'] = 100

    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def passthrough(series):
    return series

# --- Strategy Definition ---

class MarketMakerSwingTradeStrategy(Strategy):
    """
    Implements the Market Maker Swing Trade strategy based on M/W formations,
    3-day cycles, and dynamic stop loss management.
    """
    # --- Optimizable Parameters ---
    # --- Optimizable Parameters ---
    pivot_lookback = 15          # Lookback distance for finding swing points
    swing_lookback_period = 120  # How many bars of history to use for causal swing detection
    confirmation_window_bars = 5 # Bars to wait for a confirmation candle
    sl_buffer_pct = 0.01       # Percentage buffer for stop loss placement
    breakeven_profit_usd = 50  # Profit in USD to move SL to break-even (replaces 'pips')

    def init(self):
        # --- Indicators & Pre-calculated data ---
        self.prev_asia_high = self.I(passthrough, self.data.df['prev_asia_high'].values)
        self.prev_asia_low = self.I(passthrough, self.data.df['prev_asia_low'].values)

        # --- State Machine Variables ---
        self.entry_date = None
        self.breakeven_set = False
        self.trade_level = 0
        self.pending_w_pattern = None
        self.pending_m_pattern = None

    def next(self):
        current_index = len(self.data) - 1
        current_date = self.data.index[-1].date()

        # --- State Reset ---
        if not self.position and self.trade_level > 0:
            self.entry_date, self.breakeven_set, self.trade_level = None, False, 0

        # --- Invalidate Pending Patterns if Window Expires ---
        if self.pending_w_pattern and current_index > self.pending_w_pattern['window_closes_at']:
            self.pending_w_pattern = None
        if self.pending_m_pattern and current_index > self.pending_m_pattern['window_closes_at']:
            self.pending_m_pattern = None

        # --- Main Logic ---
        if not self.position:
            # 1. Check for a pending trade first
            if self.pending_w_pattern:
                if self.data.Close[-1] > self.data.Open[-1]: # Bullish confirmation
                    sl = self.pending_w_pattern['sl']
                    entry_price = self.data.Close[-1]
                    if entry_price > sl: # Pre-trade validation
                        self.buy(limit=entry_price, sl=sl, size=0.1)
                        self.entry_date, self.trade_level = current_date, 1
                    self.pending_w_pattern = None # Clear after attempting trade
                return # Don't look for new patterns until pending is resolved

            if self.pending_m_pattern:
                if self.data.Close[-1] < self.data.Open[-1]: # Bearish confirmation
                    sl = self.pending_m_pattern['sl']
                    entry_price = self.data.Close[-1]
                    if entry_price < sl: # Pre-trade validation
                        self.sell(limit=entry_price, sl=sl, size=0.1)
                        self.entry_date, self.trade_level = current_date, 1
                    self.pending_m_pattern = None # Clear after attempting trade
                return

            # 2. If no pending trade, look for new patterns
            if current_index < self.swing_lookback_period: return

            history_start_index = current_index - self.swing_lookback_period + 1
            recent_swings = _find_recent_swings(
                self.data.High[-self.swing_lookback_period:],
                self.data.Low[-self.swing_lookback_period:],
                self.pivot_lookback
            )

            if len(recent_swings) < 3: return
            s1, s2, s3 = recent_swings[-3:]

            # A swing is only 'confirmed' after pivot_lookback bars have passed
            if current_index >= (history_start_index + s3['index'] + self.pivot_lookback):
                is_w = s1['type'] == 'low' and s2['type'] == 'high' and s3['type'] == 'low'
                is_m = s1['type'] == 'high' and s2['type'] == 'low' and s3['type'] == 'high'

                if is_w:
                    self.pending_w_pattern = {
                        'sl': s3['price'] * (1 - self.sl_buffer_pct),
                        'window_closes_at': current_index + self.confirmation_window_bars
                    }
                elif is_m:
                    self.pending_m_pattern = {
                        'sl': s3['price'] * (1 + self.sl_buffer_pct),
                        'window_closes_at': current_index + self.confirmation_window_bars
                    }

        # --- Position & SL Management ---
        if self.position:
            trade = self.trades[0]
            days_in_trade = (current_date - self.entry_date).days

            # Update trade level based on days in trade
            current_level = days_in_trade + 1
            if current_level != self.trade_level:
                self.trade_level = current_level

            # Level 1 Management: Initial phase, only move to BE
            if self.trade_level == 1:
                if not self.breakeven_set and trade.pl > self.breakeven_profit_usd:
                    trade.sl = trade.entry_price
                    self.breakeven_set = True

            # Level 2 Management: Start trailing SL behind Asia range
            elif self.trade_level == 2:
                if self.position.is_long:
                    new_sl = self.prev_asia_low[-1] * (1 - self.sl_buffer_pct)
                    if new_sl > trade.sl: trade.sl = new_sl
                else:
                    new_sl = self.prev_asia_high[-1] * (1 + self.sl_buffer_pct)
                    if new_sl < trade.sl: trade.sl = new_sl

            # Level 3 Management: Look for reversal exit signal
            elif self.trade_level >= 3:
                history_highs = self.data.High[-self.swing_lookback_period:]
                history_lows = self.data.Low[-self.swing_lookback_period:]
                recent_swings = _find_recent_swings(history_highs, history_lows, self.pivot_lookback)

                if len(recent_swings) >= 3:
                    s1, s2, s3 = recent_swings[-3:]
                    confirmation_index = s3['index'] + self.pivot_lookback

                    if current_index == confirmation_index + 1:
                        # For a long trade, an M-pattern is a reversal signal
                        if self.position.is_long:
                            is_m_pattern = s1['type'] == 'high' and s2['type'] == 'low' and s3['type'] == 'high'
                            if is_m_pattern:
                                self.position.close()

                        # For a short trade, a W-pattern is a reversal signal
                        elif self.position.is_short:
                            is_w_pattern = s1['type'] == 'low' and s2['type'] == 'high' and s3['type'] == 'low'
                            if is_w_pattern:
                                self.position.close()

                # Failsafe: exit after 3 full days if no reversal pattern found
                if days_in_trade >= 4:
                    self.position.close()


# --- Backtesting Execution ---

if __name__ == '__main__':
    USE_SYNTHETIC_DATA = False # <-- Set to True to debug with ideal patterns

    if USE_SYNTHETIC_DATA:
        print("Using synthetic data for verification...")
        data = generate_synthetic_data(days=300)
    else:
        data_path = 'data/BTC-USD-15m.csv'
        if os.path.exists(data_path):
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            data.columns = [c.strip().title() for c in data.columns]
            data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        else:
            raise FileNotFoundError(f"Data file not found at {data_path}. "
                                    "Please provide the required data file.")

    # Pre-process the data
    data = preprocess_data(data)

    bt = Backtest(data, MarketMakerSwingTradeStrategy, cash=100_000, commission=.002)

    stats = bt.run()
    print(stats)

    # --- Save Results ---
    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats_obj):
        sanitized = {}
        for key, value in stats_obj.items():
            if isinstance(value, (pd.Series, pd.DataFrame, Strategy)):
                continue
            if pd.isna(value):
                sanitized[key] = None
            elif isinstance(value, (np.integer, np.int64)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                sanitized[key] = float(value)
            elif isinstance(value, pd.Timestamp):
                 sanitized[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                sanitized[key] = str(value)
            else:
                sanitized[key] = value
        return sanitized

    result_data = sanitize_stats(stats)
    result_data['strategy_name'] = 'market_maker_swing_trade'

    with open('results/temp_result.json', 'w') as f:
        json.dump(result_data, f, indent=2)

    print("Backtest results saved to results/temp_result.json")

    try:
        plot_filename = 'results/market_maker_swing_trade.html'
        bt.plot(filename=plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
