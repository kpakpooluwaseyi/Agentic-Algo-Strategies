
import backtesting
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os

from backtesting import Strategy, Backtest

# Helper function to check for a Hammer or Inverted Hammer pattern
def is_hammer(open, high, low, close, is_inverted=False):
    """
    Identifies a Hammer or Inverted Hammer candlestick pattern.
    A true return indicates the presence of the pattern.
    """
    body = abs(open - close)
    total_range = high - low

    if total_range == 0:
        return False

    if is_inverted:
        # Inverted Hammer: Long upper wick, small body at the bottom
        upper_wick = high - max(open, close)
        lower_wick = min(open, close) - low
        return upper_wick >= 2 * body and lower_wick <= 0.5 * body
    else:
        # Hammer: Long lower wick, small body at the top
        lower_wick = min(open, close) - low
        upper_wick = high - max(open, close)
        return lower_wick >= 2 * body and upper_wick <= 0.5 * body

# Helper function to check for an Engulfing pattern
def is_engulfing(open_prev, close_prev, open_curr, close_curr, is_bullish=False):
    """
    Identifies a Bullish or Bearish Engulfing pattern.
    A true return indicates the presence of the pattern.
    """
    prev_body = abs(open_prev - close_prev)
    curr_body = abs(open_curr - close_curr)

    if is_bullish:
        # Bullish Engulfing: Current bull candle engulfs previous bear candle
        is_prev_bearish = close_prev < open_prev
        is_curr_bullish = close_curr > open_curr
        engulfs = close_curr > open_prev and open_curr < close_prev and curr_body > prev_body
        return is_prev_bearish and is_curr_bullish and engulfs
    else:
        # Bearish Engulfing: Current bear candle engulfs previous bull candle
        is_prev_bullish = close_prev > open_prev
        is_curr_bearish = close_curr < open_curr
        engulfs = open_curr > close_prev and close_curr < open_prev and curr_body > prev_body
        return is_prev_bullish and is_curr_bearish and engulfs

class QuickFlipScalper(Strategy):
    """
    A scalping strategy based on exploiting an opening range "liquidity candle"
    and trading reversals back towards the range.
    """
    atr_confirmation_pct = 0.25
    time_window_minutes = 90

    def init(self):
        # State variables, reset daily
        self.today = None
        self.opening_range_high = None
        self.opening_range_low = None
        self.opening_range_candle_is_bullish = None
        self.is_liquidity_candle = False
        self.trade_window_end = None

    def next(self):
        current_time = self.data.index[-1]
        current_date = current_time.date()

        # --- Daily State Reset ---
        if self.today != current_date:
            self.today = current_date
            self.opening_range_high = None
            self.opening_range_low = None
            self.opening_range_candle_is_bullish = None
            self.is_liquidity_candle = False
            self.trade_window_end = None

        # --- Step 1: Box the Opening Range Candle (first 15min of the day) ---
        if current_time.hour == 0 and current_time.minute == 0:
            self.opening_range_high = self.data.High[-1]
            self.opening_range_low = self.data.Low[-1]
            self.opening_range_candle_is_bullish = self.data.Close[-1] > self.data.Open[-1]
            self.trade_window_end = current_time + pd.Timedelta(minutes=self.time_window_minutes)

            # --- Step 2: Confirm Liquidity Candle ---
            opening_range_size = self.opening_range_high - self.opening_range_low
            daily_atr = self.data.DAILY_ATR[-1]

            if daily_atr > 0 and (opening_range_size / daily_atr) >= self.atr_confirmation_pct:
                self.is_liquidity_candle = True
            return

        # --- Strategy Logic (only runs after the first candle and if it was a liquidity candle) ---
        # --- Dynamic Stop Loss Management ---
        if self.position:
            # Move SL to break-even if trade is halfway to TP
            for trade in self.trades:
                if trade.is_long:
                    mid_point = trade.entry_price + (trade.tp - trade.entry_price) / 2
                    if self.data.High[-1] >= mid_point:
                        trade.sl = trade.entry_price
                else: # is_short
                    mid_point = trade.entry_price - (trade.entry_price - trade.tp) / 2
                    if self.data.Low[-1] <= mid_point:
                        trade.sl = trade.entry_price
            return # Don't look for new entries if we have an open position

        if not self.is_liquidity_candle:
            return

        # Check if we are outside the trading window
        if current_time > self.trade_window_end:
            self.is_liquidity_candle = False # Invalidate for the rest of the day
            return

        # --- Step 3: Find Reversal Candle Outside the Boxed Range ---

        # Condition for Bearish Reversal (Short Entry)
        if self.opening_range_candle_is_bullish and self.data.High[-1] > self.opening_range_high:
            sl = self.data.High[-1] * 1.001  # SL is high of signal candle
            tp = self.opening_range_low      # TP is low of the range

            # Check for Inverted Hammer (market order entry)
            is_inv_hammer = is_hammer(self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1], is_inverted=True)
            if is_inv_hammer:
                if self.data.Close[-1] > tp:
                    self.sell(sl=sl, tp=tp)
                return

            # Check for Bearish Engulfing (limit order entry)
            is_bear_engulf = is_engulfing(self.data.Open[-2], self.data.Close[-2], self.data.Open[-1], self.data.Close[-1], is_bullish=False)
            if is_bear_engulf:
                limit_price = self.data.Low[-2]  # Limit entry at the low of the preceding candle
                if limit_price > tp:
                    self.sell(limit=limit_price, sl=sl, tp=tp)

        # Condition for Bullish Reversal (Long Entry)
        elif not self.opening_range_candle_is_bullish and self.data.Low[-1] < self.opening_range_low:
            sl = self.data.Low[-1] * 0.999  # SL is low of signal candle
            tp = self.opening_range_high   # TP is high of the range

            # Check for Hammer (market order entry)
            is_reg_hammer = is_hammer(self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1], is_inverted=False)
            if is_reg_hammer:
                if self.data.Close[-1] < tp:
                    self.buy(sl=sl, tp=tp)
                return

            # Check for Bullish Engulfing (limit order entry)
            is_bull_engulf = is_engulfing(self.data.Open[-2], self.data.Close[-2], self.data.Open[-1], self.data.Close[-1], is_bullish=True)
            if is_bull_engulf:
                limit_price = self.data.High[-2]  # Limit entry at the high of the preceding candle
                if limit_price < tp:
                    self.buy(limit=limit_price, sl=sl, tp=tp)


def preprocess_data(filepath='data/BTC-USD-15m.csv'):
    """Loads and preprocesses data, calculating and merging daily ATR."""
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.columns = [col.strip().capitalize() for col in df.columns]

    # Calculate Daily ATR
    df_daily = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    df_daily['DAILY_ATR'] = ta.atr(df_daily['High'], df_daily['Low'], df_daily['Close'], length=14)

    # Map daily ATR back to the 15m dataframe
    df['DAILY_ATR'] = df.index.normalize().map(df_daily['DAILY_ATR'])
    df.dropna(subset=['DAILY_ATR'], inplace=True)

    return df

def sanitize_stats(stats):
    """Converts non-serializable types in stats object to serializable types."""
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (np.int64, np.int32)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            sanitized[key] = float(value)
        elif isinstance(value, pd.Timestamp):
            sanitized[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            sanitized[key] = str(value)
        elif isinstance(value, backtesting.Strategy):
            # Skip strategy object to avoid deep recursion
            continue
        elif isinstance(value, pd.DataFrame):
            # Skip DataFrames which are not easily serializable
             continue
        elif key.startswith('_'):
             continue
        else:
            sanitized[key] = value
    return sanitized

if __name__ == '__main__':
    data = preprocess_data('data/BTC-USD-15m.csv')

    # Instantiate Backtest
    bt = Backtest(data, QuickFlipScalper, cash=100000, commission=.002)

    # Run backtest
    stats = bt.run()
    print(stats)

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Save results to JSON
    sanitized_stats = sanitize_stats(stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    # Generate plot
    plot_filename = 'results/strategy_67b6e1e2ef4a.html'
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
