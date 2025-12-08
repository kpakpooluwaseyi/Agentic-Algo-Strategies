from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import json

def SMA(array, n):
    """Return series of simple moving averages."""
    return pd.Series(array).rolling(n).mean()

def EMA(array, n):
    """Return series of exponential moving averages."""
    return pd.Series(array).ewm(span=n, adjust=False).mean()

class MmWFormationEmaRetestStrategy(Strategy):
    # Parameters for optimization
    ema_fast_period = 50
    ema_slow_period = 200
    w_formation_lookback = 20
    m_formation_lookback = 20

    def init(self):
        # Initialize indicators on the 1H timeframe
        self.data_1h = self.data.df.resample('h').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        # Calculate EMAs on 1H data and reindex to match 15M data
        ema_fast_1h = EMA(self.data_1h['Close'], self.ema_fast_period)
        ema_slow_1h = EMA(self.data_1h['Close'], self.ema_slow_period)

        self.ema_fast = ema_fast_1h.reindex(self.data.df.index, method='ffill')
        self.ema_slow = ema_slow_1h.reindex(self.data.df.index, method='ffill')

        # Helper functions for candlestick patterns
        self.is_hammer = self._is_hammer
        self.is_railroad_tracks = self._is_railroad_tracks
        self.find_w_formation = self._find_w_formation
        self.find_m_formation = self._find_m_formation

    def _is_hammer(self, i):
        """Detects a Hammer candlestick pattern at index i on the 15M data."""
        if i < 1:
            return False

        candle = self.data
        body_size = abs(candle.Close[i] - candle.Open[i])
        upper_wick = candle.High[i] - max(candle.Open[i], candle.Close[i])
        lower_wick = min(candle.Open[i], candle.Close[i]) - candle.Low[i]

        # Hammer conditions
        return (body_size > 0 and
                lower_wick > 2 * body_size and
                upper_wick < body_size)

    def _is_railroad_tracks(self, i):
        """Detects a Railroad Tracks candlestick pattern ending at index i on the 15M data."""
        if i < 1:
            return False

        candle = self.data
        current_candle_is_bullish = candle.Close[i] > candle.Open[i]
        prev_candle_is_bearish = candle.Close[i-1] < candle.Open[i-1]

        # Bullish Railroad Tracks
        if current_candle_is_bullish and prev_candle_is_bearish:
            current_body = candle.Close[i] - candle.Open[i]
            prev_body = candle.Open[i-1] - candle.Close[i-1]
            if (abs(current_body - prev_body) / prev_body < 0.2 and # Bodies are similar in size
                candle.Close[i] > candle.Open[i-1]):
                return True

        # Bearish Railroad Tracks
        current_candle_is_bearish = candle.Close[i] < candle.Open[i]
        prev_candle_is_bullish = candle.Close[i-1] > candle.Open[i-1]
        if current_candle_is_bearish and prev_candle_is_bullish:
            current_body = candle.Open[i] - candle.Close[i]
            prev_body = candle.Close[i-1] - candle.Open[i-1]
            if (abs(current_body - prev_body) / prev_body < 0.2 and
                candle.Close[i] < candle.Open[i-1]):
                return True

        return False

    def _find_w_formation(self, i):
        """Finds a W-formation structure on the 1H data."""
        if i < self.w_formation_lookback:
            return None

        window = self.data_1h.iloc[i - self.w_formation_lookback : i]

        # Find two troughs (lows)
        lows = window['Low'].nsmallest(2)
        if len(lows) < 2:
            return None

        trough1_idx, trough2_idx = lows.index[0], lows.index[1]
        trough1_val, trough2_val = lows.iloc[0], lows.iloc[1]

        # Make sure troughs are distinct and in order
        if trough1_idx == trough2_idx:
            return None

        if trough1_idx > trough2_idx:
            trough1_idx, trough2_idx = trough2_idx, trough1_idx
            trough1_val, trough2_val = trough2_val, trough1_val

        # Find peak between the troughs
        middle_section = window.loc[trough1_idx:trough2_idx]
        peak = middle_section['High'].max()

        # W-formation confirmation
        if peak > trough1_val and peak > trough2_val:
            return trough2_val # Return the second trough for SL placement

        return None

    def _find_m_formation(self, i):
        """Finds an M-formation structure on the 1H data."""
        if i < self.m_formation_lookback:
            return None

        window = self.data_1h.iloc[i - self.m_formation_lookback : i]

        # Find two peaks (highs)
        highs = window['High'].nlargest(2)
        if len(highs) < 2:
            return None

        peak1_idx, peak2_idx = highs.index[0], highs.index[1]
        peak1_val, peak2_val = highs.iloc[0], highs.iloc[1]

        # Make sure peaks are distinct and in order
        if peak1_idx == peak2_idx:
            return None

        if peak1_idx > peak2_idx:
            peak1_idx, peak2_idx = peak2_idx, peak1_idx
            peak1_val, peak2_val = peak2_val, peak1_val

        # Find trough between the peaks
        middle_section = window.loc[peak1_idx:peak2_idx]
        trough = middle_section['Low'].min()

        # M-formation confirmation
        if trough < peak1_val and trough < peak2_val:
            return peak2_val # Return the second peak for SL placement

        return None

    def next(self):
        # Ensure we have enough data for 1H timeframe
        if len(self.data_1h) < max(self.ema_fast_period, self.ema_slow_period):
            return

        # Get the current 15M and 1H timestamps
        current_time_15m = self.data.index[-1]
        try:
            current_time_1h = self.data_1h.index[self.data_1h.index <= current_time_15m][-1]
            current_1h_index = self.data_1h.index.get_loc(current_time_1h)
        except IndexError:
            return # Not enough 1H data yet

        # Time-based exit rules
        if self.position:
            # Close positions if not in profit after 2 hours
            if current_time_15m - self.position.entry_time >= pd.Timedelta(hours=2):
                if self.position.pl_pct < 0.01: # Not in 'decent profit'
                    self.position.close()

            # Close all positions before Friday 5 PM NY time
            if current_time_15m.weekday() == 4 and current_time_15m.hour >= 17:
                self.position.close()

        # Entry rules - only on Tuesday, Wednesday, Thursday
        if current_time_15m.weekday() not in [1, 2, 3]: # Monday=0, Tuesday=1, etc.
            return

        current_price = self.data.Close[-1]
        current_ema_fast = self.ema_fast[-1]
        current_ema_slow = self.ema_slow[-1]

        # Long Entry (W-Formation)
        if not self.position:
            w_trough_low = self._find_w_formation(current_1h_index)

            price_above_ema50 = current_price > current_ema_fast
            retest_of_ema50 = abs(current_price - current_ema_fast) / current_ema_fast < 0.005

            if w_trough_low and price_above_ema50 and retest_of_ema50:
                if self._is_hammer(len(self.data) - 1) or self._is_railroad_tracks(len(self.data) - 1):
                    sl = w_trough_low * 0.99
                    tp = current_ema_slow
                    if tp > current_price and current_price > sl:
                        self.buy(sl=sl, tp=tp)

        # Short Entry (M-Formation)
        if not self.position:
            m_peak_high = self._find_m_formation(current_1h_index)

            price_below_ema50 = current_price < current_ema_fast
            retest_of_ema50 = abs(current_price - current_ema_fast) / current_ema_fast < 0.005

            if m_peak_high and price_below_ema50 and retest_of_ema50:
                if self._is_railroad_tracks(len(self.data) - 1):
                    sl = m_peak_high * 1.01
                    tp = current_ema_slow
                    if tp < current_price and current_price < sl:
                        self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    # Load or generate data
    from backtesting.test import GOOG
    # The GOOG dataset is daily, so we need to resample it to a higher frequency
    # to test the 15M/1H logic. We'll use a smaller slice of data to avoid timeouts.
    data = GOOG.copy().iloc[-1000:] # Use last 1000 days
    data.index = pd.to_datetime(data.index)
    data = data.resample('15min').ffill()

    # Run backtest
    bt = Backtest(data, MmWFormationEmaRetestStrategy, cash=10000, commission=.002)

    # Run the backtest without optimization
    stats = bt.run()

    # Save results
    import os
    # Create the 'results' directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'mm_w_formation_ema_retest',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']),
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }, f, indent=2)

    # Generate plot
    # bt.plot()
