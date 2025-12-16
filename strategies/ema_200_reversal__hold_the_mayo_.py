from backtesting import Strategy
import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import find_peaks

def find_swing_points(series, order=5):
    """Finds swing highs and lows in a series."""
    series = pd.Series(series)

    # Find peaks (highs)
    peaks_indices, _ = find_peaks(series, distance=order)

    # Find troughs (lows) by inverting the series
    troughs_indices, _ = find_peaks(-series, distance=order)

    swings = pd.Series(np.nan, index=series.index)
    swings.iloc[peaks_indices] = 1  # Mark highs with 1
    swings.iloc[troughs_indices] = -1 # Mark lows with -1

    return swings.values

class Ema200ReversalHoldTheMayo(Strategy):
    ema_long_period = 200
    ema_med_period = 50
    ema_short_period1 = 13
    ema_short_period2 = 5
    rr_ratio = 2.0
    sl_buffer_pips = 0.001
    ema_proximity_percent = 0.005
    swing_order = 5 # Lookback period for swing points

    def init(self):
        # --- Indicators ---
        self.ema200 = self.I(ta.ema, pd.Series(self.data.Close), length=self.ema_long_period)
        self.ema50 = self.I(ta.ema, pd.Series(self.data.Close), length=self.ema_med_period)
        self.ema13 = self.I(ta.ema, pd.Series(self.data.Close), length=self.ema_short_period1)
        self.ema5 = self.I(ta.ema, pd.Series(self.data.Close), length=self.ema_short_period2)

        # --- Swing Points ---
        self.swing_points = self.I(find_swing_points, self.data.Close, order=self.swing_order)

        # --- State Machine for M-Pattern (Short) ---
        self.m_state = 0
        self.m_peak1_price = None
        self.m_valley_price = None
        self.m_peak2_price = None

        # --- State Machine for W-Pattern (Long) ---
        self.w_state = 0
        self.w_trough1_price = None
        self.w_rally_price = None
        self.w_trough2_price = None

    def next(self):
        # --- Candlestick Patterns ---
        candle_body = abs(self.data.Open[-1] - self.data.Close[-1])
        candle_range = self.data.High[-1] - self.data.Low[-1]
        is_hammer = candle_range > 0 and candle_body > 0 and candle_range / candle_body > 2 and (self.data.Close[-1] - self.data.Low[-1]) / candle_range > 0.6
        is_shooting_star = candle_range > 0 and candle_body > 0 and candle_range / candle_body > 2 and (self.data.High[-1] - self.data.Close[-1]) / candle_range > 0.6

        # --- M-Pattern Logic ---
        if self.swing_points[-1] == 1: # A swing high is detected
            if self.m_state == 0 and abs(self.data.High[-1] - self.ema200[-1]) / self.data.High[-1] < self.ema_proximity_percent:
                self.m_peak1_price = self.data.High[-1]
                self.m_state = 1
            elif self.m_state == 2 and self.data.High[-1] < self.m_peak1_price:
                self.m_peak2_price = self.data.High[-1]
                self.m_state = 3

        if self.swing_points[-1] == -1 and self.m_state == 1: # A swing low after peak 1
            self.m_valley_price = self.data.Low[-1]
            self.m_state = 2

        if self.m_state == 3 and is_shooting_star and not self.position:
            entry_price = self.data.Close[-1]
            stop_loss = max(self.m_peak1_price, self.m_peak2_price) * (1 + self.sl_buffer_pips)
            take_profit = entry_price - (stop_loss - entry_price) * self.rr_ratio
            if entry_price < stop_loss:
                self.sell(sl=stop_loss, tp=take_profit)
            self.m_state = 0 # Reset

        # --- W-Pattern Logic ---
        if self.swing_points[-1] == -1: # A swing low is detected
            if self.w_state == 0 and abs(self.data.Low[-1] - self.ema200[-1]) / self.data.Low[-1] < self.ema_proximity_percent:
                self.w_trough1_price = self.data.Low[-1]
                self.w_state = 1
            elif self.w_state == 2 and self.data.Low[-1] > self.w_trough1_price:
                self.w_trough2_price = self.data.Low[-1]
                self.w_state = 3

        if self.swing_points[-1] == 1 and self.w_state == 1: # A swing high after trough 1
            self.w_rally_price = self.data.High[-1]
            self.w_state = 2

        if self.w_state == 3 and is_hammer and not self.position:
            entry_price = self.data.Close[-1]
            stop_loss = min(self.w_trough1_price, self.w_trough2_price) * (1 - self.sl_buffer_pips)
            take_profit = entry_price + (entry_price - stop_loss) * self.rr_ratio
            if entry_price > stop_loss:
                self.buy(sl=stop_loss, tp=take_profit)
            self.w_state = 0

        # Invalidation logic (e.g., if price moves too far away)
        if self.m_state > 0 and self.data.Close[-1] > self.ema200[-1] * (1 + self.ema_proximity_percent * 2): self.m_state = 0
        if self.w_state > 0 and self.data.Close[-1] < self.ema200[-1] * (1 - self.ema_proximity_percent * 2): self.w_state = 0

if __name__ == '__main__':
    import os
    import json
    from backtesting import Backtest

    # --- Data Loading ---
    data_path = 'data/crypto/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data.columns = [c.strip().title() for c in data.columns] # Strip whitespace and ensure correct column names

    # --- Backtest Execution ---
    bt = Backtest(data, Ema200ReversalHoldTheMayo, cash=100_000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()
    print(stats)

    # --- Results and Plotting ---
    os.makedirs('results', exist_ok=True)
    result_dict = {
        'strategy_name': 'ema_200_reversal__hold_the_mayo_',
        'return': stats.get('Return [%]', None),
        'sharpe': stats.get('Sharpe Ratio', None),
        'max_drawdown': stats.get('Max. Drawdown [%]', None),
        'win_rate': stats.get('Win Rate [%]', None),
        'total_trades': stats.get('# Trades', None)
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    print("Backtest results saved to results/temp_result.json")

    try:
        plot_filename = 'results/ema_200_reversal__hold_the_mayo_.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
