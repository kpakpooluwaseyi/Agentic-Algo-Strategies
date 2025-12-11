from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
import numpy as np
import json
import os
from scipy.signal import find_peaks

def rsi(array, n):
    """Calculate Relative Strength Index."""
    gain = pd.Series(array).diff()
    gain[gain < 0] = 0
    loss = -pd.Series(array).diff()
    loss[loss < 0] = 0
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def find_swing_points(series, order=5):
    """Find swing highs and lows in a series."""
    # Ensure series is a plain numpy array for find_peaks
    if isinstance(series, pd.Series):
        series = series.values

    # Find peaks (highs)
    high_indices, _ = find_peaks(series, distance=order)

    # Find troughs (lows) by inverting the series
    low_indices, _ = find_peaks(-series, distance=order)

    return high_indices, low_indices

def generate_synthetic_data(n_points=500):
    """
    Generates synthetic data with a clear bullish RSI divergence, ChOC, and OB.
    """
    np.random.seed(42)
    # Start with a base price
    price = 100
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='1min')

    # Initial data with some noise
    data = np.random.randn(n_points) * 0.1

    # Create the pattern
    # 1. Downtrend
    data[100:200] = -0.1 + np.random.randn(100) * 0.05
    # 2. First Low (Price=P1, RSI=R1)
    data[200:210] = -0.5 - np.arange(10) * 0.1 # Sharp drop
    # 3. Rally
    data[210:250] = 0.1 + np.random.randn(40) * 0.05
    # 4. Second Low (Price=P2 < P1, RSI=R2 > R1) - Bullish Divergence
    data[250:260] = -0.6 - np.arange(10) * 0.1 # Lower price low
    # To create RSI divergence, we make the close less severe on the second drop
    # Let's construct OHLC manually for the key parts

    close_prices = price + np.cumsum(data)

    # Manually adjust the close for RSI divergence around point 255
    close_prices[255] = close_prices[254] - 0.05 # Smaller drop than at point 205

    # 5. Change of Character (ChOC) - Strong rally breaking previous high
    data[260:280] = 0.5 + np.random.randn(20) * 0.1
    # 6. Order Block (OB) - The last down candle before the ChOC
    # Let's define the OB candle at index 259

    # Re-cumulate prices after manual adjustment
    price_diffs = np.diff(close_prices, prepend=price)

    # Now build the OHLC DataFrame
    open_prices = close_prices - price_diffs
    high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0, 0.1, n_points)
    low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0, 0.1, n_points)

    # Ensure OHLC consistency
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    # Manually define the OB candle at index 259 to be a clear down-candle
    open_prices[259] = low_prices[258] - 0.1
    close_prices[259] = open_prices[259] - 0.3
    low_prices[259] = close_prices[259] - 0.05
    high_prices[259] = open_prices[259] + 0.05

    # Manually define the ChOC
    swing_high_price = high_prices[210:250].max()
    close_prices[265] = swing_high_price + 0.2 # Clear break and close above

    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': np.random.randint(100, 1000, n_points)
    }, index=dates)

    return df

class SmcRsiDivergenceScalperStrategy(Strategy):
    # --- Strategy Parameters ---
    rsi_period = 14
    swing_order = 15
    rr_ratio_tp1 = 2.0
    sl_buffer_pips = 5

    # --- State Variables ---
    # For initial setup detection
    bullish_divergence_confirmed = False
    bearish_divergence_confirmed = False
    choc_level = None
    choc_idx = None

    # For pending orders and active trades
    pending_order = None
    # A list of dictionaries, each holding params for an active trade
    active_trades = []

    # Store the direction of the initial trade to ensure scaling entries are in the same direction
    trade_direction = 0 # 1 for long, -1 for short

    def init(self):
        self.rsi = self.I(rsi, self.data.Close, self.rsi_period)
        self.reset_state()

    def reset_state(self):
        """Resets all state variables, preparing for a new setup search."""
        self.bullish_divergence_confirmed = False
        self.bearish_divergence_confirmed = False
        self.choc_level = None
        self.choc_idx = None
        self.pending_order = None
        self.active_trades = []
        self.trade_direction = 0

    def next(self):
        pip_size = 0.0001
        sl_buffer = self.sl_buffer_pips * pip_size

        # If there are open trades, manage them or look for scaling opportunities
        if len(self.trades) > 0:
            self.manage_open_positions()
            # If we have less than 3 trades, look for a scaling opportunity
            if len(self.trades) < 3:
                self.find_scaling_setup(sl_buffer)

        # If there are no open trades, look for a new initial setup
        else:
            # If the last trade was just closed, reset the state
            if self.trade_direction != 0:
                self.reset_state()
            self.find_new_setup(sl_buffer)

    def find_new_setup(self, sl_buffer):
        """The state machine for finding and placing a trade."""
        if not self.bullish_divergence_confirmed and not self.bearish_divergence_confirmed:
            self.detect_divergence()

        if self.bullish_divergence_confirmed or self.bearish_divergence_confirmed:
            self.detect_choc_and_create_order(sl_buffer)

        if self.pending_order:
            self.execute_pending_order()

    def find_scaling_setup(self, sl_buffer):
        """Looks for a Break of Structure (BOS) to scale into a winning position."""
        # Ensure we don't already have a pending order and the first trade is profitable
        if self.pending_order or not self.trades[0].is_long and not self.trades[0].is_short:
             return
        if self.trades[0].pl < 0:
            return

        high_indices, low_indices = find_swing_points(self.data.Close, order=self.swing_order)

        # Scaling logic for a long position
        if self.trade_direction == 1 and len(high_indices) > 0:
            last_swing_high = self.data.High[high_indices[-1]]
            if self.data.Close[-1] > last_swing_high: # Break of Structure
                # Look for the last down candle before the break
                for i in range(high_indices[-1], 0, -1):
                    if self.data.Close[i] < self.data.Open[i]:
                        entry_price = self.data.High[i]
                        sl = self.data.Low[i] - sl_buffer
                        if entry_price > sl:
                            tp1 = entry_price + (entry_price - sl) * self.rr_ratio_tp1
                            tp2 = entry_price + (entry_price - sl) * (self.rr_ratio_tp1 * 2)
                            self.pending_order = {'price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'type': 'long'}
                            break

        # Scaling logic for a short position
        elif self.trade_direction == -1 and len(low_indices) > 0:
            last_swing_low = self.data.Low[low_indices[-1]]
            if self.data.Close[-1] < last_swing_low: # Break of Structure
                # Look for the last up candle before the break
                for i in range(low_indices[-1], 0, -1):
                    if self.data.Close[i] > self.data.Open[i]:
                        entry_price = self.data.Low[i]
                        sl = self.data.High[i] + sl_buffer
                        if entry_price < sl:
                            tp1 = entry_price - (sl - entry_price) * self.rr_ratio_tp1
                            tp2 = entry_price - (sl - entry_price) * (self.rr_ratio_tp1 * 2)
                            self.pending_order = {'price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'type': 'short'}
                            break

    def detect_divergence(self):
        """Simplified divergence detection."""
        high_indices, low_indices = find_swing_points(self.data.Close, order=self.swing_order)

        if len(low_indices) >= 2:
            price_low1_idx, price_low2_idx = low_indices[-2], low_indices[-1]
            if self.data.Low[price_low2_idx] < self.data.Low[price_low1_idx] and self.rsi[price_low2_idx] > self.rsi[price_low1_idx]:
                relevant_highs = [i for i in high_indices if price_low1_idx < i < price_low2_idx]
                if relevant_highs:
                    self.bullish_divergence_confirmed = True
                    self.choc_idx = relevant_highs[-1]
                    self.choc_level = self.data.High[self.choc_idx]

        if len(high_indices) >= 2:
            price_high1_idx, price_high2_idx = high_indices[-2], high_indices[-1]
            if self.data.High[price_high2_idx] > self.data.High[price_high1_idx] and self.rsi[price_high2_idx] < self.rsi[price_high1_idx]:
                relevant_lows = [i for i in low_indices if price_high1_idx < i < price_high2_idx]
                if relevant_lows:
                    self.bearish_divergence_confirmed = True
                    self.choc_idx = relevant_lows[-1]
                    self.choc_level = self.data.Low[self.choc_idx]

    def detect_choc_and_create_order(self, sl_buffer):
        """If ChOC occurs, find the OB and create a pending order dictionary."""
        if self.bullish_divergence_confirmed and self.data.Close[-1] > self.choc_level:
            # Look back from the ChOC swing high to find the last down candle
            for i in range(self.choc_idx, 0, -1):
                if self.data.Close[i] < self.data.Open[i]:
                    entry_price = self.data.High[i]
                    sl = self.data.Low[i] - sl_buffer
                    if entry_price > sl:
                        tp1 = entry_price + (entry_price - sl) * self.rr_ratio_tp1
                        tp2 = entry_price + (entry_price - sl) * (self.rr_ratio_tp1 * 2)
                        self.pending_order = {'price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'type': 'long'}
                        self.bullish_divergence_confirmed = False # Move to next state
                        break

        elif self.bearish_divergence_confirmed and self.data.Close[-1] < self.choc_level:
            for i in range(self.choc_idx, 0, -1):
                if self.data.Close[i] > self.data.Open[i]:
                    entry_price = self.data.Low[i]
                    sl = self.data.High[i] + sl_buffer
                    if entry_price < sl:
                        tp1 = entry_price - (sl - entry_price) * self.rr_ratio_tp1
                        tp2 = entry_price - (sl - entry_price) * (self.rr_ratio_tp1 * 2)
                        self.pending_order = {'price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'type': 'short'}
                        self.bearish_divergence_confirmed = False # Move to next state
                        break

    def execute_pending_order(self):
        """If a pending order exists, place the limit order and store its params."""
        order = self.pending_order
        # Initial entry is size 0.1, scaling entries are 0.05
        size = 0.1 if self.trade_direction == 0 else 0.05

        if order['type'] == 'long':
            self.buy(limit=order['price'], sl=order['sl'], size=size)
        else: # Short
            self.sell(limit=order['price'], sl=order['sl'], size=size)

        # Add trade params to the list of active trades, tagging it with entry price and size for later identification
        trade_params = {**order, 'tp1_hit': False, 'entry_price': order['price'], 'size': size}
        self.active_trades.append(trade_params)

        # If this is the first trade, set the overall trade direction
        if self.trade_direction == 0:
            self.trade_direction = 1 if order['type'] == 'long' else -1

        self.pending_order = None

    def manage_open_positions(self):
        """Handles multi-stage TP and SL management for all active trades."""

        trades_to_remove = []

        for trade in self.trades:
            # Find the corresponding parameters in our active_trades list
            params = next((t for t in self.active_trades if t['entry_price'] == trade.entry_price and not t.get('closed')), None)

            if not params:
                continue

            # Manage TP1
            if not params['tp1_hit']:
                if (trade.is_long and self.data.High[-1] >= params['tp1']) or \
                   (trade.is_short and self.data.Low[-1] <= params['tp1']):
                    trade.close(portion=0.5)
                    trade.sl = trade.entry_price
                    params['tp1_hit'] = True

            # Manage TP2 (Final Target)
            if params['tp1_hit']:
                if (trade.is_long and self.data.High[-1] >= params['tp2']) or \
                   (trade.is_short and self.data.Low[-1] <= params['tp2']):
                    trade.close()
                    params['closed'] = True # Mark for removal
                    trades_to_remove.append(params)

        # Clean up closed trades from our tracking list
        if trades_to_remove:
            self.active_trades = [t for t in self.active_trades if not t.get('closed')]


if __name__ == '__main__':
    # Load or generate data
    data = generate_synthetic_data(n_points=500)

    # Ensure results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    # Run backtest
    bt = Backtest(data, SmcRsiDivergenceScalperStrategy, cash=10000, commission=.002)

    # Optimize
    stats = bt.optimize(
        rsi_period=range(10, 20, 2),
        swing_order=range(5, 25, 5),
        rr_ratio_tp1=np.arange(1.5, 3.0, 0.5),
        sl_buffer_pips=range(3, 10, 2),
        maximize='Sharpe Ratio',
        constraint=lambda param: param.swing_order > 2 # Basic constraint example
    )

    # Sanitize stats for JSON serialization
    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (np.integer, np.int64)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                sanitized[key] = float(value)
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                # Decide how to handle pandas objects, e.g., convert to dict or ignore
                sanitized[key] = None # Or some other representation
            elif pd.isna(value):
                sanitized[key] = None
            else:
                sanitized[key] = value
        return sanitized

    # Clean up stats dictionary
    # The stats object from optimize is a Series, let's access the values we need
    final_stats = {
        'strategy_name': 'smc_rsi_divergence_scalper',
        'return': stats.get('Return [%]', 0.0),
        'sharpe': stats.get('Sharpe Ratio', 0.0),
        'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
        'win_rate': stats.get('Win Rate [%]', 0.0),
        'total_trades': stats.get('# Trades', 0)
    }

    sanitized_final_stats = sanitize_stats(final_stats)

    # Save results
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_final_stats, f, indent=2)

    print("Backtest complete. Results saved to results/temp_result.json")

    # Generate plot
    try:
        bt.plot(filename='results/smc_rsi_divergence_scalper_plot.html')
        print("Plot saved to results/smc_rsi_divergence_scalper_plot.html")
    except Exception as e:
        print(f"Could not generate plot: {e}")
