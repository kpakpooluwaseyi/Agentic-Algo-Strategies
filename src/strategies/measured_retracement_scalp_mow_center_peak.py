from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json

# --- Custom Indicator Functions ---
# These functions must be defined outside the class
def identity(x, **kwargs):
    """A simple identity function that returns the input, ignoring other arguments."""
    return x

def swing_high(high_prices: np.ndarray, n: int, **kwargs) -> np.ndarray:
    """
    Identifies swing highs. A swing high is a high price that is greater than
    the highs of the 'n' preceding bars.
    """
    high_series = pd.Series(high_prices)
    max_in_prev_window = high_series.shift(1).rolling(n, min_periods=n).max()
    is_swing_high = high_series > max_in_prev_window
    return is_swing_high.to_numpy()

def swing_low(low_prices: np.ndarray, n: int, **kwargs) -> np.ndarray:
    """
    Identifies swing lows. A swing low is a low price that is less than
    the lows of the 'n' preceding bars.
    """
    low_series = pd.Series(low_prices)
    min_in_prev_window = low_series.shift(1).rolling(n, min_periods=n).min()
    is_swing_low = low_series < min_in_prev_window
    return is_swing_low.to_numpy()

class MeasuredRetracementScalpMowCenterPeakStrategy(Strategy):
    # --- Strategy Parameters to Optimize ---
    level_drop_lookback = 20
    min_rr_ratio = 5.0
    aoi_fib_level = 0.5
    aoi_tolerance = 0.05
    center_peak_lookback = 15

    def init(self):
        # --- Indicators for 15M Timeframe ---
        self.swing_high_15m = self.I(swing_high, self.data.High, self.level_drop_lookback, resample='15T')
        self.swing_low_15m = self.I(swing_low, self.data.Low, self.level_drop_lookback, resample='15T')

        # Use the 'identity' function to get resampled price series
        self.high_15m = self.I(identity, self.data.High, resample='15T')
        self.low_15m = self.I(identity, self.data.Low, resample='15T')

        # --- State Management ---
        self.swing_sequence = []

    def next(self):
        if self.position:
            return

        # --- Update Swing Sequence ---
        is_new_swing = False
        if self.swing_high_15m[-1] and (not self.swing_sequence or self.swing_sequence[-1][0] != 'high'):
            self.swing_sequence.append(('high', self.high_15m[-1]))
            is_new_swing = True
        elif self.swing_low_15m[-1] and (not self.swing_sequence or self.swing_sequence[-1][0] != 'low'):
            self.swing_sequence.append(('low', self.low_15m[-1]))
            is_new_swing = True

        if is_new_swing and len(self.swing_sequence) > 10:
            self.swing_sequence.pop(0)

        if len(self.swing_sequence) < 2:
            return

        # --- Trade Logic ---
        current_price = self.data.Close[-1]
        last_swing_type, last_swing_price = self.swing_sequence[-1]
        prev_swing_type, prev_swing_price = self.swing_sequence[-2]

        # --- SHORT Trade (M-formation) ---
        if last_swing_type == 'low' and prev_swing_type == 'high':
            swing_high_price = prev_swing_price
            swing_low_price = last_swing_price

            if swing_high_price <= swing_low_price: return

            fib_range = swing_high_price - swing_low_price
            aoi_level = swing_high_price - fib_range * self.aoi_fib_level
            aoi_upper = aoi_level + fib_range * self.aoi_tolerance
            aoi_lower = aoi_level - fib_range * self.aoi_tolerance

            if aoi_lower <= current_price <= aoi_upper:
                if self.data.Close[-1] < self.data.Open[-1]:
                    center_peak_high = self.data.High[-self.center_peak_lookback:].max()

                    stop_loss = center_peak_high * 1.001
                    take_profit = center_peak_high - (center_peak_high - swing_low_price) * self.aoi_fib_level

                    if take_profit >= current_price or stop_loss <= current_price: return

                    risk = stop_loss - current_price
                    reward = current_price - take_profit
                    if risk > 0 and reward / risk >= self.min_rr_ratio:
                        self.sell(sl=stop_loss, tp=take_profit)

        # --- LONG Trade (W-formation) ---
        elif last_swing_type == 'high' and prev_swing_type == 'low':
            swing_low_price = prev_swing_price
            swing_high_price = last_swing_price

            if swing_low_price >= swing_high_price: return

            fib_range = swing_high_price - swing_low_price
            aoi_level = swing_low_price + fib_range * self.aoi_fib_level
            aoi_upper = aoi_level + fib_range * self.aoi_tolerance
            aoi_lower = aoi_level - fib_range * self.aoi_tolerance

            if aoi_lower <= current_price <= aoi_upper:
                if self.data.Close[-1] > self.data.Open[-1]:
                    center_peak_low = self.data.Low[-self.center_peak_lookback:].min()

                    stop_loss = center_peak_low * 0.999
                    take_profit = center_peak_low + (swing_high_price - center_peak_low) * self.aoi_fib_level

                    if take_profit <= current_price or stop_loss >= current_price: return

                    risk = current_price - stop_loss
                    reward = take_profit - current_price
                    if risk > 0 and reward / risk >= self.min_rr_ratio:
                        self.buy(sl=stop_loss, tp=take_profit)

if __name__ == '__main__':
    # Generate synthetic 1-minute data
    n_days = 30
    rng = pd.date_range('2023-01-01', periods=n_days * 24 * 60, freq='min')
    price = 100 + np.random.randn(len(rng)).cumsum() * 0.1
    price += np.sin(np.linspace(0, n_days * np.pi, len(rng))) * 5
    data = pd.DataFrame({
        'Open': price,
        'High': price + np.random.uniform(0, 0.1, len(rng)),
        'Low': price - np.random.uniform(0, 0.1, len(rng)),
        'Close': price + np.random.uniform(-0.05, 0.05, len(rng)),
        'Volume': np.random.randint(100, 1000, len(rng))
    }, index=rng)
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

    # Run backtest
    bt = Backtest(data, MeasuredRetracementScalpMowCenterPeakStrategy, cash=100_000, commission=.002)

    # Optimize
    stats = bt.optimize(
        level_drop_lookback=range(10, 31, 5),
        min_rr_ratio=range(3, 7, 1),
        center_peak_lookback=range(10, 21, 5),
        maximize='Sharpe Ratio'
    )

    print(stats)

    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    if stats['# Trades'] > 0:
        with open('results/temp_result.json', 'w') as f:
            json.dump({
                'strategy_name': 'measured_retracement_scalp_mow_center_peak',
                'return': float(stats['Return [%]']),
                'sharpe': float(stats['Sharpe Ratio']),
                'max_drawdown': float(stats['Max. Drawdown [%]']),
                'win_rate': float(stats['Win Rate [%]']),
                'total_trades': int(stats['# Trades'])
            }, f, indent=2)

        bt.plot()
    else:
        print("No trades were executed. No results to save or plot.")
