from backtesting import Backtest, Strategy
from backtesting.test import GOOG
import pandas as pd
import numpy as np
import json
import os

# Custom EMA indicator function
def ema(arr: np.ndarray, n: int) -> np.ndarray:
    """Computes the Exponential Moving Average (EMA)."""
    return pd.Series(arr).ewm(span=n, adjust=False).mean().values

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to ensure it's JSON-serializable.
    Converts specific numpy types and pandas objects to native Python types.
    """
    sanitized = {}
    for key, value in stats.items():
        # Handle complex pandas objects first
        if isinstance(value, (pd.DataFrame, pd.Series)):
            sanitized[key] = None
        elif isinstance(value, (np.integer, np.int_)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            sanitized[key] = float(value)
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    return sanitized


class Ema50PullbackEngulfingContinuationStrategy(Strategy):
    """
    Strategy based on EMA pullback, engulfing patterns, and trend continuation.
    """
    # Optimizable parameters
    ema_period = 50
    rr_ratio = 2.0

    def init(self):
        """
        Initialize indicators and strategy variables.
        """
        # Initialize the 50-period EMA
        self.ema = self.I(ema, self.data.Close, self.ema_period)

    def next(self):
        """
        Defines the trading logic for each bar.
        """
        # Ensure we have enough data points to work with
        if len(self.data.Close) < 3:
            return

        # --- Candle Properties ---
        is_green = self.data.Close[-1] > self.data.Open[-1]
        is_red = self.data.Close[-1] < self.data.Open[-1]

        prev_is_red = self.data.Close[-2] < self.data.Open[-2]
        prev_prev_is_red = self.data.Close[-3] < self.data.Open[-3]

        prev_is_green = self.data.Close[-2] > self.data.Open[-2]
        prev_prev_is_green = self.data.Close[-3] > self.data.Open[-3]

        # --- LONG ENTRY LOGIC ---
        if not self.position and self.data.Close[-1] > self.ema[-1]:
            # 1. Trend: Price is above 50 EMA.
            # 2. Pullback: At least two consecutive Red candles.
            if prev_is_red and prev_prev_is_red:
                # 3. EMA Touch: Pullback touches the 50 EMA.
                pullback_low = min(self.data.Low[-2], self.data.Low[-3])
                if pullback_low <= self.ema[-2]:
                    # 4. Entry Signal: Advanced bullish engulfing pattern after pullback.
                    if is_green:
                        c1_open, c1_close, c1_low = self.data.Open[-2], self.data.Close[-2], self.data.Low[-2]
                        c2_open, c2_close, c2_low = self.data.Open[-1], self.data.Close[-1], self.data.Low[-1]

                        c1_body = abs(c1_open - c1_close)
                        c2_body = abs(c2_open - c2_close)

                        if c2_close > c1_open and c2_open < c1_close:
                            pattern1 = c2_body > c1_body and c2_low < c1_low
                            pattern2 = c1_low < self.data.Low[-3]

                            if pattern1 or pattern2:
                                entry_price = self.data.Close[-1]
                                stop_loss = min(c1_low, c2_low)
                                take_profit = entry_price + (entry_price - stop_loss) * self.rr_ratio

                                if entry_price > stop_loss:
                                    self.buy(sl=stop_loss, tp=take_profit)

        # --- SHORT ENTRY LOGIC ---
        elif not self.position and self.data.Close[-1] < self.ema[-1]:
            # 1. Trend: Price is below 50 EMA.
            # 2. Pullback: At least two consecutive Green candles.
            if prev_is_green and prev_prev_is_green:
                # 3. EMA Touch: Pullback touches the 50 EMA.
                pullback_high = max(self.data.High[-2], self.data.High[-3])
                if pullback_high >= self.ema[-2]:
                    # 4. Entry Signal: Advanced bearish engulfing pattern after pullback.
                    if is_red:
                        c1_open, c1_close, c1_high = self.data.Open[-2], self.data.Close[-2], self.data.High[-2]
                        c2_open, c2_close, c2_high = self.data.Open[-1], self.data.Close[-1], self.data.High[-1]

                        c1_body = abs(c1_open - c1_close)
                        c2_body = abs(c2_open - c2_close)

                        if c2_close < c1_open and c2_open > c1_close:
                            pattern1 = c2_body > c1_body and c2_high > c1_high
                            pattern2 = c1_high > self.data.High[-3]

                            if pattern1 or pattern2:
                                entry_price = self.data.Close[-1]
                                stop_loss = max(c1_high, c2_high)
                                take_profit = entry_price - (stop_loss - entry_price) * self.rr_ratio

                                if entry_price < stop_loss:
                                    self.sell(sl=stop_loss, tp=take_profit)

if __name__ == '__main__':
    # Load sample data
    data = GOOG.copy()

    # Initialize the backtest
    bt = Backtest(data, Ema50PullbackEngulfingContinuationStrategy, cash=10000, commission=.002)

    # Optimize the strategy parameters
    stats = bt.optimize(
        ema_period=range(30, 70, 10),
        rr_ratio=list(np.arange(1.5, 3.0, 0.5)),
        maximize='Sharpe Ratio'
    )

    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)

    # Sanitize the stats for JSON serialization
    sanitized_stats = sanitize_stats(stats)

    # Save the results to a JSON file
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'ema_50_pullback_engulfing_continuation',
            'return': sanitized_stats.get('Return [%]'),
            'sharpe': sanitized_stats.get('Sharpe Ratio'),
            'max_drawdown': sanitized_stats.get('Max. Drawdown [%]'),
            'win_rate': sanitized_stats.get('Win Rate [%]'),
            'total_trades': sanitized_stats.get('# Trades')
        }, f, indent=2)

    print("Backtest results saved to results/temp_result.json")

    # Generate and save the plot
    try:
        plot_filename = 'results/ema_50_pullback_engulfing_continuation.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
