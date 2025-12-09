
import json
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

def generate_forex_data():
    """
    Generates a plausible 15-minute EUR/USD-like dataset for a few months.
    This ensures the backtest runs on data that has 24-hour sessions.
    """
    start_date = "2023-01-01"
    end_date = "2023-05-01"
    date_range = pd.date_range(start=start_date, end=end_date, freq="15min", tz="UTC")

    # Remove weekends (Saturday, Sunday)
    date_range = date_range[date_range.dayofweek < 5]

    n_points = len(date_range)

    # Base price path using Geometric Brownian Motion for realism
    price_path = np.cumprod(1 + np.random.randn(n_points) * 0.0008) * 1.08

    # Create OHLC DataFrame
    data = pd.DataFrame(index=date_range)
    data['Open'] = price_path

    # Add variability to candles
    high_noise = np.random.uniform(0, 0.001, n_points)
    low_noise = np.random.uniform(0, 0.001, n_points)
    close_noise = np.random.uniform(-0.001, 0.001, n_points)

    data['High'] = data['Open'] + high_noise
    data['Low'] = data['Open'] - low_noise
    data['Close'] = data['Open'] + close_noise

    # Ensure High is the highest and Low is the lowest price point in a candle
    data['High'] = data[['Open', 'High', 'Low', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'High', 'Low', 'Close']].min(axis=1)

    # Add Volume data
    data['Volume'] = np.random.randint(1000, 5000, n_points)

    return data

class AsiaLiquidityGrabReversalUkSessionStrategy(Strategy):
    """
    Implements the Asia Liquidity Grab strategy for the UK session.
    - Identifies Asia session high (AH) and low (AL).
    - Filters for Asia ranges smaller than a given percentage.
    - Enters short on a bearish reversal after a spike above AH during the UK session.
    - Enters long on a bullish reversal after a spike below AL during the UK session.
    """
    asia_range_percent_filter = 2.0

    def init(self):
        """Initialize strategy state variables."""
        self.asia_high = None
        self.asia_low = None
        self.current_day = -1

    def next(self):
        """Core logic executed on each new data point (candle)."""
        # Ensure index is a timezone-aware timestamp
        if not isinstance(self.data.index[-1], pd.Timestamp):
             return

        timestamp = self.data.index[-1].tz_convert('UTC')
        current_hour = timestamp.hour
        current_day = timestamp.dayofyear

        # Reset session highs/lows on a new day
        if current_day != self.current_day:
            self.current_day = current_day
            self.asia_high = None
            self.asia_low = None

        # --- Asia Session (00:00 - 08:00 UTC) ---
        # Capture the high and low of the Asia range
        if 0 <= current_hour < 8:
            if self.asia_high is None or self.data.High[-1] > self.asia_high:
                self.asia_high = self.data.High[-1]
            if self.asia_low is None or self.data.Low[-1] < self.asia_low:
                self.asia_low = self.data.Low[-1]
            return

        # --- UK Session (07:00 - 16:00 UTC) ---
        if 7 <= current_hour < 16:
            # Wait until Asia session data is available
            if self.asia_high is None or self.asia_low is None:
                return

            # 1. Asia Range Filter
            asia_range = (self.asia_high - self.asia_low) / self.asia_low * 100
            if asia_range > self.asia_range_percent_filter:
                return

            # Check if we have enough data for pattern recognition
            if len(self.data.Close) < 2:
                return

            # 2. Candlestick Pattern Recognition
            current_open = self.data.Open[-1]
            current_close = self.data.Close[-1]
            prev_open = self.data.Open[-2]
            prev_close = self.data.Close[-2]

            is_bullish_engulfing = (current_close > current_open and
                                     prev_close < prev_open and
                                     current_open < prev_close and
                                     current_close > prev_open)

            is_bearish_engulfing = (current_close < current_open and
                                     prev_close > prev_open and
                                     current_open > prev_close and
                                     current_close < prev_open)

            # --- Entry and Exit Logic ---
            if not self.position:
                # Long Entry: Spike below Asia Low + Bullish Reversal
                if self.data.Low[-1] < self.asia_low and is_bullish_engulfing:
                    sl = self.data.Low[-1] * 0.999 # SL just below the grab candle's low
                    tp = self.asia_high
                    if tp > current_close: # Ensure TP is valid
                        self.buy(sl=sl, tp=tp)

                # Short Entry: Spike above Asia High + Bearish Reversal
                elif self.data.High[-1] > self.asia_high and is_bearish_engulfing:
                    sl = self.data.High[-1] * 1.001 # SL just above the grab candle's high
                    tp = self.asia_low
                    if tp < current_close: # Ensure TP is valid
                        self.sell(sl=sl, tp=tp)


if __name__ == '__main__':
    # Generate synthetic 15M Forex data for backtesting
    data = generate_forex_data()

    # Initialize and run the backtest
    bt = Backtest(data, AsiaLiquidityGrabReversalUkSessionStrategy,
                  cash=100000, commission=.0002, trade_on_close=True)

    # Optimize the strategy over a range of Asia range percentages
    stats = bt.optimize(
        asia_range_percent_filter=list(np.arange(0.5, 3.0, 0.25)),
        maximize='Sharpe Ratio'
    )

    print("Best run stats:")
    print(stats)

    # Save the results of the best run to a JSON file
    import os
    os.makedirs('results', exist_ok=True)

    # Check if any trades were made before saving results
    if stats['# Trades'] > 0:
        with open('results/temp_result.json', 'w') as f:
            json.dump({
                'strategy_name': 'asia_liquidity_grab_reversal_uk_session',
                'return': float(stats['Return [%]']),
                'sharpe': float(stats['Sharpe Ratio']),
                'max_drawdown': float(stats['Max. Drawdown [%]']),
                'win_rate': float(stats['Win Rate [%]']),
                'total_trades': int(stats['# Trades'])
            }, f, indent=2)
    else:
        print("No trades were executed in the best run. Results not saved.")

    # Generate and display the performance plot
    bt.plot()
