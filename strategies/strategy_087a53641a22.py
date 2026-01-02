import pandas as pd
from backtesting import Backtest, Strategy
import numpy as np
import json
import os
from enum import Enum

class StrategyState(Enum):
    """
    Enum to manage the state of the strategy throughout the day.
    """
    WAITING_FOR_SESSION = 1
    RANGE_DEFINED = 2
    WAITING_FOR_FVG = 3
    TRADE_EXECUTED = 4

class OneCandleSetup(Strategy):
    """
    Implements the "One Candle Setup" strategy, adapted for a 15-minute timeframe.

    NOTE: The source strategy requires a 5m chart for range definition and a 1m
    chart for entry confirmation. This implementation adapts the core logic to the
    provided 15-minute dataset, as generating 1m/5m data from 15m data is not
    possible. The principles of an initial range breakout and FVG entry are preserved.

    - Identifies the range of the first 15m candle at/after 9:30 AM New York time.
    - Waits for a candle to close outside this range (breakout).
    - Enters on a confirmed Fair Value Gap (FVG) pattern post-breakout, ensuring
      the FVG forms *outside* the initial range.
    - Exits based on a fixed 2:1 Risk-to-Reward ratio.
    """
    risk_reward_ratio = 2.0

    def init(self):
        """Initialize strategy state and daily variables."""
        self.state = StrategyState.WAITING_FOR_SESSION
        self.current_day = None
        self.range_high = None
        self.range_low = None
        self.breakout_direction = 0

    def next(self):
        """Main strategy logic executed on each bar."""
        current_time = self.data.index[-1]
        current_date = current_time.date()

        # --- Daily State Reset ---
        # If a new day starts, reset the entire state machine.
        if self.current_day != current_date:
            self.current_day = current_date
            self.state = StrategyState.WAITING_FOR_SESSION
            self.range_high = None
            self.range_low = None
            self.breakout_direction = 0

        # --- State Machine Logic ---

        # If a trade is already done for the day or we are in a position, do nothing.
        if self.state == StrategyState.TRADE_EXECUTED or self.position:
            return

        # 1. WAITING_FOR_SESSION: Look for the 9:30 AM candle to define the range.
        if self.state == StrategyState.WAITING_FOR_SESSION:
            if current_time.hour == 9 and current_time.minute >= 30:
                self.range_high = self.data.High[-1]
                self.range_low = self.data.Low[-1]
                self.state = StrategyState.RANGE_DEFINED
                # print(f"{current_time}: Range defined. High={self.range_high}, Low={self.range_low}")
                return

        # 2. RANGE_DEFINED: Wait for a breakout candle.
        if self.state == StrategyState.RANGE_DEFINED:
            # Bullish Breakout
            if self.data.Close[-1] > self.range_high:
                self.breakout_direction = 1
                self.state = StrategyState.WAITING_FOR_FVG
                # print(f"{current_time}: Bullish breakout detected.")
                return
            # Bearish Breakout
            elif self.data.Close[-1] < self.range_low:
                self.breakout_direction = -1
                self.state = StrategyState.WAITING_FOR_FVG
                # print(f"{current_time}: Bearish breakout detected.")
                return

        # 3. WAITING_FOR_FVG: Look for a valid Fair Value Gap after the breakout.
        if self.state == StrategyState.WAITING_FOR_FVG:
            # First, check if price has re-entered the range, invalidating the breakout.
            if self.breakout_direction == 1 and self.data.Close[-1] <= self.range_high:
                self.state = StrategyState.RANGE_DEFINED
                self.breakout_direction = 0
                return
            elif self.breakout_direction == -1 and self.data.Close[-1] >= self.range_low:
                self.state = StrategyState.RANGE_DEFINED
                self.breakout_direction = 0
                return

            # Need at least 3 candles to form an FVG pattern.
            if len(self.data) < 3:
                return

            entry_price = self.data.Close[-1]

            # Look for Bullish FVG (for a long trade)
            # Ensure the FVG forms *outside* the initial range.
            if self.breakout_direction == 1 and self.data.Close[-1] > self.range_high and self.data.Low[-1] > self.data.High[-3]:
                stop_loss = self.data.Low[-2] # Low of the middle FVG candle
                if entry_price > stop_loss: # Basic validation
                    take_profit = entry_price + self.risk_reward_ratio * (entry_price - stop_loss)
                    self.buy(sl=stop_loss, tp=take_profit)
                    self.state = StrategyState.TRADE_EXECUTED
                    return

            # Look for Bearish FVG (for a short trade)
            # Ensure the FVG forms *outside* the initial range.
            elif self.breakout_direction == -1 and self.data.Close[-1] < self.range_low and self.data.High[-1] < self.data.Low[-3]:
                stop_loss = self.data.High[-2] # High of the middle FVG candle
                if entry_price < stop_loss: # Basic validation
                    take_profit = entry_price - self.risk_reward_ratio * (stop_loss - entry_price)
                    self.sell(sl=stop_loss, tp=take_profit)
                    self.state = StrategyState.TRADE_EXECUTED
                    return


def load_and_preprocess_data(data_path):
    """
    Loads data, sanitizes column names, and converts the timezone
    to America/New_York for session-based logic.
    """
    if not os.path.exists(data_path):
        print(f"Data file not found at '{data_path}'. Falling back to synthetic data.")
        return generate_synthetic_data()

    print(f"Loading data from: {data_path}")
    try:
        data = pd.read_csv(
            data_path,
            index_col='datetime',
            parse_dates=True,
            header=0,
            names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume'],
            usecols=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        # The names parameter handles the column names, so no need to strip/title them again.
        data.index = data.index.tz_localize('UTC').tz_convert('America/New_York')
        return data.dropna()
    except Exception as e:
        print(f"Error loading or preprocessing data: {e}. Falling back to synthetic data.")
        return generate_synthetic_data()

def generate_synthetic_data():
    """Generates synthetic data for testing when real data is unavailable."""
    print("Generating synthetic data...")
    n_points = 5000
    index = pd.date_range('2023-01-01 00:00', periods=n_points, freq='15min', tz='America/New_York')
    price = 100 + np.random.randn(n_points).cumsum() * 0.2
    data = pd.DataFrame({
        'Open': price,
        'High': price + np.random.uniform(0, 0.5, n_points),
        'Low': price - np.random.uniform(0, 0.5, n_points),
        'Close': price + np.random.uniform(-0.2, 0.2, n_points),
        'Volume': np.random.randint(100, 1000, n_points)
    }, index=index)
    return data

def sanitize_stats(stats):
    """Removes non-serializable objects from backtest stats for JSON output."""
    sanitized = {}
    for key, value in stats.items():
        if key.startswith('_'): continue
        if isinstance(value, (pd.Series, pd.DataFrame)): continue
        try:
            json.dumps(value)
            sanitized[key] = value
        except (TypeError, OverflowError):
            sanitized[key] = str(value)
    return sanitized

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    data = load_and_preprocess_data(data_path)

    if data is not None and not data.empty:
        bt = Backtest(data, OneCandleSetup, cash=100_000, commission=.002)

        print("Running backtest...")
        stats = bt.run()
        print(stats)

        os.makedirs('results', exist_ok=True)

        # Save stats to JSON
        json_filename = 'results/temp_result.json'
        final_stats = sanitize_stats(stats)
        with open(json_filename, 'w') as f:
            json.dump(final_stats, f, indent=2)
        print(f"Backtest statistics saved to {json_filename}")

        # Generate plot
        plot_filename = 'results/strategy_087a53641a22.html'
        try:
            bt.plot(filename=plot_filename, open_browser=False)
            print(f"Backtest plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot: {e}")
    else:
        print("Could not run backtest due to data loading issues.")
