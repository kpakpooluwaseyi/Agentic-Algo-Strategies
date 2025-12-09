from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
import numpy as np
import json

def generate_forex_data(days=30):
    """Generates synthetic 24-hour OHLCV data for backtesting."""
    rng = np.random.default_rng(seed=42)
    n_points = days * 24 * 4  # 15-minute intervals
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='15min')

    # Base price movement with some volatility
    price_changes = rng.normal(0, 0.0005, n_points)
    price = 1.1000 + np.cumsum(price_changes)

    # Create OHLC data
    open_price = price
    close_price = price + rng.normal(0, 0.0002, n_points)
    high_price = np.maximum(open_price, close_price) + rng.uniform(0, 0.0005, n_points)
    low_price = np.minimum(open_price, close_price) - rng.uniform(0, 0.0005, n_points)

    # Ensure OHLC integrity
    close_price = np.clip(close_price, low_price, high_price)

    # Volume
    volume = rng.integers(100, 10000, n_points)

    data = pd.DataFrame({
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Close': close_price,
        'Volume': volume
    }, index=dates)

    return data

def preprocess_data(df):
    """Adds session indicators and other necessary columns to the data."""
    df.index = pd.to_datetime(df.index, utc=True)

    # Define session times in UTC
    asia_start_hour = 23
    asia_end_hour = 8
    uk_start_hour = 7
    uk_end_hour = 16

    df['hour'] = df.index.hour
    df['date'] = df.index.date

    # Identify sessions
    df['is_asia_session'] = (df['hour'] >= asia_start_hour) | (df['hour'] < asia_end_hour)
    df['is_uk_session'] = (df['hour'] >= uk_start_hour) & (df['hour'] < uk_end_hour)

    # Calculate Asia Session High and Low
    asia_session_data = df[df['is_asia_session']].copy()
    asia_high = asia_session_data.groupby('date')['High'].max()
    asia_low = asia_session_data.groupby('date')['Low'].min()

    # Map ASH and ASL to the main dataframe
    df['ASH'] = df['date'].map(asia_high)
    df['ASL'] = df['date'].map(asia_low)

    # Forward-fill the session levels for the entire day
    df['ASH'] = df['ASH'].ffill()
    df['ASL'] = df['ASL'].ffill()

    # Calculate Asia Session Range Percentage
    df['ASR_percent'] = (df['ASH'] - df['ASL']) / df['ASL'] * 100

    # Add ATR
    tr = np.maximum(df['High'] - df['Low'],
                    np.maximum(abs(df['High'] - df['Close'].shift()),
                               abs(df['Low'] - df['Close'].shift())))
    df['ATR'] = tr.rolling(window=14).mean()

    # Clean up and remove NaNs from the start
    df.dropna(inplace=True)

    return df

def ATR(high, low, close, n=14):
    """Custom ATR indicator function for backtesting.py"""
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    return pd.Series(tr).rolling(window=n).mean().to_numpy()

class AsiaLiquiditySweepReversalStrategy(Strategy):
    asr_max_percent = 2.0
    sl_atr_multiplier = 1.5

    def init(self):
        # Indicators
        self.atr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, 14)

        # State tracking
        self.current_day = None
        self.sweep_high_detected = False
        self.sweep_low_detected = False

        # Trade management state
        self.tp1 = None
        self.tp1_hit = False
        self.break_even_sl_set = False

    def next(self):
        # --- Daily State Reset ---
        current_date = self.data.index[-1].date()
        if self.current_day != current_date:
            self.current_day = current_date
            self.sweep_high_detected = False
            self.sweep_low_detected = False

        # --- Trade Management ---
        if self.position:
            # TP1 Management: Close half of the position
            if not self.tp1_hit:
                if (self.position.is_long and self.data.High[-1] >= self.tp1) or \
                   (self.position.is_short and self.data.Low[-1] <= self.tp1):
                    self.position.close(portion=0.5)
                    self.tp1_hit = True

            # After TP1 is hit, move SL to break-even on the remaining position.
            if self.tp1_hit and not self.break_even_sl_set and self.trades:
                # There should only be one open trade at this point (the runner)
                self.trades[0].sl = self.trades[0].entry_price
                self.break_even_sl_set = True

        # --- Entry Logic ---
        # Only look for entries if we are not in a position and it's the UK session
        if not self.position and self.data.is_uk_session[-1]:
            if self.data.ASR_percent[-1] >= self.asr_max_percent:
                return

            ash = self.data.ASH[-1]
            asl = self.data.ASL[-1]

            # --- State Management for Sweeps ---
            # Detect a new sweep
            if self.data.High[-1] > ash and not self.sweep_high_detected:
                self.sweep_high_detected = True
            if self.data.Low[-1] < asl and not self.sweep_low_detected:
                self.sweep_low_detected = True

            # Reset sweep flag if price pulls back inside the range after a sweep
            if self.sweep_high_detected and self.data.Close[-1] < ash:
                self.sweep_high_detected = False
            if self.sweep_low_detected and self.data.Close[-1] > asl:
                self.sweep_low_detected = False

            # --- Candlestick Pattern Detection ---
            is_bullish_candle = self.data.Close[-1] > self.data.Open[-1]
            is_bearish_candle = self.data.Close[-1] < self.data.Open[-1]
            body_size = abs(self.data.Close[-1] - self.data.Open[-1])

            # SHORT ENTRY (Bearish Engulfing or Shooting Star)
            if self.sweep_high_detected:
                is_prev_bullish_candle = self.data.Close[-2] > self.data.Open[-2]
                is_bearish_engulfing = is_bearish_candle and is_prev_bullish_candle and \
                                       self.data.Open[-1] >= self.data.Close[-2] and \
                                       self.data.Close[-1] < self.data.Open[-2]

                is_shooting_star = False
                if body_size > 0:
                    upper_wick = self.data.High[-1] - max(self.data.Open[-1], self.data.Close[-1])
                    lower_wick = min(self.data.Open[-1], self.data.Close[-1]) - self.data.Low[-1]
                    is_shooting_star = is_bearish_candle and (upper_wick > 2 * body_size) and (lower_wick < 0.5 * body_size)

                if is_bearish_engulfing or is_shooting_star:
                    sl = self.data.High[-1] + self.atr[-1] * self.sl_atr_multiplier
                    if sl > self.data.Close[-1]:
                        self.tp1 = asl
                        self.tp1_hit = False
                        self.break_even_sl_set = False
                        self.sell(sl=sl)
                        self.sweep_high_detected = False

            # LONG ENTRY (Bullish Engulfing or Hammer)
            if self.sweep_low_detected:
                is_prev_bearish_candle = self.data.Close[-2] < self.data.Open[-2]
                is_bullish_engulfing = is_bullish_candle and is_prev_bearish_candle and \
                                       self.data.Open[-1] <= self.data.Close[-2] and \
                                       self.data.Close[-1] > self.data.Open[-2]

                is_hammer = False
                if body_size > 0:
                    upper_wick = self.data.High[-1] - max(self.data.Open[-1], self.data.Close[-1])
                    lower_wick = min(self.data.Open[-1], self.data.Close[-1]) - self.data.Low[-1]
                    is_hammer = is_bullish_candle and (lower_wick > 2 * body_size) and (upper_wick < 0.5 * body_size)

                if is_bullish_engulfing or is_hammer:
                    sl = self.data.Low[-1] - self.atr[-1] * self.sl_atr_multiplier
                    if sl < self.data.Close[-1]:
                        self.tp1 = ash
                        self.tp1_hit = False
                        self.break_even_sl_set = False
                        self.buy(sl=sl)
                        self.sweep_low_detected = False

if __name__ == '__main__':
    # 1. Generate Data
    data = generate_forex_data(days=90)

    # 2. Preprocess Data
    data = preprocess_data(data)

    # 3. Run Backtest
    bt = Backtest(data, AsiaLiquiditySweepReversalStrategy, cash=10000, commission=.002, finalize_trades=True)

    # 4. Optimize
    stats = bt.optimize(
        asr_max_percent=list(np.arange(0.5, 2.5, 0.5)),
        sl_atr_multiplier=list(np.arange(1.0, 3.5, 0.5)),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.asr_max_percent > 0
    )

    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        # Handle cases where no trades are made
        win_rate = stats.get('Win Rate [%]', 0)
        sharpe_ratio = stats.get('Sharpe Ratio', 0)

        json.dump({
            'strategy_name': 'asia_liquidity_sweep_reversal',
            'return': float(stats.get('Return [%]', 0)),
            'sharpe': float(sharpe_ratio) if np.isfinite(sharpe_ratio) else 0,
            'max_drawdown': float(stats.get('Max. Drawdown [%]', 0)),
            'win_rate': float(win_rate),
            'total_trades': int(stats.get('# Trades', 0))
        }, f, indent=2)

    # Generate plot
    bt.plot()
