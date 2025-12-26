from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
import numpy as np
import json
import os

def fvg(highs, lows):
    """Detects Fair Value Gaps (FVG). Returns numpy arrays."""
    # Compare candle [i-2] and [i] using array slicing
    bearish_fvg_mask = lows[:-2] > highs[2:]
    bullish_fvg_mask = highs[:-2] < lows[2:]

    # Bearish FVGs
    bearish_fvg_high = np.full_like(highs, np.nan)
    bearish_fvg_low = np.full_like(highs, np.nan)
    bearish_fvg_high[1:-1][bearish_fvg_mask] = lows[:-2][bearish_fvg_mask]
    bearish_fvg_low[1:-1][bearish_fvg_mask] = highs[2:][bearish_fvg_mask]

    # Bullish FVGs
    bullish_fvg_high = np.full_like(highs, np.nan)
    bullish_fvg_low = np.full_like(highs, np.nan)
    bullish_fvg_high[1:-1][bullish_fvg_mask] = lows[2:][bullish_fvg_mask]
    bullish_fvg_low[1:-1][bullish_fvg_mask] = highs[:-2][bullish_fvg_mask]

    return bearish_fvg_high, bearish_fvg_low, bullish_fvg_high, bullish_fvg_low

def order_block(opens, closes, highs, lows):
    """Detects Order Blocks (OB). Returns numpy arrays."""
    # Compare candle [i-1] and [i] using array slicing
    is_up_candle_minus_1 = opens[:-1] < closes[:-1]
    is_down_candle_current = closes[1:] < opens[1:]
    is_down_candle_minus_1 = opens[:-1] > closes[:-1]
    is_up_candle_current = closes[1:] > opens[1:]

    # Bearish OB: Last up candle before a down move
    is_bearish_ob = is_up_candle_minus_1 & is_down_candle_current
    bearish_ob_high = np.full_like(highs, np.nan)
    bearish_ob_low = np.full_like(highs, np.nan)
    bearish_ob_high[1:][is_bearish_ob] = highs[:-1][is_bearish_ob]
    bearish_ob_low[1:][is_bearish_ob] = lows[:-1][is_bearish_ob]

    # Bullish OB: Last down candle before an up move
    is_bullish_ob = is_down_candle_minus_1 & is_up_candle_current
    bullish_ob_high = np.full_like(highs, np.nan)
    bullish_ob_low = np.full_like(highs, np.nan)
    bullish_ob_high[1:][is_bullish_ob] = highs[:-1][is_bullish_ob]
    bullish_ob_low[1:][is_bullish_ob] = lows[:-1][is_bullish_ob]

    return bearish_ob_high, bearish_ob_low, bullish_ob_high, bullish_ob_low


class IctPredictiveCandleExpansionStrategy(Strategy):
    # --- Strategy Parameters ---
    rr = 3.0 # Risk-Reward Ratio
    lookback = 20 # Lookback period for swing points
    invalidation_bars = 10 # Bars to wait for entry before invalidating

    def init(self):
        # --- Indicators ---
        self.pdh = self.I(lambda x: x, self.data.df['pdh'], name="PDH")
        self.pdl = self.I(lambda x: x, self.data.df['pdl'], name="PDL")

        self.bearish_fvg_high, self.bearish_fvg_low, self.bullish_fvg_high, self.bullish_fvg_low = self.I(
            fvg, self.data.High, self.data.Low
        )
        self.bearish_ob_high, self.bearish_ob_low, self.bullish_ob_high, self.bullish_ob_low = self.I(
            order_block, self.data.Open, self.data.Close, self.data.High, self.data.Low
        )

        # --- State Machine ---
        self.state = "WAIT_STOP_RUN"
        self.stop_run_high = 0.0
        self.stop_run_low = 0.0
        self.displacement_fvg_ob_high = 0.0
        self.displacement_fvg_ob_low = 0.0
        self.entry_bar_counter = 0

    def next(self):
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_close = self.data.Close[-1]

        # --- State: WAIT_STOP_RUN ---
        if self.state == "WAIT_STOP_RUN":
            self.entry_bar_counter = 0 # Reset counter
            # Bearish setup: Sweep of PDH
            if current_high > self.pdh[-1]:
                self.state = "WAIT_DISPLACEMENT_SHORT"
                self.stop_run_high = current_high
            # Bullish setup: Sweep of PDL
            elif current_low < self.pdl[-1]:
                self.state = "WAIT_DISPLACEMENT_LONG"
                self.stop_run_low = current_low

        # --- State: WAIT_DISPLACEMENT_SHORT ---
        elif self.state == "WAIT_DISPLACEMENT_SHORT":
            self.stop_run_high = max(self.stop_run_high, current_high)
            swing_low = self.data.Low[-self.lookback:].min()
            if current_close < swing_low:
                if not np.isnan(self.bearish_fvg_high[-1]) or not np.isnan(self.bearish_ob_high[-1]):
                    self.state = "WAIT_ENTRY_SHORT"
                    self.displacement_fvg_ob_high = self.bearish_fvg_high[-1] if not np.isnan(self.bearish_fvg_high[-1]) else self.bearish_ob_high[-1]
                    self.displacement_fvg_ob_low = self.bearish_fvg_low[-1] if not np.isnan(self.bearish_fvg_low[-1]) else self.bearish_ob_low[-1]
                    self.entry_bar_counter = 0

        # --- State: WAIT_DISPLACEMENT_LONG ---
        elif self.state == "WAIT_DISPLACEMENT_LONG":
            self.stop_run_low = min(self.stop_run_low, current_low)
            swing_high = self.data.High[-self.lookback:].max()
            if current_close > swing_high:
                if not np.isnan(self.bullish_fvg_high[-1]) or not np.isnan(self.bullish_ob_high[-1]):
                    self.state = "WAIT_ENTRY_LONG"
                    self.displacement_fvg_ob_high = self.bullish_fvg_high[-1] if not np.isnan(self.bullish_fvg_high[-1]) else self.bullish_ob_high[-1]
                    self.displacement_fvg_ob_low = self.bullish_fvg_low[-1] if not np.isnan(self.bullish_fvg_low[-1]) else self.bullish_ob_low[-1]
                    self.entry_bar_counter = 0

        # --- State: WAIT_ENTRY_SHORT ---
        elif self.state == "WAIT_ENTRY_SHORT":
            self.entry_bar_counter += 1
            if self.entry_bar_counter > self.invalidation_bars:
                self.state = "WAIT_STOP_RUN"
            elif current_high > self.displacement_fvg_ob_low and not self.position:
                sl = self.stop_run_high
                risk = abs(current_close - sl)
                if risk > 0:
                    tp = current_close - (risk * self.rr)
                    self.sell(sl=sl, tp=tp)
                self.state = "WAIT_STOP_RUN"

        # --- State: WAIT_ENTRY_LONG ---
        elif self.state == "WAIT_ENTRY_LONG":
            self.entry_bar_counter += 1
            if self.entry_bar_counter > self.invalidation_bars:
                self.state = "WAIT_STOP_RUN"
            elif current_low < self.displacement_fvg_ob_high and not self.position:
                sl = self.stop_run_low
                risk = abs(current_close - sl)
                if risk > 0:
                    tp = current_close + (risk * self.rr)
                    self.buy(sl=sl, tp=tp)
                self.state = "WAIT_STOP_RUN"

if __name__ == '__main__':
    # --- Data Generation and Preprocessing ---
    def generate_synthetic_data(days=200):
        """Generates synthetic 15-minute OHLC data with daily structure."""
        rng = np.random.default_rng(42)
        timestamps = pd.date_range('2023-01-01', periods=days * 96, freq='15min', tz='UTC') # 96 candles per day

        price = 1.0
        prices = []
        for i in range(len(timestamps)):
            # Add some daily cyclical pattern
            daily_cycle = np.sin(2 * np.pi * (timestamps[i].hour * 60 + timestamps[i].minute) / (24 * 60))
            price += daily_cycle * 0.001
            price += rng.normal(loc=0, scale=0.002)
            prices.append(price)

        df = pd.DataFrame(index=timestamps, data={'Close': prices})

        # Generate OHLC data
        df['Open'] = df['Close'].shift(1)
        df['High'] = df[['Open', 'Close']].max(axis=1) + rng.uniform(0.0005, 0.002, size=len(df))
        df['Low'] = df[['Open', 'Close']].min(axis=1) - rng.uniform(0.0005, 0.002, size=len(df))

        # Ensure OHLC consistency
        df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
        df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)

        return df.dropna()

    def preprocess_data(df):
        """Calculates Previous Day High/Low."""
        df['date'] = df.index.date

        # Calculate daily high and low
        daily_high = df.groupby('date')['High'].max()
        daily_low = df.groupby('date')['Low'].min()

        # Shift to get previous day's high and low
        df['pdh'] = df['date'].map(daily_high.shift(1))
        df['pdl'] = df['date'].map(daily_low.shift(1))

        return df.dropna()

    data = generate_synthetic_data(days=250)
    data = preprocess_data(data)

    # Run backtest
    bt = Backtest(data, IctPredictiveCandleExpansionStrategy, cash=10000, commission=.002)

    # Optimize
    stats = bt.optimize(
        rr=list(np.arange(2, 5, 0.5)),
        lookback=range(10, 40, 5),
        invalidation_bars=range(5, 20, 5),
        maximize='Sharpe Ratio'
    )

    # Save results
    os.makedirs('results', exist_ok=True)
    # Sanitize stats for JSON serialization
    sanitized_stats = {key: (float(value) if isinstance(value, (int, float, np.number)) else str(value)) for key, value in stats.items() if key != '_equity_curve' and key != '_trades'}
    sanitized_stats['_strategy'] = str(sanitized_stats['_strategy']) # Convert strategy object to string

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'ict_predictive_candle_expansion',
            'return': sanitized_stats.get('Return [%]', 0.0),
            'sharpe': sanitized_stats.get('Sharpe Ratio', 0.0),
            'max_drawdown': sanitized_stats.get('Max. Drawdown [%]', 0.0),
            'win_rate': sanitized_stats.get('Win Rate [%]', 0.0),
            'total_trades': int(sanitized_stats.get('# Trades', 0))
        }, f, indent=2)

    # Generate plot
    try:
        bt.plot()
    except TypeError as e:
        print(f"Could not generate plot due to a known issue with the plotting library: {e}")
