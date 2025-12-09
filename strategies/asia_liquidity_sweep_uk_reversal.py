import pandas as pd
import numpy as np
import json
from backtesting import Backtest, Strategy
import datetime
import os

def generate_forex_data(days=200):
    """
    Generates synthetic 15-minute forex OHLC data for a specified number of days.
    """
    np.random.seed(42)
    n = days * 24 * 4  # 15-minute intervals
    dates = pd.date_range(start='2023-01-01', periods=n, freq='15min', tz='UTC')

    price = 1.1000
    returns = np.random.randn(n) * 0.0005
    prices = price * np.exp(np.cumsum(returns))

    time_of_day_effect = np.sin(np.linspace(0, 2 * np.pi * days, n)) * 0.001
    prices += time_of_day_effect

    open_prices = prices
    high_prices = open_prices + np.random.uniform(0, 0.001, n)
    low_prices = open_prices - np.random.uniform(0, 0.001, n)
    close_prices = open_prices + np.random.randn(n) * 0.0003

    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices
    }, index=dates)

    return df

def add_indicators(df):
    """
    Pre-processes the OHLC data to add session information and other indicators.
    """
    asia_start = datetime.time(20, 0)
    asia_end = datetime.time(9, 30)
    uk_start = datetime.time(7, 0)
    uk_end = datetime.time(16, 0)

    df['is_uk_session'] = ((df.index.time >= uk_start) & (df.index.time < uk_end)).astype(int)

    df['session_date'] = df.index.date
    df.loc[df.index.time >= asia_start, 'session_date'] += pd.Timedelta(days=1)

    is_asia_hours = (df.index.time >= asia_start) | (df.index.time < asia_end)
    asia_session_bars = df[is_asia_hours]

    asia_stats = asia_session_bars.groupby('session_date').agg(
        asia_high=('High', 'max'),
        asia_low=('Low', 'min')
    )

    df = pd.merge(df, asia_stats, on='session_date', how='left')
    df['asia_high'] = df['asia_high'].ffill()
    df['asia_low'] = df['asia_low'].ffill()
    df['asia_range_pct'] = (df['asia_high'] - df['asia_low']) / df['asia_low'] * 100

    # Avoid lookahead bias by shifting session data
    df['prev_asia_high'] = df['asia_high'].shift()
    df['prev_asia_low'] = df['asia_low'].shift()
    df['prev_asia_range_pct'] = df['asia_range_pct'].shift()

    # Drop rows with NaN values that were introduced by shifting
    df = df.dropna()
    return df

class AsiaLiquiditySweepUkReversalStrategy(Strategy):
    """
    A trading strategy that looks for liquidity sweeps of the Asia session high/low
    followed by a reversal pattern during the UK session.
    """
    asia_range_max_pct = 2.0  # Max Asia range in percent
    sl_buffer_pct = 0.05      # Stop loss buffer in percent

    def init(self):
        pass

    def _is_bearish_engulfing(self):
        """Checks for a bearish engulfing pattern on the last two candles."""
        if len(self.data.Close) < 2: return False
        open1, close1 = self.data.Open[-1], self.data.Close[-1]
        open2, close2 = self.data.Open[-2], self.data.Close[-2]
        return (close2 > open2 and open1 > close1 and open1 >= close2 and close1 <= open2)

    def _is_bullish_engulfing(self):
        """Checks for a bullish engulfing pattern on the last two candles."""
        if len(self.data.Close) < 2: return False
        open1, close1 = self.data.Open[-1], self.data.Close[-1]
        open2, close2 = self.data.Open[-2], self.data.Close[-2]
        return (close2 < open2 and open1 < close1 and open1 <= close2 and close1 >= open2)

    def next(self):
        if self.position:
            return

        is_uk = self.data.df.iloc[-1]['is_uk_session']
        asia_high = self.data.df.iloc[-1]['prev_asia_high']
        asia_low = self.data.df.iloc[-1]['prev_asia_low']
        asia_range = self.data.df.iloc[-1]['prev_asia_range_pct']

        if pd.isna(asia_high) or pd.isna(asia_low):
            return

        if is_uk and asia_range < self.asia_range_max_pct:

            sweep_up = self.data.High[-2] > asia_high
            if sweep_up and self._is_bearish_engulfing():
                sl_price = max(self.data.High[-1], self.data.High[-2]) * (1 + self.sl_buffer_pct / 100)
                tp_price = asia_low
                self.sell(sl=sl_price, tp=tp_price)

            sweep_down = self.data.Low[-2] < asia_low
            if sweep_down and self._is_bullish_engulfing():
                sl_price = min(self.data.Low[-1], self.data.Low[-2]) * (1 - self.sl_buffer_pct / 100)
                tp_price = asia_high
                self.buy(sl=sl_price, tp=tp_price)

if __name__ == '__main__':
    # Load and prepare data
    data = generate_forex_data(days=100) # Using a smaller dataset for faster optimization
    data = add_indicators(data)

    # Run backtest optimization
    bt = Backtest(data, AsiaLiquiditySweepUkReversalStrategy, cash=100000, commission=.002)

    stats = bt.optimize(
        asia_range_max_pct=list(np.arange(0.5, 2.5, 0.5)),
        sl_buffer_pct=list(np.arange(0.01, 0.1, 0.02)),
        maximize='Sharpe Ratio'
    )

    print("--- Best Optimization Stats ---")
    print(stats)

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        # Check for NaN and replace with None for valid JSON
        sharpe = stats['Sharpe Ratio']
        json.dump({
            'strategy_name': 'asia_liquidity_sweep_uk_reversal',
            'return': float(stats['Return [%]']),
            'sharpe': float(sharpe) if not np.isnan(sharpe) else None,
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades'])
        }, f, indent=2)

    # Generate plot
    # The plot is saved to the root directory by default.
    # We specify a filename to save it in the allowed 'results' directory.
    plot_filename = 'results/asia_liquidity_sweep_uk_reversal.html'
    bt.plot(filename=plot_filename)
    print(f"\nGenerated plot saved to {plot_filename}")
