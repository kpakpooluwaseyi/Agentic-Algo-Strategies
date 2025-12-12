import json
import os

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy


def generate_synthetic_data(days=500):
    """
    Generates synthetic 24-hour forex-like data for backtesting.
    """
    rng = np.random.default_rng(42)
    n_points = days * 24 * 4  # 15-min intervals
    dates = pd.date_range(start='2022-01-01', periods=n_points, freq='15min')

    # Base price with a random walk
    price = 1.1000
    returns = rng.normal(loc=0, scale=0.0005, size=n_points)
    price_path = price * (1 + returns).cumprod()

    # Add some volatility spikes
    for _ in range(int(n_points / 100)):
        idx = rng.integers(1, n_points - 1)
        spike = rng.normal(0, 0.005)
        price_path[idx:] *= (1 + spike)

    # Create DataFrame
    df = pd.DataFrame(index=dates)
    df['Open'] = price_path
    df['High'] = df['Open'] + rng.uniform(0, 0.001, size=n_points)
    df['Low'] = df['Open'] - rng.uniform(0, 0.001, size=n_points)
    df['Close'] = df['Open'] + rng.normal(0, 0.0005, size=n_points)

    # Ensure High is the max and Low is the min
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)

    return df


def preprocess_data(df):
    """
    Adds session information and historical data points to the DataFrame using a map-based approach
    to avoid multiprocessing errors during backtest optimization.
    """
    # Define session times in UTC
    asia_start_hour = 0
    asia_end_hour = 8
    london_start_hour = 9
    london_end_hour = 17

    df['hour'] = df.index.hour
    df['day'] = df.index.date
    # Use a unique week identifier (year_week) to prevent mapping issues
    df['week_str'] = df.index.isocalendar().year.astype(str) + '_' + df.index.isocalendar().week.astype(str)

    # Identify sessions
    df['is_asia'] = (df['hour'] >= asia_start_hour) & (df['hour'] < asia_end_hour)
    df['is_london'] = (df['hour'] >= london_start_hour) & (df['hour'] < london_end_hour)

    # --- Calculate Daily/Weekly Levels ---
    daily_high_map = df.groupby('day')['High'].max().shift(1)
    daily_low_map = df.groupby('day')['Low'].min().shift(1)
    df['Prev_Day_High'] = df['day'].map(daily_high_map)
    df['Prev_Day_Low'] = df['day'].map(daily_low_map)
    df['Prev_Day_50'] = (df['Prev_Day_High'] + df['Prev_Day_Low']) / 2

    weekly_high_map = df.groupby('week_str')['High'].max().shift(1)
    weekly_low_map = df.groupby('week_str')['Low'].min().shift(1)
    df['Prev_Week_High'] = df['week_str'].map(weekly_high_map)
    df['Prev_Week_Low'] = df['week_str'].map(weekly_low_map)
    df['Prev_Week_50'] = (df['Prev_Week_High'] + df['Prev_Week_Low']) / 2

    # --- Calculate Asia Session Levels ---
    asia_high_map = df[df['is_asia']].groupby('day')['High'].max()
    asia_low_map = df[df['is_asia']].groupby('day')['Low'].min()
    df['Asia_High'] = df['day'].map(asia_high_map)
    df['Asia_Low'] = df['day'].map(asia_low_map)

    # Forward fill the session data to make it available throughout the day
    df['Asia_High'] = df['Asia_High'].ffill()
    df['Asia_Low'] = df['Asia_Low'].ffill()

    # Calculate Asia Range
    df['Asia_Range'] = df['Asia_High'] - df['Asia_Low']
    df['Asia_Range_Pct'] = (df['Asia_Range'] / df['Asia_Low']) * 100

    df.dropna(inplace=True)
    # clean up helper columns
    df.drop(columns=['hour', 'day', 'week_str'], inplace=True, errors='ignore')
    return df


class SessionLiquidityGrabReversalStrategy(Strategy):
    """
    A strategy that looks for liquidity grabs above/below the Asia session range
    during the London open, confirmed by an engulfing candle reversal.
    """
    # Optimization parameters
    asia_range_max_pct = 2.0
    sl_buffer_pct = 0.01
    risk_pct = 1.0 # Risk 1% of equity per trade

    def init(self):
        # Using a simple flag to manage the multi-TP state.
        self.tp_hit = self.I(lambda: np.zeros_like(self.data.Close), name="tp_hit_flag")

    def _calculate_position_size(self, sl_price):
        """Calculates position size based on fixed fractional risk."""
        risk_per_trade = self.equity * (self.risk_pct / 100)
        sl_distance_pips = abs(self.data.Close[-1] - sl_price)
        if sl_distance_pips == 0:
            return 0 # Avoid division by zero

        # Assuming a forex-like environment where 1 lot = 100,000 units
        # and pip value is related to the quote currency. For simplicity, we'll
        # treat the asset as the base currency and calculate size in units.
        position_size = risk_per_trade / sl_distance_pips
        return position_size / 10000 # Convert to a smaller, more testable unit size

    def next(self):
        # --- TRADE MANAGEMENT ---
        if self.position:
            trade = self.trades[-1]
            if self.tp_hit[-2] == 0:
                if (self.position.is_long and self.data.High[-1] >= trade.tp) or \
                   (self.position.is_short and self.data.Low[-1] <= trade.tp):
                    trade.close(portion=0.5)
                    trade.sl = trade.entry_price
                    if self.position.is_long:
                        trade.tp = self.data.Prev_Day_50[-1] if not np.isnan(self.data.Prev_Day_50[-1]) else trade.tp
                    else:
                        trade.tp = self.data.Prev_Day_50[-1] if not np.isnan(self.data.Prev_Day_50[-1]) else trade.tp
                    self.tp_hit[-1] = 1

        # --- ENTRY LOGIC ---
        if not self.position and self.data.is_london[-1]:
            is_asia_range_small = self.data.Asia_Range_Pct[-1] < self.asia_range_max_pct

            # --- SHORT ENTRY LOGIC ---
            grabbed_asia_high = self.data.High[-1] > self.data.Asia_High[-1]
            # More robust engulfing: current candle's high/low engulfs previous high/low
            is_bearish_engulfing = (self.data.Close[-1] < self.data.Open[-1] and
                                    self.data.High[-1] > self.data.High[-2] and
                                    self.data.Low[-1] < self.data.Low[-2])

            if is_asia_range_small and grabbed_asia_high and is_bearish_engulfing:
                sl = self.data.High[-1] * (1 + self.sl_buffer_pct / 100)
                tp1 = self.data.Asia_Low[-1]
                size = self._calculate_position_size(sl)

                if tp1 < self.data.Close[-1] and size > 0:
                    self.sell(sl=sl, tp=tp1, size=size)

            # --- LONG ENTRY LOGIC ---
            grabbed_asia_low = self.data.Low[-1] < self.data.Asia_Low[-1]
            # More robust engulfing: current candle's high/low engulfs previous high/low
            is_bullish_engulfing = (self.data.Close[-1] > self.data.Open[-1] and
                                    self.data.High[-1] > self.data.High[-2] and
                                    self.data.Low[-1] < self.data.Low[-2])

            if is_asia_range_small and grabbed_asia_low and is_bullish_engulfing:
                sl = self.data.Low[-1] * (1 - self.sl_buffer_pct / 100)
                tp1 = self.data.Asia_High[-1]
                size = self._calculate_position_size(sl)

                if tp1 > self.data.Close[-1] and size > 0:
                    self.buy(sl=sl, tp=tp1, size=size)


if __name__ == '__main__':
    # Generate and preprocess data
    data = generate_synthetic_data(days=365 * 2) # 2 years of data
    data = preprocess_data(data)

    # Run backtest
    bt = Backtest(data, SessionLiquidityGrabReversalStrategy, cash=100_000, commission=.002, finalize_trades=True)

    # Optimize
    stats = bt.optimize(
        asia_range_max_pct=np.arange(0.5, 3.0, 0.5).tolist(),
        sl_buffer_pct=np.arange(0.01, 0.1, 0.02).tolist(),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.asia_range_max_pct > 0
    )

    print("Best stats:", stats)

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save results to JSON
    # Handle cases where no trades are made
    final_stats = {
        'strategy_name': 'session_liquidity_grab_reversal',
        'return': stats.get('Return [%]', 0.0),
        'sharpe': stats.get('Sharpe Ratio', 0.0),
        'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
        'win_rate': stats.get('Win Rate [%]', 0.0),
        'total_trades': stats.get('# Trades', 0)
    }

    with open('results/temp_result.json', 'w') as f:
        # Cast numpy types to native python types for JSON serialization
        for key, value in final_stats.items():
            if isinstance(value, (np.int64, np.int32)):
                final_stats[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                final_stats[key] = float(value)
        json.dump(final_stats, f, indent=2)

    # Generate plot
    plot_filename = 'results/session_liquidity_grab_reversal.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"Plot saved to {plot_filename}")
