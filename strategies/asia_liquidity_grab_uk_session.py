
import json
import os

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

# --- Helper Functions for Candlestick Patterns ---

def is_bearish_engulfing(df, i):
    """Checks for a bearish engulfing pattern at index i."""
    if i == 0:
        return False
    current = df.iloc[i]
    previous = df.iloc[i - 1]
    return (current['Open'] > previous['Close'] and
            current['Close'] < previous['Open'] and
            current['Close'] < current['Open'] and
            previous['Close'] > previous['Open'])

def is_bullish_engulfing(df, i):
    """Checks for a bullish engulfing pattern at index i."""
    if i == 0:
        return False
    current = df.iloc[i]
    previous = df.iloc[i - 1]
    return (current['Open'] < previous['Close'] and
            current['Close'] > previous['Open'] and
            current['Close'] > current['Open'] and
            previous['Close'] < previous['Open'])


# --- Data Preprocessing ---

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds session information (Asia, UK) and calculates
    High of Asia (HOA) and Low of Asia (LOA) for each day.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    df_copy = df.copy()

    # Define session times in UTC
    asia_start_hour = 0
    asia_end_hour = 8
    uk_start_hour = 8
    uk_end_hour = 17

    # Create session flags
    df_copy['is_asia_session'] = (df_copy.index.hour >= asia_start_hour) & (df_copy.index.hour < asia_end_hour)
    df_copy['is_uk_session'] = (df_copy.index.hour >= uk_start_hour) & (df_copy.index.hour < uk_end_hour)

    # Group by date to calculate daily Asia session HOA/LOA
    asia_session_data = df_copy[df_copy['is_asia_session']]
    daily_asia_stats = asia_session_data.groupby(asia_session_data.index.strftime('%Y-%m-%d')).agg(
        HOA=('High', 'max'),
        LOA=('Low', 'min')
    )

    # Map HOA/LOA to each row of the original dataframe using a string-based map
    date_map_series = pd.Series(df_copy.index.strftime('%Y-%m-%d'), index=df_copy.index)
    df_copy['HOA'] = date_map_series.map(daily_asia_stats['HOA'])
    df_copy['LOA'] = date_map_series.map(daily_asia_stats['LOA'])

    # Calculate Asia Range and its percentage
    df_copy['Asia_Range'] = df_copy['HOA'] - df_copy['LOA']
    df_copy['Asia_Range_Percent'] = (df_copy['Asia_Range'] / df_copy['LOA']) * 100

    # Forward-fill HOA/LOA so they are available throughout the day
    df_copy[['HOA', 'LOA', 'Asia_Range', 'Asia_Range_Percent']] = df_copy[['HOA', 'LOA', 'Asia_Range', 'Asia_Range_Percent']].ffill()

    return df_copy.dropna()


# --- Strategy Implementation ---

class AsiaLiquidityGrabUkSessionStrategy(Strategy):
    """
    A strategy that trades liquidity grabs of the Asia session range
    during the UK session.
    """
    asia_range_percent_max = 2.0  # Optimizable: Max Asia range in percent

    def init(self):
        # State variables to track setups
        self.short_setup_active = False
        self.long_setup_active = False
        self.breakout_high = None
        self.breakout_low = None
        self.current_day = None

    def next(self):
        # Reset state at the start of each new day
        current_date = self.data.index[-1].date()
        if self.current_day != current_date:
            self.current_day = current_date
            self.short_setup_active = False
            self.long_setup_active = False
            self.breakout_high = None
            self.breakout_low = None

        # --- Strategy Filters ---
        # Only trade during the UK session
        if not self.data.is_uk_session[-1]:
            return

        # Filter by Asia session range size
        if self.data.Asia_Range_Percent[-1] > self.asia_range_percent_max:
            return

        # Ensure we have valid HOA/LOA levels
        hoa = self.data.HOA[-1]
        loa = self.data.LOA[-1]
        if pd.isna(hoa) or pd.isna(loa):
            return

        # Avoid trading if a position is already open
        if self.position:
            return

        # --- Short Entry Logic ---
        # 1. Price spikes above HOA
        if self.data.High[-1] > hoa and not self.short_setup_active:
            self.short_setup_active = True
            self.long_setup_active = False # Invalidate long setup
            self.breakout_high = self.data.High[-1]

        # 2. Bearish engulfing candle forms and closes back below HOA
        if self.short_setup_active:
            # Update breakout high if a new high is made
            if self.data.High[-1] > self.breakout_high:
                self.breakout_high = self.data.High[-1]

            is_engulfing = is_bearish_engulfing(self.data.df, len(self.data.df) - 1)
            if is_engulfing and self.data.Close[-1] < hoa:
                sl = self.breakout_high
                tp = loa
                # Ensure TP is valid (below current price)
                if tp < self.data.Close[-1]:
                    self.sell(sl=sl, tp=tp)
                self.short_setup_active = False # Reset after entry

        # --- Long Entry Logic ---
        # 1. Price spikes below LOA
        if self.data.Low[-1] < loa and not self.long_setup_active:
            self.long_setup_active = True
            self.short_setup_active = False # Invalidate short setup
            self.breakout_low = self.data.Low[-1]

        # 2. Bullish engulfing candle forms and closes back above LOA
        if self.long_setup_active:
            # Update breakout low if a new low is made
            if self.data.Low[-1] < self.breakout_low:
                self.breakout_low = self.data.Low[-1]

            is_engulfing = is_bullish_engulfing(self.data.df, len(self.data.df) - 1)
            if is_engulfing and self.data.Close[-1] > loa:
                sl = self.breakout_low
                tp = hoa
                # Ensure TP is valid (above current price)
                if tp > self.data.Close[-1]:
                    self.buy(sl=sl, tp=tp)
                self.long_setup_active = False # Reset after entry


# --- Main Execution Block ---

if __name__ == '__main__':

    def generate_synthetic_data():
        """Generates synthetic 15-min data modeling the strategy's pattern."""
        n_days = 50
        periods_per_day = 96  # 24 * 4
        total_periods = n_days * periods_per_day

        # Start with a base price and random walk
        base_price = 1.2000
        np.random.seed(42)
        price_changes = np.random.randn(total_periods) * 0.0005
        prices = base_price + np.cumsum(price_changes)

        # Create timestamp index
        timestamps = pd.to_datetime(pd.date_range('2023-01-01', periods=total_periods, freq='15min', tz='UTC'))

        df = pd.DataFrame(index=timestamps)
        df['Open'] = prices
        df['High'] = prices + np.random.uniform(0, 0.001, size=total_periods)
        df['Low'] = prices - np.random.uniform(0, 0.001, size=total_periods)
        df['Close'] = prices + np.random.uniform(-0.0005, 0.0005, size=total_periods)
        df['Volume'] = np.random.randint(100, 1000, size=total_periods)

        # Force Asia session to be a tight range and UK to break out
        for day in range(n_days):
            day_start_idx = day * periods_per_day
            asia_end_idx = day_start_idx + (8 * 4) # 8 hours * 4 periods/hour
            uk_start_idx = asia_end_idx

            # Make Asia session range-bound
            asia_prices = df['Close'].iloc[day_start_idx:asia_end_idx]
            avg_asia_price = asia_prices.mean()
            df.loc[asia_prices.index, ['Open', 'High', 'Low', 'Close']] *= 0.999
            df.loc[asia_prices.index, ['Open', 'High', 'Low', 'Close']] += avg_asia_price * 0.001

            # Create a breakout pattern for UK session on some days
            if day % 4 == 0: # Bearish setup
                hoa = df['High'].iloc[day_start_idx:asia_end_idx].max()
                breakout_idx = uk_start_idx + 2
                df.loc[df.index[breakout_idx], 'High'] = hoa + 0.0015
                df.loc[df.index[breakout_idx], 'Open'] = hoa + 0.0005
                df.loc[df.index[breakout_idx], 'Close'] = hoa - 0.0005 # Close below
            elif day % 4 == 2: # Bullish setup
                loa = df['Low'].iloc[day_start_idx:asia_end_idx].min()
                breakout_idx = uk_start_idx + 2
                df.loc[df.index[breakout_idx], 'Low'] = loa - 0.0015
                df.loc[df.index[breakout_idx], 'Open'] = loa - 0.0005
                df.loc[df.index[breakout_idx], 'Close'] = loa + 0.0005 # Close above

        return df

    # Generate and preprocess data
    data = generate_synthetic_data()
    processed_data = preprocess_data(data)

    # Initialize and run backtest
    bt = Backtest(processed_data, AsiaLiquidityGrabUkSessionStrategy, cash=100_000, commission=.0002)

    # Optimize
    print("Running optimization...")
    stats = bt.optimize(
        asia_range_percent_max=list(np.arange(0.5, 3.0, 0.5)),
        maximize='Sharpe Ratio'
    )

    print("Best stats:")
    print(stats)

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Save results to JSON
    # Handle cases where no trades were made and metrics are NaN
    if stats['# Trades'] > 0:
        win_rate = float(stats['Win Rate [%]'])
        sharpe = float(stats['Sharpe Ratio'])
    else:
        win_rate = 0.0
        sharpe = 0.0

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'asia_liquidity_grab_uk_session',
            'return': float(stats['Return [%]']),
            'sharpe': sharpe,
            'max_drawdown': float(stats['Max. Drawdown [%]']),
            'win_rate': win_rate,
            'total_trades': int(stats['# Trades'])
        }, f, indent=2)

    print("Results saved to results/temp_result.json")

    # Generate and save plot
    plot_filename = 'results/asia_liquidity_grab_uk_session.html'
    bt.plot(filename=plot_filename)
    print(f"Plot saved to {plot_filename}")
