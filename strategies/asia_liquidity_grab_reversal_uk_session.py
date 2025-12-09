from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json

def generate_forex_data():
    """
    Generates synthetic 15-minute forex data with specific patterns
    for the Asia liquidity grab strategy.
    """
    # Create a 15-minute date range for a few weeks
    dates = pd.date_range(start='2023-01-01', end='2023-01-21', freq='15min')
    n = len(dates)

    # Base price movement with some randomness
    price = 1.1000 + np.random.randn(n).cumsum() * 0.0001

    # Create DataFrame
    data = pd.DataFrame({
        'Open': price,
        'High': price,
        'Low': price,
        'Close': price,
        'Volume': np.random.randint(100, 1000, size=n)
    }, index=dates)

    # Generate OHLC from the price series
    data['Open'] = data['Close'].shift(1)
    data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 0.0005, size=n)
    data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 0.0005, size=n)

    # Create specific patterns
    # Pattern 1: Textbook liquidity grab and reversal
    pattern1_day = '2023-01-10'
    asia_start = f'{pattern1_day} 00:00'
    asia_end = f'{pattern1_day} 07:59'
    uk_start = f'{pattern1_day} 08:00'

    # Asia Session: Narrow range
    data.loc[asia_start:asia_end, 'High'] = 1.1050
    data.loc[asia_start:asia_end, 'Low'] = 1.1040
    data.loc[asia_start:asia_end, ['Open', 'Close']] = 1.1045

    # UK Session: Liquidity grab
    grab_time = f'{pattern1_day} 08:15'
    data.loc[grab_time, 'High'] = 1.1060  # Spike above Asia high
    data.loc[grab_time, 'Open'] = 1.1048
    data.loc[grab_time, 'Close'] = 1.1055
    data.loc[grab_time, 'Low'] = 1.1047

    # UK Session: Bearish engulfing reversal
    reversal_time = f'{pattern1_day} 08:30'
    data.loc[reversal_time, 'Open'] = 1.1058 # Opens above previous close
    data.loc[reversal_time, 'Close'] = 1.1042 # Closes below previous open
    data.loc[reversal_time, 'High'] = 1.1059
    data.loc[reversal_time, 'Low'] = 1.1041

    # Pattern 2: Asia range is too wide
    pattern2_day = '2023-01-12'
    asia_start2 = f'{pattern2_day} 00:00'
    asia_end2 = f'{pattern2_day} 07:59'
    data.loc[asia_start2:asia_end2, 'High'] = 1.1300 # 3% range
    data.loc[asia_start2:asia_end2, 'Low'] = 1.1000

    data.dropna(inplace=True)
    return data

def preprocess_data(df):
    """
    Adds strategy-specific columns to the dataframe.
    - Asia Session High/Low/Range
    - UK Session flag
    """
    # Define session times (UTC)
    asia_session_end = '08:00:00'
    uk_session_start = '08:00:00'
    uk_session_end = '16:00:00'

    # Calculate Asia session data
    df_asia = df.between_time('00:00:00', asia_session_end)

    # Group by date to get daily Asia high and low
    daily_asia_metrics = df_asia.groupby(df_asia.index.date).agg(
        asia_high=('High', 'max'),
        asia_low=('Low', 'min')
    )

    # Calculate Asia range and map back to main df
    daily_asia_metrics['asia_range_pct'] = \
        (daily_asia_metrics['asia_high'] - daily_asia_metrics['asia_low']) / \
        daily_asia_metrics['asia_low'] * 100

    # Map the daily metrics to each row in the original dataframe
    df['asia_high'] = pd.Series(df.index.date, index=df.index).map(daily_asia_metrics['asia_high'])
    df['asia_low'] = pd.Series(df.index.date, index=df.index).map(daily_asia_metrics['asia_low'])
    df['asia_range_pct'] = pd.Series(df.index.date, index=df.index).map(daily_asia_metrics['asia_range_pct'])

    # Add UK session flag
    df['is_uk_session'] = df.index.to_series().between_time(uk_session_start, uk_session_end).apply(lambda x: True if x else False).reindex(df.index).fillna(False)

    df.dropna(inplace=True)
    return df

# Pass-through function for custom data
def pass_through(series):
    return series

class AsiaLiquidityGrabReversalUkSessionStrategy(Strategy):
    # Optimizable parameters
    asia_range_max_pct = 2.0
    sl_buffer_pct = 0.05

    def init(self):
        # Make pre-calculated data available to the strategy
        self.asia_high = self.I(pass_through, self.data.df['asia_high'])
        self.asia_low = self.I(pass_through, self.data.df['asia_low'])
        self.asia_range_pct = self.I(pass_through, self.data.df['asia_range_pct'])
        self.is_uk_session = self.I(pass_through, self.data.df['is_uk_session'])

        # State tracking for multi-stage TP
        self.tp1_hit = False

    def next(self):
        # --- TRADE MANAGEMENT ---
        if self.position:
            # Check for TP1
            if not self.tp1_hit and self.data.Low[-1] <= self.asia_low[-1]:
                self.position.close(portion=0.5) # Close 50%
                self.tp1_hit = True
                # Move SL to breakeven for the remaining position
                self.trades[0].sl = self.trades[0].entry_price

            # NOTE: The logic for the second TP (e.g., daily/weekly levels)
            # is not implemented as it would require more complex data
            # (daily/weekly levels, vector candles) not available in the
            # current synthetic data generation. The remaining position
            # is managed by the trailing stop (moved to BE).
            return

        # --- SHORT ENTRY RULES ---
        # Reset TP flag for new trades
        self.tp1_hit = False

        if not self.is_uk_session[-1]:
            return

        if self.asia_range_pct[-1] >= self.asia_range_max_pct:
            return

        is_bullish_prev = self.data.Close[-2] > self.data.Open[-2]
        is_bearish_curr = self.data.Close[-1] < self.data.Open[-1]
        engulfs = self.data.Open[-1] > self.data.Close[-2] and \
                  self.data.Close[-1] < self.data.Open[-2]

        if not (is_bullish_prev and is_bearish_curr and engulfs):
            return

        engulfing_high = max(self.data.High[-1], self.data.High[-2])
        if engulfing_high <= self.asia_high[-1]:
            return

        buffer = engulfing_high * (self.sl_buffer_pct / 100)
        sl = engulfing_high + buffer

        # For this strategy, we will manage TP programmatically, so no TP on entry
        self.sell(sl=sl)

# --- Main Execution Block ---
if __name__ == '__main__':
    data = generate_forex_data()
    data = preprocess_data(data)

    # Initialize and run the backtest
    bt = Backtest(data, AsiaLiquidityGrabReversalUkSessionStrategy,
                  cash=100_000, commission=.002)

    # Optimize the strategy
    stats = bt.optimize(
        asia_range_max_pct=np.arange(0.5, 3.0, 0.5).tolist(),
        sl_buffer_pct=np.arange(0.01, 0.1, 0.02).tolist(),
        maximize='Sharpe Ratio'
    )

    print("Best optimization results:")
    print(stats)

    # Save results to JSON
    import os
    os.makedirs('results', exist_ok=True)

    # --- JSON Serialization Helper ---
    def sanitize_for_json(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize_for_json(i) for i in obj]
        return obj

    results_dict = {
        'strategy_name': 'asia_liquidity_grab_reversal_uk_session',
        'return': stats.get('Return [%]', 0.0),
        'sharpe': stats.get('Sharpe Ratio', 0.0),
        'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
        'win_rate': stats.get('Win Rate [%]', 0.0),
        'total_trades': stats.get('# Trades', 0)
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitize_for_json(results_dict), f, indent=2)

    print("\nResults saved to results/temp_result.json")

    # Generate and save the plot
    try:
        bt.plot(filename='results/strategy_plot.html', open_browser=False)
        print("Plot saved to results/strategy_plot.html")
    except TypeError as e:
        print(f"\nCould not generate plot due to a known issue with backtesting.py and pandas: {e}")
        print("Continuing without the plot.")
