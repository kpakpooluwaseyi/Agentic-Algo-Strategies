
import pandas as pd
import numpy as np
import json
import os
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Helper function for Simple Moving Average using pandas_ta
def SMA(array, n):
    """Return series of simple moving averages."""
    return ta.sma(pd.Series(array), length=n)

# Helper function for calculating the slope of a series
def SLOPE(array, n):
    """Calculate the slope of a series over n periods."""
    return pd.Series(array).diff(n) / n

# Helper function for rolling maximum (resistance)
def rolling_max(array, n):
    return pd.Series(array).rolling(n).max()

# Helper function for rolling minimum (support)
def rolling_min(array, n):
    return pd.Series(array).rolling(n).min()

class WeinsteinStage2Trader(Strategy):
    """
    Implements Stan Weinstein's Stage 2 continuation breakout strategy.
    This strategy buys stocks that are already in a clear Stage 2 uptrend,
    consolidate, and then break out again to continue the trend.
    """
    # Timeframe and MA parameters
    ma_30_period = 30  # 30-week MA for stage analysis
    ma_10_period = 10  # 10-week MA for traders
    ma_slope_period = 5 # Period to calculate the slope of the 30-week MA

    # Consolidation and Breakout parameters
    consolidation_period = 52 # Lookback period for consolidation range (in weeks)
    volume_ma_period = 20 # Period for volume moving average
    volume_multiplier = 1.5 # Breakout volume must be X times the average

    # Risk Management
    min_rr = 2.0 # Minimum Risk-to-Reward ratio required for entry

    def init(self):
        # Weekly indicators calculated from the pre-processed data
        self.weekly_close = self.I(lambda x: x, self.data.Weekly_Close, overlay=True)
        self.ma_30_week = self.I(SMA, self.data.Weekly_Close, self.ma_30_period, overlay=True)
        self.ma_10_week = self.I(SMA, self.data.Weekly_Close, self.ma_10_period, overlay=True)
        self.ma_30_slope = self.I(SLOPE, self.ma_30_week, self.ma_slope_period, overlay=False)
        self.volume_ma_week = self.I(SMA, self.data.Weekly_Volume, self.volume_ma_period, overlay=False)

        # Dynamically find consolidation support and resistance
        self.resistance = self.I(rolling_max, self.data.Weekly_High, self.consolidation_period, overlay=True)
        self.support = self.I(rolling_min, self.data.Weekly_Low, self.consolidation_period, overlay=True)

    def next(self):
        # Run trading logic only on the first bar of each new week.
        if self.data.index[-1].week != self.data.index[-2].week:

            # === ENTRY CONDITIONS ===
            is_stage_2 = (self.weekly_close[-1] > self.ma_30_week[-1] and
                          self.ma_10_week[-1] > self.ma_30_week[-1])
            is_ma_rising = self.ma_30_slope[-1] > 0
            is_breakout = crossover(self.data.Close, self.resistance)
            # Use previous week's volume to avoid lookahead bias
            is_volume_spike = self.data.Weekly_Volume[-2] > self.volume_ma_week[-2] * self.volume_multiplier

            if not self.position and is_stage_2 and is_ma_rising and is_breakout and is_volume_spike:

                stop_loss = self.support[-1]
                entry_price = self.data.Close[-1]

                # Simple R:R based Take Profit (as per-book exit is discretionary)
                # We can target the height of the consolidation range projected upwards
                risk = entry_price - stop_loss
                if risk <= 0: return # Invalid risk

                reward = risk * self.min_rr
                take_profit = entry_price + reward

                if take_profit > entry_price:
                     self.buy(sl=stop_loss, tp=take_profit)

            # === EXIT CONDITIONS ===
            # Exit if the 30-week MA starts to flatten or decline (Stage 3)
            if self.position and self.ma_30_slope[-1] <= 0:
                self.position.close()


def sanitize_stats(stats):
    """
    Cleans up the backtest stats Series for JSON serialization.
    Handles non-finite numbers and NumPy types.
    """
    results = {}
    for key, value in stats.items():
        if isinstance(value, (np.integer, np.int64)):
            value = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            # Handle NaN, inf, -inf
            if not np.isfinite(value):
                value = None
            else:
                value = float(value)
        elif isinstance(value, pd.Timestamp):
            value = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            value = str(value)
        results[key] = value
    # Select only the required fields for the final JSON
    final_results = {
        'strategy_name': 'weinstein_stage2_trader_continuation_buy',
        'return': results.get('Return [%]'),
        'sharpe': results.get('Sharpe Ratio'),
        'max_drawdown': results.get('Max. Drawdown [%]'),
        'win_rate': results.get('Win Rate [%]'),
        'total_trades': results.get('# Trades')
    }
    return final_results


if __name__ == '__main__':
    data_path = 'data/crypto/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        # 1. Load Data
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Standardize column names to 'Open', 'High', 'Low', 'Close', 'Volume'
        df.columns = [col.strip().title() for col in df.columns]
        # Drop the empty column that results from a trailing comma in the CSV
        df.drop(columns=['Unnamed: 6'], inplace=True, errors='ignore')

        # 2. Multi-Timeframe Preprocessing
        # Calculate weekly aggregations
        weekly_agg = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })

        # Convert weekly_agg index to period for correct mapping
        weekly_agg.index = weekly_agg.index.to_period('W')

        # Map weekly data back to the 15m dataframe
        period_index = df.index.to_period('W')
        df['Weekly_Open'] = period_index.map(weekly_agg['Open'])
        df['Weekly_High'] = period_index.map(weekly_agg['High'])
        df['Weekly_Low'] = period_index.map(weekly_agg['Low'])
        df['Weekly_Close'] = period_index.map(weekly_agg['Close'])
        df['Weekly_Volume'] = period_index.map(weekly_agg['Volume'])

        # Forward-fill the weekly data to fill non-market hours and weekends
        df.ffill(inplace=True)
        df.bfill(inplace=True) # backfill for the start
        df.dropna(inplace=True)

        # 3. Run Backtest
        print("Running backtest...")
        bt = Backtest(df, WeinsteinStage2Trader, cash=100000, commission=.002)
        stats = bt.run()
        print(stats)

        # 4. Save Results
        os.makedirs('results', exist_ok=True)
        sanitized_results = sanitize_stats(stats)

        with open('results/temp_result.json', 'w') as f:
            json.dump(sanitized_results, f, indent=4)

        print("\nBacktest stats saved to results/temp_result.json")

        # 5. Generate Plot
        plot_filename = 'results/weinstein_stage2_trader_continuation_buy_plot.html'
        print(f"Generating plot... saved to {plot_filename}")
        try:
            bt.plot(filename=plot_filename, open_browser=False)
        except Exception as e:
            print(f"Could not generate plot: {e}")
