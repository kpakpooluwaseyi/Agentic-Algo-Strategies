from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import numpy as np
import json

def ema_indicator(series, length):
    """
    Calculates EMA using pandas_ta and returns a writeable numpy array.
    This is necessary to avoid "ValueError: output array is read-only"
    with FractionalBacktest.
    """
    return ta.ema(pd.Series(series), length=length).values.copy()


class GannPrinciplesTrendFollowing(Strategy):
    # --- Strategy Parameters ---
    main_trend_ema_period = 200
    volume_ema_period = 20
    stop_loss_pct = 0.03  # 3%
    trailing_stop_pct = 0.04 # 4%

    # --- Pyramiding Parameters ---
    initial_position_size = 0.1 # Start with 10% of equity
    max_pyramid_entries = 4  # Max number of entries (initial + 3 pyramids)
    pyramid_entry_size_decay = 0.5 # Each pyramid entry is 50% of the previous one
    pyramid_trigger_pct = 0.02 # Add to position after a 2% move in our favor

    def init(self):
        # --- Indicators ---
        self.main_trend_ema = self.I(ema_indicator, self.data.Close, self.main_trend_ema_period)
        self.volume_ema = self.I(ema_indicator, self.data.Volume, self.volume_ema_period)

        # --- Pre-calculate Day of Week ---
        # 0=Monday, 1=Tuesday, ..., 6=Sunday
        # .copy() is needed to make the array writeable for FractionalBacktest
        self.day_of_week = self.I(lambda x: x, self.data.index.dayofweek.copy())

        # --- State Variables ---
        self.last_entry_price = None


    def next(self):
        # --- Exit Logic ---
        # 1. Trend Reversal Exit
        if self.position.is_long and crossover(self.main_trend_ema, self.data.Close):
            self.position.close()
            self.last_entry_price = None # Reset on close
            return

        if self.position.is_short and crossover(self.data.Close, self.main_trend_ema):
            self.position.close()
            self.last_entry_price = None # Reset on close
            return

        # 2. Pyramiding and Trailing Stop-Loss
        if self.position.is_long:
            # Trailing stop
            new_stop = self.data.Close[-1] * (1 - self.trailing_stop_pct)
            # Update SL for all open trades in the position
            for trade in self.trades:
                if new_stop > trade.sl:
                    trade.sl = new_stop

            # Pyramiding
            if self.last_entry_price and len(self.trades) < self.max_pyramid_entries:
                if self.data.Close[-1] > self.last_entry_price * (1 + self.pyramid_trigger_pct):
                    # Calculate new size as a decaying fraction of the initial size
                    new_size = self.initial_position_size * (self.pyramid_entry_size_decay ** len(self.trades))
                    sl = self.data.Close[-1] * (1 - self.stop_loss_pct)
                    self.buy(size=new_size, sl=sl)
                    self.last_entry_price = self.data.Close[-1]

        elif self.position.is_short:
            # Trailing stop
            new_stop = self.data.Close[-1] * (1 + self.trailing_stop_pct)
            for trade in self.trades:
                if new_stop < trade.sl:
                    trade.sl = new_stop

            # Pyramiding
            if self.last_entry_price and len(self.trades) < self.max_pyramid_entries:
                if self.data.Close[-1] < self.last_entry_price * (1 - self.pyramid_trigger_pct):
                    new_size = self.initial_position_size * (self.pyramid_entry_size_decay ** len(self.trades))
                    sl = self.data.Close[-1] * (1 + self.stop_loss_pct)
                    self.sell(size=new_size, sl=sl)
                    self.last_entry_price = self.data.Close[-1]

        # --- Initial Entry Logic ---
        if self.position:
            return

        # --- Main Trend Confirmation ---
        is_bullish_trend = self.data.Close[-1] > self.main_trend_ema[-1]
        is_bearish_trend = self.data.Close[-1] < self.main_trend_ema[-1]

        # --- Volume Confirmation ---
        is_strong_volume = self.data.Volume[-1] > self.volume_ema[-1]

        # --- Monday Entry Logic ---
        # Monday is 0
        if self.day_of_week[-1] == 0:
            # Look back to the previous Friday. Assuming 15-min data,
            # a full weekend is 2 days * 24 hours/day * 4 bars/hour = 192 bars
            # A friday would be 3 days before -> 3 * 24 * 4 = 288 bars ago.
            # This is a rough approximation due to market closures and data gaps.
            # A more robust method would be to analyze the previous day's data,
            # but for this implementation we will check the recent trend.

            # Corrected check: Look at the trend over the last 3 days (approx Friday)
            friday_lookback = 3 * 96 # 3 days of 15-min bars
            if len(self.data.Close) > friday_lookback:
                recent_trend = self.data.Close[-1] - self.data.Close[-friday_lookback]

                # LONG Entry: Trend is bullish, but recent move (Fri-Mon) was weak/down.
                if is_bullish_trend and recent_trend < 0 and is_strong_volume:
                    # Wait for the first hour to pass (4 bars on 15m timeframe)
                    if self.data.index[-1].hour >= 1:
                        sl = self.data.Close[-1] * (1 - self.stop_loss_pct)
                        self.buy(size=self.initial_position_size, sl=sl)
                        self.last_entry_price = self.data.Close[-1]
                        return

                # SHORT Entry: Trend is bearish, but recent move (Fri-Mon) was strong/up.
                if is_bearish_trend and recent_trend > 0 and is_strong_volume:
                    # Wait for the first hour to pass
                    if self.data.index[-1].hour >= 1:
                        sl = self.data.Close[-1] * (1 + self.stop_loss_pct)
                        self.sell(size=self.initial_position_size, sl=sl)
                        self.last_entry_price = self.data.Close[-1]
                        return

        # --- General Trend Following Entry (for other days) ---
        # Simple entry for now: enter on trend confirmation if volume is good
        # This part can be expanded with more sophisticated Gann analysis
        else:
            if is_bullish_trend and is_strong_volume:
                sl = self.data.Close[-1] * (1 - self.stop_loss_pct)
                self.buy(size=self.initial_position_size, sl=sl)
                self.last_entry_price = self.data.Close[-1]

            elif is_bearish_trend and is_strong_volume:
                sl = self.data.Close[-1] * (1 + self.stop_loss_pct)
                self.sell(size=self.initial_position_size, sl=sl)
                self.last_entry_price = self.data.Close[-1]

if __name__ == '__main__':
    import os
    from backtesting import Backtest

    # --- Data Loading ---
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        print(f"Loading data from: {data_path}")
        # Add `skipinitialspace=True` to handle malformed headers with leading spaces
        data = pd.read_csv(data_path, index_col=0, parse_dates=True, skipinitialspace=True)
        # Remove any unnamed columns that may result from trailing commas in the header
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        # Clean and standardize column names
        data.columns = [c.strip().title() for c in data.columns]

        # --- Backtesting ---
        # The strategy requires a custom Backtest class for fractional sizing
        from backtesting.lib import FractionalBacktest
        bt = FractionalBacktest(data, GannPrinciplesTrendFollowing, cash=100000, commission=.002)

        stats = bt.run()

        # --- Results & Plotting ---
        print("--- Backtest Results ---")
        print(stats)

        # A robust way to sanitize stats for JSON serialization
        def sanitize_stats(stats):
            # This handles cases where metrics might be missing or have non-serializable types
            sanitized = {
                'strategy_name': 'gann_principles_trend_following',
                'return': stats.get('Return [%]', 0.0),
                'sharpe': stats.get('Sharpe Ratio', 0.0),
                'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
                'win_rate': stats.get('Win Rate [%]', 0.0),
                'total_trades': stats.get('# Trades', 0)
            }
            for key, value in sanitized.items():
                if isinstance(value, (np.floating, np.integer)):
                    sanitized[key] = float(value) if np.isfinite(value) else None
                elif isinstance(value, int):
                     sanitized[key] = int(value)
                elif pd.isna(value):
                    sanitized[key] = None
            return sanitized

        os.makedirs('results', exist_ok=True)
        final_stats = sanitize_stats(stats)

        with open('results/temp_result.json', 'w') as f:
            json.dump(final_stats, f, indent=2)
        print("\nBacktest results saved to results/temp_result.json")

        try:
            plot_filename = 'results/gann_principles_trend_following.html'
            bt.plot(filename=plot_filename, open_browser=False)
            print(f"Backtest plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot: {e}")
