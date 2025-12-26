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


class Strategy077077237895(Strategy):
    # --- Strategy Parameters ---
    main_trend_ema_period = 200
    volume_ema_period = 20
    stop_loss_pct = 0.05  # 5%
    take_profit_pct = 0.10 # 10%
    position_size = 0.1 # 10% of equity

    def init(self):
        # --- Indicators ---
        self.main_trend_ema = self.I(ema_indicator, self.data.Close, self.main_trend_ema_period)
        self.volume_ema = self.I(ema_indicator, self.data.Volume, self.volume_ema_period)

    def next(self):
        # --- Exit Logic ---
        # If position exists, it will be closed by SL or TP.
        # We can add other exit logic here, like a trend reversal.
        if self.position.is_long and crossover(self.main_trend_ema, self.data.Close):
            self.position.close()
            return

        if self.position.is_short and crossover(self.data.Close, self.main_trend_ema):
            self.position.close()
            return

        # --- Initial Entry Logic ---
        if self.position:
            return

        # --- Main Trend Confirmation ---
        is_bullish_trend = self.data.Close[-1] > self.main_trend_ema[-1]
        is_bearish_trend = self.data.Close[-1] < self.main_trend_ema[-1]

        # --- Volume Confirmation ---
        is_strong_volume = self.data.Volume[-1] > self.volume_ema[-1]

        # --- General Trend Following Entry ---
        if is_bullish_trend and is_strong_volume:
            sl = self.data.Close[-1] * (1 - self.stop_loss_pct)
            tp = self.data.Close[-1] * (1 + self.take_profit_pct)
            self.buy(size=self.position_size, sl=sl, tp=tp)

        elif is_bearish_trend and is_strong_volume:
            sl = self.data.Close[-1] * (1 + self.stop_loss_pct)
            tp = self.data.Close[-1] * (1 - self.take_profit_pct)
            self.sell(size=self.position_size, sl=sl, tp=tp)


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
        bt = FractionalBacktest(data, Strategy077077237895, cash=100000, commission=.002)

        stats = bt.run()

        # --- Results & Plotting ---
        print("--- Backtest Results ---")
        print(stats)

        # A robust way to sanitize stats for JSON serialization
        def sanitize_stats(stats):
            # This handles cases where metrics might be missing or have non-serializable types
            sanitized = {
                'strategy_name': 'Strategy077077237895',
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
            plot_filename = 'results/strategy_077077237895.html'
            bt.plot(filename=plot_filename, open_browser=False)
            print(f"Backtest plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot: {e}")
