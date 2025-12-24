import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


def swing_high(series_input, period: int) -> pd.Series:
    """Returns the rolling swing high."""
    series = pd.Series(series_input)
    return series.rolling(period).max().copy()

def swing_low(series_input, period: int) -> pd.Series:
    """Returns the rolling swing low."""
    series = pd.Series(series_input)
    return series.rolling(period).min().copy()

class Strategy_9b06d97de898(Strategy):
    """
    Swing Breakout Strategy
    Inspired by W.D. Gann's principles of trading significant price movements.
    This strategy identifies swing highs and lows and trades the breakouts.
    Risk management is based on the Average True Range (ATR).
    """
    # Optimizable parameters
    swing_lookback = 30
    atr_period = 14
    atr_multiplier_sl = 2.0
    atr_multiplier_tp = 4.0

    def init(self):
        # Wrapper for ATR to handle _Array type and ensure writability
        def atr_wrapper(high, low, close, length):
            high_s = pd.Series(high)
            low_s = pd.Series(low)
            close_s = pd.Series(close)
            atr = ta.atr(high=high_s, low=low_s, close=close_s, length=length)
            return atr.copy()

        # Pre-calculate indicators
        self.atr = self.I(atr_wrapper, self.data.High, self.data.Low, self.data.Close, length=self.atr_period)
        self.swing_high = self.I(swing_high, self.data.High, self.swing_lookback)
        self.swing_low = self.I(swing_low, self.data.Low, self.swing_lookback)

    def next(self):
        # Get the most recent values
        price = self.data.Close[-1]
        atr_value = self.atr[-1]

        # Define stop-loss and take-profit levels
        sl_long = price - atr_value * self.atr_multiplier_sl
        tp_long = price + atr_value * self.atr_multiplier_tp

        sl_short = price + atr_value * self.atr_multiplier_sl
        tp_short = price - atr_value * self.atr_multiplier_tp

        # Entry conditions
        # Only trade if not already in a position
        if not self.position:
            # Long entry: price breaks above the swing high of the previous bar
            if crossover(self.data.High, self.swing_high[-2]):
                # Validate that TP > price to avoid errors
                if tp_long > price:
                    self.buy(sl=sl_long, tp=tp_long, size=0.001)

            # Short entry: price breaks below the swing low of the previous bar
            elif crossover(self.swing_low[-2], self.data.Low):
                # Validate that TP < price to avoid errors
                if tp_short < price:
                    self.sell(sl=sl_short, tp=tp_short, size=0.001)

# --- Backtesting Harness ---
if __name__ == '__main__':
    import json
    import os

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Load data
    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True, skipinitialspace=True)
        # Sanitize headers
        data.columns = [c.strip().title() for c in data.columns]
        # Drop unnamed columns that might be created by trailing commas
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found.")
        # As a fallback, create some synthetic data to allow the script to run
        print("Generating synthetic data for demonstration.")
        from backtesting.test import GOOG
        data = GOOG.iloc[-2000:].copy()
        data.index.name = 'datetime'

    # Instantiate and run the backtest
    bt = Backtest(data, Strategy_9b06d97de898, cash=100000, commission=.002, finalize_trades=True)
    stats = bt.run()

    # Print the stats
    print(stats)

    # Save stats to a JSON file
    stats_dict = dict(stats)

    # Remove non-serializable objects
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    # Sanitize the rest of the stats object for JSON serialization
    for key, value in list(stats_dict.items()):
        if isinstance(value, pd.Series):
            stats_dict[key] = value.to_dict()
        elif isinstance(value, pd.DataFrame):
            stats_dict.pop(key, None)
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
             stats_dict[key] = str(value)
        elif pd.isna(value) or value is pd.NA:
            stats_dict[key] = None
        elif isinstance(value, (float, int, str, bool)) or value is None:
            continue # Already serializable
        else:
            # Fallback for other types, like numpy integers
            try:
                stats_dict[key] = value.item()
            except AttributeError:
                stats_dict[key] = str(value)


    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)

    # Generate the plot
    plot_filename = 'results/strategy_9b06d97de898.html'
    try:
        bt.plot(filename=plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
