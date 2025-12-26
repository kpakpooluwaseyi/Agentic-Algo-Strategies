import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
import json
import os

def sanitize_stats(stats):
    """Sanitizes the stats object for JSON serialization."""
    # Create a copy to avoid modifying the original
    stats_dict = dict(stats).copy()

    # List of keys to remove
    keys_to_remove = [
        '_strategy',
        '_equity_curve',
        '_trades',
        'Equity Final [$]',
        'Equity Peak [$]',
        'Start',
        'End',
        'Duration'
    ]
    for key in keys_to_remove:
        stats_dict.pop(key, None)

    # Convert specific types to JSON-serializable formats
    for key, value in list(stats_dict.items()):
        if pd.isna(value) or value is None:
            stats_dict[key] = None
        elif isinstance(value, pd.Timestamp):
            stats_dict[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            stats_dict[key] = str(value)
        elif isinstance(value, (int, float, str, bool)):
            # These types are already serializable
            continue
        else:
            # For other types (like numpy int64), convert to Python native types
            try:
                stats_dict[key] = value.item()
            except AttributeError:
                 stats_dict[key] = str(value) # Fallback to string representation

    return stats_dict

class NordenFuturesOrderflowScalping(Strategy):
    """
    A proxy strategy for Norden's order flow scalping. It uses Bollinger Bands
    and high volume to identify potential exhaustion points where 'weak players'
    might be trapped, creating a reversal opportunity.
    """
    # Bollinger Bands parameters
    bb_period = 20
    bb_std = 2.0

    # Volume filter parameters
    volume_ma_period = 20
    volume_multiplier = 2.5

    # Risk management parameters
    sl_pct = 0.015  # 1.5% stop loss
    tp_pct = 0.02   # 2% take profit

    # Time-based exit
    hold_bars = 4 # Exit after 4 bars if not stopped out

    def init(self):
        # Calculate Bollinger Bands
        bbands = self.data.df.ta.bbands(length=self.bb_period, std=self.bb_std)
        # Find column names dynamically to avoid issues with float representation
        upper_col = [col for col in bbands.columns if col.startswith(f'BBU_{self.bb_period}')][0]
        lower_col = [col for col in bbands.columns if col.startswith(f'BBL_{self.bb_period}')][0]
        self.bb_upper = self.I(lambda: bbands[upper_col])
        self.bb_lower = self.I(lambda: bbands[lower_col])

        # Calculate Volume Moving Average
        volume_ma = self.data.df['Volume'].rolling(window=self.volume_ma_period).mean()
        self.volume_ma = self.I(lambda: volume_ma)

        # Track bars since entry
        self.bars_since_entry = 0

    def next(self):
        # === Position Management ===
        if self.position:
            self.bars_since_entry += 1
            # Time-based exit
            if self.bars_since_entry >= self.hold_bars:
                self.position.close()
            return

        # Reset counter if no position
        self.bars_since_entry = 0

        # === Entry Conditions ===
        high_volume = self.data.Volume[-1] > self.volume_ma[-1] * self.volume_multiplier

        # Short Entry: Price closes above upper Bollinger Band on high volume
        if self.data.Close[-1] > self.bb_upper[-1] and high_volume:
            sl = self.data.Close[-1] * (1 + self.sl_pct)
            tp = self.data.Close[-1] * (1 - self.tp_pct)
            self.sell(sl=sl, tp=tp)

        # Long Entry: Price closes below lower Bollinger Band on high volume
        elif self.data.Close[-1] < self.bb_lower[-1] and high_volume:
            sl = self.data.Close[-1] * (1 - self.sl_pct)
            tp = self.data.Close[-1] * (1 + self.tp_pct)
            self.buy(sl=sl, tp=tp)


if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    # Load data
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # Clean and capitalize column names
    data.columns = [col.strip().title() for col in data.columns]

    # Backtest
    bt = Backtest(data, NordenFuturesOrderflowScalping, cash=100000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()

    print(stats)

    # Save results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Sanitize stats for JSON
    sanitized = sanitize_stats(stats)

    with open(f'{results_dir}/temp_result.json', 'w') as f:
        json.dump(sanitized, f, indent=4)

    print(f"Results saved to {results_dir}/temp_result.json")

    # Plot
    plot_filename = f'{results_dir}/norden_futures_orderflow_scalping.html'
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
