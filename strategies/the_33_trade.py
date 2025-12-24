from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
import os
from scipy.signal import find_peaks

def find_swing_points(data, prominence=2, width=1):
    """
    Identifies swing highs and lows in the data.
    Returns an array with 1 for swing highs, -1 for swing lows, and 0 otherwise.
    """
    peaks, _ = find_peaks(data, prominence=prominence, width=width)
    troughs, _ = find_peaks(-data, prominence=prominence, width=width)

    swings = np.zeros(len(data))
    swings[peaks] = 1
    swings[troughs] = -1
    return swings

def deduplicate_swings(swings):
    """
    Removes consecutive duplicate swing signals, keeping only the first occurrence.
    """
    if len(swings) == 0:
        return swings

    deduplicated = np.copy(swings)
    for i in range(1, len(deduplicated)):
        if deduplicated[i] != 0 and deduplicated[i] == deduplicated[i-1]:
            deduplicated[i] = 0
    return deduplicated

def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object to ensure it's JSON-serializable.
    Converts specific numpy types and pandas objects to native Python types.
    """
    sanitized = {}
    for key, value in stats.items():
        if key == '_strategy':
            continue
        if isinstance(value, (pd.DataFrame, pd.Series)):
            sanitized[key] = None
        elif isinstance(value, (np.integer, np.int_)):
            sanitized[key] = int(value)
        elif isinstance(value, np.floating):
            sanitized[key] = float(value)
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    return sanitized


class The33TradeStrategy(Strategy):
    """
    Strategy based on the "3-3 trade" concept of multi-day and intraday cycle peaks.
    """
    # Optimizable parameters
    rr_ratio = 2.0
    prominence = 2
    width = 1
    day_lookback = 288 # 3 days of 15m candles
    intraday_lookback = 12 # 3 hours of 15m candles

    def init(self):
        """
        Initialize indicators and strategy variables.
        """
        # Intraday (15m) swing points
        self.swing_points_15m = self.I(find_swing_points, self.data.Close, prominence=self.prominence, width=self.width)

        # Multi-day (1h) swing points
        df_1h = self.data.df.resample('1H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()

        swing_points_1h_series = pd.Series(find_swing_points(df_1h['Close'].values, prominence=self.prominence, width=self.width), index=df_1h.index)

        # Map 1h swing points back to 15m timeframe
        self.data.df['swing_points_1h'] = self.data.df.index.floor('H').map(swing_points_1h_series)
        self.data.df['swing_points_1h'].fillna(0, inplace=True)
        self.swing_points_1h = self.I(lambda: self.data.df['swing_points_1h'].values)


    def next(self):
        """
        Defines the trading logic for each bar.
        """
        # --- Multi-Day Count ---
        if len(self.swing_points_1h) < self.day_lookback:
            return

        recent_swings_1h = self.swing_points_1h[-self.day_lookback:]
        deduplicated_swings_1h = deduplicate_swings(recent_swings_1h)
        three_level_rise_1h = np.sum(deduplicated_swings_1h == 1) >= 3

        # --- Intraday Count ---
        if len(self.swing_points_15m) < self.intraday_lookback:
            return

        recent_swings_15m = self.swing_points_15m[-self.intraday_lookback:]
        three_level_rise_15m = np.sum(recent_swings_15m == 1) >= 3

        if three_level_rise_1h and three_level_rise_15m and not self.position:
            # Find the most recent swing high for stop loss placement
            recent_swing_high_indices = np.where(self.swing_points_15m[-self.intraday_lookback:] == 1)[0]
            if len(recent_swing_high_indices) > 0:
                last_swing_high_index = len(self.data.Close) - self.intraday_lookback + recent_swing_high_indices[-1]
                stop_loss = self.data.High[last_swing_high_index]
                entry_price = self.data.Close[-1]
                take_profit = entry_price - (stop_loss - entry_price) * self.rr_ratio

                if entry_price < stop_loss:
                    self.sell(sl=stop_loss, tp=take_profit)

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Clean up column names
        data.columns = [c.strip().title() for c in data.columns]
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    else:
        raise FileNotFoundError(f"Data file not found at {data_path}")

    bt = Backtest(data, The33TradeStrategy, cash=100000, commission=.002)

    stats = bt.run()

    os.makedirs('results', exist_ok=True)

    sanitized_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")

    try:
        plot_filename = 'results/the_33_trade.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
