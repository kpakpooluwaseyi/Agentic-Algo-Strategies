import json
import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class RajanDalTrendFollowing(Strategy):
    """
    Implementation of the Rajan Dal Trend Following Pullback Strategy.
    """
    # Strategy parameters
    ema_period = 21
    rsi_period = 14
    rsi_pullback_long = 30
    rsi_pullback_short = 70
    sl_lookback = 20
    rr_ratio = 2.0

    def init(self):
        # Pre-calculated indicators from the input DataFrame
        self.daily_ema = self.I(lambda x: x, self.data.df['daily_ema'], name='daily_ema')
        self.daily_rsi = self.I(lambda x: x, self.data.df['daily_rsi'], name='daily_rsi')
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), length=self.rsi_period)

    def next(self):
        price = self.data.Close[-1]

        # Trend Confirmation (Higher Timeframe)
        is_uptrend = price > self.daily_ema[-1] and self.daily_rsi[-1] > 50
        is_downtrend = price < self.daily_ema[-1] and self.daily_rsi[-1] < 50

        # Entry Logic
        if not self.position:
            if is_uptrend and crossover(self.rsi, self.rsi_pullback_long):
                # Long entry on pullback
                stop_loss = self.data.Low[-self.sl_lookback:].min()
                take_profit = price + (price - stop_loss) * self.rr_ratio
                self.buy(sl=stop_loss, tp=take_profit)

            elif is_downtrend and crossover(self.rsi_pullback_short, self.rsi):
                # Short entry on pullback
                stop_loss = self.data.High[-self.sl_lookback:].max()
                take_profit = price - (stop_loss - price) * self.rr_ratio
                self.sell(sl=stop_loss, tp=take_profit)

def sanitize_stats(stats):
    """
    Sanitizes the stats object from a backtest run to make it JSON serializable.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = float(value)
        elif isinstance(value, (pd.DataFrame, pd.Series, Strategy)):
            continue  # Skip non-serializable types
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    return sanitized

if __name__ == '__main__':
    # Load and preprocess data
    data_path = 'data/BTC-USD-15m.csv'
    try:
        data = pd.read_csv(data_path)
        # Sanitize column names: strip spaces and make them title case for consistency
        data.columns = [c.strip().title() for c in data.columns]
        if 'Datetime' in data.columns:
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            data.set_index('Datetime', inplace=True)
        else:
            raise KeyError("'datetime' column not found after sanitization.")

        # Drop the empty column created by the trailing comma in the header
        data.dropna(axis=1, how='all', inplace=True)

    except FileNotFoundError:
        print(f"Data file not found at {data_path}. Please ensure the file exists.")
        exit()

    # --- Pre-computation of Daily Indicators ---
    daily_data = data.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    daily_data['daily_ema'] = ta.ema(daily_data['Close'], length=21)
    daily_data['daily_rsi'] = ta.rsi(daily_data['Close'], length=14)

    # Use .shift(1) to avoid lookahead bias
    data['daily_ema'] = daily_data['daily_ema'].resample('15min').ffill().reindex(data.index, method='ffill').shift(1)
    data['daily_rsi'] = daily_data['daily_rsi'].resample('15min').ffill().reindex(data.index, method='ffill').shift(1)

    data.dropna(inplace=True)

    # Run backtest
    bt = Backtest(data, RajanDalTrendFollowing, cash=100000, commission=.002)
    stats = bt.run()

    print(stats)

    # Save results
    results_path = 'results/temp_result.json'
    plot_path = 'results/rajan_dal_trend_following_pullback.html'

    sanitized_stats = sanitize_stats(stats)
    with open(results_path, 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    try:
        bt.plot(filename=plot_path)
    except Exception as e:
        print(f"Could not generate plot: {e}")

    print(f"\nBacktest complete. Results saved to {results_path}")
    print(f"Plot saved to {plot_path}")
