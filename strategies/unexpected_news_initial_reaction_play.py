
import json
import os
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np


def preprocess_data(df: pd.DataFrame, gap_threshold=0.01, volume_lookback=20, volume_multiplier=1.5):
    """
    Pre-processes the data to identify potential "news" days based on
    opening gaps and unusual opening volume.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    daily_df = df.resample('D').agg({
        'Open': 'first',
        'Close': 'last',
        'Volume': 'first'
    }).dropna()

    daily_df['prev_close'] = daily_df['Close'].shift(1)
    daily_df['gap_pct'] = (daily_df['Open'] - daily_df['prev_close']) / daily_df['prev_close']
    daily_df['avg_open_volume'] = daily_df['Volume'].shift(1).rolling(window=volume_lookback).mean()
    daily_df['is_high_volume'] = daily_df['Volume'] > daily_df['avg_open_volume'] * volume_multiplier
    daily_df['is_news_day'] = (daily_df['gap_pct'].abs() > gap_threshold) & daily_df['is_high_volume']
    daily_df['gap_direction'] = np.sign(daily_df['gap_pct'])

    daily_signals = daily_df[['is_news_day', 'gap_direction']]
    df['date'] = df.index.date
    daily_signals.index = daily_signals.index.date
    df = pd.merge(df, daily_signals, left_on='date', right_index=True, how='left')
    df['is_news_day'] = df['is_news_day'].fillna(False)
    df['gap_direction'] = df['gap_direction'].fillna(0)
    df = df.drop(columns=['date'])
    return df


class UnexpectedNewsInitialReactionPlay(Strategy):
    observation_minutes = 60
    sl_pct_from_range = 0.25
    min_rr = 2.0

    def init(self):
        self.trade_day_status = "WAITING"
        self.observation_high = -np.inf
        self.observation_low = np.inf
        self.observation_end_bar = -1
        self.observation_bars = self.observation_minutes // 15 if self.observation_minutes > 0 else 1
        self.daily_gap_direction = 0

    def next(self):
        current_time = self.data.index[-1]
        current_bar_index = len(self.data.Close) - 1
        is_first_bar_of_day = current_time.hour == 0 and current_time.minute == 0

        if is_first_bar_of_day:
            self.trade_day_status = "WAITING"
            self.observation_high = -np.inf
            self.observation_low = np.inf
            if self.data.is_news_day[-1]:
                self.trade_day_status = "OBSERVING"
                self.daily_gap_direction = self.data.gap_direction[-1]
                self.observation_end_bar = current_bar_index + self.observation_bars - 1

        if self.trade_day_status == "OBSERVING":
            self.observation_high = max(self.observation_high, self.data.High[-1])
            self.observation_low = min(self.observation_low, self.data.Low[-1])
            if current_bar_index >= self.observation_end_bar:
                self.trade_day_status = "TRADING"

        if self.position:
            # Correct exit logic
            if self.position.is_short and self.data.Close[-1] > self.observation_high:
                self.position.close()
            elif self.position.is_long and self.data.Close[-1] < self.observation_low:
                self.position.close()
            return

        if self.trade_day_status == "TRADING":
            range_size = self.observation_high - self.observation_low
            if range_size <= 0:
                self.trade_day_status = "DONE"
                return

            sl_buffer = range_size * self.sl_pct_from_range

            if self.daily_gap_direction == 1: # Gap up, look for short
                entry_price = self.data.Close[-1]
                stop_loss = self.observation_high + sl_buffer
                take_profit = entry_price - (stop_loss - entry_price) * self.min_rr
                if take_profit < entry_price:
                    self.sell(sl=stop_loss, tp=take_profit)
                    self.trade_day_status = "DONE"

            elif self.daily_gap_direction == -1: # Gap down, look for long
                entry_price = self.data.Close[-1]
                stop_loss = self.observation_low - sl_buffer
                take_profit = entry_price + (entry_price - stop_loss) * self.min_rr
                if take_profit > entry_price:
                    self.buy(sl=stop_loss, tp=take_profit)
                    self.trade_day_status = "DONE"


if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        print("Error: Data file not found. Using dummy data.")
        data = pd.DataFrame(index=pd.to_datetime(pd.date_range('2023-01-01', periods=4000, freq='15min')))
        data['Open'] = 100 + np.random.randn(4000).cumsum()
        data['High'] = data['Open'] + np.random.rand(4000)
        data['Low'] = data['Open'] - np.random.rand(4000)
        data['Close'] = data['Open'] + np.random.randn(4000)
        data['Volume'] = np.random.randint(100, 1000, 4000)
    data.columns = [c.title() for c in data.columns]

    print("Preprocessing data...")
    processed_data = preprocess_data(data.copy())

    bt = Backtest(processed_data, UnexpectedNewsInitialReactionPlay, cash=100_000, commission=.002)

    print("Running backtest...")
    stats = bt.run()
    print(stats)

    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if key.startswith('_'):
                continue
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                sanitized[key] = str(value)
            elif pd.isna(value):
                sanitized[key] = None
            elif isinstance(value, (np.integer, np.int64)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                sanitized[key] = float(value)
            else:
                sanitized[key] = value
        return sanitized

    results = sanitize_stats(stats)
    results['strategy_name'] = 'unexpected_news_initial_reaction_play'
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump(results, f, indent=2)
        f.write('\n')
    print("Results saved to results/temp_result.json")

    try:
        plot_filename = 'results/unexpected_news_initial_reaction_play.html'
        bt.plot(filename=plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
