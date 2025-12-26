
import pandas as pd
from backtesting import Backtest, Strategy
import pandas_ta as ta
import json
import numpy as np
import os

class WeinsteinStageAnalysis(Strategy):
    """
    Implementation of Stan Weinstein's Stage Analysis method.
    This strategy buys when the market enters a Stage 2 uptrend
    and sells when it enters a Stage 4 downtrend, based on weekly analysis.
    """
    def init(self):
        """
        Initializes the strategy by creating aliases for the pre-calculated weekly data.
        """
        self.weekly_ema = self.I(lambda x: x, self.data.weekly_ema30)
        self.stage = self.I(lambda x: x, self.data.stage)

    def next(self):
        """
        Defines the trading logic for each bar.
        """
        is_stage2_entry = self.stage[-1] == 2 and self.stage[-2] != 2

        if not self.position and is_stage2_entry:
            self.buy()

        is_stage4_exit = self.stage[-1] == 4 and self.stage[-2] != 4

        if self.position and is_stage4_exit:
            self.position.close()

def preprocess_data(df):
    """
    Resamples data to weekly, calculates indicators, and merges them back
    using a robust time-series join.
    """
    weekly_df = df.resample('W-MON').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    weekly_df['ema30'] = ta.ema(weekly_df['Close'], length=30)
    weekly_df['ema30_slope'] = weekly_df['ema30'].diff()

    weekly_df['stage'] = 0
    weekly_df.loc[(weekly_df['Close'] > weekly_df['ema30']) & (weekly_df['ema30_slope'] > 0), 'stage'] = 2
    weekly_df.loc[(weekly_df['Close'] < weekly_df['ema30']) & (weekly_df['ema30_slope'] < 0), 'stage'] = 4

    weekly_features = weekly_df[['ema30', 'stage']].shift(1)
    weekly_features.rename(columns={'ema30': 'weekly_ema30'}, inplace=True)

    df.sort_index(inplace=True)
    weekly_features.sort_index(inplace=True)

    df = pd.merge_asof(
        left=df,
        right=weekly_features,
        left_index=True,
        right_index=True,
        direction='backward'
    )

    df.dropna(inplace=True)
    return df

if __name__ == '__main__':
    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True, skipinitialspace=True)
        data.columns = [col.strip().capitalize() for col in data.columns]
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    except FileNotFoundError:
        print("Error: Data file not found. Please ensure 'data/BTC-USD-15m.csv' exists.")
        exit()

    data = preprocess_data(data)

    if data.empty:
        print("Error: No data left after preprocessing. Cannot run backtest.")
        exit()

    bt = Backtest(data, WeinsteinStageAnalysis, cash=100000, commission=.002, finalize_trades=True)
    stats = bt.run()

    print(stats)

    plot_filename = 'results/weinstein_investor_stage_analysis.html'
    bt.plot(filename=plot_filename)
    print(f"Plot saved to {plot_filename}")

    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (np.integer, np.int64)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                sanitized[key] = float(value)
            elif isinstance(value, pd.Timestamp):
                sanitized[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                sanitized[key] = str(value)
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                continue
            elif key == '_strategy':
                continue
            else:
                sanitized[key] = value
        return sanitized

    os.makedirs('results', exist_ok=True)

    sanitized_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    print("Backtest statistics saved to results/temp_result.json")
