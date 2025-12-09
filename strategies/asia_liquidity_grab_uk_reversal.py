import json
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

def generate_forex_data(periods=5000):
    """
    Generates synthetic 15-minute Forex data with engineered patterns
    for the Asia Liquidity Grab UK Reversal strategy.
    """
    rng = np.random.default_rng(seed=42)
    index = pd.to_datetime('2023-01-01', utc=True) + pd.to_timedelta(np.arange(periods) * 15, unit='min')

    base_price = 1.10
    returns = rng.normal(loc=0.00005, scale=0.0005, size=periods)
    price = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame(index=index)
    df['Open'] = price
    df['Close'] = price + rng.normal(loc=0, scale=0.0003, size=periods)
    df['High'] = np.maximum(df['Open'], df['Close']) + rng.uniform(0, 0.0005, size=periods)
    df['Low'] = np.minimum(df['Open'], df['Close']) - rng.uniform(0, 0.0005, size=periods)
    df['Volume'] = rng.integers(100, 1000, size=periods)

    # --- Engineer a Short Entry Pattern ---
    short_day = '2023-01-03'
    short_day_indices = np.where(df.index.date == pd.to_datetime(short_day).date())[0]
    if len(short_day_indices) > 33:
        asia_start_idx, uk_start_idx = short_day_indices[0], short_day_indices[32]
        asia_end_idx = uk_start_idx
        hoa = df.iloc[asia_start_idx:asia_end_idx]['High'].max()
        df.loc[df.index[uk_start_idx], 'High'] = hoa * 1.001
        df.loc[df.index[uk_start_idx], 'Close'] = hoa * 0.999
        df.loc[df.index[uk_start_idx+1], 'Open'] = df.iloc[uk_start_idx]['Close'] + 0.0001
        df.loc[df.index[uk_start_idx+1], 'Close'] = df.iloc[uk_start_idx]['Open'] - 0.0001

    # --- Engineer a Long Entry Pattern ---
    long_day = '2023-01-05'
    long_day_indices = np.where(df.index.date == pd.to_datetime(long_day).date())[0]
    if len(long_day_indices) > 33:
        asia_start_idx_2, uk_start_idx_2 = long_day_indices[0], long_day_indices[32]
        asia_end_idx_2 = uk_start_idx_2
        loa = df.iloc[asia_start_idx_2:asia_end_idx_2]['Low'].min()
        df.loc[df.index[uk_start_idx_2], 'Low'] = loa * 0.999
        df.loc[df.index[uk_start_idx_2], 'Close'] = loa * 1.001
        df.loc[df.index[uk_start_idx_2+1], 'Open'] = df.iloc[uk_start_idx_2]['Close'] - 0.0001
        df.loc[df.index[uk_start_idx_2+1], 'Close'] = df.iloc[uk_start_idx_2]['Open'] + 0.0001

    return df

def preprocess_data(df: pd.DataFrame):
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    df['date'] = df.index.date
    df['hour'] = df.index.hour

    is_asia = (df['hour'] >= 0) & (df['hour'] < 8)
    asia_range = df[is_asia].groupby('date').agg(HOA=('High', 'max'), LOA=('Low', 'min'))

    df = df.merge(asia_range, left_on='date', right_index=True, how='left')
    df[['HOA', 'LOA']] = df[['HOA', 'LOA']].ffill()

    df['daily_high'] = df.groupby('date')['High'].transform('max')
    df['daily_low'] = df.groupby('date')['Low'].transform('min')
    df['daily_50'] = (df['daily_high'] + df['daily_low']) / 2

    weekly_grouper = pd.Grouper(freq='W-SUN', closed='right', label='right')
    df['weekly_high'] = df.groupby(weekly_grouper)['High'].transform('max')
    df['weekly_low'] = df.groupby(weekly_grouper)['Low'].transform('min')
    df['weekly_50'] = (df['weekly_high'] + df['weekly_low']) / 2

    prev_open = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)

    df['is_bearish_engulfing'] = (prev_close > prev_open) & (df['Open'] > df['Close']) & \
                                (df['Open'] > prev_close) & (df['Close'] < prev_open)

    df['is_bullish_engulfing'] = (prev_open > prev_close) & (df['Close'] > df['Open']) & \
                                (df['Close'] > prev_open) & (df['Open'] < prev_close)

    df.dropna(inplace=True)
    return df

class AsiaLiquidityGrabUkReversalStrategy(Strategy):
    asia_range_max_percent = 2.0

    def init(self):
        self.reset_entry_state()
        self.reset_trade_management_state()

    def reset_entry_state(self):
        self.breakout_type = None
        self.liquidity_grab_price = None

    def reset_trade_management_state(self):
        self.trade_entry_price = None
        self.tp1_target = None
        self.tp1_hit = False
        self.final_tp_target = None

    def next(self):
        # --- State Resets ---
        if self.data.index[-1].date() != self.data.index[-2].date():
            self.reset_entry_state()
        if not self.position:
            self.reset_trade_management_state()

        # --- Manage Open Position ---
        if self.position:
            # First bar of the trade, initialize management state
            if self.trade_entry_price is None:
                self.trade_entry_price = self.trades[0].entry_price
                if self.position.is_long:
                    self.tp1_target = self.data.HOA[-1]
                else: # Short
                    self.tp1_target = self.data.LOA[-1]

            # TP1 Logic
            if not self.tp1_hit and self.tp1_target:
                if self.position.is_long and self.data.High[-1] >= self.tp1_target:
                    self.position.close(portion=0.5)
                    self.trades[0].sl = self.trade_entry_price
                    self.tp1_hit = True
                elif self.position.is_short and self.data.Low[-1] <= self.tp1_target:
                    self.position.close(portion=0.5)
                    self.trades[0].sl = self.trade_entry_price
                    self.tp1_hit = True

            # Final TP Logic
            if self.tp1_hit:
                if self.final_tp_target is None:
                    # Set the closer of daily or weekly 50% as the final target
                    if self.position.is_long:
                        self.final_tp_target = min(self.data.daily_50[-1], self.data.weekly_50[-1])
                    else:
                        self.final_tp_target = max(self.data.daily_50[-1], self.data.weekly_50[-1])

                if self.position.is_long and self.data.High[-1] >= self.final_tp_target:
                    self.position.close()
                elif self.position.is_short and self.data.Low[-1] <= self.final_tp_target:
                    self.position.close()
            return

        # --- Entry Logic (No changes from previous step) ---
        if pd.isna(self.data.HOA[-1]) or pd.isna(self.data.LOA[-1]):
            return

        current_time = self.data.index[-1]
        is_uk_session = 8 <= current_time.hour < 16

        asia_range = (self.data.HOA[-1] - self.data.LOA[-1]) / self.data.LOA[-1] * 100
        if not (0 < asia_range < self.asia_range_max_percent):
            return

        if is_uk_session and self.breakout_type is None:
            if self.data.High[-1] > self.data.HOA[-1]:
                self.breakout_type = 'short'
                self.liquidity_grab_price = self.data.High[-1]
            elif self.data.Low[-1] < self.data.LOA[-1]:
                self.breakout_type = 'long'
                self.liquidity_grab_price = self.data.Low[-1]

        if self.breakout_type and not self.position:
            sl = self.liquidity_grab_price
            if self.breakout_type == 'short' and self.data.is_bearish_engulfing[-1] and self.data.Close[-1] < self.data.HOA[-1]:
                self.sell(sl=sl)
                self.reset_entry_state()
            elif self.breakout_type == 'long' and self.data.is_bullish_engulfing[-1] and self.data.Close[-1] > self.data.LOA[-1]:
                self.buy(sl=sl)
                self.reset_entry_state()

if __name__ == '__main__':
    data = generate_forex_data(periods=10000)
    data = preprocess_data(data.copy())

    bt = Backtest(data, AsiaLiquidityGrabUkReversalStrategy, cash=100_000, commission=.002, finalize_trades=True)

    stats = bt.optimize(
        asia_range_max_percent=np.arange(1.0, 3.5, 0.5).tolist(),
        maximize='Sharpe Ratio',
        constraint=lambda p: p.asia_range_max_percent > 0
    )

    print(stats)

    results = {
        'strategy_name': 'asia_liquidity_grab_uk_reversal',
        'return': None, 'sharpe': None, 'max_drawdown': None,
        'win_rate': None, 'total_trades': 0
    }

    if stats['# Trades'] > 0:
        results.update({
            'return': float(stats.get('Return [%]', 0)),
            'sharpe': float(stats.get('Sharpe Ratio', None) or 0),
            'max_drawdown': float(stats.get('Max. Drawdown [%]', 0)),
            'win_rate': float(stats.get('Win Rate [%]', 0)),
            'total_trades': int(stats.get('# Trades', 0))
        })

    import os
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump(results, f, indent=2)

    bt.plot(filename='results/asia_liquidity_grab_uk_reversal.html')
