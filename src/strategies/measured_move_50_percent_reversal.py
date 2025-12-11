from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
import os

def EMA(series, n):
    """Exponential Moving Average"""
    return pd.Series(series).ewm(span=n, adjust=False).mean()

class MeasuredMove50PercentReversalStrategy(Strategy):
    # Optimizable parameters
    ema_period = 50
    level_drop_lookback = 240
    level_drop_pct = 2.0
    confluence_tolerance_pct = 0.5
    min_rr_ratio = 2.0 # Minimum Risk-to-Reward ratio

    def init(self):
        self.data_15m = self.data.df.resample('15T').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()

        self.ema_15m = self.I(EMA, pd.Series(self.data_15m['Close']), self.ema_period)

        self.lod = self.data.LOD if hasattr(self.data, 'LOD') else pd.Series(np.nan, index=self.data.index)
        self.asia_mid = self.data.AsiaMid if hasattr(self.data, 'AsiaMid') else pd.Series(np.nan, index=self.data.index)

        self.level_drop_start_price = None
        self.level_drop_low_price = None
        self.r50_aoi = None
        self.in_aoi = False
        self.entry_high = None

    def next(self):
        if self.position:
            return

        if len(self.data_15m.index) < self.level_drop_lookback:
            return

        lookback_data = self.data_15m['High'].iloc[-self.level_drop_lookback:]
        high_point_price = np.max(lookback_data)
        high_point_index = np.argmax(lookback_data)

        low_since_high = np.min(self.data_15m['Low'].iloc[-self.level_drop_lookback + high_point_index:])
        price_drop_pct = (high_point_price - low_since_high) / high_point_price * 100

        if price_drop_pct >= self.level_drop_pct:
            self.r50_aoi = low_since_high + (high_point_price - high_point_price * 0.5)

            tolerance = self.r50_aoi * (self.confluence_tolerance_pct / 100)

            ema_conf = abs(self.ema_15m[-1] - self.r50_aoi) < tolerance if not np.isnan(self.ema_15m[-1]) else False
            lod_conf = abs(self.lod[-1] - self.r50_aoi) < tolerance if not np.isnan(self.lod[-1]) else False
            asia_conf = abs(self.asia_mid[-1] - self.r50_aoi) < tolerance if not np.isnan(self.asia_mid[-1]) else False

            if not (ema_conf or lod_conf or asia_conf):
                self.r50_aoi = None
                return

            self.level_drop_start_price = high_point_price
            self.level_drop_low_price = low_since_high
        else:
            self.r50_aoi = None
            self.in_aoi = False
            return

        current_price = self.data.Close[-1]

        if self.r50_aoi and current_price >= self.r50_aoi:
            self.in_aoi = True
            if self.entry_high is None or current_price > self.entry_high:
                self.entry_high = current_price

        if self.in_aoi and current_price < self.r50_aoi and self.entry_high is not None:
            sl = self.entry_high * 1.001
            prior_low_for_tp = self.level_drop_low_price
            tp = self.entry_high - (self.entry_high - prior_low_for_tp) * 0.5

            # --- R:R Filter ---
            risk = sl - current_price
            reward = current_price - tp
            if risk <= 0 or reward <= 0: return # Avoid division by zero

            rr_ratio = reward / risk

            if rr_ratio >= self.min_rr_ratio and tp < current_price:
                self.sell(sl=sl, tp=tp)

            self.in_aoi = False
            self.entry_high = None

def add_prev_day_low(df: pd.DataFrame) -> pd.DataFrame:
    daily_lows = df['Low'].resample('D').min().shift(1)
    df['LOD'] = df.index.normalize().map(daily_lows)
    df['LOD'] = df['LOD'].ffill()
    return df

def add_asia_session_mid(df: pd.DataFrame, start_hour=23, end_hour=8) -> pd.DataFrame:
    df_copy = df.copy()
    if df_copy.index.tz is None:
        df_copy.index = df_copy.index.tz_localize('UTC')

    session_hours = (df_copy.index.hour >= start_hour) | (df_copy.index.hour < end_hour)

    df_copy['session_group'] = (df_copy.index - pd.Timedelta(hours=end_hour)).date

    session_data = df_copy[session_hours]

    if not session_data.empty:
        session_high = session_data.groupby('session_group')['High'].max()
        session_low = session_data.groupby('session_group')['Low'].min()
        session_mid = (session_high + session_low) / 2

        df['AsiaMid'] = df_copy['session_group'].map(session_mid)
        df['AsiaMid'] = df['AsiaMid'].ffill()
    else:
        df['AsiaMid'] = np.nan

    return df

def generate_synthetic_data(days=30):
    n_minutes = days * 24 * 60
    start_date = pd.to_datetime('2023-01-01')
    index = pd.date_range(start_date, periods=n_minutes, freq='T')

    price = 1.5 + np.random.randn(n_minutes).cumsum() / 500

    data = pd.DataFrame(index=index)
    data['Open'] = price
    data['High'] = data['Open'] + np.random.uniform(0, 0.001, n_minutes)
    data['Low'] = data['Open'] - np.random.uniform(0, 0.001, n_minutes)
    data['Close'] = data['Open'] + np.random.uniform(-0.0005, 0.0005, n_minutes)
    data['Volume'] = np.random.randint(100, 1000, n_minutes)
    return data

if __name__ == '__main__':
    data = generate_synthetic_data(days=90)

    data = add_prev_day_low(data)
    data = add_asia_session_mid(data)
    data.dropna(subset=['AsiaMid', 'LOD'], inplace=True)

    if not data.empty:
        bt = Backtest(data, MeasuredMove50PercentReversalStrategy, cash=100000, commission=.002)

        stats = bt.optimize(
            ema_period=range(20, 101, 20),
            level_drop_lookback=range(120, 361, 120),
            level_drop_pct=np.arange(1.5, 3.1, 0.5),
            confluence_tolerance_pct=np.arange(0.1, 1.1, 0.2),
            min_rr_ratio=np.arange(1.5, 5.1, 0.5),
            maximize='Sharpe Ratio'
        )

        os.makedirs('results', exist_ok=True)
        if stats is not None and stats['# Trades'] > 0:
            stats_dict = {
                'strategy_name': 'measured_move_50_percent_reversal',
                'return': float(stats['Return [%]']),
                'sharpe': float(stats['Sharpe Ratio']),
                'max_drawdown': float(stats['Max. Drawdown [%]']),
                'win_rate': float(stats['Win Rate [%]']),
                'total_trades': int(stats['# Trades'])
            }
        else:
            stats_dict = {
                'strategy_name': 'measured_move_50_percent_reversal',
                'return': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0,
                'win_rate': 0.0, 'total_trades': 0
            }

        with open('results/temp_result.json', 'w') as f:
            json.dump(stats_dict, f, indent=2)

        print("Backtest results saved to results/temp_result.json")
        bt.plot()
    else:
        print("Data for backtest is empty after pre-processing. Skipping backtest.")
