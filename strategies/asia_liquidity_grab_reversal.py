
import json
import os

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy


# Helper function to generate synthetic 24-hour data with higher volatility
def generate_forex_data(days=100):
    """
    Generates synthetic 24-hour OHLCV data with increased volatility
    to better test the Asia Liquidity Grab strategy.
    """
    rng = np.random.default_rng(seed=42)
    n_points = days * 24 * 4  # 15-minute intervals

    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='15min', tz='UTC')

    price_changes = rng.standard_normal(n_points) * 0.0025

    num_shocks = int(n_points / 100)
    shock_indices = rng.choice(n_points, num_shocks, replace=False)
    shock_magnitudes = (rng.random(num_shocks) - 0.5) * 0.015
    np.add.at(price_changes, shock_indices, shock_magnitudes)

    price = 1.10 + np.cumsum(price_changes)

    open_price = price[:-1]
    close_price = price[1:]

    high_price = np.maximum(open_price, close_price) + rng.random(n_points - 1) * 0.001
    low_price = np.minimum(open_price, close_price) - rng.random(n_points - 1) * 0.001

    volume = rng.integers(100, 1000, size=n_points - 1)

    data = pd.DataFrame({
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Close': close_price,
        'Volume': volume
    }, index=dates[:-1])

    return data

def preprocess_data(df):
    """
    Identifies trading sessions and calculates Asia session high/low.
    """
    df['hour'] = df.index.hour

    is_asia = (df['hour'] >= 0) & (df['hour'] < 8)
    is_uk = (df['hour'] >= 8) & (df['hour'] < 16)
    is_us = (df['hour'] >= 13) & (df['hour'] < 21)

    df['session'] = 'N/A'
    df.loc[is_asia, 'session'] = 'Asia'
    df.loc[is_uk, 'session'] = 'UK'
    df.loc[is_us, 'session'] = 'US'

    df['date'] = df.index.date
    asia_session_data = df[df['session'] == 'Asia']

    daily_asia_high = asia_session_data.groupby('date')['High'].max()
    daily_asia_low = asia_session_data.groupby('date')['Low'].min()

    df['asia_high'] = df['date'].map(daily_asia_high)
    df['asia_low'] = df['date'].map(daily_asia_low)

    df['asia_range_pct'] = (df['asia_high'] / df['asia_low'] - 1) * 100

    df[['asia_high', 'asia_low', 'asia_range_pct']] = df[['asia_high', 'asia_low', 'asia_range_pct']].ffill()

    df = df.dropna()
    return df


class AsiaLiquidityGrabReversalStrategy(Strategy):
    asia_range_max = 2.0
    sl_buffer_pct = 0.05
    rr_ratio_tp2 = 2.0

    def init(self):
        self.grabbed_high = False
        self.grabbed_low = False
        self.current_day = None
        self.tp1_hit = False

    def next(self):
        if self.data.index[-1].date() != self.current_day:
            self.current_day = self.data.index[-1].date()
            self.grabbed_high = False
            self.grabbed_low = False
            self.tp1_hit = False

        if self.position and not self.tp1_hit:
            if self.position.is_long and self.data.High[-1] >= self.data.asia_high[-1]:
                self.position.close(portion=0.5)
                self.tp1_hit = True
            elif self.position.is_short and self.data.Low[-1] <= self.data.asia_low[-1]:
                self.position.close(portion=0.5)
                self.tp1_hit = True

        if self.position:
            return

        if self.data.asia_range_pct[-1] > self.asia_range_max:
            return

        if self.data.session[-1] not in ['UK', 'US']:
            return

        # Short Entry
        if not self.grabbed_low:
            if self.data.High[-1] > self.data.asia_high[-1]:
                self.grabbed_high = True

            if self.grabbed_high:
                # Relaxed bearish reversal: a strong bearish candle
                candle_range = self.data.High[-1] - self.data.Low[-1]
                body_size = self.data.Open[-1] - self.data.Close[-1]
                is_strong_bearish = body_size > candle_range * 0.5 if candle_range > 0 else False

                if is_strong_bearish:
                    entry_price = self.data.Close[-1]
                    sl = self.data.High[-1] * (1 + self.sl_buffer_pct / 100)
                    risk = abs(entry_price - sl)
                    tp2 = entry_price - (risk * self.rr_ratio_tp2)

                    if tp2 < entry_price:
                        self.sell(sl=sl, tp=tp2)
                        self.tp1_hit = False

        # Long Entry
        if not self.grabbed_high:
            if self.data.Low[-1] < self.data.asia_low[-1]:
                self.grabbed_low = True

            if self.grabbed_low:
                # Relaxed bullish reversal: a strong bullish candle
                candle_range = self.data.High[-1] - self.data.Low[-1]
                body_size = self.data.Close[-1] - self.data.Open[-1]
                is_strong_bullish = body_size > candle_range * 0.5 if candle_range > 0 else False

                if is_strong_bullish:
                    entry_price = self.data.Close[-1]
                    sl = self.data.Low[-1] * (1 - self.sl_buffer_pct / 100)
                    risk = abs(entry_price - sl)
                    tp2 = entry_price + (risk * self.rr_ratio_tp2)

                    if tp2 > entry_price:
                        self.buy(sl=sl, tp=tp2)
                        self.tp1_hit = False


if __name__ == '__main__':
    data = generate_forex_data(days=200)
    processed_data = preprocess_data(data.copy())

    bt = Backtest(processed_data, AsiaLiquidityGrabReversalStrategy, cash=10000, commission=.0002, finalize_trades=True)

    stats = bt.optimize(
        asia_range_max=list(np.arange(1.0, 3.5, 0.5)),
        sl_buffer_pct=list(np.arange(0.01, 0.1, 0.02)),
        rr_ratio_tp2=list(np.arange(1.5, 3.0, 0.5)),
        maximize='Sharpe Ratio'
    )

    print(stats)

    os.makedirs('results', exist_ok=True)

    results_dict = {
        'strategy_name': 'asia_liquidity_grab_reversal',
        'return': float(stats.get('Return [%]', 0.0)),
        'sharpe': float(np.nan_to_num(stats.get('Sharpe Ratio', 0.0))),
        'max_drawdown': float(stats.get('Max. Drawdown [%]', 0.0)),
        'win_rate': float(np.nan_to_num(stats.get('Win Rate [%]', 0.0))),
        'total_trades': int(stats.get('# Trades', 0))
    }

    with open('results/temp_result.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    try:
        if stats['# Trades'] > 0:
            bt.plot(filename='results/asia_liquidity_grab_reversal.html', open_browser=False)
        else:
            print("No trades were executed, skipping plot generation.")
    except Exception as e:
        print(f"Could not generate plot: {e}")
