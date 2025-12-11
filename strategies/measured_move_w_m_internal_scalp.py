import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import json
import os
from backtesting import Strategy, Backtest

def generate_synthetic_data(periods=3000):
    """
    Generates synthetic 1-minute OHLC data with a very clear M-formation,
    designed to meet the strategy's R:R criteria.
    """
    rng = np.random.default_rng(42)
    price = np.full(periods, 100, dtype=float)

    # M-formation values designed for a >5:1 R:R
    p1_high_val = 120
    t_low_val = 80
    aoi_level = t_low_val + (p1_high_val - t_low_val) * 0.5  # Exactly 100
    p2_high_val = 110 # Center peak, lower than p1

    # Timeline of events
    price[100:200] = np.linspace(100, p1_high_val, 100)
    price[200:300] = np.linspace(p1_high_val, t_low_val, 100)
    price[300:400] = np.linspace(t_low_val, aoi_level, 100)
    price[400:500] = np.linspace(aoi_level, p2_high_val, 100)
    price[500:600] = np.linspace(p2_high_val, aoi_level, 100) # Reaches AOI here
    price[600:800] = np.linspace(aoi_level, 75, 200) # Subsequent drop

    dates = pd.date_range(start='2023-01-01', periods=periods, freq='min')
    df = pd.DataFrame(index=dates)

    # Create realistic OHLC
    open_price = price + rng.normal(0, 0.02, periods)
    close_price = price + rng.normal(0, 0.02, periods)
    high_price = np.maximum(open_price, close_price) + rng.uniform(0, 0.05, periods)
    low_price = np.minimum(open_price, close_price) - rng.uniform(0, 0.05, periods)

    df['Open'] = open_price
    df['High'] = high_price
    df['Low'] = low_price
    df['Close'] = close_price

    return df

def preprocess_data(df_1m, prominence_param=5):
    """
    Preprocesses 1-minute data to identify 15-minute M-formations and the AOI.
    """
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()

    peaks, _ = find_peaks(df_15m['High'], prominence=prominence_param, distance=3)
    troughs, _ = find_peaks(-df_15m['Low'], prominence=prominence_param, distance=3)

    setups = []
    for p1_idx in peaks:
        next_troughs = troughs[troughs > p1_idx]
        if not next_troughs.any(): continue
        t_idx = next_troughs[0]

        next_peaks = peaks[peaks > t_idx]
        if not next_peaks.any(): continue
        p2_idx = next_peaks[0]

        p1_high = df_15m['High'].iloc[p1_idx]
        t_low = df_15m['Low'].iloc[t_idx]
        p2_high = df_15m['High'].iloc[p2_idx]

        if p1_high > p2_high:
            setups.append((df_15m.index[p2_idx], p1_high, t_low, p2_high))

    if not setups:
        for col in ['aoi_level', 'initial_high', 'initial_low', 'center_peak_high']:
            df_1m[col] = np.nan
        return df_1m

    setup_df = pd.DataFrame(setups, columns=['p2_time', 'initial_high', 'initial_low', 'center_peak_high'])
    setup_df['aoi_level'] = setup_df['initial_low'] + (setup_df['initial_high'] - setup_df['initial_low']) * 0.5
    setup_df = setup_df.set_index('p2_time')

    merged_df = pd.merge_asof(
        df_1m.sort_index(), setup_df.sort_index(),
        left_index=True, right_index=True, direction='backward'
    )
    merged_df.ffill(inplace=True)
    return merged_df

class MeasuredMoveWMInternalScalpStrategy(Strategy):
    prominence_param = 1 # Set a default

    def init(self):
        self.aoi_level = self.I(lambda: self.data.df.get('aoi_level'))
        self.initial_low = self.I(lambda: self.data.df.get('initial_low'))
        self.center_peak_high = self.I(lambda: self.data.df.get('center_peak_high'))

        self.in_aoi = False
        self.last_trade_was_short_tp = False
        self.last_closed_trade = None

    def next(self):
        if self.last_trade_was_short_tp:
            if self.data.Close[-1] > self.data.Open[-1]:
                self.buy()
                self.last_trade_was_short_tp = False
            return

        if len(self.closed_trades) > 0 and self.closed_trades[-1] != self.last_closed_trade:
            self.last_closed_trade = self.closed_trades[-1]
            if not self.last_closed_trade.is_long and self.last_closed_trade.pl > 0:
                self.last_trade_was_short_tp = True

        if self.position:
            return

        price = self.data.Close[-1]
        aoi = self.aoi_level[-1]

        if pd.notna(aoi) and abs(price - aoi) < 1.0:
            self.in_aoi = True

        if self.in_aoi:
            if self.data.Close[-1] < self.data.Open[-1]:
                sl = self.center_peak_high[-1] + 0.2

                fib2_high = self.center_peak_high[-1]
                fib2_low = self.initial_low[-1]
                tp = fib2_low + (fib2_high - fib2_low) * 0.5

                risk = abs(price - sl)
                reward = abs(price - tp)

                if risk > 0 and reward / risk >= 5:
                    self.sell(sl=sl, tp=tp)
                    self.in_aoi = False

if __name__ == '__main__':
    data_1m = generate_synthetic_data()
    processed_data = preprocess_data(data_1m, prominence_param=1)

    bt = Backtest(processed_data, MeasuredMoveWMInternalScalpStrategy,
                  cash=10000, commission=.002, finalize_trades=True)

    stats = bt.optimize(prominence_param=[1, 3, 5, 7, 9])

    print(stats)

    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'measured_move_w_m_internal_scalp',
            'return': float(stats.get('Return [%]', 0.0)),
            'sharpe': float(stats.get('Sharpe Ratio', 0.0)),
            'max_drawdown': float(stats.get('Max. Drawdown [%]', 0.0)),
            'win_rate': float(stats.get('Win Rate [%]', 0.0)),
            'total_trades': int(stats.get('# Trades', 0))
        }, f, indent=2)

    bt.plot()
