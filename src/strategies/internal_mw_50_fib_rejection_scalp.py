from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json

def generate_synthetic_data(n_patterns=20):
    """Generates synthetic 1-minute OHLC data with M and W patterns."""
    rng = np.random.default_rng(seed=42)
    price_data = []
    base_price = 100

    for i in range(n_patterns):
        noise_duration = rng.integers(50, 150); noise = rng.normal(0, 0.05, size=noise_duration)
        ranging_prices = base_price + np.cumsum(noise); price_data.extend(ranging_prices)
        base_price = ranging_prices[-1]

        pattern_type = rng.choice(['M', 'W'])

        if pattern_type == 'M':
            peak_a_height = rng.uniform(2, 5); move1_duration = rng.integers(15, 30); move1 = np.linspace(0, peak_a_height, move1_duration)
            low_b_drop = peak_a_height * rng.uniform(0.6, 0.9); move2_duration = rng.integers(15, 30); move2 = np.linspace(0, -low_b_drop, move2_duration)
            retrace_level = rng.choice([0.5, 0.65, 0.8]); peak_c_retracement = low_b_drop * retrace_level
            move3_duration = rng.integers(15, 30); move3 = np.linspace(0, peak_c_retracement, move3_duration)
            final_drop = peak_c_retracement * rng.uniform(1.2, 2.5); move4_duration = rng.integers(15, 30); move4 = np.linspace(0, -final_drop, move4_duration)
            pattern = np.concatenate([move1, move2, move3, move4])
        else:
            trough_a_depth = rng.uniform(2, 5); move1_duration = rng.integers(15, 30); move1 = np.linspace(0, -trough_a_depth, move1_duration)
            high_b_rise = trough_a_depth * rng.uniform(0.6, 0.9); move2_duration = rng.integers(15, 30); move2 = np.linspace(0, high_b_rise, move2_duration)
            retrace_level = rng.choice([0.5, 0.65, 0.8]); trough_c_retracement = high_b_rise * retrace_level
            move3_duration = rng.integers(15, 30); move3 = np.linspace(0, -trough_c_retracement, move3_duration)
            final_rise = trough_c_retracement * rng.uniform(1.2, 2.5); move4_duration = rng.integers(15, 30); move4 = np.linspace(0, final_rise, move4_duration)
            pattern = np.concatenate([move1, move2, move3, move4])

        pattern_prices = base_price + np.cumsum(pattern); price_data.extend(pattern_prices)
        base_price = pattern_prices[-1]

    total_bars = len(price_data)
    dates = pd.date_range(start='2023-01-01', periods=total_bars, freq='T', tz='UTC')
    close_prices = np.array(price_data)
    open_prices = np.roll(close_prices, 1); open_prices[0] = close_prices[0]
    high_prices = np.maximum(open_prices, close_prices) + rng.uniform(0, 0.02, size=total_bars)
    low_prices = np.minimum(open_prices, close_prices) - rng.uniform(0, 0.02, size=total_bars)

    return pd.DataFrame({'Open': open_prices, 'High': high_prices, 'Low': low_prices, 'Close': close_prices}, index=dates)

def preprocess_data(df_1m, swing_period=10, ema_period=20):
    df_15m = df_1m.resample('15min').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()

    # --- Confluence Indicators ---
    # 1. 15-min EMA
    df_15m['EMA'] = df_15m['Close'].ewm(span=ema_period, adjust=False).mean()

    # 2. Low of Day (on 1M data)
    df_1m['LOD'] = df_1m.groupby(df_1m.index.date)['Low'].cummin()

    # 3. Asia Session Range (00:00-08:00 UTC)
    asia_session = df_1m.between_time('00:00', '08:00')
    asia_range = asia_session.groupby(asia_session.index.date).agg(asia_high=('High', 'max'), asia_low=('Low', 'min'))
    asia_range['asia_50_pct'] = (asia_range['asia_high'] + asia_range['asia_low']) / 2
    df_1m['asia_50_pct'] = df_1m.index.normalize().map(asia_range['asia_50_pct'])
    df_1m['asia_50_pct'].ffill(inplace=True)

    # --- M/W Setup Detection ---
    n = swing_period
    peaks_idx = [df_15m.index[i] for i in range(n, len(df_15m) - n) if df_15m['High'].iloc[i] == df_15m.iloc[i-n:i+n+1]['High'].max()]
    troughs_idx = [df_15m.index[i] for i in range(n, len(df_15m) - n) if df_15m['Low'].iloc[i] == df_15m.iloc[i-n:i+n+1]['Low'].min()]

    swings = pd.concat([pd.Series('peak', index=pd.to_datetime(peaks_idx)), pd.Series('trough', index=pd.to_datetime(troughs_idx))]).sort_index()
    swings = swings[~swings.index.duplicated(keep='first') & (swings != swings.shift(1))]

    cols = ['setup_type', 'peak_A', 'trough_B', 'trough_A', 'peak_B', 'fib_382', 'fib_500', 'fib_618', 'fib_786']
    setup_df = pd.DataFrame(index=df_15m.index, columns=cols)

    for i in range(1, len(swings)):
        prev_idx, prev_type, curr_idx, curr_type = swings.index[i-1], swings.iloc[i-1], swings.index[i], swings.iloc[i]
        if prev_type == 'peak' and curr_type == 'trough':
            peak_A = df_15m.loc[prev_idx, 'High']; trough_B = df_15m.loc[curr_idx, 'Low']; price_range = peak_A - trough_B
            fibs = {'fib_382': trough_B + price_range * 0.382, 'fib_500': trough_B + price_range * 0.5, 'fib_618': trough_B + price_range * 0.618, 'fib_786': trough_B + price_range * 0.786}
            setup_df.loc[curr_idx:, ['setup_type', 'peak_A', 'trough_B', *fibs.keys()]] = ['M', peak_A, trough_B, *fibs.values()]
        elif prev_type == 'trough' and curr_type == 'peak':
            trough_A = df_15m.loc[prev_idx, 'Low']; peak_B = df_15m.loc[curr_idx, 'High']; price_range = peak_B - trough_A
            fibs = {'fib_382': peak_B - price_range * 0.382, 'fib_500': peak_B - price_range * 0.5, 'fib_618': peak_B - price_range * 0.618, 'fib_786': peak_B - price_range * 0.786}
            setup_df.loc[curr_idx:, ['setup_type', 'trough_A', 'peak_B', *fibs.keys()]] = ['W', trough_A, peak_B, *fibs.values()]

    # Merge all pre-processed data
    combined_df = pd.merge_asof(df_1m, setup_df.ffill(), left_index=True, right_index=True)
    combined_df = pd.merge_asof(combined_df, df_15m[['EMA']], left_index=True, right_index=True)
    return combined_df.dropna()

class InternalMw50FibRejectionScalpStrategy(Strategy):
    retrace_leeway_pct = 2
    confluence_leeway_pct = 1

    def init(self):
        self.last_peak_A, self.last_trough_A = None, None
        self.setup_invalid, self.m_setup_active, self.w_setup_active = False, False, False
        self.m_center_peak_high, self.w_center_trough_low = 0, float('inf')

    def next(self):
        if len(self.data.Close) < 2: return

        aoi = self.data.fib_500[-1]

        # Confluence Check
        confluence_levels = [self.data.EMA[-1], self.data.LOD[-1], self.data.asia_50_pct[-1]]
        is_confluent = any(abs(aoi - level) / aoi < (self.confluence_leeway_pct / 100) for level in confluence_levels if level)
        if not is_confluent: return

        setup_type = self.data.setup_type[-1]
        if setup_type == 'M':
            if self.data.peak_A[-1] != self.last_peak_A:
                self.last_peak_A = self.data.peak_A[-1]; self.m_setup_active, self.setup_invalid = False, False
            if self.setup_invalid or self.position.is_short: return

            leeway = aoi * (self.retrace_leeway_pct / 100)
            if not self.m_setup_active and self.data.High[-1] >= aoi - leeway:
                self.m_setup_active, self.m_center_peak_high = True, self.data.High[-1]

            if self.m_setup_active:
                if self.data.High[-1] > self.data.fib_618[-1]: self.setup_invalid, self.m_setup_active = True, False; return
                self.m_center_peak_high = max(self.m_center_peak_high, self.data.High[-1])
                if self.data.Close[-1] < aoi - leeway and self.data.Close[-2] >= aoi - leeway:
                    sl, peak_C, low_B = self.m_center_peak_high * 1.001, self.m_center_peak_high, self.data.trough_B[-1]
                    tp = peak_C - (peak_C - low_B) * 0.5
                    if tp < self.data.Close[-1]: self.sell(sl=sl, tp=tp)
                    self.m_setup_active = False
        elif setup_type == 'W':
            if self.data.trough_A[-1] != self.last_trough_A:
                self.last_trough_A = self.data.trough_A[-1]; self.w_setup_active, self.setup_invalid = False, False
            if self.setup_invalid or self.position.is_long: return

            leeway = aoi * (self.retrace_leeway_pct / 100)
            if not self.w_setup_active and self.data.Low[-1] <= aoi + leeway:
                self.w_setup_active, self.w_center_trough_low = True, self.data.Low[-1]
            if self.w_setup_active:
                if self.data.Low[-1] < self.data.fib_618[-1]: self.setup_invalid, self.w_setup_active = True, False; return
                self.w_center_trough_low = min(self.w_center_trough_low, self.data.Low[-1])
                if self.data.Close[-1] > aoi + leeway and self.data.Close[-2] <= aoi + leeway:
                    sl, trough_C, high_B = self.w_center_trough_low * 0.999, self.w_center_trough_low, self.data.peak_B[-1]
                    tp = trough_C + (high_B - trough_C) * 0.5
                    if tp > self.data.Close[-1]: self.buy(sl=sl, tp=tp)
                    self.w_setup_active = False

if __name__ == '__main__':
    data_1m = generate_synthetic_data(n_patterns=100) # More patterns for better optimization
    data_processed = preprocess_data(data_1m)

    bt = Backtest(data_processed, InternalMw50FibRejectionScalpStrategy, cash=100_000, commission=.002)

    stats = bt.optimize(
        retrace_leeway_pct=np.arange(1, 4, 0.5).tolist(),
        confluence_leeway_pct=np.arange(0.5, 2, 0.5).tolist(),
        maximize='Sharpe Ratio'
    )

    print(stats)

    import os
    os.makedirs('results', exist_ok=True)
    best_params = stats._strategy
    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'internal_mw_50_fib_rejection_scalp',
            'return': float(stats['Return [%]']),
            'sharpe': float(stats['Sharpe Ratio']) if np.isfinite(stats['Sharpe Ratio']) else None,
            'max_drawdown': float(stats['Max. Drawdown [%]']), 'win_rate': float(stats['Win Rate [%]']),
            'total_trades': int(stats['# Trades']),
            'parameters': {'retrace_leeway_pct': best_params.retrace_leeway_pct, 'confluence_leeway_pct': best_params.confluence_leeway_pct}
        }, f, indent=2)

    bt.plot(filename="results/internal_mw_50_fib_rejection_scalp.html")
