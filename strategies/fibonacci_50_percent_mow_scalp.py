
import json
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks
import os

# --- Data Generation ---

def generate_synthetic_data(num_bars=7000, m_pattern_start=1000, w_pattern_start=4000):
    """
    Generates synthetic 1-minute OHLC data with explicit M and W patterns
    to reliably test the Fibonacci retracement logic.
    """
    dates = pd.date_range(start='2023-01-01', periods=num_bars, freq='min')
    price = np.zeros(num_bars)
    price[0] = 100
    volatility = 0.001

    # Baseline random walk
    for i in range(1, num_bars):
        price[i] = price[i-1] + np.random.normal(0, volatility)

    # --- Inject M-Pattern (for short trade setup) ---
    p1, p2, p3, p4, p5, p6, p7 = m_pattern_start, m_pattern_start+300, m_pattern_start+400, m_pattern_start+600, m_pattern_start+1000, m_pattern_start+1200, m_pattern_start+1500
    price[p1:p2] = np.linspace(price[p1], 110, p2-p1)
    price[p2:p3] = np.linspace(110, 108, p3-p2)
    price[p3:p4] = np.linspace(108, 112, p4-p3)
    price[p4:p5] = np.linspace(112, 95, p5-p4)
    price[p5:p6] = np.linspace(95, 103.5, p6-p5)
    price[p6:p7] = np.linspace(103.5, 98, p7-p6)

    # --- Inject W-Pattern (for long trade setup) ---
    p1, p2, p3, p4, p5, p6, p7 = w_pattern_start, w_pattern_start+300, w_pattern_start+400, w_pattern_start+600, w_pattern_start+1000, w_pattern_start+1200, w_pattern_start+1500
    price[p1:p2] = np.linspace(price[p1], 90, p2-p1)
    price[p2:p3] = np.linspace(90, 92, p3-p2)
    price[p3:p4] = np.linspace(92, 88, p4-p3)
    price[p4:p5] = np.linspace(88, 105, p5-p4)
    price[p5:p6] = np.linspace(105, 96.5, p6-p5)
    price[p6:p7] = np.linspace(96.5, 102, p7-p6)

    df = pd.DataFrame(index=dates)
    df['Open'] = price
    df['High'] = price + np.abs(np.random.normal(0, volatility*5, num_bars))
    df['Low'] = price - np.abs(np.random.normal(0, volatility*5, num_bars))
    df['Close'] = price + np.random.normal(0, volatility, num_bars)
    df['Volume'] = np.random.randint(100, 1000, num_bars)
    return df.dropna()

# --- Pre-processing ---

def preprocess_data_15m(data_1m, swing_lookback=20, min_move_pct=2.0):
    df_15m = data_1m['Close'].resample('15min').ohlc().dropna()
    min_move_pct_val = min_move_pct / 100.0
    peaks, _ = find_peaks(df_15m['high'], distance=swing_lookback)
    troughs, _ = find_peaks(-df_15m['low'], distance=swing_lookback)
    setups = []
    for peak_idx in peaks:
        peak_price = df_15m['high'].iloc[peak_idx]
        preceding_troughs = troughs[troughs < peak_idx]
        if preceding_troughs.size > 0:
            trough_idx = preceding_troughs[-1]
            trough_price = df_15m['low'].iloc[trough_idx]
            if (peak_price - trough_price) / trough_price > min_move_pct_val:
                aoi_level = trough_price + (peak_price - trough_price) * 0.5
                setups.append({'time': df_15m.index[peak_idx], 'setup_high': peak_price, 'setup_low': trough_price,
                               'aoi_high': aoi_level * 1.01, 'aoi_low': aoi_level * 0.99, 'setup_dir': 1})
    for trough_idx in troughs:
        trough_price = df_15m['low'].iloc[trough_idx]
        preceding_peaks = peaks[peaks < trough_idx]
        if preceding_peaks.size > 0:
            peak_idx = preceding_peaks[-1]
            peak_price = df_15m['high'].iloc[peak_idx]
            if (peak_price - trough_price) / peak_price > min_move_pct_val:
                aoi_level = peak_price - (peak_price - trough_price) * 0.5
                setups.append({'time': df_15m.index[trough_idx], 'setup_high': peak_price, 'setup_low': trough_price,
                               'aoi_high': aoi_level * 1.01, 'aoi_low': aoi_level * 0.99, 'setup_dir': -1})
    if not setups: return pd.DataFrame(index=data_1m.index, columns=['aoi_high', 'aoi_low', 'setup_high', 'setup_low', 'setup_dir'])
    setup_df = pd.DataFrame(setups).set_index('time').sort_index()
    return setup_df.reindex(data_1m.index, method='ffill')

# --- Strategy Definition ---

class Fibonacci50PercentMowScalpStrategy(Strategy):
    swing_lookback = 15
    min_move_pct = 1.5

    def init(self):
        analysis_df = preprocess_data_15m(self.data.df, int(self.swing_lookback), self.min_move_pct)
        self.data.df['aoi_high'] = analysis_df['aoi_high']
        self.data.df['aoi_low'] = analysis_df['aoi_low']
        self.data.df['setup_high'] = analysis_df['setup_high']
        self.data.df['setup_low'] = analysis_df['setup_low']
        self.data.df['setup_dir'] = analysis_df['setup_dir'].fillna(0)

        self.state = 'SEARCHING'
        self.entry_extremum = None

    def next(self):
        if self.position: return

        if self.state == 'SEARCHING':
            if self.data.df['setup_dir'].iloc[-1] != 0 and \
               self.data.df['aoi_low'].iloc[-1] <= self.data.Close[-1] <= self.data.df['aoi_high'].iloc[-1]:
                self.state = 'CONFIRMING'
                self.entry_extremum = self.data.Close[-1]

        elif self.state == 'CONFIRMING':
            setup_dir = self.data.df['setup_dir'].iloc[-1]
            if setup_dir == -1:
                self.entry_extremum = max(self.entry_extremum, self.data.High[-1])
                if self.data.Close[-1] < self.data.Open[-1]:
                    sl = self.entry_extremum * 1.001
                    tp = self.entry_extremum - (self.entry_extremum - self.data.df['setup_low'].iloc[-1]) * 0.5
                    if tp < self.data.Close[-1]: self.sell(sl=sl, tp=tp)
                    self.state = 'SEARCHING'
            elif setup_dir == 1:
                self.entry_extremum = min(self.entry_extremum, self.data.Low[-1])
                if self.data.Close[-1] > self.data.Open[-1]:
                    sl = self.entry_extremum * 0.999
                    tp = self.entry_extremum + (self.data.df['setup_high'].iloc[-1] - self.entry_extremum) * 0.5
                    if tp > self.data.Close[-1]: self.buy(sl=sl, tp=tp)
                    self.state = 'SEARCHING'

            if not (self.data.df['aoi_low'].iloc[-1] <= self.data.Close[-1] <= self.data.df['aoi_high'].iloc[-1]):
                self.state = 'SEARCHING'
                self.entry_extremum = None

# --- Main Execution Block ---
if __name__ == '__main__':
    data = generate_synthetic_data()
    bt = Backtest(data, Fibonacci50PercentMowScalpStrategy, cash=100_000, commission=.002)

    print("Running optimization...")
    stats = bt.optimize(
        swing_lookback=range(10, 25, 5),
        min_move_pct=np.arange(1.0, 2.6, 0.5).tolist(),
        maximize='Sharpe Ratio'
    )

    print("\n--- Best Run Stats ---")
    print(stats)
    print("\n--- Best Parameters ---")
    print(stats._strategy)

    os.makedirs('results', exist_ok=True)
    results_path = 'results/temp_result.json'

    with open(results_path, 'w') as f:
        json.dump({
            'strategy_name': 'fibonacci_50_percent_mow_scalp',
            'return': float(stats.get('Return [%]', 0)),
            'sharpe': float(stats.get('Sharpe Ratio', 0)),
            'max_drawdown': float(stats.get('Max. Drawdown [%]', 0)),
            'win_rate': float(stats.get('Win Rate [%]', 0)),
            'total_trades': int(stats.get('# Trades', 0))
        }, f, indent=2)

    print(f"\nResults saved to {results_path}")

    plot_path = 'results/fibonacci_50_percent_mow_scalp.html'
    bt.plot(filename=plot_path, open_browser=False)
    print(f"Plot saved to {plot_path}")
