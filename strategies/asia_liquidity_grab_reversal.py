from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
import os

# --- Helper Functions ---

def generate_synthetic_data(days=100):
    """
    Generates synthetic 15-minute OHLC data tailored to produce
    the Asia Liquidity Grab pattern for both long and short setups.
    """
    n_periods = days * 24 * 4
    rng = np.random.default_rng(seed=42)

    price = 1.5000
    base_returns = rng.normal(loc=0, scale=0.0002, size=n_periods)

    # Alternate between short and long pattern days
    for i in range(96, n_periods, 192):
        # --- SHORT Pattern Day ---
        if i + 96 < n_periods:
            base_returns[i : i+32] *= 0.3      # Quiet Asia
            base_returns[i+33] = 0.0025     # Spike up
            base_returns[i+34] = -0.0030    # Reversal down
            base_returns[i+35 : i+45] -= 0.0001

        # --- LONG Pattern Day ---
        if i + 192 < n_periods:
            long_day_start = i + 96
            base_returns[long_day_start : long_day_start+32] *= 0.3 # Quiet Asia
            base_returns[long_day_start+33] = -0.0025   # Spike down
            base_returns[long_day_start+34] = 0.0030   # Reversal up
            base_returns[long_day_start+35 : long_day_start+45] += 0.0001

    price_path = price * (1 + np.cumsum(base_returns))

    timestamps = pd.date_range(start='2023-01-01', periods=n_periods, freq='15min', tz='UTC')
    df = pd.DataFrame(index=timestamps)
    df['Open'] = price_path
    df['High'] = df['Open'] + rng.uniform(0.0001, 0.0003, size=n_periods)
    df['Low'] = df['Open'] - rng.uniform(0.0001, 0.0003, size=n_periods)
    df['Close'] = df['Open'] + rng.normal(loc=0, scale=0.0002, size=n_periods)
    df['Volume'] = rng.integers(100, 1000, size=n_periods)

    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)

    return df

def preprocess_data(df):
    """
    Identifies trading sessions and calculates daily Asia session high/low.
    """
    df = df.copy()
    if df.index.tz is None: df.index = pd.to_datetime(df.index).tz_localize('UTC')

    conditions = [(df.index.hour >= 0) & (df.index.hour < 8), (df.index.hour >= 8) & (df.index.hour < 17)]
    choices = ['Asia', 'UK']
    df['Session'] = np.select(conditions, choices, default='US')

    asia_session = df[df['Session'] == 'Asia']
    daily_asia_stats = asia_session.groupby(asia_session.index.date).agg(HOA=('High', 'max'), LOA=('Low', 'min')).dropna()

    df['Date'] = df.index.date
    df = df.merge(daily_asia_stats, left_on='Date', right_index=True, how='left').ffill().dropna()
    return df

# --- Strategy Definition ---

class AsiaLiquidityGrabReversalStrategy(Strategy):
    asia_range_threshold_pct = 2.0
    rr_ratio = 1.5

    def init(self):
        self.session = self.I(lambda x: x, self.data.df['Session'].astype('category').cat.codes)
        self.hoa = self.I(lambda x: x, self.data.HOA)
        self.loa = self.I(lambda x: x, self.data.LOA)
        self.session_cat = self.data.df['Session'].astype('category')
        self.session_map = dict(enumerate(self.session_cat.cat.categories))
        self.state = 'WAITING'

    def is_bearish_engulfing(self):
        if len(self.data.Close) < 2: return False
        prev_body_min, prev_body_max = min(self.data.Open[-2], self.data.Close[-2]), max(self.data.Open[-2], self.data.Close[-2])
        curr_open, curr_close = self.data.Open[-1], self.data.Close[-1]
        return curr_open > curr_close and curr_open >= prev_body_max and curr_close <= prev_body_min

    def is_bullish_engulfing(self):
        if len(self.data.Close) < 2: return False
        prev_body_min, prev_body_max = min(self.data.Open[-2], self.data.Close[-2]), max(self.data.Open[-2], self.data.Close[-2])
        curr_open, curr_close = self.data.Open[-1], self.data.Close[-1]
        return curr_open < curr_close and curr_open <= prev_body_min and curr_close >= prev_body_max

    def next(self):
        current_session = self.session_map.get(int(self.session[-1]), 'Unknown')

        if current_session != 'UK':
            self.state = 'WAITING'
            return

        hoa, loa = self.hoa[-1], self.loa[-1]
        if hoa == 0 or loa == 0 or ((hoa - loa) / loa * 100) >= self.asia_range_threshold_pct:
            self.state = 'WAITING'
            return

        high, low, close = self.data.High[-1], self.data.Low[-1], self.data.Close[-1]

        if self.state == 'WAITING':
            if high > hoa and high > self.data.High[-2]: self.state = 'GRAB_HIGH_DETECTED'
            elif low < loa and low < self.data.Low[-2]: self.state = 'GRAB_LOW_DETECTED'

        elif self.state == 'GRAB_HIGH_DETECTED':
            if self.is_bearish_engulfing() and not self.position:
                sl, tp, entry = self.data.High[-2], loa, close
                if tp < entry < sl and (entry - tp) / (sl - entry) >= self.rr_ratio:
                    self.sell(limit=entry, sl=sl, tp=tp)
            self.state = 'WAITING'

        elif self.state == 'GRAB_LOW_DETECTED':
            if self.is_bullish_engulfing() and not self.position:
                sl, tp, entry = self.data.Low[-2], hoa, close
                if sl < entry < tp and (tp - entry) / (entry - sl) >= self.rr_ratio:
                    self.buy(limit=entry, sl=sl, tp=tp)
            self.state = 'WAITING'

if __name__ == '__main__':
    print("Generating and preprocessing data...")
    data = generate_synthetic_data(days=200)
    data = preprocess_data(data)

    bt = Backtest(data, AsiaLiquidityGrabReversalStrategy, cash=100_000, commission=.002, exclusive_orders=True)

    print("Running optimization...")
    stats = bt.optimize(
        asia_range_threshold_pct=np.arange(1.0, 4.5, 0.5).tolist(),
        rr_ratio=np.arange(1.0, 3.0, 0.5).tolist(),
        maximize='Sharpe Ratio', max_tries=200
    )

    print("\nBest Run Stats:")
    print(stats)

    os.makedirs('results', exist_ok=True)
    final_stats = {
        'strategy_name': 'asia_liquidity_grab_reversal',
        'return': stats.get('Return [%]', 0.0), 'sharpe': stats.get('Sharpe Ratio', 0.0),
        'max_drawdown': stats.get('Max. Drawdown [%]', 0.0), 'win_rate': stats.get('Win Rate [%]', 0.0),
        'total_trades': stats.get('# Trades', 0)
    }
    for key, value in final_stats.items():
        if pd.isna(value): final_stats[key] = None
        elif isinstance(value, (np.generic, float)): final_stats[key] = float(value)
        elif isinstance(value, (np.integer, int)): final_stats[key] = int(value)

    with open('results/temp_result.json', 'w') as f: json.dump(final_stats, f, indent=2)
    print(f"\nResults saved to results/temp_result.json")

    if stats.get('# Trades', 0) > 0:
        try:
            bt.plot(filename='results/asia_liquidity_grab_reversal_plot.html', plot_volume=True, open_browser=False)
            print(f"Plot saved to results/asia_liquidity_grab_reversal_plot.html")
        except TypeError as e:
            print(f"\n---\nPlotting failed due to a library incompatibility: {e}")
            print("This is often due to a mismatch between Backtesting.py and pandas versions.")
            print("The main backtest results have been saved successfully.\n---")
    else:
        print("No trades were executed. Skipping plot generation.")
