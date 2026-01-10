import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def PtaAtr(high, low, close, length=14):
    """Wrapper for pandas_ta.atr."""
    atr = ta.atr(high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), length=length)
    if atr is None or atr.empty:
        return np.nan
    return atr.values

def generate_synthetic_data():
    """Generates synthetic data for testing."""
    n_points = 2000
    index = pd.to_datetime(pd.date_range('2023-01-01', periods=n_points, freq='15min'))
    price = 100 + pd.Series(np.random.randn(n_points).cumsum() * 0.2)
    price += np.sin(np.linspace(0, 200, n_points)) * 5
    data = pd.DataFrame({
        'Open': price, 'High': price * 1.01, 'Low': price * 0.99,
        'Close': price, 'Volume': np.random.randint(500, 2000, n_points)
    }, index=index)
    return data

class SpxOptionsCalendarSpread(Strategy):
    short_atr_period = 14
    long_atr_period = 100
    entry_threshold_ratio = 1.5
    exit_threshold_ratio = 1.0
    time_exit_bars = 96
    sl_atr_multiplier = 3.0

    def init(self):
        self.short_atr = self.I(PtaAtr, self.data.High, self.data.Low, self.data.Close, self.short_atr_period)
        self.long_atr = self.I(PtaAtr, self.data.High, self.data.Low, self.data.Close, self.long_atr_period)
        self.entry_bar = 0

    def next(self):
        if self.position and (len(self.data) - self.entry_bar >= self.time_exit_bars):
            self.position.close()
            return

        vol_ratio = self.short_atr[-1] / self.long_atr[-1]
        price = self.data.Close[-1]

        if self.position:
            if self.position.is_long and vol_ratio <= self.exit_threshold_ratio:
                self.position.close()
            elif self.position.is_short and vol_ratio >= self.exit_threshold_ratio:
                self.position.close()
        elif not self.position:
            if vol_ratio > self.entry_threshold_ratio:
                sl = price - self.short_atr[-1] * self.sl_atr_multiplier
                self.sell(sl=sl)
                self.entry_bar = len(self.data)
            elif vol_ratio < (1 / self.entry_threshold_ratio):
                sl = price + self.short_atr[-1] * self.sl_atr_multiplier
                self.buy(sl=sl)
                self.entry_bar = len(self.data)

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if os.path.exists(data_path):
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        data.columns = [col.strip().title() for col in data.columns]
    else:
        print("Data file not found, generating synthetic data.")
        data = generate_synthetic_data()

    bt = Backtest(data, SpxOptionsCalendarSpread, cash=100_000, commission=.002)
    stats = bt.run()

    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Series, pd.DataFrame)) or key.startswith('_'): continue
            if pd.isna(value) or not np.isfinite(value): sanitized[key] = None
            else: sanitized[key] = value
        return {k: v for k, v in sanitized.items() if v is not None}

    final_stats = sanitize_stats(stats)
    with open('results/temp_result.json', 'w') as f:
        json.dump(final_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")
    print(stats)

    try:
        bt.plot(filename='results/spx_options_calendar_spread.html')
    except Exception as e:
        print(f"Could not generate plot: {e}")
