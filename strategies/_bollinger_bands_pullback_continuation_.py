import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def PtaBollingerBands(close_series, length=20, std=2.0):
    """
    Wrapper for pandas_ta.bbands that returns a tuple of numpy arrays
    as expected by backtesting.py's self.I().
    """
    bbands = ta.bbands(close=pd.Series(close_series), length=length, std=std, append=False)
    lower_col = f'BBL_{length}_{std}'
    middle_col = f'BBM_{length}_{std}'
    upper_col = f'BBU_{length}_{std}'

    # pandas-ta might return slightly different column names, so we find them dynamically
    lower_col = [col for col in bbands.columns if col.startswith('BBL_')][0]
    middle_col = [col for col in bbands.columns if col.startswith('BBM_')][0]
    upper_col = [col for col in bbands.columns if col.startswith('BBU_')][0]

    return bbands[lower_col].values, bbands[middle_col].values, bbands[upper_col].values

def generate_synthetic_data():
    """Generates synthetic data for testing the strategy."""
    print("Generating synthetic data...")
    n_points = 2000
    index = pd.to_datetime(pd.date_range('2023-01-01', periods=n_points, freq='15min'))
    price = 100 + pd.Series(np.random.randn(n_points).cumsum() * 0.1)
    price += np.sin(np.linspace(0, 200, n_points)) * 2
    data = pd.DataFrame({
        'Open': price, 'High': price * 1.005, 'Low': price * 0.995,
        'Close': price, 'Volume': np.random.randint(100, 1000, n_points)
    }, index=index)
    return data

class BollingerBandsPullbackContinuation(Strategy):
    """
    Strategy to trade pullbacks to the middle Bollinger Band in a trending market.
    """
    bb_period = 20
    bb_std_dev = 2.0
    risk_reward_ratio = 2.0 # Optimizable parameter for risk-reward

    def init(self):
        self.lower_band, self.middle_band, self.upper_band = self.I(
            PtaBollingerBands, self.data.Close, self.bb_period, self.bb_std_dev
        )

    def next(self):
        if len(self.data) < 2:
            return

        # Condition 1: Market is trending
        is_uptrend = self.data.Close[-1] > self.middle_band[-1]
        is_downtrend = self.data.Close[-1] < self.middle_band[-1]

        if self.position:
            return

        # Condition 2: Price pulls back to the middle Bollinger Band
        pullback_long = self.data.Low[-1] <= self.middle_band[-1]
        pullback_short = self.data.High[-1] >= self.middle_band[-1]

        # Condition 3: Confirmation candlestick pattern (causal)
        is_bullish_engulfing = (
            self.data.Close[-2] < self.data.Open[-2] and  # Previous is bearish
            self.data.Close[-1] > self.data.Open[-1] and   # Current is bullish
            self.data.Close[-1] > self.data.Open[-2] and
            self.data.Open[-1] < self.data.Close[-2]
        )

        is_bearish_engulfing = (
            self.data.Close[-2] > self.data.Open[-2] and  # Previous is bullish
            self.data.Close[-1] < self.data.Open[-1] and   # Current is bearish
            self.data.Close[-1] < self.data.Open[-2] and
            self.data.Open[-1] > self.data.Close[-2]
        )

        if is_uptrend and pullback_long and is_bullish_engulfing:
            sl = self.data.Low[-1]
            tp = self.data.Close[-1] + (self.data.Close[-1] - sl) * self.risk_reward_ratio
            if tp > sl:
                self.buy(sl=sl, tp=tp)

        elif is_downtrend and pullback_short and is_bearish_engulfing:
            sl = self.data.High[-1]
            tp = self.data.Close[-1] - (sl - self.data.Close[-1]) * self.risk_reward_ratio
            if tp < sl:
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        try:
            data = pd.read_csv(
                data_path, index_col='datetime', parse_dates=True,
            )
            # Sanitize column names
            data.columns = [col.strip().title() for col in data.columns]
        except Exception as e:
            print(f"Error loading CSV, falling back to synthetic data: {e}")
            data = generate_synthetic_data()
    else:
        print(f"Data file not found at '{data_path}'.")
        data = generate_synthetic_data()

    bt = Backtest(data, BollingerBandsPullbackContinuation, cash=100_000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()

    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Series, pd.DataFrame)): continue
            if isinstance(value, (np.floating, np.integer)):
                sanitized[key] = float(value) if np.isfinite(value) else None
            elif isinstance(value, int): sanitized[key] = int(value)
            elif isinstance(value, pd.Timestamp): sanitized[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta): sanitized[key] = str(value)
            elif pd.isna(value): sanitized[key] = None
            elif key.startswith('_'): continue
            else: sanitized[key] = value
        return sanitized

    final_stats = sanitize_stats(stats)

    results_filename = 'results/temp_result.json'
    with open(results_filename, 'w') as f:
        json.dump(final_stats, f, indent=2)

    print(f"Backtest results saved to {results_filename}")
    print(stats)

    try:
        plot_filename = 'results/_bollinger_bands_pullback_continuation_.html'
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
