import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def pta_ema(series, length):
    """
    Wrapper for pandas_ta.ema that returns a numpy array
    as expected by backtesting.py's self.I().
    """
    ema = ta.ema(close=pd.Series(series), length=length)
    return ema.values

def PtaPSAR(high_series, low_series, step=0.02, max_step=0.2):
    """
    Wrapper for pandas_ta.psar that returns a single SAR line
    as a numpy array.
    """
    psar_df = ta.psar(high=pd.Series(high_series), low=pd.Series(low_series), af=step, max=max_step)
    # Dynamically find the long and short column names
    long_col = [col for col in psar_df.columns if col.startswith('PSARl_')]
    short_col = [col for col in psar_df.columns if col.startswith('PSARs_')]

    if not long_col or not short_col:
        raise ValueError("Could not find PSAR long/short columns in pandas_ta output.")

    # Combine the long and short SAR values into a single series
    psar_series = psar_df[long_col[0]].fillna(psar_df[short_col[0]])
    return psar_series.values

class ParabolicSar200EmaTrendFollowing(Strategy):
    """
    A trend-following strategy using Parabolic SAR and a 200-period EMA.
    - Enters long when price is above 200 EMA and SAR flips below the price.
    - Enters short when price is below 200 EMA and SAR flips above the price.
    - Exits based on a fixed risk-reward ratio.
    """
    ema_period = 200
    psar_step = 0.02
    psar_max_step = 0.2
    risk_pct = 0.007
    risk_reward_ratio = 1.5

    def init(self):
        self.ema200 = self.I(
            pta_ema, self.data.Close, self.ema_period
        )
        self.psar = self.I(
            PtaPSAR, self.data.High, self.data.Low, self.psar_step, self.psar_max_step
        )

    def next(self):
        price = self.data.Close[-1]

        # If a position is already open, do nothing.
        if self.position:
            return

        # Long entry conditions
        is_bullish_trend = price > self.ema200[-1]
        # SAR flips when the previous SAR was above the low, and the current is below.
        sar_flipped_up = self.psar[-2] > self.data.Low[-2] and self.psar[-1] < self.data.Low[-1]

        if is_bullish_trend and sar_flipped_up:
            sl = self.psar[-1]

            # Ensure stop-loss is valid (below the current price for a long)
            if price <= sl:
                return

            risk_per_unit = price - sl
            if risk_per_unit == 0:
                return

            # Calculate position size based on risk percentage
            size = int((self.equity * self.risk_pct) / risk_per_unit)
            if size <= 0:
                return

            tp = price + risk_per_unit * self.risk_reward_ratio
            self.buy(sl=sl, tp=tp, size=size)
            return

        # Short entry conditions
        is_bearish_trend = price < self.ema200[-1]
        # SAR flips when the previous SAR was below the high, and the current is above.
        sar_flipped_down = self.psar[-2] < self.data.High[-2] and self.psar[-1] > self.data.High[-1]

        if is_bearish_trend and sar_flipped_down:
            sl = self.psar[-1]

            # Ensure stop-loss is valid (above the current price for a short)
            if price >= sl:
                return

            risk_per_unit = sl - price
            if risk_per_unit == 0:
                return

            # Calculate position size based on risk percentage
            size = int((self.equity * self.risk_pct) / risk_per_unit)
            if size <= 0:
                return

            tp = price - risk_per_unit * self.risk_reward_ratio
            self.sell(sl=sl, tp=tp, size=size)

def generate_synthetic_data():
    """Generates synthetic data for testing the strategy."""
    print("Generating synthetic data...")
    n_points = 5000
    index = pd.to_datetime(pd.date_range('2023-01-01', periods=n_points, freq='15min'))
    price = 100 + pd.Series(np.random.randn(n_points).cumsum() * 0.1)
    # Add a clear trend for part of the data
    trend = np.linspace(0, 50, n_points)
    price += trend
    data = pd.DataFrame({
        'Open': price, 'High': price * 1.005, 'Low': price * 0.995,
        'Close': price, 'Volume': np.random.randint(100, 1000, n_points)
    }, index=index)
    # Sanitize column names
    data.columns = [c.capitalize() for c in data.columns]
    return data

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        try:
            data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
            # Sanitize column names to match 'Open', 'High', 'Low', 'Close', 'Volume'
            data.columns = [c.strip().capitalize() for c in data.columns]
        except Exception as e:
            print(f"Error loading CSV, falling back to synthetic data: {e}")
            data = generate_synthetic_data()
    else:
        print(f"Data file not found at '{data_path}'.")
        data = generate_synthetic_data()

    bt = Backtest(data, ParabolicSar200EmaTrendFollowing, cash=100_000, commission=.002)

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

    with open('results/temp_result.json', 'w') as f:
        json.dump(final_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")
    print(stats)

    try:
        plot_filename = 'results/parabolic_sar_200_ema_trend_following.html'
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
