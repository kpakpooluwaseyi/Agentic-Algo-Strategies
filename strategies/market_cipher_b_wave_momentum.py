import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def PtaEMA(close_series, length):
    """Wrapper for pandas_ta.ema that returns a numpy array."""
    return ta.ema(close=pd.Series(close_series), length=length).values

def PtaMFI(high, low, close, volume, length):
    """Wrapper for pandas_ta.mfi that returns a numpy array."""
    return ta.mfi(high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), volume=pd.Series(volume), length=length).values

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

class MarketCipherBWaveMomentum(Strategy):
    """
    NOTE: This is a proxy implementation of the Market Cipher B Wave Momentum strategy.
    The requested 'VuManChuCipherB' indicator was not found in the repository.
    Therefore, this strategy uses standard indicators to approximate the logic:
    - Momentum Wave is proxied by an EMA crossover (fast EMA crossing slow EMA).
    - Money Flow is proxied by the Money Flow Index (MFI).
    """
    # EMA settings for Momentum Wave proxy
    fast_ema_period = 12
    slow_ema_period = 26

    # MFI settings for Money Flow proxy
    mfi_period = 14
    mfi_threshold = 50

    def init(self):
        # Proxy for Momentum Wave
        self.fast_ema = self.I(PtaEMA, self.data.Close, self.fast_ema_period)
        self.slow_ema = self.I(PtaEMA, self.data.Close, self.slow_ema_period)

        # Proxy for Money Flow
        self.mfi = self.I(PtaMFI, self.data.High, self.data.Low, self.data.Close, self.data.Volume, self.mfi_period)

    def next(self):
        # Define entry conditions
        long_momentum_signal = crossover(self.fast_ema, self.slow_ema)
        long_money_flow_signal = self.mfi[-1] > self.mfi_threshold
        long_entry_condition = long_momentum_signal and long_money_flow_signal

        short_momentum_signal = crossover(self.slow_ema, self.fast_ema)
        short_money_flow_signal = self.mfi[-1] < self.mfi_threshold
        short_entry_condition = short_momentum_signal and short_money_flow_signal

        # Check for exits first
        if self.position:
            if self.position.is_long and short_momentum_signal:
                self.position.close()
            elif self.position.is_short and long_momentum_signal:
                self.position.close()

        # Check for entries if no position is open
        if not self.position:
            if long_entry_condition:
                sl = self.data.Low[-1]
                # Ensure risk is not zero before calculating size
                risk = self.data.Close[-1] - sl
                if risk > 0:
                    size = self.equity * 0.01 / risk
                    if size > 0:
                        self.buy(sl=sl, size=int(size))

            elif short_entry_condition:
                sl = self.data.High[-1]
                # Ensure risk is not zero before calculating size
                risk = sl - self.data.Close[-1]
                if risk > 0:
                    size = self.equity * 0.01 / risk
                    if size > 0:
                        self.sell(sl=sl, size=int(size))

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        try:
            # Using names and usecols to handle potential header issues
            data = pd.read_csv(
                data_path,
                header=0,
                names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                usecols=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                index_col='datetime',
                parse_dates=True
            )
            # Ensure column names are capitalized as required by backtesting.py
            data.columns = [col.capitalize() for col in data.columns]
        except Exception as e:
            print(f"Error loading CSV, falling back to synthetic data: {e}")
            data = generate_synthetic_data()
    else:
        print(f"Data file not found at '{data_path}'.")
        data = generate_synthetic_data()

    bt = Backtest(data, MarketCipherBWaveMomentum, cash=100_000, commission=.002)

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
        plot_filename = 'results/market_cipher_b_wave_momentum.html'
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
