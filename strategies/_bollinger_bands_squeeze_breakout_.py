import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os
from backtesting import Backtest, Strategy

def PtaBollingerBands(close_series, length=20, std=2.0):
    """
    Wrapper for pandas_ta.bbands. Ensures input is a pandas Series
    and returns numpy arrays as expected by backtesting.py's self.I().
    """
    close_series_pd = pd.Series(close_series)
    bbands = ta.bbands(close=close_series_pd, length=length, std=std)

    # Dynamically find column names to avoid issues with library updates
    upper_col = [col for col in bbands.columns if col.startswith('BBU_')][0]
    middle_col = [col for col in bbands.columns if col.startswith('BBM_')][0]
    lower_col = [col for col in bbands.columns if col.startswith('BBL_')][0]

    return bbands[upper_col].values, bbands[middle_col].values, bbands[lower_col].values

def BollingerBandwidth(upper_band, lower_band, middle_band):
    """Calculates Bollinger Bandwidth."""
    # Add a small epsilon to avoid division by zero
    return (upper_band - lower_band) / (middle_band + 1e-9)

def CandleBodySize(open_price, close_price):
    """Calculates the absolute size of the candle body."""
    return abs(close_price - open_price)

def generate_synthetic_data():
    """Generates synthetic data for testing the strategy."""
    print("Generating synthetic data...")
    n_points = 2000
    index = pd.to_datetime(pd.date_range('2023-01-01', periods=n_points, freq='15min'))

    # Generate a base price series with some trend and noise
    price = 100 + pd.Series(np.random.randn(n_points).cumsum() * 0.1)

    # Inject periods of low volatility (squeeze) followed by breakouts
    for i in range(5):
        squeeze_start = np.random.randint(200, n_points - 200)
        squeeze_end = squeeze_start + np.random.randint(50, 100)
        price[squeeze_start:squeeze_end] = np.linspace(
            price[squeeze_start-1],
            price[squeeze_start-1] + np.random.randn(),
            num=squeeze_end-squeeze_start
        )
        breakout_direction = 1 if np.random.rand() > 0.5 else -1
        breakout_end = squeeze_end + 10
        price[squeeze_end:breakout_end] = price[squeeze_end-1] + np.linspace(0, breakout_direction * 5, num=10)

    data = pd.DataFrame({
        'Open': price, 'High': price + abs(np.random.randn(n_points) * 0.5),
        'Low': price - abs(np.random.randn(n_points) * 0.5),
        'Close': price + np.random.randn(n_points) * 0.1,
        'Volume': np.random.randint(100, 1000, n_points)
    }, index=index)
    return data

class BollingerBandsSqueezeBreakout(Strategy):
    """
    A breakout strategy that identifies low-volatility "squeezes" using
    Bollinger Bands and enters on high-momentum breakouts.
    """
    # Bollinger Bands settings
    bb_period = 20
    bb_std_dev = 2.0

    # Squeeze identification settings
    squeeze_lookback = 40
    squeeze_percentile = 5  # Squeeze if bandwidth is in the bottom 5%

    # Momentum check settings
    momentum_lookback = 20
    momentum_factor = 2.0  # Breakout candle must be N times the avg body size

    # Risk management settings
    risk_reward_ratio = 2.5

    def init(self):
        # Initialize Bollinger Bands and Bandwidth indicators
        self.upper_band, self.middle_band, self.lower_band = self.I(
            PtaBollingerBands, self.data.Close, self.bb_period, self.bb_std_dev
        )
        self.bandwidth = self.I(
            BollingerBandwidth, self.upper_band, self.lower_band, self.middle_band
        )
        self.body_size = self.I(
            CandleBodySize, self.data.Open, self.data.Close
        )

    def next(self):
        # Ensure we have enough data for our lookbacks
        if len(self.data) < max(self.squeeze_lookback, self.momentum_lookback):
            return

        # A trade is already open, do nothing
        if self.position:
            return

        # 1. Squeeze Check
        historical_bandwidth = self.bandwidth[-self.squeeze_lookback:-1]
        squeeze_threshold = np.percentile(historical_bandwidth, self.squeeze_percentile)
        is_in_squeeze = self.bandwidth[-2] < squeeze_threshold

        if not is_in_squeeze:
            return

        # 2. Breakout and Momentum Check
        avg_body_size = np.mean(self.body_size[-self.momentum_lookback:-1])
        current_body_size = self.body_size[-1]
        has_momentum = current_body_size > avg_body_size * self.momentum_factor

        if not has_momentum:
            return

        # 3. Entry Logic
        # Long Entry: Green candle closes above the upper band
        if self.data.Close[-1] > self.upper_band[-1] and self.data.Close[-1] > self.data.Open[-1]:
            sl = self.data.Low[-1]
            tp = self.data.Close[-1] + (self.data.Close[-1] - sl) * self.risk_reward_ratio
            if tp > self.data.Close[-1]:  # Sanity check
                self.buy(sl=sl, tp=tp)

        # Short Entry: Red candle closes below the lower band
        elif self.data.Close[-1] < self.lower_band[-1] and self.data.Close[-1] < self.data.Open[-1]:
            sl = self.data.High[-1]
            tp = self.data.Close[-1] - (sl - self.data.Close[-1]) * self.risk_reward_ratio
            if tp < self.data.Close[-1]:  # Sanity check
                self.sell(sl=sl, tp=tp)


if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        try:
            data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
            # Sanitize column names
            data.columns = [c.strip().title() for c in data.columns]
        except Exception as e:
            print(f"Error loading CSV, falling back to synthetic data: {e}")
            data = generate_synthetic_data()
    else:
        print(f"Data file not found at '{data_path}'.")
        data = generate_synthetic_data()

    # Ensure data has the required OHLCV columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_cols):
        print("Data does not contain all required columns. Generating synthetic data.")
        data = generate_synthetic_data()


    bt = Backtest(data, BollingerBandsSqueezeBreakout, cash=100_000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()

    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Series, pd.DataFrame)): continue
            if isinstance(value, (np.floating, np.integer)):
                sanitized[key] = float(value) if np.isfinite(value) else None
            elif isinstance(value, int): sanitized[key] = value
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
        plot_filename = 'results/bollinger_bands_squeeze_breakout.html'
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
