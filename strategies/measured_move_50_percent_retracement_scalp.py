import json
import os
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from scipy.signal import find_peaks

# --- 1. Synthetic Data Generation ---
def generate_synthetic_data(periods=5000):
    """
    Generates synthetic 1-minute data with "M" patterns.
    An "M" pattern is characterized by a drop, a retracement, and another drop.
    """
    rng = np.random.default_rng(42)
    price = 100
    data = []

    # Generate a base trend
    trend = np.cumsum(rng.normal(0, 0.1, periods))

    # Overlay "M" patterns
    for i in range(periods):
        price = 100 + trend[i]

        # Create a sharp drop (first leg of M)
        if 500 < i < 600:
            price -= (i - 500) * 0.1
        # Retracement (center of M)
        elif 600 <= i < 700:
            price -= (600 - 500) * 0.1
            price += (i - 600) * 0.05
        # Second drop
        elif 700 <= i < 800:
            price -= (600 - 500) * 0.1
            price += (700 - 600) * 0.05
            price -= (i - 700) * 0.1

        # Another M pattern
        if 2500 < i < 2600:
            price -= (i - 2500) * 0.1
        elif 2600 <= i < 2750:
            price -= (2600 - 2500) * 0.1
            price += (i - 2600) * 0.05
        elif 2750 <= i < 2850:
            price -= (2600 - 2500) * 0.1
            price += (2750 - 2600) * 0.05
            price -= (i - 2750) * 0.1

        # Add noise
        open_price = price + rng.normal(0, 0.05)
        high_price = max(open_price, price) + rng.uniform(0, 0.1)
        low_price = min(open_price, price) - rng.uniform(0, 0.1)
        close_price = price + rng.normal(0, 0.05)

        data.append([open_price, high_price, low_price, close_price])

    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'])
    df.index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=periods, freq='1min'))

    return df

# --- 2. Multi-Timeframe Preprocessing ---
def preprocess_data(df_1m, prominence=1, ema_period=50):
    """
    Preprocesses 1M data to create 15M analysis timeframe data and merges it back.
    """
    # Resample to 15M
    df_15m = df_1m.resample('15min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()

    # Find swing highs and lows on 15M
    high_peaks_idx, _ = find_peaks(df_15m['High'], prominence=prominence)
    low_peaks_idx, _ = find_peaks(-df_15m['Low'], prominence=prominence)

    swing_highs = df_15m.iloc[high_peaks_idx]
    swing_lows = df_15m.iloc[low_peaks_idx]

    # Identify price legs (from a high to a low)
    legs = []
    for high_idx, high in swing_highs.iterrows():
        # Find the next low after this high
        potential_lows = swing_lows[swing_lows.index > high_idx]
        if not potential_lows.empty:
            next_low = potential_lows.iloc[[0]]
            if not next_low.empty:
                legs.append((high, next_low.iloc[0]))

    # Calculate AOI (50% Fib) for each leg and merge back to 15M data
    df_15m['aoi'] = np.nan
    df_15m['leg_low'] = np.nan
    df_15m['leg_high'] = np.nan

    for high, low in legs:
        fib_50 = high['High'] - (high['High'] - low['Low']) * 0.5
        df_15m.loc[high.name:low.name, 'aoi'] = fib_50
        df_15m.loc[high.name:low.name, 'leg_low'] = low['Low']
        df_15m.loc[high.name:low.name, 'leg_high'] = high['High']

    # Calculate 15M EMA
    df_15m['ema'] = df_15m['Close'].ewm(span=ema_period, adjust=False).mean()

    # Merge 15M data into 1M data
    df_merged = pd.merge(df_1m, df_15m[['aoi', 'leg_low', 'leg_high', 'ema']],
                         left_index=True, right_index=True, how='left')
    df_merged.ffill(inplace=True)
    df_merged.dropna(inplace=True)

    return df_merged

# --- 3. Strategy Class ---
class MeasuredMove50PercentRetracementScalpStrategy(Strategy):
    ema_period = 50
    min_rr = 5

    def init(self):
        # Pre-calculated data will be accessed via self.data
        # Ensure additional columns are available
        self.aoi = self.data.df['aoi']
        self.leg_low = self.data.df['leg_low']
        self.ema = self.data.df['ema']

    def next(self):
        price = self.data

        # --- Entry Logic ---
        # If we are already in a position, do nothing
        if self.position:
            return

        # Check for short trade setup
        # 1. Price must be in the Area of Interest (AOI)
        # 2. Confluence: Price should be near the 15M EMA
        # 3. Entry Trigger: Bearish engulfing on 1M
        is_in_aoi = price.Low[-1] <= self.aoi.iloc[-1] <= price.High[-1]
        is_near_ema = abs(price.Close[-1] - self.ema.iloc[-1]) / self.ema.iloc[-1] < 0.001 # within 0.1%

        # Bearish engulfing: current candle's body engulfs previous candle's body
        is_bearish_engulfing = (price.Open[-1] > price.Close[-1] and # current is red
                                price.Open[-2] < price.Close[-2] and # previous is green
                                price.Open[-1] >= price.Close[-2] and
                                price.Close[-1] <= price.Open[-2])

        if is_in_aoi and is_near_ema and is_bearish_engulfing:
            # --- Risk Management ---
            stop_loss = price.High[-1] * 1.001 # SL above the entry candle high

            # TP is 50% retracement of the move from leg_low to entry high
            entry_high = price.High[-1]
            take_profit = entry_high - (entry_high - self.leg_low.iloc[-1]) * 0.5

            # Check R:R
            risk = abs(price.Close[-1] - stop_loss)
            reward = abs(price.Close[-1] - take_profit)

            if risk == 0: return

            rr = reward / risk

            if rr >= self.min_rr:
                self.sell(sl=stop_loss, tp=take_profit)

# --- 4. Backtesting and Optimization ---
if __name__ == '__main__':
    # Load or generate data
    data_1m = generate_synthetic_data(periods=10000)

    # Preprocess data
    data_processed = preprocess_data(data_1m, prominence=10, ema_period=50)

    # Run backtest
    bt = Backtest(data_processed, MeasuredMove50PercentRetracementScalpStrategy,
                  cash=100000, commission=.002)

    # Optimize
    stats = bt.optimize(ema_period=range(20, 100, 10),
                        min_rr=range(3, 8, 1),
                        maximize='Sharpe Ratio',
                        constraint=lambda p: p.ema_period > 0) # Simple constraint example

    print(stats)

    # Save results
    os.makedirs('results', exist_ok=True)

    # Safely access stats and handle NaN values for JSON serialization
    win_rate = stats.get('Win Rate [%]', 0.0)
    if np.isnan(win_rate):
        win_rate = 0.0

    sharpe = stats.get('Sharpe Ratio', 0.0)
    if np.isnan(sharpe):
        sharpe = None # Represent NaN as null in JSON

    with open('results/temp_result.json', 'w') as f:
        json.dump({
            'strategy_name': 'measured_move_50_percent_retracement_scalp',
            'return': stats.get('Return [%]', 0.0),
            'sharpe': sharpe,
            'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
            'win_rate': win_rate,
            'total_trades': stats.get('# Trades', 0)
        }, f, indent=2)

    # Generate plot
    # The plot may not show custom pre-calculated indicators like 'aoi' correctly by default
    bt.plot(filename="results/measured_move_50_percent_retracement_scalp.html")
