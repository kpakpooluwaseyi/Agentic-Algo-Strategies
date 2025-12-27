import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import json
import os

def generate_synthetic_universe(base_data, num_assets=10, vol_scale=0.02):
    """
    Generates a universe of synthetic assets based on a provided base dataset.

    Args:
        base_data (pd.DataFrame): The base data, must have a 'Close' column.
        num_assets (int): The number of synthetic assets to generate.
        vol_scale (float): Volatility scaling factor for the random noise.

    Returns:
        pd.DataFrame: A DataFrame where each column is the close price of a synthetic asset.
    """
    universe = pd.DataFrame(index=base_data.index)
    base_returns = base_data['Close'].pct_change().fillna(0)

    for i in range(num_assets):
        # Introduce varying momentum factors
        momentum_factor = 1 + (np.random.rand() - 0.5) * 0.1  # e.g., 0.95 to 1.05

        # Add random noise
        noise = np.random.normal(0, base_returns.std() * vol_scale, size=len(base_data))

        # Create synthetic returns and price series
        synthetic_returns = base_returns * momentum_factor + noise
        synthetic_price = (1 + synthetic_returns).cumprod() * base_data['Close'].iloc[0]

        universe[f'asset_{i}'] = synthetic_price

    return universe

def preprocess_for_cross_section(data, lookback_period=252, rank_threshold=0.8):
    """
    Prepares the data for the cross-sectional strategy by adding a signal column.

    Args:
        data (pd.DataFrame): The main asset's OHLCV data.
        lookback_period (int): The lookback period for momentum calculation.
        rank_threshold (float): The percentile rank required to be a "top performer".

    Returns:
        pd.DataFrame: The original data with a 'signal' column added.
    """
    # 1. Generate a synthetic universe
    synthetic_universe = generate_synthetic_universe(data)

    # 2. Combine the main asset with the universe
    combined_universe = synthetic_universe.copy()
    combined_universe['main_asset'] = data['Close']

    # 3. Calculate past returns for all assets
    returns = combined_universe.pct_change(lookback_period).dropna()

    # 4. Rank assets at each timestep
    ranks = returns.rank(axis=1, pct=True)

    # 5. Create the signal for the main asset
    main_asset_rank = ranks['main_asset']

    # Generate signals: 1 for top performer, 0 for neutral, -1 for bottom performer
    signal = np.zeros(len(main_asset_rank))
    signal[main_asset_rank >= rank_threshold] = 1  # Top 20%
    signal[main_asset_rank <= (1 - rank_threshold)] = -1 # Bottom 20%

    # Add the signal to the original dataframe
    data_with_signal = data.copy()
    data_with_signal['signal'] = pd.Series(signal, index=main_asset_rank.index).reindex(data.index).ffill().fillna(0)

    return data_with_signal

class CrossSectionalMomentum(Strategy):
    """
    Executes a true cross-sectional momentum strategy.
    The core logic is pre-calculated in the preprocessing step. This strategy
    simply acts on the 'signal' column.
    """
    sl_pct = 0.10  # 10% stop-loss
    tp_pct = 0.30  # 30% take-profit

    def init(self):
        # The signal is pre-calculated, so we just need to access it.
        self.signal = self.I(lambda: self.data.signal)

    def next(self):
        price = self.data.Close[-1]

        # Exit logic: Close position if the signal is no longer active (i.e., neutral)
        if self.position and self.signal[-1] == 0:
            self.position.close()

        # Entry logic
        if not self.position:
            if self.signal[-1] == 1:  # Buy signal
                sl = price * (1 - self.sl_pct)
                tp = price * (1 + self.tp_pct)
                self.buy(sl=sl, tp=tp)
            elif self.signal[-1] == -1:  # Short signal
                sl = price * (1 + self.sl_pct)
                tp = price * (1 - self.tp_pct)
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Please ensure the file is present.")

    # Load and resample data
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    daily_data = data.resample('D').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    daily_data.columns = [c.capitalize() for c in daily_data.columns]

    # Apply the cross-sectional preprocessing
    processed_data = preprocess_for_cross_section(daily_data)

    # Run the backtest
    bt = Backtest(processed_data, CrossSectionalMomentum, cash=100_000, commission=.002)
    stats = bt.run()

    print("--- Backtest Results ---")
    print(stats)

    os.makedirs('results', exist_ok=True)

    # Save results to JSON
    result = {
        'strategy_name': 'cross_sectional_momentum',
        'return': float(stats.get('Return [%]', 0)),
        'sharpe': float(stats.get('Sharpe Ratio', 0)),
        'max_drawdown': float(stats.get('Max. Drawdown [%]', 0)),
        'win_rate': float(stats.get('Win Rate [%]', 0)),
        'total_trades': int(stats.get('# Trades', 0))
    }
    result_path = 'results/temp_result.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_path}")

    # Generate and save the plot
    plot_path = 'results/cross_sectional_momentum.html'
    bt.plot(filename=plot_path, open_browser=False)
    print(f"Plot saved to {plot_path}")