
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import json
import os

def generate_synthetic_sector(base_series, num_assets=4, correlation=0.8, volatility_factor=0.01):
    """
    Generates a DataFrame of synthetic, correlated asset prices.

    Args:
        base_series (pd.Series): The base price series (e.g., BTC Close).
        num_assets (int): Number of additional assets to generate.
        correlation (float): Correlation factor with the base series.
        volatility_factor (float): Factor to control noise/randomness.

    Returns:
        pd.DataFrame: DataFrame with synthetic asset prices.
    """
    base_returns = base_series.pct_change().dropna()
    synthetic_assets = {}

    for i in range(num_assets):
        # Correlated noise
        correlated_noise = correlation * base_returns + (1 - correlation) * np.random.randn(len(base_returns)) * volatility_factor

        # Start price based on a random factor of the base series' start price
        start_price = base_series.iloc[0] * (1 + (np.random.rand() - 0.5) * 0.2)

        # Create the synthetic price series
        synthetic_price = start_price * (1 + correlated_noise).cumprod()
        synthetic_assets[f'SYNTH_{i+1}'] = synthetic_price

    df = pd.DataFrame(synthetic_assets, index=base_returns.index)
    # Add the base asset for a complete sector view
    df['BASE_ASSET'] = base_series[df.index]
    return df

def preprocess_data(df, pred_period=1):
    """
    Prepares the data for the VAR Sector Neutral strategy.

    1. Generates a synthetic sector of correlated assets.
    2. Calculates the 'predicted return' (using simple past return as a proxy) for all assets.
    3. Calculates the mean return of the sector for each day.
    4. Adds the primary asset's predicted return and the sector's mean return to the main DataFrame.
    """

    # 1. Generate synthetic sector
    sector_df = generate_synthetic_sector(df['Close'])

    # 2. Calculate "predicted return" (proxy) for all assets in the sector
    # Using simple returns as a proxy for a VAR(1) model prediction
    predicted_returns = sector_df.pct_change(periods=pred_period).shift(-pred_period)

    # 3. Calculate the mean return of the sector
    df['sector_mean_return'] = predicted_returns.mean(axis=1)

    # 4. Add the primary asset's predicted return to the main DataFrame
    df['predicted_return'] = predicted_returns['BASE_ASSET']

    # Align and drop NaNs
    df = df.dropna()

    return df


class VarSectorNeutralStocks(Strategy):
    """
    A proxy implementation of a VAR Sector-Neutral Stocks strategy.

    This strategy simulates a multi-asset, sector-neutral approach using
    synthetically generated data for the "sector" based on a primary asset.
    It is designed to be run on a single-asset dataset (e.g., BTC-USD).

    Entry Rules:
    - Long: If the primary asset's predicted return is greater than the
            mean predicted return of the synthetic sector.
    - Short: If the primary asset's predicted return is less than the
             mean predicted return of the synthetic sector.

    Exit Rules:
    - Hold positions for one day (rebalanced daily).
    """

    pred_period = 1 # How many bars ahead to "predict" returns

    def init(self):
        # The core logic is based on pre-calculated columns, so we just need access to them.
        self.predicted_return = self.I(lambda: self.data.predicted_return, name="predicted_return")
        self.sector_mean_return = self.I(lambda: self.data.sector_mean_return, name="sector_mean_return")

    def next(self):
        # --- Daily Rebalancing ---
        # Close any open position at the start of each new bar (simulates a 1-day hold)
        if self.position:
            self.position.close()

        # --- Entry Logic ---
        # Get the latest signal values
        asset_pred = self.predicted_return[-1]
        sector_mean_pred = self.sector_mean_return[-1]

        # Go long if the asset is predicted to outperform the sector average
        if asset_pred > sector_mean_pred:
            self.buy()

        # Go short if the asset is predicted to underperform the sector average
        elif asset_pred < sector_mean_pred:
            self.sell()


if __name__ == '__main__':
    # --- Configuration ---
    DATA_PATH = 'data/BTC-USD-15m.csv'
    STRATEGY = VarSectorNeutralStocks
    CASH = 100_000
    COMMISSION = .002

    # --- Data Loading ---
    try:
        data = pd.read_csv(DATA_PATH)
        data.columns = [col.strip().capitalize() for col in data.columns]
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data = data.set_index('Datetime')
        # Resample to 1D for the daily timeframe specified in the request
        data = data.resample('1D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        print(f"Loaded and resampled {len(data)} rows from {DATA_PATH}")
    except FileNotFoundError:
        print(f"Data file not found. Generating synthetic data.")
        dates = pd.date_range(start='2017-01-01', periods=2000, freq='1D')
        price = 10000 + np.cumsum(np.random.randn(2000)) * 100
        data = pd.DataFrame({
            'Open': price, 'High': price + 50, 'Low': price - 50,
            'Close': price + np.random.randn(2000) * 20,
            'Volume': np.random.randint(100, 5000, 2000)
        }, index=dates)
        data.index.name = 'Datetime'

    # --- Preprocessing ---
    # NOTE: This adds lookahead bias by calculating future returns.
    # This is a simplified proxy for a predictive VAR model as requested.
    data = preprocess_data(data, pred_period=STRATEGY.pred_period)

    # --- Backtesting ---
    bt = Backtest(data, STRATEGY, cash=CASH, commission=COMMISSION)
    stats = bt.run()
    print("\n--- Backtest Results ---")
    print(stats)

    # --- Save Results ---
    os.makedirs('results', exist_ok=True)
    def sanitize_stats(stats_obj):
        sanitized = {k: v for k, v in stats_obj.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
        for key, value in sanitized.items():
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                sanitized[key] = str(value)
            elif isinstance(value, (np.integer, np.floating)):
                sanitized[key] = float(value)
            elif pd.isna(value):
                sanitized[key] = None
        return sanitized

    results_path = 'results/temp_result.json'
    with open(results_path, 'w') as f:
        json.dump(sanitize_stats(stats), f, indent=4)
    print(f"\nResults saved to {results_path}")

    # --- Plotting ---
    plot_path = f"results/{STRATEGY.__name__}.html"
    try:
        bt.plot(filename=plot_path, open_browser=False)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")
