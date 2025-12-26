# Fama-French Factor Prediction Strategy
#
# Original Source: "Machine Trading" by Ernest P. Chan (Strategy 2)
#
# This strategy attempts to predict next-day stock returns based on the Fama-French
# three-factor model. However, the original strategy is cross-sectional, meaning it
# ranks and trades a universe of assets (e.g., SPX components) daily.
#
# --- Implementation for Single-Asset Backtesting ---
# A direct implementation is not possible with the backtesting.py framework, which
# is designed for single-instrument time-series analysis. To adapt the core logic,
# this script follows a robust simulation approach:
#
# 1.  **Synthetic Universe Creation:** The provided single-asset data (BTC-USD) is
#     used as a "market" proxy. A synthetic universe of correlated assets and the
#     three Fama-French factor series (Mkt-RF, SMB, HML) are generated.
#
# 2.  **Cross-Sectional Simulation:** In a pre-processing step, the script simulates
#     the Fama-French strategy on this synthetic universe for each day in the
#     backtest period. It performs rolling regressions to estimate factor loadings,
#     predicts next-day returns for all assets, and forms a long-short portfolio
#     based on these predictions.
#
# 3.  **Equity Curve Backtest:** The simulation's output is the daily performance
#     of the long-short portfolio, which is converted into a cumulative equity
#     curve. This curve is then treated as a single tradable asset.
#
# 4.  **Final Analysis:** A simple 'Buy and Hold' strategy is run on this final
#     equity curve using backtesting.py. This allows for the analysis of the
#     Fama-French model's performance using the framework's standard metrics and
#     plotting tools, providing a meaningful, albeit simulated, result.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from backtesting import Strategy, Backtest
import json
import os


# --- Main Strategy Logic (Pre-Backtest Simulation) ---

def generate_synthetic_data(base_data, n_assets=200, n_factors=3):
    """
    Generates a synthetic universe of assets and factor returns based on the
    provided single-asset data.
    """
    # Resample to daily and calculate returns for the "market"
    daily_data = base_data['Close'].resample('D').last().dropna()
    market_returns = daily_data.pct_change().dropna()

    # Generate synthetic factors (uncorrelated with each other)
    factor_returns = pd.DataFrame(
        np.random.normal(loc=0, scale=market_returns.std(), size=(len(market_returns), n_factors)),
        index=market_returns.index,
        columns=['Mkt-RF', 'SMB', 'HML']
    )
    # Make the first factor correlated with the market
    factor_returns['Mkt-RF'] = market_returns * 0.7 + np.random.normal(loc=0, scale=market_returns.std() * 0.3, size=len(market_returns))


    # Generate synthetic asset returns with random factor exposures
    np.random.seed(42)
    asset_betas = np.random.uniform(-0.5, 1.5, size=(n_assets, n_factors))
    asset_specific_vol = np.random.uniform(market_returns.std() * 0.5, market_returns.std() * 2, size=n_assets)

    asset_returns_list = []
    for i in range(n_assets):
        systematic_return = (factor_returns * asset_betas[i]).sum(axis=1)
        idiosyncratic_return = np.random.normal(loc=0, scale=asset_specific_vol[i], size=len(market_returns))
        total_return = systematic_return + idiosyncratic_return
        asset_returns_list.append(total_return)

    asset_returns = pd.concat(asset_returns_list, axis=1)
    asset_returns.columns = [f'Asset_{i}' for i in range(n_assets)]

    return asset_returns, factor_returns


def simulate_fama_french_portfolio(asset_returns, factor_returns, window=60, top_n=50):
    """
    Simulates the Fama-French prediction and portfolio construction process.
    """
    predictions = {}
    model = LinearRegression()

    # Iterate through time to make predictions
    for i in range(window, len(asset_returns)):
        current_date = asset_returns.index[i]

        # Define training window for regression
        train_assets = asset_returns.iloc[i-window:i]
        train_factors = factor_returns.iloc[i-window:i]

        asset_predictions = {}
        for asset in train_assets.columns:
            # Fit model: Asset_Return ~ Factor_Returns
            model.fit(train_factors, train_assets[asset])

            # Predict next-day return using today's factor returns
            # Note: In a real scenario, you'd predict factors first. Here we use
            # today's factors as a simple proxy for the prediction input.
            predicted_return = model.predict(factor_returns.iloc[i:i+1])[0]
            asset_predictions[asset] = predicted_return

        predictions[current_date] = asset_predictions

    # Convert predictions to a DataFrame
    pred_df = pd.DataFrame.from_dict(predictions, orient='index')

    # Rank assets and form portfolio
    ranks = pred_df.rank(axis=1, ascending=False)

    longs = ranks[ranks <= top_n].fillna(0).astype(bool)
    shorts = ranks[ranks > (len(asset_returns.columns) - top_n)].fillna(0).astype(bool)

    # Calculate portfolio returns
    # Shift returns by 1 to align predictions with the actual returns they were for
    actual_returns_aligned = asset_returns.shift(-1)

    long_returns = (actual_returns_aligned[longs].mean(axis=1)).fillna(0)
    short_returns = (actual_returns_aligned[shorts].mean(axis=1)).fillna(0)

    # Portfolio is long the top N and short the bottom N
    portfolio_returns = long_returns - short_returns
    portfolio_returns = portfolio_returns.dropna() / (2 * top_n) # Normalize by number of assets

    # Create cumulative equity curve
    equity_curve = (1 + portfolio_returns).cumprod()

    # Convert to OHLC for backtesting.py
    ohlc = pd.DataFrame({
        'Open': equity_curve,
        'High': equity_curve,
        'Low': equity_curve,
        'Close': equity_curve,
        'Volume': 0 # No volume data for the equity curve
    })

    # Resample can introduce NaNs if dates are missing, forward fill them
    ohlc = ohlc.resample('D').last().ffill()

    return ohlc.dropna()


# --- Backtesting.py Strategy ---

class BuyAndHold(Strategy):
    """
    A simple strategy that buys on the first bar and holds until the end.
    Used to analyze the performance of a pre-computed equity curve.
    """
    def init(self):
        pass

    def next(self):
        if not self.position:
            self.buy()


def sanitize_stats(stats):
    """
    Removes non-serializable objects from the backtesting stats dictionary.
    """
    # Work on a copy
    stats_dict = dict(stats)

    # Remove non-serializable objects
    for key in ['_strategy', '_equity_curve', '_trades']:
        if key in stats_dict:
            del stats_dict[key]

    # Sanitize individual values that might be numpy types
    for key, value in list(stats_dict.items()):
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
             stats_dict[key] = str(value)
        elif isinstance(value, (np.int64, np.int32, np.float64)):
            stats_dict[key] = value.item()
        elif isinstance(value, pd.NA.__class__):
            stats_dict[key] = None
        elif pd.isna(value):
            stats_dict[key] = None


    return stats_dict


# --- Main Execution Block ---

if __name__ == '__main__':
    # Configuration
    data_path = 'data/BTC-USD-15m.csv'
    results_dir = 'results'

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    print("Loading and preparing base data...")
    # Load base data to drive the simulation
    try:
        data = pd.read_csv(
            data_path,
            index_col='datetime',
            parse_dates=True
        )
        # Sanitize column names: strip whitespace and capitalize
        data.columns = [c.strip().capitalize() for c in data.columns]
        # Keep only the 'Close' column
        data = data[['Close']]
    except (FileNotFoundError, KeyError):
        print(f"Warning: Main data file not found at {data_path}. Generating synthetic data as a fallback.")
        # Create a fallback date range if the file doesn't exist
        date_range = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
        close_prices = 100 + np.random.randn(len(date_range)).cumsum()
        data = pd.DataFrame({'Close': close_prices}, index=date_range)
        data.index.name = 'datetime'

    print("Generating synthetic asset universe and factor returns...")
    asset_returns, factor_returns = generate_synthetic_data(data)

    print("Simulating Fama-French portfolio strategy...")
    portfolio_equity_curve = simulate_fama_french_portfolio(asset_returns, factor_returns)

    if portfolio_equity_curve.empty:
        print("Strategy simulation resulted in an empty equity curve. No trades were generated.")
        # Create a dummy result file
        results = {"error": "No trades generated during simulation."}
        with open(os.path.join(results_dir, 'temp_result.json'), 'w') as f:
            json.dump(results, f, indent=4)
    else:
        print("Running backtest on the simulated portfolio equity curve...")
        # Backtest the resulting equity curve
        bt = Backtest(
            portfolio_equity_curve,
            BuyAndHold,
            cash=100_000,
            commission=.002,
            finalize_trades=True
        )

        stats = bt.run()
        print(stats)

        # Save results
        results_file = os.path.join(results_dir, 'temp_result.json')
        plot_file = os.path.join(results_dir, 'fama_french_factor_prediction.html')

        print(f"Saving stats to {results_file}")
        sanitized = sanitize_stats(stats)
        with open(results_file, 'w') as f:
            json.dump(sanitized, f, indent=4)

        print(f"Saving plot to {plot_file}")
        try:
            bt.plot(filename=plot_file, open_browser=False)
        except Exception as e:
            print(f"Could not generate plot due to an error: {e}")
