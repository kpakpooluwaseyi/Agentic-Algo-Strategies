import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from sklearn.linear_model import LinearRegression
import os
import json

# --- Synthetic Data Generation ---

def generate_synthetic_data(base_data, num_assets=100, years=5):
    """
    Generates a synthetic dataset of multiple stocks and Fama-French factors.
    - The market factor (Mkt-Rf) is derived from the base data (BTC).
    - SMB and HML factors are generated as random walks.
    - Individual stock returns are generated based on the Fama-French model
      with randomized factor loadings for each stock.
    """
    print("Generating synthetic data...")
    # Resample base data to daily
    base_data['datetime'] = pd.to_datetime(base_data['datetime'])
    base_data = base_data.set_index('datetime')
    daily_data = base_data['close'].resample('D').last().dropna()

    # Limit data to a specific number of years to manage simulation time
    daily_data = daily_data.last(f'{365*years}D')

    # Calculate market factor (Mkt-Rf) from the base data
    mkt_rf = daily_data.pct_change().fillna(0)

    # Generate SMB and HML factors as random walks
    num_days = len(mkt_rf)
    smb = pd.Series(np.random.normal(0.0001, 0.005, num_days), index=mkt_rf.index).cumsum()
    hml = pd.Series(np.random.normal(0.0001, 0.005, num_days), index=mkt_rf.index).cumsum()

    # Combine factors into a single DataFrame
    factors = pd.DataFrame({'Mkt-Rf': mkt_rf, 'SMB': smb, 'HML': hml})

    # Generate individual stock data
    asset_prices = {}
    for i in range(num_assets):
        # Assign random factor loadings (betas) for each stock
        beta_mkt = np.random.normal(1.0, 0.3)
        beta_smb = np.random.normal(0.0, 0.5)
        beta_hml = np.random.normal(0.0, 0.5)

        # Generate idiosyncratic noise
        noise = np.random.normal(0, 0.01, num_days)

        # Calculate stock returns based on the FF model
        asset_return = (factors['Mkt-Rf'] * beta_mkt +
                        factors['SMB'] * beta_smb +
                        factors['HML'] * beta_hml +
                        noise)

        # Create a price series from the returns
        price_series = 100 * (1 + asset_return).cumprod()
        asset_prices[f'Asset_{i}'] = price_series

    asset_df = pd.DataFrame(asset_prices)

    # Combine asset prices and factors
    full_df = pd.concat([asset_df, factors], axis=1).dropna()

    print("Synthetic data generation complete.")
    return full_df


# --- Portfolio Simulation ---

def run_portfolio_simulation(data, training_window=120, num_long=50, num_short=50):
    """
    Runs the Fama-French cross-sectional portfolio simulation.
    - Iterates through the data day-by-day.
    - On each day, trains a regression model for each asset using a lookback window.
    - Predicts next-day returns for all assets.
    - Forms a long-short portfolio by selecting the top and bottom predicted performers.
    - Calculates the daily portfolio return and compounds it into an equity curve.
    """
    print("Running portfolio simulation...")
    asset_cols = [col for col in data.columns if col.startswith('Asset_')]
    factor_cols = ['Mkt-Rf', 'SMB', 'HML']

    portfolio_returns = []

    # We start the simulation after the first training window period
    for i in range(training_window, len(data) - 1):

        # Define the training data for the lookback period
        train_data = data.iloc[i - training_window:i]

        # Prepare target variable (returns) and features (factors)
        y_train = train_data[asset_cols].pct_change().dropna()
        X_train = train_data[factor_cols].loc[y_train.index]

        # Current day's factors for prediction
        X_pred_today = data[factor_cols].iloc[i:i+1]

        predictions = []
        for asset in asset_cols:
            # Check for sufficient data
            if y_train[asset].count() < 30:
                continue

            # Train a model for each asset
            model = LinearRegression()
            model.fit(X_train, y_train[asset])

            # Predict next day's return
            predicted_return = model.predict(X_pred_today)[0]
            predictions.append({'asset': asset, 'predicted_return': predicted_return})

        if not predictions:
            portfolio_returns.append(0)
            continue

        # Rank assets by predicted return
        pred_df = pd.DataFrame(predictions).sort_values(by='predicted_return', ascending=False)

        # Select long and short portfolios
        long_assets = pred_df.head(num_long)['asset'].tolist()
        short_assets = pred_df.tail(num_short)['asset'].tolist()

        # Calculate next day's actual portfolio return
        actual_returns_next_day = data[asset_cols].pct_change().iloc[i+1]

        long_return = actual_returns_next_day[long_assets].mean()
        short_return = actual_returns_next_day[short_assets].mean()

        # Dollar-neutral portfolio return
        daily_portfolio_return = (long_return - short_return) / 2
        portfolio_returns.append(daily_portfolio_return)

        if (i - training_window) % 100 == 0:
             print(f"  Simulated day {i - training_window} of {len(data) - training_window -1}...")

    # Create the portfolio equity curve
    portfolio_series = pd.Series(portfolio_returns, index=data.index[training_window+1:])
    equity_curve = 100 * (1 + portfolio_series).cumprod()

    print("Portfolio simulation complete.")
    return equity_curve.dropna()


# --- Strategy Definition ---

class FamaFrenchPortfolio(Strategy):
    """
    This is a proxy strategy. It doesn't trade in the conventional sense.
    Instead, it runs on a pre-computed equity curve from a simulated
    Fama-French long-short portfolio. The goal is to use backtesting.py's
    analytics and plotting capabilities on the results of the simulation.
    """
    def init(self):
        # A simple "buy and hold" on the portfolio's equity curve
        self.buy()

    def next(self):
        pass

# --- Main Execution Block ---

if __name__ == '__main__':
    # Configuration
    data_path = 'data/BTC-USD-15m.csv'
    results_dir = 'results'
    results_file = os.path.join(results_dir, 'temp_result.json')
    plot_file = os.path.join(results_dir, 'fama_french_three_factor_predictive_long_short_equity.html')

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    print("Strategy execution started.")
    print("This strategy involves a complex simulation and may take a few minutes to run.")

    # 1. Load base data
    try:
        base_data = pd.read_csv(data_path)
        # Sanitize column names: strip whitespace and convert to lowercase
        base_data.columns = [col.strip().lower() for col in base_data.columns]
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        # Create a dummy file to avoid crashing the runner
        with open(results_file, 'w') as f:
            json.dump({"error": "Data file not found"}, f)
        exit()

    # 2. Generate synthetic data
    # Using a smaller number of years for quicker execution
    synthetic_data = generate_synthetic_data(base_data, num_assets=100, years=3)

    # 3. Run portfolio simulation
    equity_curve = run_portfolio_simulation(synthetic_data, training_window=120, num_long=50, num_short=50)

    if equity_curve.empty:
        print("Equity curve is empty. Skipping backtest.")
        stats = {"message": "Could not generate an equity curve from the simulation."}
    else:
        # 4. Prepare data for backtesting.py
        # The framework needs an OHLC DataFrame. We'll create one from our equity curve.
        bt_data = pd.DataFrame({
            'Open': equity_curve,
            'High': equity_curve,
            'Low': equity_curve,
            'Close': equity_curve,
            'Volume': 100 # Add some dummy volume
        })
        bt_data.index.name = 'datetime' # Ensure index has a name if it's a DatetimeIndex

        # 5. Run the backtest
        print("Running backtest on the simulated portfolio equity curve...")
        bt = Backtest(bt_data, FamaFrenchPortfolio, cash=1_000_000, commission=.002, finalize_trades=True)
        stats = bt.run()
        print(stats)

        # 6. Plot the results
        try:
            bt.plot(filename=plot_file, open_browser=False)
            print(f"Plot saved to {plot_file}")
        except Exception as e:
            print(f"Could not generate plot: {e}")

    # 7. Save results
    # Sanitize stats for JSON serialization
    stats_dict = dict(stats)
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    sanitized_stats = {}
    for key, value in stats_dict.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized_stats[key] = str(value)
        elif isinstance(value, (np.int64, np.float64, np.bool_)):
            sanitized_stats[key] = value.item()
        elif pd.isna(value):
            sanitized_stats[key] = None
        else:
            sanitized_stats[key] = value

    with open(results_file, 'w') as f:
        json.dump(sanitized_stats, f, indent=4)

    print(f"Final results saved to {results_file}")
