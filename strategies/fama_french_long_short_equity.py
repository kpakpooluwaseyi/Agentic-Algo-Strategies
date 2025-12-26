import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from sklearn.linear_model import LinearRegression
import json
import os

# --- Plausible Synthetic Data Generation ---

def generate_plausible_fama_french_factors(primary_asset_returns):
    """
    Generates a DataFrame with plausible synthetic Fama-French factors.

    This is a necessary simulation step because the Fama-French model requires
    market-wide factors that are not available in a single-asset backtest.
    We derive these factors from the primary asset's own data to create a
    logically consistent, albeit simulated, environment.

    Args:
        primary_asset_returns (pd.Series): The daily returns of the primary asset.

    Returns:
        pd.DataFrame: A DataFrame containing the synthetic 'Mkt-RF', 'SMB', and 'HML' factors.
    """
    dates = primary_asset_returns.index
    n = len(dates)

    # Mkt-RF: The primary asset's return is the best proxy for the "market" in this single-asset context.
    mkt_rf = primary_asset_returns.copy()

    # SMB (Small Minus Big): Proxy using rolling return volatility. Higher volatility can be a proxy for smaller, riskier assets.
    smb = primary_asset_returns.rolling(30).std().fillna(0) * np.random.normal(0.1, 0.05, n)

    # HML (High Minus Low): Proxy using rolling momentum. Strong momentum can be a proxy for "growth" (low HML).
    hml = -primary_asset_returns.rolling(90).mean().fillna(0) * np.random.normal(0.5, 0.1, n)

    factors = pd.DataFrame({'Mkt-RF': mkt_rf, 'SMB': smb, 'HML': hml}, index=dates)
    return factors

def generate_synthetic_stock_returns(primary_asset_returns, n_stocks=100):
    """
    Generates a universe of synthetic stock returns correlated with the primary asset.

    This simulation is required to perform the cross-sectional ranking at the
    heart of the Fama-French strategy. We create a "market" of stocks that
    behave similarly to our primary asset but with some random variation.

    Args:
        primary_asset_returns (pd.Series): The daily returns of the primary asset.
        n_stocks (int): The number of synthetic stocks to generate.

    Returns:
        pd.DataFrame: A DataFrame where each column represents a synthetic stock's return series.
    """
    n_days = len(primary_asset_returns)
    synthetic_returns = {}
    for i in range(n_stocks):
        correlation = np.random.uniform(0.2, 0.8)
        noise = np.random.normal(0, np.std(primary_asset_returns) * np.random.uniform(0.5, 2.0), n_days)
        synthetic_returns[f'STOCK_{i}'] = (primary_asset_returns * correlation) + noise
    return pd.DataFrame(synthetic_returns, index=primary_asset_returns.index)

# --- Data Preprocessing ---

def preprocess_data_for_fama_french(data, training_window, n_stocks, n_select):
    """
    Preprocesses the raw data to generate trading signals for the Fama-French strategy simulation.

    The process is as follows:
    1. Resample the primary asset's data to a daily frequency.
    2. Generate plausible synthetic Fama-French factors derived from the asset's own history.
    3. Generate a synthetic universe of stocks with returns correlated to the primary asset.
    4. On each day, perform a rolling regression for every stock in the synthetic universe
       to predict its next-day return based on the current factors.
    5. Rank all stocks (including the primary asset) based on their predicted returns.
    6. Generate a trade signal for the primary asset:
       - LONG (1) if it ranks in the top `n_select` stocks.
       - SHORT (-1) if it ranks in the bottom `n_select` stocks.
       - NEUTRAL (0) otherwise.
    7. Map this daily signal back to the original data's timeframe.

    Args:
        data (pd.DataFrame): The input OHLCV data.
        training_window (int): The rolling window for the regression model.
        n_stocks (int): The total number of synthetic stocks to create.
        n_select (int): The number of top/bottom stocks that define a long/short signal.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'signal' column.
    """
    # 1. Resample to daily and calculate returns for the primary asset
    daily_data = data['Close'].resample('D').last().to_frame()
    daily_data['returns'] = daily_data['Close'].pct_change()
    daily_data.dropna(inplace=True)

    # 2. & 3. Generate the synthetic universe and factors
    factors = generate_plausible_fama_french_factors(daily_data['returns'])
    synthetic_returns = generate_synthetic_stock_returns(daily_data['returns'], n_stocks=n_stocks)

    all_returns = pd.concat([daily_data[['returns']].rename(columns={'returns': 'PRIMARY_ASSET'}), synthetic_returns], axis=1)
    full_df = pd.concat([all_returns, factors], axis=1).dropna()

    signals = pd.Series(index=full_df.index, dtype=float).fillna(0)

    # 4. Perform rolling cross-sectional regression and ranking
    for t in range(training_window, len(full_df)):
        # Define the training window for this step
        window = full_df.iloc[t - training_window : t]
        X_train = window[['Mkt-RF', 'SMB', 'HML']]

        predictions = {}
        # Train a model for each asset and predict its next-day return
        for asset in all_returns.columns:
            y_train = window[asset]
            if y_train.isnull().sum() > training_window * 0.5: continue

            model = LinearRegression()
            model.fit(X_train, y_train)
            # Predict using the most recent factor values
            X_pred = full_df.iloc[t:t+1][['Mkt-RF', 'SMB', 'HML']]
            predictions[asset] = model.predict(X_pred)[0]

        if not predictions: continue

        # 5. & 6. Rank assets and generate signal for the primary asset
        predicted_returns = pd.Series(predictions)
        rank = predicted_returns.rank(ascending=False) # Rank descending: lower rank is better
        primary_asset_rank = rank.get('PRIMARY_ASSET')

        if primary_asset_rank is None: continue
        elif primary_asset_rank <= n_select:
            signals.iloc[t] = 1  # Long if in top N
        elif primary_asset_rank > (len(predicted_returns) - n_select):
            signals.iloc[t] = -1 # Short if in bottom N

    daily_data['signal'] = signals.shift(1) # Shift to avoid lookahead bias

    # 7. Map the daily signal back to the original timeframe
    mapped_signals = data.index.normalize().map(daily_data['signal'])
    data['signal'] = pd.Series(mapped_signals, index=data.index).ffill().fillna(0)

    return data

# --- Strategy Implementation ---

class FamaFrenchLongShortEquity(Strategy):
    # --- Strategy Parameters ---
    # These are now configurable and can be optimized.
    n_stocks = 500         # Total size of the synthetic stock universe.
    n_select = 50          # Number of top/bottom stocks to select for long/short signals.
    training_window = 100  # Rolling window for the regression model.

    def init(self):
        self.signal = self.I(lambda x: x, self.data.signal)

    def next(self):
        if self.position and self.signal[-1] == 0:
            self.position.close()

        signal = self.signal[-1]
        if signal == 1 and not self.position.is_long:
            if self.position.is_short: self.position.close()
            self.buy()
        elif signal == -1 and not self.position.is_short:
            if self.position.is_long: self.position.close()
            self.sell()

# --- Main execution block ---

def sanitize_stats(stats):
    sanitized = stats.to_dict()
    for k in ['_strategy', '_equity_curve', '_trades']:
        if k in sanitized: del sanitized[k]
    for k, v in sanitized.items():
        if isinstance(v, (pd.Timestamp, pd.Timedelta)): sanitized[k] = str(v)
        elif isinstance(v, np.integer): sanitized[k] = int(v)
        elif isinstance(v, np.floating): sanitized[k] = float(v)
    return sanitized

if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'BTC-USD-15m.csv')

    try:
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        data.columns = [c.strip().title() for c in data.columns]
    except FileNotFoundError:
        print("Warning: Data file not found. Generating synthetic data.")
        dates = pd.date_range(start='2022-01-01', periods=20000, freq='15T')
        data = pd.DataFrame({
            'Open': 100 + np.random.randn(20000).cumsum(), 'High': 100 + np.random.randn(20000).cumsum() + 1,
            'Low': 100 + np.random.randn(20000).cumsum() - 1, 'Close': 100 + np.random.randn(20000).cumsum(),
            'Volume': np.random.randint(100, 1000, 20000)}, index=dates)
        data.index.name = 'datetime'

    # Instantiate the strategy to access its parameters
    strategy_params = FamaFrenchLongShortEquity

    # Pass the strategy's parameters to the preprocessing function
    processed_data = preprocess_data_for_fama_french(
        data,
        training_window=strategy_params.training_window,
        n_stocks=strategy_params.n_stocks,
        n_select=strategy_params.n_select
    )

    bt = Backtest(processed_data, FamaFrenchLongShortEquity, cash=100_000, commission=.002)
    stats = bt.run()

    print(stats)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, 'temp_result.json')
    with open(json_path, 'w') as f:
        json.dump(sanitize_stats(stats), f, indent=4)
    print(f"\nSaved stats to {json_path}")

    plot_path = os.path.join(results_dir, 'fama_french_long_short_equity.html')
    try:
        bt.plot(filename=plot_path, open_browser=False)
        print(f"Saved plot to {plot_path}")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")
