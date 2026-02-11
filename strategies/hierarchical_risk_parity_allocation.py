
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from backtesting import Strategy, Backtest


def generate_synthetic_portfolio(df, n_assets=10, vol_scale=0.2):
    """
    Generates a synthetic portfolio of correlated assets from a single price series.

    Args:
        df (pd.DataFrame): DataFrame with a 'Close' price column.
        n_assets (int): The total number of assets in the final portfolio.
        vol_scale (float): The amount of random noise to add to generate new assets.

    Returns:
        pd.DataFrame: A DataFrame where each column is a price series of a synthetic asset.
    """
    if 'Close' not in df.columns:
        raise ValueError("Input DataFrame must have a 'Close' column.")

    base_asset_price = df['Close'].copy().rename('Asset_0')
    returns = base_asset_price.pct_change().dropna()

    # Align index of returns with the main df
    aligned_returns_index = base_asset_price.index[1:]

    synthetic_assets = [base_asset_price]
    np.random.seed(42) # for reproducibility

    for i in range(1, n_assets):
        # Create correlated returns by adding scaled random noise
        noise = np.random.normal(0, returns.std() * vol_scale, size=len(returns))
        synthetic_returns = returns + noise

        # Convert returns back to a price series
        initial_price = base_asset_price.iloc[0]

        # Create a new series for the cumulative product
        cumulative_returns = (1 + synthetic_returns).cumprod()

        # Prepend the initial price
        synthetic_price = pd.concat([pd.Series([initial_price]), initial_price * cumulative_returns])

        # Set the correct index
        synthetic_price.index = base_asset_price.index[:len(synthetic_price)]

        synthetic_price.name = f'Asset_{i}'
        synthetic_assets.append(synthetic_price)

    portfolio_df = pd.concat(synthetic_assets, axis=1).dropna()
    return portfolio_df


# --- HRP Algorithm Implementation ---

def get_cluster_var(cov, cluster_assets):
    """Calculates the variance of a cluster of assets."""
    cluster_cov = cov.loc[cluster_assets, cluster_assets]
    weights = np.ones(len(cluster_assets)) / len(cluster_assets)
    return np.dot(weights, np.dot(cluster_cov, weights))

def get_quasi_diag(link):
    """Sorts the assets based on the hierarchical clustering tree."""
    link = link.astype(int)
    sort_ix = pd.Series([link[0, 0], link[0, 1]])
    num_items = link[0, 3]

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()

def get_rec_bipart(cov, sort_ix):
    """Recursively bisects the portfolio to allocate weights."""
    w = pd.Series(1, index=sort_ix)
    c_items = [sort_ix]

    while len(c_items) > 0:
        c_items = [
            i[j:k]
            for i in c_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            w[c_items0] *= alpha
            w[c_items1] *= 1 - alpha
    return w

def hrp_allocation(asset_returns):
    """
    Calculates asset weights using the Hierarchical Risk Parity algorithm.
    """
    if not isinstance(asset_returns, pd.DataFrame) or asset_returns.empty:
        # Return equal weights if no data is available
        return pd.Series(1./len(asset_returns.columns), index=asset_returns.columns)

    cov = asset_returns.cov()
    corr = asset_returns.corr()

    dist = np.sqrt((1 - corr) / 2)
    link = sch.linkage(squareform(dist), 'single')

    sort_ix = get_quasi_diag(link)
    sorted_assets = corr.index[sort_ix].tolist()

    weights = get_rec_bipart(cov, sorted_assets)
    return weights.sort_index()


def preprocess_data(df, rebalance_period=30, lookback_period=90, n_assets=15):
    """
    Generates the HRP portfolio equity curve to be backtested.
    """
    # 1. Generate synthetic portfolio
    portfolio_prices = generate_synthetic_portfolio(df, n_assets=n_assets)
    portfolio_returns = portfolio_prices.pct_change().dropna()

    # 2. Perform periodic rebalancing
    portfolio_equity = [100000] # Initial capital
    last_rebalance_day = None
    weights = pd.Series(1./n_assets, index=portfolio_prices.columns)

    for i in range(1, len(portfolio_returns)):
        current_date = portfolio_returns.index[i].date()

        # Rebalance at the start of a new month (approximates monthly rebalancing)
        if last_rebalance_day is None or (current_date.month != portfolio_returns.index[i-1].date().month):

            lookback_data = portfolio_returns.iloc[max(0, i - lookback_period):i]
            if not lookback_data.empty:
                weights = hrp_allocation(lookback_data)
            last_rebalance_day = current_date

        # Calculate portfolio return for the current period
        period_return = (portfolio_returns.iloc[i] * weights).sum()

        # Update portfolio equity
        current_equity = portfolio_equity[-1] * (1 + period_return)
        portfolio_equity.append(current_equity)

    # 3. Create a DataFrame for the backtester
    equity_curve = pd.Series(portfolio_equity, index=portfolio_returns.index)

    # Create a synthetic OHLC DataFrame for backtesting.py
    synthetic_ohlc = pd.DataFrame(index=equity_curve.index)
    synthetic_ohlc['Open'] = equity_curve.shift(1).fillna(method='bfill')
    synthetic_ohlc['High'] = equity_curve
    synthetic_ohlc['Low'] = equity_curve.shift(1).fillna(method='bfill')
    synthetic_ohlc['Close'] = equity_curve
    synthetic_ohlc['Volume'] = 100 # Add dummy volume

    return synthetic_ohlc.dropna()


class HierarchicalRiskParityAllocation(Strategy):
    def init(self):
        pass

    def next(self):
        pass

def sanitize_stats(stats):
    """
    Sanitizes the backtesting stats object to be JSON serializable,
    handling specific pandas and numpy types.
    """
    if stats is None:
        return None

    # If stats is a Series, convert it to a dict
    if isinstance(stats, pd.Series):
        stats = stats.to_dict()

    # Sanitize individual items in the dictionary
    clean_stats = {}
    for key, value in stats.items():
        if isinstance(key, pd.Timestamp):
            key = key.isoformat()

        # Handle problematic DataFrame values
        if isinstance(value, (pd.DataFrame, pd.Series)):
            continue # Skip DataFrames like _equity_curve and _trades

        # Handle numpy types
        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, np.bool_):
            value = bool(value)

        # Handle pandas Timestamps and Timedeltas
        if isinstance(value, pd.Timestamp):
            value = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            value = str(value)

        # Handle NA/NaN values
        if pd.isna(value) or value is np.nan:
            value = None

        clean_stats[key] = value

    # Remove keys that are known to be problematic
    clean_stats.pop('_strategy', None)
    clean_stats.pop('_equity_curve', None)
    clean_stats.pop('_trades', None)

    return clean_stats

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    try:
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        exit()

    print("Preprocessing data and generating HRP equity curve...")
    processed_df = preprocess_data(df)

    if processed_df.empty:
        print("Error: Preprocessing resulted in an empty DataFrame. Cannot run backtest.")
        exit()

    print("Running backtest...")
    bt = Backtest(processed_df, HierarchicalRiskParityAllocation, cash=100_000, commission=.002)
    stats = bt.run()

    print("\nBacktest Results:")
    print(stats)

    # Save results
    results_path = 'results/temp_result.json'
    plot_path = 'results/hierarchical_risk_parity_allocation.html'

    print(f"\nSaving stats to {results_path}")
    cleaned_stats = sanitize_stats(stats)
    with open(results_path, 'w') as f:
        json.dump(cleaned_stats, f, indent=4)

    print(f"Saving plot to {plot_path}")
    try:
        bt.plot(filename=plot_path, open_browser=False)
    except Exception as e:
        print(f"Could not generate plot due to an error: {e}")
