
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from sklearn.linear_model import LinearRegression
import json
import os

# --- Step 1: Synthetic Data Generation ---
def generate_synthetic_universe(base_data, n_assets=100, correlation=0.7):
    base_returns = base_data['Close'].pct_change().dropna()
    asset_names = [f'asset_{i}' for i in range(n_assets)]
    synthetic_universe = pd.DataFrame(columns=asset_names, index=base_data.index)

    for asset in asset_names:
        noise = np.random.normal(0, base_returns.std(), len(base_returns))
        correlated_returns = correlation * base_returns + (1 - correlation) * noise
        start_price = base_data['Close'].iloc[0] * np.random.uniform(0.5, 1.5)
        price_series = start_price * (1 + correlated_returns).cumprod()
        synthetic_universe[asset] = price_series.reindex(base_data.index).ffill().bfill()

    synthetic_universe.dropna(axis=1, how='any', inplace=True)
    asset_chars = pd.DataFrame(index=synthetic_universe.columns)
    asset_chars['size'] = np.random.lognormal(10, 2, len(synthetic_universe.columns))
    asset_chars['book_to_market'] = np.random.uniform(0.5, 2.5, len(synthetic_universe.columns))
    return synthetic_universe, asset_chars

# --- Step 2: Full Portfolio Simulation ---
def simulate_fama_french_portfolio(synthetic_universe, asset_chars, lookback=60, top_n=20):
    daily_prices = synthetic_universe.resample('D').last()
    daily_returns = daily_prices.pct_change().dropna()

    # Calculate Factors
    mkt_rf = daily_returns.mean(axis=1)
    median_size = asset_chars['size'].median()
    small_caps = asset_chars[asset_chars['size'] < median_size].index
    big_caps = asset_chars[asset_chars['size'] >= median_size].index
    smb = daily_returns[small_caps].mean(axis=1) - daily_returns[big_caps].mean(axis=1)
    median_bm = asset_chars['book_to_market'].median()
    high_bm = asset_chars[asset_chars['book_to_market'] >= median_bm].index
    low_bm = asset_chars[asset_chars['book_to_market'] < median_bm].index
    hml = daily_returns[high_bm].mean(axis=1) - daily_returns[low_bm].mean(axis=1)
    factors = pd.DataFrame({'Mkt_RF': mkt_rf, 'SMB': smb, 'HML': hml})

    # --- Rolling Prediction and Portfolio Return Calculation ---
    portfolio_returns = []

    if len(daily_returns) < lookback + 1:
        return None # Not enough data

    for i in range(len(daily_returns) - lookback - 1):
        train_end_idx = i + lookback
        pred_idx = train_end_idx + 1

        factors_train = factors.iloc[i:train_end_idx]
        returns_train = daily_returns.iloc[i:train_end_idx]

        predictions = {}
        for asset in daily_returns.columns:
            model = LinearRegression()
            model.fit(factors_train, returns_train[asset])
            predictions[asset] = model.predict(factors.iloc[train_end_idx:train_end_idx+1])[0]

        pred_series = pd.Series(predictions).sort_values(ascending=False)

        long_portfolio = pred_series.head(top_n).index
        short_portfolio = pred_series.tail(top_n).index

        long_return = daily_returns.iloc[pred_idx][long_portfolio].mean()
        short_return = daily_returns.iloc[pred_idx][short_portfolio].mean()

        # Net return of the dollar-neutral long/short portfolio
        portfolio_returns.append(long_return - short_return)

    # --- Create a synthetic equity curve for backtesting ---
    if not portfolio_returns:
        return None

    portfolio_daily_returns = pd.Series(portfolio_returns, index=daily_returns.index[lookback + 1:])
    equity_curve = (1 + portfolio_daily_returns).cumprod() * 10000 # Start with 10k capital

    # Create a DataFrame in OHLC format for backtesting.py
    synthetic_ohlc = pd.DataFrame(index=equity_curve.index)
    synthetic_ohlc['Open'] = equity_curve.shift(1).fillna(10000)
    synthetic_ohlc['Close'] = equity_curve
    synthetic_ohlc['High'] = equity_curve
    synthetic_ohlc['Low'] = equity_curve.shift(1).fillna(10000)
    synthetic_ohlc['Volume'] = 100 # Dummy volume

    return synthetic_ohlc.dropna()

# --- Step 3: "Buy and Hold" Strategy for Analysis ---
class PortfolioAnalyzer(Strategy):
    def init(self):
        pass
    def next(self):
        if not self.position:
            self.buy()

# --- New Sanitization Function ---
def sanitize_stats_for_json(stats):
    """
    Sanitizes the backtest stats object for JSON serialization using a strict allowlist.
    """
    allowed_keys = {
        'Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]',
        'Equity Peak [$]', 'Return [%]', 'Buy & Hold Return [%]', 'Return (Ann.) [%]',
        'Volatility (Ann.) [%]', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
        'Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration',
        'Avg. Drawdown Duration', '# Trades', 'Win Rate [%]', 'Best Trade [%]',
        'Worst Trade [%]', 'Avg. Trade [%]', 'Max. Trade Duration',
        'Avg. Trade Duration', 'Profit Factor', 'Expectancy [%]', 'SQN'
    }

    sanitized = {}
    for key in allowed_keys:
        if key in stats:
            value = stats[key]
            # Handle non-native types
            if isinstance(value, (np.integer, np.int64)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                sanitized[key] = float(value) if pd.notna(value) else None
            elif isinstance(value, pd.Timestamp):
                sanitized[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                sanitized[key] = str(value)
            elif pd.isna(value):
                sanitized[key] = None
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            # Skip any other types, including complex objects
    return sanitized


# --- Step 4: Backtesting Execution ---
if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    try:
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        data.columns = [col.strip().title() for col in data.columns]
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    except FileNotFoundError:
        print("Data file not found. Using synthetic GOOG data.")
        from backtesting.test import GOOG
        data = GOOG.copy()

    # 1. Generate universe
    synthetic_universe, asset_chars = generate_synthetic_universe(data, n_assets=100)

    # 2. Simulate portfolio and get its equity curve
    portfolio_equity_curve = simulate_fama_french_portfolio(
        synthetic_universe, asset_chars, lookback=60, top_n=20
    )

    if portfolio_equity_curve is None or portfolio_equity_curve.empty:
        print("Could not generate portfolio equity curve. Not enough data.")
    else:
        # 3. Run backtest on the synthetic portfolio performance
        bt = Backtest(portfolio_equity_curve, PortfolioAnalyzer, cash=100000, commission=0.0, finalize_trades=True)
        stats = bt.run()
        print(stats)

        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, 'fama_french_next_day_predictive.html')
        bt.plot(filename=plot_filename)

        # Sanitize and save stats using the new function
        sanitized_stats = sanitize_stats_for_json(stats)

        with open(os.path.join(output_dir, 'temp_result.json'), 'w') as f:
            json.dump(sanitized_stats, f, indent=4)
        print(f"Backtest finished. Plot: {plot_filename}")
