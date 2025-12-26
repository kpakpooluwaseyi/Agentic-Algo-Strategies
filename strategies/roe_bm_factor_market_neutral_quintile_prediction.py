import pandas as pd
import numpy as np
import statsmodels.api as sm
from backtesting import Strategy, Backtest
import json
import os

def generate_synthetic_stock_universe(dates):
    """
    Generates a synthetic monthly stock universe for the given date index.

    Args:
        dates (pd.DatetimeIndex): The monthly dates for which to generate data.

    Returns:
        pd.DataFrame: A DataFrame with ['date', 'ticker', 'ret', 'roe', 'bm'].
    """
    np.random.seed(42) # for reproducibility
    n_stocks = 200
    n_months = len(dates)

    tickers = [f'STOCK_{i}' for i in range(n_stocks)]

    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    df = pd.DataFrame(index=index).reset_index()

    df['roe'] = np.random.uniform(0.01, 0.25, size=len(df))
    df['bm'] = np.random.uniform(0.1, 2.5, size=len(df))

    log_roe = np.log(df['roe'])
    log_bm = np.log(df['bm'])
    noise = np.random.normal(0, 0.05, size=len(df))

    # The underlying factor model the regression will try to find
    df['ret'] = 0.05 * log_roe + 0.02 * log_bm + noise

    # Simulate earnings announcements (used for the activity filter)
    announcements = pd.Series(
        np.random.randint(0, 15, size=n_months),
        index=dates
    )
    df['earnings_announcements'] = df['date'].map(announcements)

    return df.sort_values(by=['date', 'ticker']).reset_index(drop=True)


def run_factor_simulation(data_timeline):
    """
    This function contains the core logic of the multi-asset factor simulation.
    It's designed to be called once from the Strategy's init() method.
    """
    # 1. Generate the synthetic universe based on the backtest timeline
    synthetic_data = generate_synthetic_stock_universe(dates=data_timeline.index)

    unique_dates = data_timeline.index
    portfolio_returns = []

    # 2. Loop through each month to run the regression and form portfolios
    for i in range(len(unique_dates) - 1):
        current_date = unique_dates[i]
        next_date = unique_dates[i+1]

        current_month_data = synthetic_data[synthetic_data['date'] == current_date].copy()
        next_month_data = synthetic_data[synthetic_data['date'] == next_date].copy()

        # Activity Filter
        if current_month_data['earnings_announcements'].iloc[0] < 5:
            portfolio_returns.append({'date': next_date, 'portfolio_ret': 0})
            continue

        # Merge next month's returns to use as the dependent variable `y`
        merged_data = pd.merge(
            current_month_data,
            next_month_data[['ticker', 'ret']],
            on='ticker',
            suffixes=('', '_next_month')
        )

        if merged_data.empty or len(merged_data) < 10: # Need enough data to fit model
            portfolio_returns.append({'date': next_date, 'portfolio_ret': 0})
            continue

        # Fit linear regression model
        X = merged_data[['roe', 'bm']].copy()
        X['log_roe'] = np.log(X.pop('roe'))
        X['log_bm'] = np.log(X.pop('bm'))
        X = sm.add_constant(X)
        y = merged_data['ret_next_month']
        model = sm.OLS(y, X).fit()

        # Predict returns for all stocks in the current month
        predict_X = current_month_data[['roe', 'bm']].copy()
        predict_X['log_roe'] = np.log(predict_X.pop('roe'))
        predict_X['log_bm'] = np.log(predict_X.pop('bm'))
        predict_X = sm.add_constant(predict_X)
        current_month_data['predicted_ret'] = model.predict(predict_X)

        # Form quintiles and identify long/short portfolios
        current_month_data['quintile'] = pd.qcut(
            current_month_data['predicted_ret'], 5, labels=False, duplicates='drop'
        )
        long_stocks = current_month_data[current_month_data['quintile'] == 4]['ticker']
        short_stocks = current_month_data[current_month_data['quintile'] == 0]['ticker']

        # Calculate this month's portfolio return
        long_returns = next_month_data[next_month_data['ticker'].isin(long_stocks)]['ret']
        short_returns = next_month_data[next_month_data['ticker'].isin(short_stocks)]['ret']

        monthly_ret = long_returns.mean() - short_returns.mean() if not long_returns.empty and not short_returns.empty else 0
        portfolio_returns.append({'date': next_date, 'portfolio_ret': monthly_ret})

    # 3. Create the final equity curve
    if not portfolio_returns:
        return np.ones(len(data_timeline)) # Return flat curve if no returns

    returns_df = pd.DataFrame(portfolio_returns).set_index('date')
    equity_curve = (1 + returns_df['portfolio_ret']).cumprod().fillna(1)

    # Reindex to the backtest timeline to ensure it aligns perfectly
    final_curve = equity_curve.reindex(data_timeline.index, method='ffill').fillna(1)

    return final_curve.values


class MarketNeutralFactorStrategy(Strategy):
    """
    This strategy encapsulates the entire multi-asset factor model simulation.
    The logic is executed once in `init()` to generate a portfolio equity curve,
    which is then traded in `next()`.
    """
    def init(self):
        # Run the entire simulation and get the final equity curve
        # The result is stored as a custom indicator
        self.equity_curve = self.I(run_factor_simulation, self.data)

    def next(self):
        # Simple logic to trade the pre-computed equity curve
        # If the curve is rising, be in a long position
        if self.equity_curve[-1] > self.equity_curve[-2]:
            if not self.position.is_long:
                self.buy()
        # If the curve is falling or flat, close the position
        elif self.position.is_long:
            self.position.close()


if __name__ == '__main__':
    # --- Data Loading and Preparation ---
    data_path = 'data/BTC-USD-15m.csv'
    try:
        data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
        data.columns = [c.strip().capitalize() for c in data.columns]
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}, using sample data.")
        from backtesting.test import EURUSD
        data = EURUSD.copy()

    # The strategy logic operates on a monthly basis, so we resample to proper OHLCV candles
    data = data.resample('ME').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # --- Backtest Execution ---
    bt = Backtest(data, MarketNeutralFactorStrategy, cash=100_000, commission=.002, finalize_trades=True)
    stats = bt.run()
    print(stats)

    # --- Output Generation ---
    if not os.path.exists('results'):
        os.makedirs('results')

    plot_filename = 'results/roe_bm_factor_market_neutral_quintile_prediction.html'
    bt.plot(filename=plot_filename)

    # Sanitize and save stats to JSON
    stats_dict = dict(stats)
    for key, value in list(stats_dict.items()):
        if isinstance(value, (pd.Series, pd.DataFrame, Strategy)):
            stats_dict.pop(key)
        elif isinstance(value, pd.Timestamp):
            stats_dict[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            stats_dict[key] = str(value)
        elif pd.isna(value) or np.isnan(value):
            stats_dict[key] = None

    results_filename = 'results/temp_result.json'
    with open(results_filename, 'w') as f:
        json.dump(stats_dict, f, indent=4)
