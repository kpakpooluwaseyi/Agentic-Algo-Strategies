import json
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Strategy Note:
# This strategy is named `random_forest_spy_daily_returns` based on the source material.
# However, it has been implemented to run on BTC-USD data as per the user's instructions.

class RandomForestSpyDailyReturns(Strategy):
    """
    A strategy that uses a Random Forest model to predict the next day's return.
    """

    def init(self):
        """
        Initialize the strategy.
        """
        self.signal = self.I(lambda x: x, self.data.df['signal'], name="Signal")

    def next(self):
        """
        Define the trading logic for the next iteration.
        """
        # Enforce a 1-day holding period.
        if self.position:
            self.position.close()

        # Get the latest signal.
        signal = self.signal[-1]

        # Calculate size for a fixed capital amount
        fixed_capital_amount = 40000 # Must be > asset price
        size = int(fixed_capital_amount / self.data.Close[-1])

        # Entry logic
        if signal > 0:
            self.buy(size=size)
        elif signal < 0:
            self.sell(size=size)


def preprocess_data(data_path):
    """
    Loads and preprocesses the data.
    """
    df = pd.read_csv(data_path, skipinitialspace=True)
    # Clean column names
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains('^unnamed')] # Drop unnamed columns

    # Harden data loading by ensuring OHLCV columns are numeric
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Coerce to numeric, and if any values fail (become NaN), raise an error.
            original_nas = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().sum() > original_nas:
                raise ValueError(
                    f"Column '{col}' contains non-numeric values that could not be parsed. "
                    "This may indicate a malformed or malicious CSV file."
                )

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # Resample to daily timeframe.
    ohlc = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    daily_df = df.resample('D').agg(ohlc)
    daily_df.dropna(inplace=True)

    # Feature Engineering
    for lag in [1, 2, 5, 20]:
        daily_df[f'ret{lag}'] = daily_df['close'].pct_change(lag)

    # Target Variable
    daily_df['target'] = daily_df['close'].pct_change(1).shift(-1)
    daily_df.dropna(inplace=True)

    return daily_df


def train_and_predict(df):
    """
    Trains the Random Forest model and generates predictions.
    """
    features = [col for col in df.columns if 'ret' in col]
    target = 'target'

    # Using a rolling window approach to avoid lookahead bias
    predictions = pd.Series(index=df.index, dtype=float)

    # Define a reasonable training window size
    train_window = 252 # Approximately one year of trading days

    for i in range(train_window, len(df)):
        train_set = df.iloc[i-train_window:i]
        test_set = df.iloc[i:i+1]

        X_train = train_set[features]
        y_train = train_set[target]
        X_test = test_set[features]

        if y_train.isnull().any() or X_train.isnull().values.any():
            continue

        model = RandomForestRegressor(n_estimators=5, min_samples_leaf=100, random_state=42)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        predictions.iloc[i] = pred[0]

    df['signal'] = predictions
    return df


def run_backtest(df):
    """
    Runs the backtest.
    """
    bt = Backtest(df, RandomForestSpyDailyReturns, cash=100000, commission=.002)
    stats = bt.run()

    print(stats)
    bt.plot(filename='results/random_forest_spy_daily_returns.html')

    # Save stats to json using a whitelist of safe metrics
    safe_metrics = [
        'Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]',
        'Equity Peak [$]', 'Return [%]', 'Buy & Hold Return [%]', 'Return (Ann.) [%]',
        'Volatility (Ann.) [%]', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
        '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
        'Avg. Trade [%]', 'Profit Factor', 'Expectancy [%]', 'SQN'
    ]

    json_stats = {}
    for key in safe_metrics:
        value = stats.get(key)

        if value is None:
            continue

        if isinstance(value, pd.Timestamp):
            json_stats[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            json_stats[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            json_stats[key] = value.item()
        elif isinstance(value, (int, float, str, bool)):
            json_stats[key] = value
        else:
            json_stats[key] = str(value)

    with open('results/temp_result.json', 'w') as f:
        json.dump(json_stats, f, indent=4)


if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    processed_df = preprocess_data(data_path)
    final_df = train_and_predict(processed_df)
    final_df.dropna(inplace=True)
    # Rename columns to the format required by backtesting.py
    final_df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)

    run_backtest(final_df)
