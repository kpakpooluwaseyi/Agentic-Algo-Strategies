import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import os
import json

def preprocess_and_generate_signals(data_path, training_window=252):
    """
    Loads and preprocesses the data, then generates trading signals using a rolling regression model.
    """
    # Load and prepare data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data.columns = [c.strip().title() for c in data.columns]

    # Resample to daily and calculate returns
    daily_data = data['Close'].resample('D').last().dropna().to_frame()
    daily_data['Return'] = daily_data['Close'].pct_change()

    # Create predictors (past returns) and response (future return)
    for lag in [1, 2, 5, 20]:
        daily_data[f'ret{lag}'] = daily_data['Return'].shift(lag)
    daily_data['retFut1'] = daily_data['Return'].shift(-1)

    # Drop rows with NaN values created by shifting
    daily_data = daily_data.dropna()

    # Ensure we have enough data to perform a single training run
    if len(daily_data) < training_window:
        print(f"Warning: Data length ({len(daily_data)}) is less than the training window ({training_window}). No trades will be generated.")
        daily_data['signal'] = 0
        # Add required OHLC columns for backtesting even if no signals
        ohlc = data[['Open', 'High', 'Low', 'Close']].resample('D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()
        final_df = ohlc.join(daily_data[['signal']], how='inner')
        return final_df

    # --- Rolling Prediction ---
    predictions = []

    # Define the model: 5 regression trees with MinLeafSize of 100
    base_estimator = DecisionTreeRegressor(min_samples_leaf=100)
    model = BaggingRegressor(estimator=base_estimator, n_estimators=5, random_state=42)

    X = daily_data[[f'ret{lag}' for lag in [1, 2, 5, 20]]]
    y = daily_data['retFut1']

    for i in range(training_window, len(daily_data)):
        X_train = X.iloc[i-training_window : i]
        y_train = y.iloc[i-training_window : i]
        X_test = X.iloc[i:i+1]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        predictions.append(pred)

    # Align predictions with the correct dates in the original DataFrame
    prediction_dates = daily_data.index[training_window:]
    prediction_series = pd.Series(predictions, index=prediction_dates)

    daily_data['predicted_return'] = prediction_series

    # Generate signals: +1 for positive predicted return, -1 for negative
    daily_data['signal'] = np.sign(daily_data['predicted_return']).fillna(0)

    # Merge signals with the original OHLC data for backtesting
    ohlc = data[['Open', 'High', 'Low', 'Close']].resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()

    final_df = ohlc.join(daily_data[['signal']], how='inner')
    final_df['signal'] = final_df['signal'].fillna(0) # Ensure no NaNs in signal column

    return final_df

def passthrough(series):
    return series

class BaggedRegressionReturnsStrategy(Strategy):
    """
    A strategy that uses an ensemble of regression trees to predict
    the next day's return and trades based on the predicted direction.
    """
    def init(self):
        # The signal is pre-calculated, so we just need to load it.
        self.signal = self.I(passthrough, self.data.signal, name="Signal")

    def next(self):
        # Enforce a 1-day holding period by closing any open position at the start of a new bar.
        if self.position:
            self.position.close()

        # Get the signal for the current day
        current_signal = self.signal[-1]

        if current_signal == 1:
            # Positive predicted return -> go long
            self.buy()
        elif current_signal == -1:
            # Negative predicted return -> go short
            self.sell()

def sanitize_stats(stats):
    """Prepares the backtest stats object for JSON serialization."""
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Series, pd.DataFrame, type(pd.NA))):
            continue
        elif pd.isna(value):
            sanitized[key] = None
        elif isinstance(value, (np.int64, np.int32)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            sanitized[key] = float(value)
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        else:
            sanitized[key] = value
    return sanitized

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    print("Preprocessing data and generating signals...")
    try:
        processed_data = preprocess_and_generate_signals(data_path, training_window=252)

        if processed_data.empty or processed_data['signal'].eq(0).all():
            print("No trading signals were generated. Backtest will not run.")
            # Create a dummy result file for compatibility with runners
            result = {
                'strategy_name': 'spy_bagged_regression_tree_returns',
                'return': 0.0, 'sharpe': None, 'max_drawdown': 0.0,
                'win_rate': None, 'total_trades': 0
            }
        else:
            print("Running backtest...")
            bt = Backtest(processed_data, BaggedRegressionReturnsStrategy, cash=100_000, commission=.002)
            stats = bt.run()

            print("Backtest complete. Results:")
            print(stats)

            result = {
                'strategy_name': 'spy_bagged_regression_tree_returns',
                'return': stats.get('Return [%]', 0.0),
                'sharpe': stats.get('Sharpe Ratio'),
                'max_drawdown': stats.get('Max. Drawdown [%]', 0.0),
                'win_rate': stats.get('Win Rate [%]'),
                'total_trades': stats.get('# Trades', 0)
            }

            # Generate plot if there were trades
            if result['total_trades'] > 0:
                print("Generating plot...")
                plot_filename = 'results/spy_bagged_regression_tree_returns.html'
                try:
                    bt.plot(filename=plot_filename, open_browser=False)
                    print(f"Plot saved to {plot_filename}")
                except Exception as e:
                    print(f"Could not generate plot: {e}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        result = {'strategy_name': 'spy_bagged_regression_tree_returns', 'error': str(e)}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        result = {'strategy_name': 'spy_bagged_regression_tree_returns', 'error': str(e)}

    # Save results to a JSON file
    os.makedirs('results', exist_ok=True)
    with open('results/temp_result.json', 'w') as f:
        json.dump(sanitize_stats(result), f, indent=4)

    print("\nResults saved to results/temp_result.json")
