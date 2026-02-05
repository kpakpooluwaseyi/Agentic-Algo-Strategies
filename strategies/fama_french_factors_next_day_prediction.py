import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LinearRegression
from backtesting import Backtest, Strategy

# Helper function to sanitize stats for JSON serialization
def sanitize_stats(stats):
    """
    Sanitizes the backtest stats object by converting non-serializable
    types to native Python types.
    """
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (np.int64, np.integer)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.float64, np.floating)):
            sanitized[key] = float(value)
        elif isinstance(value, pd.Timestamp):
            sanitized[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            sanitized[key] = str(value)
        elif key == '_strategy' or key == '_equity_curve' or key == '_trades':
            continue  # Skip internal objects
        else:
            sanitized[key] = value
    return sanitized

def preprocess_and_predict(filepath, n_lags=5):
    """
    Loads and preprocesses data, then trains a linear regression model to predict
    the next day's return based on its own past returns.

    This serves as a single-asset adaptation of the multi-asset Fama-French
    factor model concept.
    """
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = [c.strip().rstrip(',').title() for c in df.columns]
    if 'Unnamed: 6' in df.columns:
        df.drop(columns=['Unnamed: 6'], inplace=True)

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)

    daily_df = df.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna()

    # Create features and target
    daily_df['ret1'] = daily_df['Close'].pct_change()
    for i in range(1, n_lags + 1):
        daily_df[f'ret1_lag{i}'] = daily_df['ret1'].shift(i)

    # The target is the *next* day's return
    daily_df['target'] = daily_df['ret1'].shift(-1)
    daily_df.dropna(inplace=True)

    # Split data for training
    train_size = int(len(daily_df) * 0.5)
    train_set = daily_df.iloc[:train_size]

    X_train = train_set[[f'ret1_lag{i}' for i in range(1, n_lags + 1)]]
    y_train = train_set['target']

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Generate predictions on the entire dataset
    X_full = daily_df[[f'ret1_lag{i}' for i in range(1, n_lags + 1)]]
    daily_df['predicted_ret'] = model.predict(X_full)

    return daily_df

class FamaFrenchNextDayPrediction(Strategy):
    """
    A strategy that trades based on a linear regression model's prediction of
    the next day's return, using past returns as features. This is an
    adaptation of the Fama-French factor model for a single asset.
    """
    def init(self):
        # The pre-calculated predictions are accessed as a custom indicator
        self.predicted_ret = self.I(lambda: self.data.df['predicted_ret'])

    def next(self):
        # Enforce a 1-day holding period by closing any open position.
        if self.position:
            self.position.close()

        # Get the prediction for the upcoming bar. The model was trained to
        # predict the return of the next bar using data from the current bar.
        prediction = self.predicted_ret[-1]

        # Trade based on the prediction
        if prediction > 0:
            self.buy()
        elif prediction < 0:
            self.sell()

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        # Preprocess the data to generate model predictions
        data_with_predictions = preprocess_and_predict(data_path)

        # Instantiate the Backtest
        bt = Backtest(data_with_predictions, FamaFrenchNextDayPrediction, cash=100000, commission=.002, finalize_trades=True)

        # Run the backtest
        print("Running backtest...")
        stats = bt.run()
        print(stats)

        # Sanitize and save the results
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        sanitized = sanitize_stats(stats)
        with open(os.path.join(results_dir, 'temp_result.json'), 'w') as f:
            json.dump(sanitized, f, indent=4)
        print(f"\nResults saved to {results_dir}/temp_result.json")

        # Generate the plot
        plot_filename = os.path.join(results_dir, 'fama_french_factors_next_day_prediction.html')
        try:
            bt.plot(filename=plot_filename)
            print(f"Plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot due to error: {e}")
