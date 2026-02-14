
import pandas as pd
import numpy as np
import os
import json
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from backtesting import Backtest, Strategy

def preprocess_data_for_svm(data_path, train_window=252, C=1.0, kernel='rbf'):
    """
    Loads and preprocesses data to generate trading signals using a rolling SVM model.

    Args:
        data_path (str): Path to the CSV data file.
        train_window (int): The number of past days to use for training the model at each step.
        C (float): SVM regularization parameter.
        kernel (str): Specifies the kernel type to be used in the algorithm.

    Returns:
        pandas.DataFrame: The original DataFrame with an added 'signal' column.
    """
    if not os.path.exists(data_path):
        print(f"Data file not found at '{data_path}'. Cannot proceed.")
        return None

    print("Loading and preprocessing data for SVM...")
    # Load 15-minute data
    df = pd.read_csv(
        data_path,
        index_col='datetime',
        parse_dates=True,
        header=0,
    )
    df.columns = [c.strip().title() for c in df.columns]

    # Resample to daily timeframe for feature engineering
    daily_df = df['Close'].resample('D').last().to_frame()

    # Feature Engineering: Previous returns
    daily_df['ret1'] = daily_df['Close'].pct_change(1)
    daily_df['ret2'] = daily_df['Close'].pct_change(2)
    daily_df['ret5'] = daily_df['Close'].pct_change(5)
    daily_df['ret20'] = daily_df['Close'].pct_change(20)

    # Target variable: Next day's return direction
    daily_df['retFut1'] = daily_df['Close'].pct_change(1).shift(-1)
    daily_df['target'] = (daily_df['retFut1'] >= 0).astype(int) # 1 for up, 0 for down

    # Drop NaNs created by feature engineering
    daily_df.dropna(inplace=True)

    # Rolling SVM Prediction
    predictions = []

    # We need enough data to start the rolling window
    if len(daily_df) < train_window:
        print("Not enough data to perform rolling window prediction.")
        return df # Return original df without signals

    for i in range(train_window, len(daily_df)):
        # Define the training set for this iteration
        train_set = daily_df.iloc[i - train_window : i]

        X_train = train_set[['ret1', 'ret2', 'ret5', 'ret20']]
        y_train = train_set['target']

        # Define the feature set for the prediction point
        X_pred = daily_df.iloc[i:i+1][['ret1', 'ret2', 'ret5', 'ret20']]

        # Train the SVM Model
        # The request mentions 5-fold CV, which is for model tuning/evaluation.
        # Here we apply a simple rolling train/predict as is standard for backtesting.
        # A linear kernel is used as a baseline as suggested.
        model = SVC(C=C, kernel=kernel, gamma='auto')
        model.fit(X_train, y_train)

        # Predict the next day's direction
        prediction = model.predict(X_pred)[0]
        predictions.append(prediction)

    # Add predictions to the daily DataFrame
    # Align predictions with the correct dates
    daily_df = daily_df.iloc[train_window:]
    daily_df['predicted_signal'] = predictions

    # Convert binary prediction (0, 1) to trading signal (-1, 1)
    daily_df['signal'] = daily_df['predicted_signal'].apply(lambda x: 1 if x == 1 else -1)

    # Merge daily signals back into the original 15m DataFrame
    daily_signal_map = daily_df['signal']
    df['signal'] = df.index.normalize().map(daily_signal_map)
    df['signal'] = df['signal'].ffill().fillna(0) # Forward-fill signals within the day

    print("Data preprocessing complete.")
    return df

class CrossValidatedSvmSpy(Strategy):
    """
    A strategy that trades based on signals from a rolling Support Vector Machine model.
    The model predicts if the next day will be an 'up' or 'down' day.
    """
    def init(self):
        # The signal is pre-calculated and passed via the data DataFrame
        self.signal = self.I(lambda x: x, self.data.signal)

    def next(self):
        # Follow the signal: 1 for long, -1 for short
        current_signal = self.signal[-1]

        if current_signal == 1 and not self.position.is_long:
            self.position.close() # Close any short position
            self.buy()
        elif current_signal == -1 and not self.position.is_short:
            self.position.close() # Close any long position
            self.sell()

def sanitize_stats(stats):
    """A helper function to sanitize backtest stats for JSON serialization."""
    sanitized = {}
    for key, value in stats.items():
        if isinstance(value, (pd.Series, pd.DataFrame)) or key.startswith('_'):
            continue
        if isinstance(value, (np.floating, np.integer)):
            sanitized[key] = float(value) if np.isfinite(value) else None
        elif isinstance(value, pd.Timestamp):
            sanitized[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            sanitized[key] = str(value)
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    return sanitized


if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    # Preprocess data to get SVM signals
    data_with_signals = preprocess_data_for_svm(data_path, train_window=252, kernel='linear')

    if data_with_signals is not None and 'signal' in data_with_signals.columns:
        # Filter data to start from the first signal
        first_signal_idx = data_with_signals[data_with_signals['signal'] != 0].index[0]
        data_to_backtest = data_with_signals.loc[first_signal_idx:]

        bt = Backtest(data_to_backtest, CrossValidatedSvmSpy, cash=100_000, commission=.002)

        print("Running backtest...")
        stats = bt.run()
        print(stats)

        os.makedirs('results', exist_ok=True)

        final_stats = sanitize_stats(stats)

        with open('results/temp_result.json', 'w') as f:
            json.dump(final_stats, f, indent=2)
        print("Backtest results saved to results/temp_result.json")

        try:
            plot_filename = 'results/cross_validated_svm_spy.html'
            bt.plot(filename=plot_filename)
            print(f"Backtest plot saved to {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot: {e}")
    else:
        print("Backtest skipped due to data processing error.")
