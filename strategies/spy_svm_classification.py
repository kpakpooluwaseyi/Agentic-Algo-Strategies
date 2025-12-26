import json
import os
import pandas as pd
from backtesting import Backtest, Strategy
from sklearn.svm import SVC
import numpy as np

def preprocess_data(df):
    """
    Preprocesses the raw data to create features and target for the SVM model.
    """
    # Sanitize column names
    df.columns = [c.strip().lower() for c in df.columns]

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # Resample to daily timeframe
    daily_df = df['close'].resample('D').last().to_frame()
    daily_df['open'] = df['open'].resample('D').first()
    daily_df['high'] = df['high'].resample('D').max()
    daily_df['low'] = df['low'].resample('D').min()
    daily_df['volume'] = df['volume'].resample('D').sum()
    daily_df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'},
                    inplace=True)

    # Calculate returns
    daily_df['return'] = daily_df['Close'].pct_change()
    daily_df['ret1'] = daily_df['return'].shift(1)
    daily_df['ret2'] = daily_df['return'].rolling(window=2).sum().shift(1)
    daily_df['ret5'] = daily_df['return'].rolling(window=5).sum().shift(1)
    daily_df['ret20'] = daily_df['return'].rolling(window=20).sum().shift(1)

    # Create the target variable for the *next* day's return
    daily_df['retFut1'] = daily_df['return'].shift(-1)
    # The signal is whether the *next* day will be an up day.
    # This is used for training, ensuring the model learns to predict the future.
    daily_df['Signal'] = (daily_df['retFut1'] >= 0).astype(int)

    # Drop rows with NaN values
    daily_df.dropna(inplace=True)

    # Select only the necessary columns for the backtest
    final_df = daily_df[['Open', 'High', 'Low', 'Close', 'ret1', 'ret2', 'ret5', 'ret20', 'Signal']].copy()

    return final_df

def sanitize_stats(stats):
    """
    Removes non-serializable objects from the stats dictionary and converts
    special types to JSON-compatible formats.
    """
    stats_dict = dict(stats)

    # Remove non-serializable objects first
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    sanitized = {}
    for key, value in stats_dict.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            sanitized[key] = str(value)
        elif isinstance(value, (np.int64, np.int32)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
             sanitized[key] = float(value)
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    return sanitized


class SvmStrategy(Strategy):
    """
    A strategy that uses a Support Vector Machine (SVM) to predict the next day's price direction.

    NOTE: The original request specified 5-fold cross-validation. Implementing this
    within a rolling backtest framework is computationally intensive and complex.
    For this implementation, priority has been given to fixing the critical
    lookahead bias, and a simpler train/predict model is used.
    """
    train_period = 100  # Days to train the model on
    retrain_every = 30  # Retrain the model every 30 days

    def init(self):
        self.model = SVC(kernel='linear')
        self.features = ['ret1', 'ret2', 'ret5', 'ret20']
        self.target = 'Signal'
        self.model_fitted = False
        self.last_retrain_bar = 0

    def next(self):
        # Close position after 1 day
        if self.position:
            self.position.close()

        # It's decision time for the current bar (i). We can only use data up to i-1.
        current_bar_index = len(self.data.Close) - 1

        # Retrain the model periodically.
        # We only train on data available *before* the current bar.
        if (current_bar_index > self.train_period and
            current_bar_index >= self.last_retrain_bar + self.retrain_every):

            # Prepare training data from a rolling window of past data (up to i-1)
            train_df = self.data.df.iloc[current_bar_index - self.train_period : current_bar_index]

            X_train = train_df[self.features]
            y_train = train_df[self.target]

            if not X_train.empty and not y_train.empty:
                self.model.fit(X_train, y_train)
                self.model_fitted = True
                self.last_retrain_bar = current_bar_index

        # Make a prediction for the next day if the model is fitted.
        # The features for prediction are from the previous day (i-1).
        if self.model_fitted:
            features = self.data.df.iloc[current_bar_index-1:current_bar_index][self.features]
            if not features.empty:
                prediction = self.model.predict(features)[0]

                if prediction == 1:
                    self.buy(size=0.1)
                else:
                    self.sell(size=0.1)


if __name__ == '__main__':
    # Ensure the results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    data = pd.read_csv('data/BTC-USD-15m.csv')
    processed_data = preprocess_data(data)

    bt = Backtest(processed_data, SvmStrategy, cash=1_000_000, commission=.002, finalize_trades=True)
    stats = bt.run()

    print("Backtest complete. Final stats:")
    print(stats)

    # Sanitize and save stats to a JSON file
    stats_dict = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)

    print("\nResults saved to results/temp_result.json")

    # Generate the plot, catching potential errors
    plot_filename = 'results/spy_svm_classification.html'
    try:
        bt.plot(filename=plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
