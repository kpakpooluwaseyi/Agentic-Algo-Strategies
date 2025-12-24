import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
from backtesting import Strategy, Backtest
import os
import json

class TimeSeriesPredictiveHold(Strategy):
    """
    A time-series predictive strategy using a rolling hold period.

    This strategy trains a linear regression model on a periodic basis to
    predict the return over a fixed future `hold_period`. It then enters a
    position based on the prediction and holds it for the entire period,
    only making a new decision after the hold period is over.
    This approach is a time-series adaptation of a predictive model and
    avoids lookahead bias and performance issues by retraining periodically.
    """
    # 63 trading days * 96 bars/day (for 15m data) = 6048
    hold_period = 6048
    train_window = 1000   # Bars of historical data to use for training the model

    def init(self):
        # Initialize indicators
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), length=14)
        self.sma20 = self.I(ta.sma, pd.Series(self.data.Close), length=20)
        self.sma50 = self.I(ta.sma, pd.Series(self.data.Close), length=50)
        self.atr = self.I(ta.atr, pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), length=14)

        self.model = None
        self.last_train_bar = -self.hold_period # Ensure training on the first possible bar

    def next(self):
        current_bar = len(self.data) - 1

        # --- Periodic Model Retraining ---
        # Only retrain if the hold period has passed since the last training
        if self.model is None or (current_bar - self.last_train_bar >= self.hold_period):
            if len(self.data.Close) < self.train_window:
                return

            self.last_train_bar = current_bar

            # 1. Prepare training data from historical window
            start_idx = max(0, current_bar - self.train_window)
            end_idx = current_bar

            features_data = {
                'rsi': self.rsi[start_idx:end_idx],
                'sma20': self.sma20[start_idx:end_idx],
                'sma50': self.sma50[start_idx:end_idx],
                'atr': self.atr[start_idx:end_idx]
            }
            features = pd.DataFrame(features_data)

            prices = pd.Series(self.data.Close[start_idx:end_idx])
            target = prices.pct_change(periods=self.hold_period).shift(-self.hold_period)

            aligned_data = features.join(target.rename('target')).dropna()
            X_train = aligned_data.drop('target', axis=1)
            y_train = aligned_data['target']

            if len(X_train) < 20:
                self.model = None # Not enough data, can't train
                return

            # 2. Train and store the model
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)

        # --- Trading Logic ---
        # Only act if a model is available and we are not in a position
        if self.model is not None and not self.position:
            # Predict future return based on the latest data
            current_features = np.array([
                self.rsi[-1], self.sma20[-1], self.sma50[-1], self.atr[-1]
            ]).reshape(1, -1)

            prediction = self.model.predict(current_features)[0]

            # Entry Rule with a fixed hold period
            size = 0.1
            if prediction > 0:
                self.buy(size=size)
            elif prediction < 0:
                self.sell(size=size)

        # Time-based exit
        elif self.position:
            if (current_bar - self.trades[0].entry_bar) >= self.hold_period:
                self.position.close()


def sanitize_stats(stats):
    """Sanitizes the stats object for JSON serialization."""
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
        elif key.startswith('_'):
             continue
        else:
            sanitized[key] = value
    return sanitized

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        print("Loading data...")
        data = pd.read_csv(
            data_path,
            index_col='datetime',
            parse_dates=True,
            skipinitialspace=True
        )
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        data.columns = [c.strip().capitalize() for c in data.columns]

        # The new hold period is very long, so we need a large dataset
        # Slicing the last 15k bars to make it runnable for verification
        data = data.iloc[-15000:]

        print("Running backtest...")
        bt = Backtest(data, TimeSeriesPredictiveHold, cash=10000, commission=.002)

        stats = bt.run()
        print(stats)

        os.makedirs('results', exist_ok=True)
        results_path = 'results/temp_result.json'

        final_stats = sanitize_stats(stats)

        with open(results_path, 'w') as f:
            json.dump(final_stats, f, indent=4)
        print(f"Stats saved to {results_path}")

        plot_path = 'results/time_series_predictive_hold.html'
        try:
            bt.plot(filename=plot_path)
            print(f"Plot saved to {plot_path}")
        except Exception as e:
            print(f"Could not generate plot: {e}")
