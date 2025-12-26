from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression

def preprocess_data(df, short_window=20, long_window=60, train_window=90, hold_period=7):
    """
    Preprocesses data to create technical factors and a predictive model.
    This strategy uses the ratio of price to its moving averages as predictive factors.
    A rolling linear regression model is trained to predict the return over a
    specified holding period.
    """
    # 1. Create technical factors
    df['short_ma'] = df['Close'].rolling(short_window).mean()
    df['long_ma'] = df['Close'].rolling(long_window).mean()

    # Use log of the price-to-MA ratio to stabilize variance and create factors
    df['price_to_short_ma'] = np.log(df['Close'] / df['short_ma'])
    df['price_to_long_ma'] = np.log(df['Close'] / df['long_ma'])

    # 2. Create target variable (future return)
    df['future_return'] = df['Close'].pct_change(periods=hold_period).shift(-hold_period)
    df.dropna(inplace=True)

    # 3. Rolling linear regression to generate predictions
    predictions = []
    features = ['price_to_short_ma', 'price_to_long_ma']
    for i in range(train_window, len(df)):
        train_df = df.iloc[i-train_window:i]
        X_train = train_df[features]
        y_train = train_df['future_return']

        model = LinearRegression()
        model.fit(X_train, y_train)

        X_pred = df.iloc[i:i+1][features]
        prediction = model.predict(X_pred)[0]
        predictions.append(prediction)

    df = df.iloc[train_window:].copy()
    df['predicted_return'] = predictions
    return df

def passthrough(data):
    return data

class TimeSeriesPredictiveHoldStrategy(Strategy):
    hold_period = 7 # Approx. 1 week in trading days

    def init(self):
        self.predicted_return = self.I(passthrough, self.data.df['predicted_return'].values)
        self.entry_bar = -1

    def next(self):
        # Rebalancing logic: close position after holding period
        if self.position:
            if (len(self.data) - 1) - self.entry_bar >= self.hold_period:
                self.position.close()

        if self.position:
            return

        prediction = self.predicted_return[-1]

        if prediction > 0:
            self.buy()
            self.entry_bar = len(self.data) - 1
        elif prediction < 0:
            self.sell()
            self.entry_bar = len(self.data) - 1

if __name__ == '__main__':
    import os

    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data.columns = [c.strip().title() for c in data.columns]

        # Resample to daily to match the strategy's intended timeframe
        data = data.resample('D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        data = preprocess_data(data)

        bt = Backtest(data, TimeSeriesPredictiveHoldStrategy, cash=100000, commission=.002, finalize_trades=True)

        stats = bt.run()
        print(stats)

        os.makedirs('results', exist_ok=True)

        def sanitize_stats(stats):
            sanitized = {}
            for key, value in stats.items():
                if isinstance(value, (pd.Series, pd.DataFrame, pd.Timestamp, Strategy)):
                    continue
                elif isinstance(value, pd.Timedelta):
                    sanitized[key] = str(value)
                elif pd.isna(value):
                    sanitized[key] = None
                elif isinstance(value, (np.int64, np.int32)):
                    sanitized[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    sanitized[key] = float(value)
                else:
                    sanitized[key] = value
            return sanitized

        clean_stats = sanitize_stats(stats)

        with open('results/temp_result.json', 'w') as f:
            json.dump(clean_stats, f, indent=2)

        print("Backtest results saved to results/temp_result.json")

        try:
            bt.plot(filename='results/time_series_predictive_hold.html')
        except Exception as e:
            print(f"Could not generate plot: {e}")

    else:
        print(f"Data file not found at {data_path}")
