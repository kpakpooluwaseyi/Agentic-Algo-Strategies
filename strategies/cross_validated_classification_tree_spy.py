
import pandas as pd
from backtesting import Strategy, Backtest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np

def preprocess_data(df, **params):
    """
    Prepares the data by calculating features and the target variable.
    """
    # Ensure column names are standardized
    df.columns = [col.strip().title() for col in df.columns]

    # Calculate returns for features
    df['ret1'] = df['Close'].pct_change(1)
    df['ret2'] = df['Close'].pct_change(2)
    df['ret5'] = df['Close'].pct_change(5)
    df['ret20'] = df['Close'].pct_change(20)

    # Define the target variable (next day's return direction)
    df['retFut1'] = df['Close'].pct_change(1).shift(-1)
    df['target'] = (df['retFut1'] > 0).astype(int)

    import pandas_ta as ta

    # Calculate indicators for filters and risk management
    df.ta.atr(length=14, append=True)
    df.ta.sma(length=80, append=True) # 80-period SMA to simulate a 20-period SMA on a 4x timeframe
    df.ta.sma(length=20, close='volume', append=True) # 20-period SMA of volume

    # Rename indicator columns for clarity
    df.rename(columns={
        'ATRr_14': 'atr',
        'SMA_80': 'sma_trend',
        'SMA_20_volume': 'volume_sma'
    }, inplace=True)

    # Drop rows with NaN values resulting from the calculations
    df.dropna(inplace=True)

    return df

# The user's request specified inheriting from MoonDevStrategy, but this file does not exist.
# Inheriting from the standard backtesting.Strategy as per other examples in the repository.
class CrossValidatedClassificationTree(Strategy):
    """
    A strategy that uses a classification tree to predict the next day's price movement.
    It trains a model on a rolling window of past data to predict the direction of the next bar.
    """
    min_leaf_size = 100
    train_window = 1000  # Number of bars to use for training
    retrain_interval = 20 # Retrain the model every 20 bars

    def init(self):
        """
        Initialize the strategy.
        """
        # Feature columns to be used for prediction
        self.feature_names = ['ret1', 'ret2', 'ret5', 'ret20']

        # Model setup
        base_estimator = DecisionTreeClassifier(min_samples_leaf=self.min_leaf_size, criterion='gini')
        self.model = BaggingClassifier(estimator=base_estimator, n_estimators=5, random_state=42)

        # Retraining counter
        self.retrain_counter = 0

        # Initialize filter indicators
        self.volume_sma = self.I(lambda: self.data.volume_sma, name="volume_sma")
        self.sma_trend = self.I(lambda: self.data.sma_trend, name="sma_trend")
        self.atr = self.I(lambda: self.data.atr, name="atr")

    def next(self):
        """
        Define the trading logic.
        """
        # --- FILTERS ---
        volume_ok = self.data.Volume[-1] > self.volume_sma[-1]
        long_trend_ok = self.data.Close[-1] > self.sma_trend[-1]
        short_trend_ok = self.data.Close[-1] < self.sma_trend[-1]

        # --- TRADING LOGIC ---

        # Only check for entries if there is no open position
        if not self.position:
            # Ensure we have enough data
            if len(self.data.Close) < self.train_window:
                return

            # Retrain the model periodically
            if self.retrain_counter % self.retrain_interval == 0:
                train_df = self.data.df.iloc[-self.train_window:]
                X_train = train_df[self.feature_names]
                y_train = train_df['target']
                self.model.fit(X_train, y_train)

            self.retrain_counter += 1

            # Make a prediction
            X_pred = self.data.df.iloc[-1:][self.feature_names]
            prediction = self.model.predict(X_pred)[0]

            # Execute trade with risk management
            current_price = self.data.Close[-1]
            current_atr = self.atr[-1]

            if prediction == 1 and volume_ok and long_trend_ok:
                sl = current_price - (2 * current_atr)
                tp = current_price + (3 * current_atr)
                self.buy(sl=sl, tp=tp)
            elif prediction == 0 and volume_ok and short_trend_ok:
                sl = current_price + (2 * current_atr)
                tp = current_price - (3 * current_atr)
                self.sell(sl=sl, tp=tp)


if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Data file not found at {data_path}. Generating synthetic data for testing.")
        # Generate synthetic data
        n_periods = 5000
        dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='15min')
        close_prices = 10000 + np.random.randn(n_periods).cumsum() * 10
        data = pd.DataFrame({
            'datetime': dates,
            'open': close_prices - np.random.uniform(0, 5, size=n_periods),
            'high': close_prices + np.random.uniform(0, 5, size=n_periods),
            'low': close_prices - np.random.uniform(0, 5, size=n_periods),
            'close': close_prices,
            'volume': np.random.uniform(100, 1000, size=n_periods)
        })

    # Set 'datetime' as the index
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

    # Resample to 1D timeframe
    data = data.resample('1D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # Preprocess the data
    data = preprocess_data(data)

    # Ensure the dataframe is not empty after preprocessing
    if data.empty:
        print("Dataframe is empty after preprocessing. Cannot run backtest.")
    else:
        print("Running backtest...")
        # Backtest the strategy
        bt = Backtest(data, CrossValidatedClassificationTree, cash=100_000, commission=.002)
        stats = bt.run()

        print("\n--- Backtest Results ---")
        print(stats)

        # Save results to a JSON file
        stats_json = stats.to_json()
        with open('results/temp_result.json', 'w') as f:
            f.write(stats_json)
        print("\nResults saved to results/temp_result.json")

        # Generate and save the plot
        plot_filename = 'results/cross_validated_classification_tree_spy.html'
        bt.plot(filename=plot_filename, open_browser=False)
        print(f"Plot saved to {plot_filename}")
