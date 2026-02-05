"""
PCA Predictive Statistical Factors Strategy
===========================================
This strategy adapts the cross-sectional PCA factor model for a single time series.
It uses PCA on a lookback window of returns to create statistical factors and then
uses a rolling linear regression to predict the next period's return.
"""

import pandas as pd
from backtesting import Strategy, Backtest
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import numpy as np
import json

def preprocess_data(df, lookback_period, n_components, model_window):
    """
    Preprocesses the data to generate trading signals using a rolling PCA
    and linear regression model.

    NOTE: This is a time-series adaptation of a cross-sectional strategy.
    TODO: The rolling for-loop is inefficient and should be vectorized for performance.

    Args:
        df (pd.DataFrame): The input OHLCV data.
        lookback_period (int): The number of past returns to use as features.
        n_components (int): The number of principal components to extract.
        model_window (int): The size of the rolling window for model training.

    Returns:
        pd.DataFrame: The dataframe with an added 'signal' column.
    """
    df['returns'] = df['Close'].pct_change()

    # Create features (past returns)
    for i in range(1, lookback_period + 1):
        df[f'return_lag_{i}'] = df['returns'].shift(i)

    # Create the target variable (next period's return)
    df['target'] = df['returns'].shift(-1)

    # Drop rows with NaNs created by lagging
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True) # Reset index after drop

    feature_names = [f'return_lag_{i}' for i in range(1, lookback_period + 1)]
    X = df[feature_names].values
    y = df['target'].values

    predictions = np.full(len(df), np.nan)

    # Rolling prediction
    for i in range(model_window, len(df)):
        X_train_window = X[i - model_window:i]
        y_train_window = y[i - model_window:i]

        # 1. Fit PCA on the window
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_window)

        # 2. Fit Linear Regression on PCA factors
        model = LinearRegression()
        model.fit(X_train_pca, y_train_window)

        # 3. Predict the next return using the current period's factors
        X_current = X[i:i+1] # Features for the current time step
        X_current_pca = pca.transform(X_current)
        pred = model.predict(X_current_pca)
        predictions[i] = pred[0]

    df['signal'] = predictions

    # Set the original datetime index back
    if 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)

    return df


class PCAPredictiveStatisticalFactors(Strategy):
    """
    Strategy class for the PCA Predictive Statistical Factors model.
    """
    def init(self):
        """
        Initialize the strategy.
        """
        self.signal = self.I(lambda: self.data.signal)

    def next(self):
        """
        Define the trading logic for the next tick.
        """
        # Close any open position to enforce a 1-day hold period.
        if self.position:
            self.position.close()

        # Go long if the signal is positive, short if negative.
        if self.signal[-1] > 0:
            self.buy()
        elif self.signal[-1] < 0:
            self.sell()


if __name__ == '__main__':
    # This block will be filled in later.
    pass
