
import numpy as np
import pandas as pd
import pandas_ta as ta
from src.strategies.base import MoonDevStrategy
from hmmlearn.hmm import CategoricalHMM
import warnings

# Suppress ConvergenceWarning from hmmlearn
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def preprocess_data(df, params):
    """
    Preprocesses the data to generate HMM-based trading signals and indicators.
    """
    # Resample to daily timeframe for HMM model
    daily_df = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # Calculate daily returns and discretize them (0 for down/same, 1 for up)
    daily_df['Return'] = daily_df['Close'].pct_change()
    daily_df['Observation'] = (daily_df['Return'] > 0).astype(int)
    daily_df.dropna(inplace=True)

    observations = daily_df['Observation'].values.reshape(-1, 1)

    # Train HMM once on the initial training window for performance
    training_window = params['training_window']
    train_set = observations[:training_window]

    if len(np.unique(train_set)) < 2:
        # If the initial training set is not diverse, we cannot train the model.
        # Return the df with indicators but no signals.
        daily_df['Signal'] = 0
    else:
        model = CategoricalHMM(n_components=params['n_components'],
                               n_iter=params['n_iter'],
                               tol=params['tol'],
                               random_state=42)
        model.fit(train_set)

        # Use the single trained model to predict for the entire dataset
        # This is a simplification for performance; it assumes static parameters.
        state_sequence = model.predict(observations)

        # Simple signal: if the model predicts state 0, go short, if state 1, go long.
        # This is a common interpretation where states correspond to regimes (e.g., bear/bull).
        signals = np.where(state_sequence == 1, 1, -1)
        daily_df['Signal'] = pd.Series(signals, index=daily_df.index)



    # Merge the daily signal back into the original dataframe
    df['Date'] = df.index.date
    daily_df['Date'] = daily_df.index.date
    merged_df = pd.merge(df, daily_df[['Date', 'Signal']], on='Date', how='left')
    merged_df.set_index(df.index, inplace=True)
    merged_df['Signal'].fillna(method='ffill', inplace=True)
    merged_df.drop(columns=['Date'], inplace=True)

    # Add indicators
    merged_df['SMA200'] = ta.sma(merged_df['Close'], length=200)
    merged_df['VolumeSMA20'] = ta.sma(merged_df['Volume'], length=20)
    merged_df['ATR14'] = ta.atr(merged_df['High'], merged_df['Low'], merged_df['Close'], length=14)

    return merged_df


class HiddenMarkovModelSpy(MoonDevStrategy):
    """
    Implements a trading strategy based on a Hidden Markov Model to predict
    the direction of the next day's price movement, with ATR-based risk management.
    """
    # Optimizable HMM parameters
    training_window = 252
    hmm_n_components = 2
    hmm_n_iter = 100
    hmm_tol = 0.01

    def init(self):
        params = {
            'training_window': self.training_window,
            'n_components': self.hmm_n_components,
            'n_iter': self.hmm_n_iter,
            'tol': self.hmm_tol
        }

        # Preprocess data using the HMM parameters
        processed_df = preprocess_data(self.data.df.copy(), params)

        self.signal = self.I(lambda x: x, processed_df['Signal'])
        self.sma200 = self.I(lambda x: x, processed_df['SMA200'])
        self.volume_sma20 = self.I(lambda x: x, processed_df['VolumeSMA20'])
        self.atr14 = self.I(lambda x: x, processed_df['ATR14'])

    def next(self):
        if self.position:
            return

        current_signal = self.signal[-1]
        is_volume_confirmed = self.data.Volume[-1] > self.volume_sma20[-1]
        current_price = self.data.Close[-1]
        atr_value = self.atr14[-1]

        if pd.isna(atr_value) or pd.isna(self.sma200[-1]):
            return

        sl_distance = 2 * atr_value
        tp_distance = 3 * atr_value

        if (current_signal == 1 and
            current_price > self.sma200[-1] and
            is_volume_confirmed):
            sl = current_price - sl_distance
            tp = current_price + tp_distance
            self.buy(sl=sl, tp=tp)

        elif (current_signal == -1 and
              current_price < self.sma200[-1] and
              is_volume_confirmed):
            sl = current_price + sl_distance
            tp = current_price - tp_distance
            self.sell(sl=sl, tp=tp)
