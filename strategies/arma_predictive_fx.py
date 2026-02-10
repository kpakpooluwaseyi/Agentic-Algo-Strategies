
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from statsmodels.tsa.arima.model import ARIMA
import warnings
import json
import os

# Suppress warnings from statsmodels
warnings.filterwarnings("ignore")

class ArmaPredictiveFx(Strategy):
    """
    A strategy that uses an ARMA model to predict the next price movement.
    - Fits an ARMA(p, q) model on a rolling window of past returns.
    - Buys if the predicted price is higher than the current price.
    - Sells if the predicted price is lower than the current price.
    - Closes the position if the prediction reverses.
    """
    # Optimizable parameters for the ARMA model
    p = 2  # AR order
    q = 5  # MA order
    model_fit_period = 100  # Number of bars to use for fitting the model

    def init(self):
        # The model is fitted on each step, so no indicators are needed here.
        pass

    def next(self):
        # Wait for enough data to be available to fit the model
        if len(self.data.Close) < self.model_fit_period:
            return

        # Get the historical data for model fitting
        history = self.data.Close[-self.model_fit_period:]
        current_price = self.data.Close[-1]

        try:
            # Fit an ARMA(p,q) model, which is an ARIMA(p,0,q) model
            model = ARIMA(history, order=(self.p, 0, self.q))
            model_fit = model.fit()

            # Forecast 1 step ahead
            prediction = model_fit.forecast(steps=1)[0]

            # --- Trading Logic ---

            # If a position is open, check if the prediction has reversed
            if self.position.is_long and prediction < current_price:
                self.position.close()
            elif self.position.is_short and prediction > current_price:
                self.position.close()

            # If no position is open, check for a new trading signal
            if not self.position:
                if prediction > current_price:
                    self.buy()
                elif prediction < current_price:
                    self.sell()

        except Exception as e:
            # The model may fail to converge, especially with noisy data.
            # In such cases, we skip this bar and do nothing.
            # print(f"Skipping bar {len(self.data.Close)} due to model fitting error: {e}")
            pass

def generate_synthetic_data():
    """Generates synthetic data for testing the strategy."""
    print("Generating synthetic data...")
    n_points = 2000
    index = pd.to_datetime(pd.date_range('2023-01-01', periods=n_points, freq='15min'))
    price = 100 + pd.Series(np.random.randn(n_points).cumsum() * 0.1)
    data = pd.DataFrame({
        'Open': price, 'High': price * 1.005, 'Low': price * 0.995,
        'Close': price, 'Volume': np.random.randint(100, 1000, n_points)
    }, index=index)
    return data

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'

    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        try:
            data = pd.read_csv(
                data_path, index_col='datetime', parse_dates=True, header=0,
                names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unnamed'],
                usecols=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            )
        except Exception as e:
            print(f"Error loading CSV, falling back to synthetic data: {e}")
            data = generate_synthetic_data()
    else:
        print(f"Data file not found at '{data_path}'.")
        data = generate_synthetic_data()

    # Use a smaller slice of data to speed up the backtest, as ARMA is slow
    data = data.iloc[-1000:]

    bt = Backtest(data, ArmaPredictiveFx, cash=100_000, commission=.002)

    print("Running backtest with default parameters...")
    stats = bt.run()

    os.makedirs('results', exist_ok=True)

    def sanitize_stats(stats):
        sanitized = {}
        for key, value in stats.items():
            if isinstance(value, (pd.Series, pd.DataFrame)): continue
            if isinstance(value, (np.floating, np.integer)):
                sanitized[key] = float(value) if np.isfinite(value) else None
            elif isinstance(value, int): sanitized[key] = int(value)
            elif isinstance(value, pd.Timestamp): sanitized[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta): sanitized[key] = str(value)
            elif pd.isna(value): sanitized[key] = None
            elif key.startswith('_'): continue
            else: sanitized[key] = value
        return sanitized

    final_stats = sanitize_stats(stats)

    with open('results/temp_result.json', 'w') as f:
        json.dump(final_stats, f, indent=2)

    print("Backtest results saved to results/temp_result.json")
    print(stats)

    try:
        plot_filename = 'results/arma_predictive_fx.html'
        bt.plot(filename=plot_filename)
        print(f"Backtest plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
