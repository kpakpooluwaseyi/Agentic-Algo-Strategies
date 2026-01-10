
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
from sklearn.linear_model import LinearRegression
import os
import json

def preprocess_data(filepath, momentum_window=252, value_window=252, holding_period=21):
    """
    Loads and preprocesses the data to create factors for the prediction model.
    """
    try:
        df = pd.read_csv(filepath, index_col='datetime', parse_dates=True)
    except (FileNotFoundError, KeyError, ValueError):
        print(f"Warning: Could not find or parse {filepath}. Generating synthetic data.")
        dates = pd.date_range(start='2017-01-01', periods=2000, freq='D')
        close = 10000 + np.cumsum(np.random.randn(2000) * 100)
        df = pd.DataFrame({
            'Open': close, 'High': close, 'Low': close, 'Close': close, 'Volume': np.random.rand(2000) * 1000
        }, index=dates)

    # Sanitize column names immediately after loading
    df.columns = [col.strip().capitalize() for col in df.columns]

    if pd.infer_freq(df.index) not in ['D', 'B']:
        agg_rules = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        df = df.resample('D').apply(agg_rules).dropna()

    df['momentum'] = df['Close'].pct_change(momentum_window).shift(1)
    df['sma'] = df['Close'].rolling(window=value_window).mean()
    df['value'] = (df['Close'] / df['sma']) - 1
    df['value'] = df['value'].shift(1)
    df['future_return'] = df['Close'].pct_change(holding_period).shift(-holding_period)
    df = df.dropna()
    return df

def get_predictions(df):
    """
    Fits a linear regression model and generates predictions.
    """
    if 'momentum' not in df.columns or 'value' not in df.columns or 'future_return' not in df.columns:
        raise ValueError("DataFrame must contain 'momentum', 'value', and 'future_return' columns.")

    X = df[['momentum', 'value']]
    y = df['future_return']
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    df['predicted_return'] = predictions
    return df

class FactorPredictionStrategy(Strategy):
    """
    A strategy that trades based on a predicted future return signal from a
    linear regression model using time-series factor proxies.
    """
    holding_period = 21

    def init(self):
        self.predicted_return = self.I(lambda x: x, self.data.predicted_return)
        self.exit_time = None

    def next(self):
        if self.position:
            if self.exit_time is not None and self.data.index[-1] >= self.exit_time:
                self.position.close()
                self.exit_time = None
            return

        if not self.position:
            prediction = self.predicted_return[-1]
            if prediction > 0:
                self.buy()
                self.exit_time = self.data.index[-1] + pd.Timedelta(days=self.holding_period)
            elif prediction < 0:
                self.sell()
                self.exit_time = self.data.index[-1] + pd.Timedelta(days=self.holding_period)

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    processed_df = preprocess_data(data_path)
    final_df = get_predictions(processed_df)

    bt = Backtest(final_df, FactorPredictionStrategy, cash=100_000, commission=.002, finalize_trades=True)
    stats = bt.run()

    print("------ Backtest Results ------")
    print(stats)

    os.makedirs('results', exist_ok=True)

    plot_filename = 'results/roe_bm_factor_next_month_prediction.html'
    bt.plot(filename=plot_filename, open_browser=False)
    print(f"\nPlot saved to {plot_filename}")

    stats_dict = dict(stats)
    # Remove non-serializable objects
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    # Sanitize for JSON
    for key, value in stats_dict.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            stats_dict[key] = str(value)
        elif isinstance(value, (np.integer, np.int64)):
            stats_dict[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            stats_dict[key] = float(value) if not np.isnan(value) else None


    results_filename = 'results/temp_result.json'
    with open(results_filename, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print(f"Stats saved to {results_filename}")
