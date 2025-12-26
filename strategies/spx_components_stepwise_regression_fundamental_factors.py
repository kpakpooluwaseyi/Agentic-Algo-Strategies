# SPX Components Stepwise Regression Fundamental Factors Strategy
# (Adapted for single-instrument time-series using a relative strength factor model)

import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.linear_model import LinearRegression
from backtesting import Backtest, Strategy
import json
import os

def create_time_series_factors(df, holding_period_bars, sma_period=50):
    """
    Calculates technical indicators as factor proxies and the future return target.
    """
    df_copy = df.copy()

    # --- Factor Creation ---
    df_copy['rel_strength'] = df_copy['Close'] / ta.sma(df_copy['Close'], length=sma_period)
    df_copy['rsi'] = ta.rsi(df_copy['Close'], length=14)
    df_copy['atr_pct'] = ta.atr(df_copy['High'], df_copy['Low'], df_copy['Close'], length=14) / df_copy['Close']

    # --- Target Variable ---
    df_copy['target'] = df_copy['Close'].pct_change(periods=holding_period_bars).shift(-holding_period_bars)

    # --- Clean Data ---
    return df_copy.dropna()

def passthrough(series, *args, **kwargs):
    return series

class TimeSeriesFactorStrategy(Strategy):
    holding_period = 63
    retrain_every = 63
    lookback_window = 100

    def init(self):
        self.model = LinearRegression()
        self.factor_columns = ['rel_strength', 'rsi', 'atr_pct']
        self.factors = {col: self.I(passthrough, self.data.df[col]) for col in self.factor_columns}
        self.trade_entry_bar = -1
        self.retrain_counter = 0

    def next(self):
        current_bar = len(self.data.Close) - 1

        if self.position:
            if current_bar >= self.trade_entry_bar + self.holding_period:
                self.position.close()
            return

        if current_bar < self.lookback_window:
            return

        if self.retrain_counter % self.retrain_every == 0:
            train_data = self.data.df.iloc[current_bar - self.lookback_window : current_bar]
            X_train = train_data[self.factor_columns]
            y_train = train_data['target']
            if len(X_train) > 1:
                self.model.fit(X_train, y_train)

        if hasattr(self.model, 'coef_'):
            current_factors = np.array([self.factors[col][-1] for col in self.factor_columns]).reshape(1, -1)
            prediction = self.model.predict(current_factors)[0]

            if prediction > 0.01 and not self.position:
                self.buy()
                self.trade_entry_bar = current_bar
            elif prediction < -0.01 and not self.position:
                self.sell()
                self.trade_entry_bar = current_bar

        self.retrain_counter += 1

def sanitize_stats(stats):
    stats_dict = stats.to_dict() if hasattr(stats, 'to_dict') else dict(stats)
    stats_dict.pop('_strategy', None); stats_dict.pop('_equity_curve', None); stats_dict.pop('_trades', None)
    sanitized = {}
    for key, value in stats_dict.items():
        if pd.isna(value): sanitized[key] = None
        elif isinstance(value, np.integer): sanitized[key] = int(value)
        elif isinstance(value, np.floating): sanitized[key] = float(value)
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)): sanitized[key] = str(value)
        else: sanitized[key] = value
    return sanitized

if __name__ == '__main__':
    data = pd.read_csv('data/BTC-USD-15m.csv', index_col=0, parse_dates=True)
    data.columns = [c.strip().title() for c in data.columns]
    data = data.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

    holding_period_days = 63
    processed_data = create_time_series_factors(data, holding_period_bars=holding_period_days, sma_period=50)

    if processed_data.empty:
        print("Error: No data left after pre-processing. Exiting.")
        exit()

    bt = Backtest(processed_data, TimeSeriesFactorStrategy, cash=100_000, commission=.002, finalize_trades=True)
    stats = bt.run(holding_period=holding_period_days)
    print(stats)

    os.makedirs('results', exist_ok=True)
    clean_stats = sanitize_stats(stats)
    with open('results/temp_result.json', 'w') as f: json.dump(clean_stats, f, indent=2)
    print("Backtest results saved to results/temp_result.json")

    try:
        plot_filename = 'results/spx_components_stepwise_regression_fundamental_factors.html'
        bt.plot(filename=plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
