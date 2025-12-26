import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from backtesting import Strategy, Backtest
import os
import json

def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    """
    Perform a forward-backward stepwise selection procedure
    to select the best predictors for a linear regression model.
    """
    included = list(initial_list)
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add  {best_feature} with p-value {best_pval:.6f}')

        # Backward step
        if len(included) > 0:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                if verbose:
                    print(f'Drop {worst_feature} with p-value {worst_pval:.6f}')

        if not changed:
            break
    return included

def prepare_data(data: pd.DataFrame, hold_period: int) -> pd.DataFrame:
    """
    Prepares the data by adding predictor and target columns for the regression model.
    """
    periods = {'ret1': 96, 'ret2': 192, 'ret5': 480, 'ret20': 1920}
    for name, period in periods.items():
        data[name] = data['Close'].pct_change(periods=period)
    data['retFut1'] = data['Close'].pct_change(periods=hold_period).shift(-hold_period)
    data.dropna(inplace=True)
    return data

class SpyStepwiseRegressionReturns(Strategy):
    """
    This strategy is an implementation of a stepwise regression model for mean-reversion,
    originally designed for SPY on a daily timeframe.

    NOTE: This implementation is tested on BTC-USD 15-minute data, which has different
    market dynamics. The "daily" periods have been translated to their 15-minute equivalents
    (e.g., 1 day = 96 bars). The core logic remains the same, but performance may differ.
    """
    hold_period = 96
    retrain_period = 96
    train_window = 2000 # Reduced window to speed up
    entry_threshold = 0.0001
    threshold_in = 0.01
    threshold_out = 0.05

    def init(self):
        self.model = None
        self.last_train_bar = -self.retrain_period
        self.selected_features = []

    def next(self):
        current_bar_index = len(self.data) - 1

        if self.position:
            if (current_bar_index - self.trades[0].entry_bar) >= self.hold_period:
                self.position.close()
            return

        if (current_bar_index - self.last_train_bar) >= self.retrain_period:
            self.last_train_bar = current_bar_index

            start_idx = max(0, current_bar_index - self.train_window)
            end_idx = current_bar_index

            if (end_idx - start_idx) < 500:
                self.model = None
                return

            train_df = self.data.df.iloc[start_idx:end_idx]
            features = ['ret1', 'ret2', 'ret5', 'ret20']
            target = 'retFut1'

            X_train = train_df[features]
            y_train = train_df[target]

            if X_train.empty or y_train.empty:
                self.model = None
                return

            self.selected_features = stepwise_selection(X_train, y_train, threshold_in=self.threshold_in, threshold_out=self.threshold_out, verbose=False)
            if not self.selected_features:
                self.model = None
                return

            self.model = LinearRegression()
            self.model.fit(X_train[self.selected_features], y_train)

        if not self.position and self.model is not None and self.selected_features:
            feature_values = [getattr(self.data, feature)[-1] for feature in self.selected_features]
            latest_features = np.array(feature_values).reshape(1, -1)

            predicted_return = self.model.predict(latest_features)[0]

            if predicted_return > self.entry_threshold:
                self.buy(size=0.1)
            elif predicted_return < -self.entry_threshold:
                self.sell(size=0.1)

def sanitize_stats(stats):
    sanitized = {}
    for key, value in stats.items():
        if key.startswith('_') or isinstance(value, pd.DataFrame):
            continue
        if isinstance(value, (np.int64, int)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.float64, float)):
            sanitized[key] = float(value)
        elif isinstance(value, pd.Timestamp):
            sanitized[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            sanitized[key] = str(value)
        else:
            sanitized[key] = value
    return sanitized

if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data.columns = [c.strip().capitalize() for c in data.columns]

    data = prepare_data(data, hold_period=96)
    # Using a much smaller slice for faster optimization
    data_slice = data.iloc[-3000:]

    if data_slice.empty:
        print("Data slice is empty after preparation. Exiting.")
    else:
        bt = Backtest(data_slice, SpyStepwiseRegressionReturns, cash=100_000, commission=.002, finalize_trades=True)

        print("Optimizing strategy with a reduced set...")
        stats = bt.optimize(
            entry_threshold=[0.0, 0.0005],
            threshold_in=[0.05, 0.1],
            threshold_out=[0.1, 0.15],
            maximize='Equity Final [$]',
            max_tries=100 # Add max_tries to avoid infinite loop
        )
        print("Best stats:")
        print(stats)

        os.makedirs('results', exist_ok=True)
        results_path = 'results/temp_result.json'
        final_stats = sanitize_stats(stats.to_dict())
        with open(results_path, 'w') as f:
            json.dump(final_stats, f, indent=4)
        print(f"Stats saved to {results_path}")

        plot_path = 'results/spy_stepwise_regression_returns.html'
        try:
            bt.plot(filename=plot_path)
            print(f"Plot saved to {plot_path}")
        except Exception as e:
            print(f"Could not generate plot: {e}")
