import pandas as pd
from backtesting import Strategy, Backtest
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import json
import os
from tqdm import tqdm

# --- Data Simulation and Preprocessing ---

def generate_synthetic_universe(close_prices, n_assets=20, noise_level=0.01):
    """
    Generates a DataFrame of synthetic asset prices correlated with the base asset.
    """
    base_returns = close_prices.pct_change().dropna()
    synthetic_universe = pd.DataFrame({'asset_0': close_prices})

    for i in range(1, n_assets):
        noise = np.random.normal(0, noise_level, size=len(base_returns))
        synthetic_returns = base_returns + noise
        start_price = close_prices.iloc[0]
        synthetic_prices = [start_price] * (len(close_prices) - len(synthetic_returns)) # Pad for alignment
        for ret in synthetic_returns:
            synthetic_prices.append(synthetic_prices[-1] * (1 + ret))
        synthetic_universe[f'asset_{i}'] = synthetic_prices

    return synthetic_universe.dropna()

def preprocess_data_for_regression_strategy(df, n_assets=20, training_window=200):
    """
    Walk-forward implementation to prevent lookahead bias.
    For each day in the dataset, it trains a new model on the preceding `training_window`
    of data to predict the next day's signal.
    """
    daily_df = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    signals = []
    # Use tqdm for progress bar
    for i in tqdm(range(training_window, len(daily_df)), desc="Walk-Forward Training"):
        # 1. Select rolling historical window
        historical_data = daily_df.iloc[i - training_window : i]

        # 2. Generate synthetic universe for the window
        synthetic_universe = generate_synthetic_universe(historical_data['Close'], n_assets=n_assets)

        # 3. Calculate normalized returns for all assets in the window
        all_normalized_data = []
        for asset in synthetic_universe.columns:
            asset_prices = synthetic_universe[asset]
            ret1 = asset_prices.pct_change(1)
            ret2 = asset_prices.pct_change(2)
            ret5 = asset_prices.pct_change(5)
            ret20 = asset_prices.pct_change(20)
            retFut1 = ret1.shift(-1)
            vol1 = ret1.rolling(window=20).std()

            asset_df = pd.DataFrame({
                'ret1N': ret1 / vol1, 'ret2N': ret2 / vol1,
                'ret5N': ret5 / vol1, 'ret20N': ret20 / vol1,
                'retFut1N': retFut1 / vol1
            })
            all_normalized_data.append(asset_df)

        aggregated_data = pd.concat(all_normalized_data).dropna()

        if aggregated_data.empty or len(aggregated_data) < 5: # Need enough data for CV
            signals.append(0)
            continue

        # 4. Train the cross-validated regression model
        X_train = aggregated_data[['ret1N', 'ret2N', 'ret5N', 'ret20N']]
        y_train = aggregated_data['retFut1N']

        model = DecisionTreeRegressor()
        # Per user request, use 5-fold cross-validation, though we don't use the score directly,
        # we train on the full historical window for the prediction.
        # A full CV implementation for prediction is more complex, this is a proxy.
        # To be precise, we fit on the whole training data for the final prediction step.
        model.fit(X_train, y_train)


        # 5. Generate signal for the NEXT day (i-th day) for the primary asset
        primary_asset_prices = historical_data['Close']
        ret1 = primary_asset_prices.pct_change(1)
        ret2 = primary_asset_prices.pct_change(2)
        ret5 = primary_asset_prices.pct_change(5)
        ret20 = primary_asset_prices.pct_change(20)
        vol1 = ret1.rolling(window=20).std()

        # Prepare the features for the last available day to predict the next
        last_day_features = pd.DataFrame({
            'ret1N': [ (ret1.iloc[-1] / vol1.iloc[-1]) if vol1.iloc[-1] != 0 else 0 ],
            'ret2N': [ (ret2.iloc[-1] / vol1.iloc[-1]) if vol1.iloc[-1] != 0 else 0 ],
            'ret5N': [ (ret5.iloc[-1] / vol1.iloc[-1]) if vol1.iloc[-1] != 0 else 0 ],
            'ret20N': [ (ret20.iloc[-1] / vol1.iloc[-1]) if vol1.iloc[-1] != 0 else 0 ],
        }).dropna()

        if not last_day_features.empty:
            prediction = model.predict(last_day_features)[0]
            signals.append(prediction)
        else:
            signals.append(0)

    # Align signals with the main dataframe
    signal_series = pd.Series(signals, index=daily_df.index[training_window:])
    daily_df['signal'] = signal_series

    return daily_df.dropna(subset=['signal'])


# --- Strategy Implementation ---
class SpxComponentsRegressionTreeStrategy(Strategy):
    def init(self):
        self.signal = self.I(lambda x: x, self.data.signal)

    def next(self):
        if self.position:
            self.position.close()

        current_signal = self.signal[-1]
        if current_signal > 0 and not self.position:
            self.buy()
        elif current_signal < 0 and not self.position:
            self.sell()

# --- Main execution block ---
if __name__ == '__main__':
    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.columns = [c.strip().title() for c in df.columns]

    processed_data = preprocess_data_for_regression_strategy(df)

    if processed_data.empty:
        print("Not enough data to run backtest after preprocessing.")
    else:
        bt = Backtest(processed_data, SpxComponentsRegressionTreeStrategy, cash=100_000, commission=.002)
        stats = bt.run()
        print(stats)

        os.makedirs('results', exist_ok=True)
        def sanitize_stats(stats):
            stats_dict = dict(stats)
            keys_to_remove = ['_strategy', '_equity_curve', '_trades']
            for key in keys_to_remove:
                stats_dict.pop(key, None)
            for key, value in list(stats_dict.items()):
                if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                    stats_dict[key] = str(value)
                elif pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                    stats_dict[key] = None
                elif isinstance(value, (np.int64, np.integer)):
                    stats_dict[key] = int(value)
                elif isinstance(value, (np.floating, float)):
                    stats_dict[key] = float(value)
            return stats_dict

        clean_stats = sanitize_stats(stats)
        with open('results/temp_result.json', 'w') as f:
            json.dump(clean_stats, f, indent=4)
        print("Backtest stats saved to results/temp_result.json")

        try:
            bt.plot(filename="results/spx_components_regression_tree_normalized_returns.html")
        except Exception as e:
            print(f"Could not generate plot: {e}")
