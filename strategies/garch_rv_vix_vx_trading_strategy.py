
from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import pandas_ta as ta

def preprocess_data(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Adds all indicators to the DataFrame.
    """
    # Proxies for GARCH/VIX
    df['short_term_atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=params.get('short_atr_period', 7))
    df['long_term_atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=params.get('long_atr_period', 28))

    # Multi-Timeframe Trend Filter (4H EMA)
    df_4h = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).copy()
    df_4h['ema_200'] = ta.ema(df_4h['Close'], length=200)
    df_4h['htf_uptrend'] = df_4h['Close'] > df_4h['ema_200']
    df['htf_uptrend'] = df_4h['htf_uptrend'].reindex(df.index, method='ffill')

    # Volume Confirmation
    df['volume_ma'] = ta.sma(df['Volume'], length=params.get('volume_ma_period', 20))

    # ATR for risk management
    df['atr_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    return df


class GarchRvVixVxTradingStrategy(Strategy):
    """
    A proxy for the GARCH-RV-VIX-VX strategy, adapted for BTC-USD data.

    This strategy models the original concept by using ATRs as proxies:
    - Short-term ATR: Proxy for GARCH predicted realized volatility.
    - Long-term ATR: Proxy for VIX implied volatility.

    The core logic is to trade the spread between these two volatility measures,
    while adhering to the project's mandatory development guidelines.
    """

    # Optimizable parameters for volatility proxies
    short_atr_period = 7
    long_atr_period = 28

    # Optimizable parameters for mandatory guidelines
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    volume_ma_period = 20

    def init(self):
        # Pre-calculated indicators from preprocess_data
        self.short_term_atr = self.I(lambda: self.data.df['short_term_atr'], name='short_term_atr')
        self.long_term_atr = self.I(lambda: self.data.df['long_term_atr'], name='long_term_atr')
        self.htf_uptrend = self.I(lambda: self.data.df['htf_uptrend'], name='htf_uptrend')
        self.volume_ma = self.I(lambda: self.data.df['volume_ma'], name='volume_ma')
        self.atr = self.I(lambda: self.data.df['atr_14'], name='atr_14')


    def next(self):
        # Guideline 1: Multi-Timeframe Trend Filter
        is_htf_uptrend = self.htf_uptrend[-1] == 1
        is_htf_downtrend = self.htf_uptrend[-1] == 0

        # Guideline 2: Volume Confirmation
        has_volume_confirmation = self.data.Volume[-1] > self.volume_ma[-1]

        # Core Strategy Logic: Volatility spread crossover
        predicted_vol_gt_implied = crossover(self.short_term_atr, self.long_term_atr)
        implied_vol_gt_predicted = crossover(self.long_term_atr, self.short_term_atr)

        # Entry Conditions
        if not self.position:
            # Long Entry: Predicted volatility > Implied volatility + Guideline checks
            if predicted_vol_gt_implied and is_htf_uptrend and has_volume_confirmation:
                sl = self.data.Close[-1] - (self.atr_sl_multiplier * self.atr[-1])
                tp = self.data.Close[-1] + (self.atr_tp_multiplier * self.atr[-1])
                if sl < self.data.Close[-1] and tp > self.data.Close[-1]:
                    self.buy(sl=sl, tp=tp)

            # Short Entry: Implied volatility > Predicted volatility + Guideline checks
            elif implied_vol_gt_predicted and is_htf_downtrend and has_volume_confirmation:
                sl = self.data.Close[-1] + (self.atr_sl_multiplier * self.atr[-1])
                tp = self.data.Close[-1] - (self.atr_tp_multiplier * self.atr[-1])
                if sl > self.data.Close[-1] and tp < self.data.Close[-1]:
                    self.sell(sl=sl, tp=tp)


if __name__ == '__main__':
    from backtesting import Backtest
    import json
    import os

    # --- Configuration ---
    DATA_PATH = 'data/BTC-USD-15m.csv'
    OUTPUT_DIR = 'results'
    OUTPUT_FILENAME = 'garch_rv_vix_vx_trading_strategy'

    # --- Data Loading and Preprocessing ---
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    # Load data, explicitly using only the first 6 columns to avoid issues with malformed CSVs.
    data = pd.read_csv(DATA_PATH, usecols=range(6))
    # Sanitize column names to handle potential whitespace and ensure correct capitalization.
    data.columns = [c.strip().capitalize() for c in data.columns]
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

    # Ensure data is sorted
    data.sort_index(inplace=True)

    # Preprocess data with default parameters
    params = {
        'short_atr_period': 7,
        'long_atr_period': 28,
        'volume_ma_period': 20
    }
    data = preprocess_data(data, **params)
    data.dropna(inplace=True)


    # --- Backtesting ---
    if not data.empty:
        bt = Backtest(data, GarchRvVixVxTradingStrategy, cash=100_000, commission=.002)

        stats = bt.run()

        # --- Results and Output ---
        print(stats)

        # Ensure results directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save plot
        plot_filepath = os.path.join(OUTPUT_DIR, f'{OUTPUT_FILENAME}.html')
        bt.plot(filename=plot_filepath)

        # Sanitize and save stats to JSON
        stats_dict = dict(stats)
        # Remove non-serializable items
        stats_dict.pop('_strategy', None)
        stats_dict.pop('_equity_curve', None)
        stats_dict.pop('_trades', None)

        for key, value in stats_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                stats_dict[key] = float(value)
            elif isinstance(value, pd.Timestamp):
                stats_dict[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                 stats_dict[key] = str(value)


        json_filepath = os.path.join(OUTPUT_DIR, 'temp_result.json')
        with open(json_filepath, 'w') as f:
            json.dump(stats_dict, f, indent=4)

        print(f"\nBacktest results saved to {json_filepath}")
        print(f"Plot saved to {plot_filepath}")
    else:
        print("Data is empty after preprocessing. Could not run backtest.")
