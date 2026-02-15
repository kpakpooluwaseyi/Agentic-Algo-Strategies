
from backtesting import Strategy, Backtest, lib
import pandas as pd
import pandas_ta as ta
import sys
import os

# Add parent directory to path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.indicators.vumanchu import cipher_b

def preprocess_data(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Applies all necessary indicators and filters to the raw dataframe.
    """
    # 1. Calculate all 15m-based indicators first
    df = cipher_b(df)
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['volume_ma20'] = ta.sma(df['Volume'], length=20)

    # 2. Create and process the 4H timeframe data
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()

    if len(df_4h) > 200:
        df_4h['ema200'] = ta.ema(df_4h['Close'], length=200)
        df_4h['htf_downtrend'] = df_4h['Close'] < df_4h['ema200']

        # 3. Map the 4H trend back to the 15m index (robust method)
        df['htf_downtrend'] = df.index.floor('4H').map(df_4h['htf_downtrend'])
        df['htf_downtrend'] = df['htf_downtrend'].ffill()
    else:
        df['htf_downtrend'] = False

    # 4. Final cleanup of all NaNs
    df.dropna(inplace=True)

    return df

# Note on Inheritance:
# The auto-generated request specified inheriting from `src.strategies.base.MoonDevStrategy`.
# However, that base class is designed for a different framework (signal generation) and is
# incompatible with the `backtesting.py` library's `init`/`next` structure required by
# all other strategy development guidelines. The reference implementation in
# `strategies/vumanchu_cipher_b.py` also uses `backtesting.Strategy`.
# Therefore, `backtesting.Strategy` is used here to create a functional and compliant backtest.
class MarketCipherBShortContinuation(Strategy):
    """
    A trend-following strategy that enters short positions during
    strong downward momentum phases identified by Market Cipher B.

    Entry Rules:
    - Market Cipher B blue wave (wt1) crosses below the -60 threshold.
    - 4H timeframe is in a downtrend (Close < EMA200).
    - Volume is above its 20-period moving average.

    Exit Rules:
    - Market Cipher B blue wave (wt1) crosses back above the -60 threshold.
    - ATR-based stop loss or take profit is hit.
    """
    # Optimizable Parameters
    atr_sl_multiplier = 2.0
    atr_tp_multiplier = 3.0
    oversold_threshold = -60

    def init(self):
        self.wt1 = self.I(lambda: self.data.wt1, name='wt1')
        self.atr = self.I(lambda: self.data.atr, name='ATR')
        self.htf_downtrend = self.I(lambda: self.data.htf_downtrend, name='htf_downtrend')
        self.volume_ma = self.I(lambda: self.data.volume_ma20, name='volume_ma20')

    def next(self):
        price = self.data.Close[-1]

        if self.position.is_short and self.wt1[-1] > self.oversold_threshold:
            self.position.close()

        if not self.position:
            is_htf_downtrend = self.htf_downtrend[-1] == 1.0
            is_volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1]
            is_wt1_crossunder = lib.crossover(self.oversold_threshold, self.wt1)

            if is_htf_downtrend and is_volume_confirmed and is_wt1_crossunder:
                stop_loss = price + self.atr_sl_multiplier * self.atr[-1]
                take_profit = price - self.atr_tp_multiplier * self.atr[-1]

                # Add a check to ensure TP is valid
                if take_profit > 0 and take_profit < price:
                    self.sell(sl=stop_loss, tp=take_profit)

if __name__ == '__main__':
    try:
        data = pd.read_csv('data/BTC-USD-15m.csv', index_col='datetime', parse_dates=True)
        # FIX: Standardize column names and remove empty "Unnamed" column
        data.columns = [col.strip().title() for col in data.columns]
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    except FileNotFoundError:
        print("Error: data/BTC-USD-15m.csv not found.")
        from backtesting.test import GOOG
        data = GOOG.copy()
        data = data.resample('15min').last().ffill()

    processed_data = preprocess_data(data)

    if not processed_data.empty:
        bt = Backtest(processed_data, MarketCipherBShortContinuation, cash=100_000, commission=.002)

        stats = bt.run()
        print(stats)

        output_filename = 'results/market_cipher_b_large_timeframe_short_continuation.html'
        bt.plot(filename=output_filename, open_browser=False)
        print(f"Plot saved to {output_filename}")

        try:
            # A more robust way to handle the stats object for JSON serialization
            stats_dict = {}
            if stats:
                for key, val in stats.items():
                    if isinstance(val, pd.Timestamp):
                        stats_dict[key] = val.isoformat()
                    elif isinstance(val, pd.Timedelta):
                        stats_dict[key] = str(val)
                    elif not isinstance(val, (pd.DataFrame, pd.Series, type(None))) and pd.notna(val):
                        if hasattr(val, 'item'):
                            stats_dict[key] = val.item()
                        else:
                            stats_dict[key] = val

                with open("results/temp_result.json", 'w') as f:
                    import json
                    json.dump(stats_dict, f, indent=4)
                print("Results saved to results/temp_result.json")
        except Exception as e:
            print(f"Could not save results to JSON: {e}")
    else:
        print("Data is empty after preprocessing. Halting backtest.")
