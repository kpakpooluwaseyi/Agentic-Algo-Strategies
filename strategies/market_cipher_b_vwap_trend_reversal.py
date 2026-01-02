import pandas as pd
import pandas_ta as ta
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
import os
import json

def ATR(high, low, close, length=14):
    """Calculate ATR using pandas_ta."""
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    return ta.atr(high=high, low=low, close=close, length=length).values

def market_cipher_b(high, low, close, volume, vwap_length=50, trigger_length=20, rsi_length=14, mfi_length=14):
    """Custom indicator function to approximate Market Cipher B components."""
    high, low, close, volume = pd.Series(high), pd.Series(low), pd.Series(close), pd.Series(volume)
    vwap = ta.vwap(high=high, low=low, close=close, volume=volume, length=vwap_length)
    if vwap is None:
        nan_series = pd.Series([float('nan')] * len(close))
        return nan_series.values, nan_series.values, nan_series.values, nan_series.values
    trigger_line = ta.ema(vwap, length=trigger_length)
    rsi = ta.rsi(close=close, length=rsi_length)
    mfi = ta.mfi(high=high, low=low, close=close, volume=volume, length=mfi_length)
    return vwap.values, trigger_line.values, rsi.values, mfi.values

class MarketCipherBVwapTrendReversal(Strategy):
    # Parameters
    vwap_length = 50
    trigger_length = 20
    rsi_length = 14
    mfi_length = 14
    atr_period = 14
    atr_multiplier = 2.5
    risk_reward_ratio = 2.0

    def init(self):
        self.vwap, self.trigger_line, self.rsi, self.mfi = self.I(market_cipher_b, self.data.High, self.data.Low, self.data.Close, self.data.Volume, vwap_length=self.vwap_length, trigger_length=self.trigger_length, rsi_length=self.rsi_length, mfi_length=self.mfi_length)
        self.atr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        price = self.data.Close[-1]

        # Define trend state based on VWAP vs. Trigger line
        is_uptrend = self.vwap[-1] > self.trigger_line[-1]
        is_downtrend = self.vwap[-1] < self.trigger_line[-1]

        # --- Entry Logic: Bounce/Rejection with momentum confirmation ---
        if not self.position:
            # Long entry on bounce during an uptrend
            if (is_uptrend and
                self.data.Low[-1] <= self.vwap[-1] and
                price > self.vwap[-1] and
                self.rsi[-1] > self.rsi[-2] and # RSI curving up
                self.mfi[-1] > self.mfi[-2]):  # MFI curving up
                sl = price - self.atr[-1] * self.atr_multiplier
                tp = price + (price - sl) * self.risk_reward_ratio
                if tp > price and sl < price: # Basic validation
                    self.buy(sl=sl, tp=tp)

            # Short entry on rejection during a downtrend
            elif (is_downtrend and
                  self.data.High[-1] >= self.vwap[-1] and
                  price < self.vwap[-1] and
                  self.rsi[-1] < self.rsi[-2] and # RSI curving down
                  self.mfi[-1] < self.mfi[-2]):  # MFI curving down
                sl = price + self.atr[-1] * self.atr_multiplier
                tp = price - (sl - price) * self.risk_reward_ratio
                if tp < price and sl > price: # Basic validation
                    self.sell(sl=sl, tp=tp)

        # --- Exit conditions on major trend reversal ---
        if self.position.is_long and crossover(self.trigger_line, self.vwap):
            self.position.close()
        if self.position.is_short and crossover(self.vwap, self.trigger_line):
            self.position.close()

if __name__ == '__main__':
    data_path, strategy_name, output_dir = 'data/BTC-USD-15m.csv', MarketCipherBVwapTrendReversal.__name__, 'results'
    if not os.path.exists(data_path): raise FileNotFoundError(f"Data file not found: {data_path}")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True, skipinitialspace=True)
    data.sort_index(inplace=True)
    data.columns = [c.strip().title() for c in data.columns]
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    bt = Backtest(data, MarketCipherBVwapTrendReversal, cash=100_000, commission=.002)
    stats = bt.run()
    print(stats)
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'{strategy_name}.html')
    try: bt.plot(filename=plot_filename, open_browser=False)
    except Exception as e: print(f"Error plotting: {e}")
    stats_dict = dict(stats)
    for key, value in list(stats_dict.items()):
        if isinstance(value, (pd.DataFrame, pd.Series)): stats_dict.pop(key)
        elif isinstance(value, pd.Timestamp): stats_dict[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta): stats_dict[key] = str(value)
        elif pd.isna(value): stats_dict[key] = None
        elif not isinstance(value, (int, float, str, bool, list, dict, type(None))): stats_dict.pop(key, None)
    json_filename = os.path.join(output_dir, 'temp_result.json')
    with open(json_filename, 'w') as f: json.dump(stats_dict, f, indent=4)
    print(f"\nBacktest results saved to {output_dir}/\n- Stats: {json_filename}\n- Plot: {plot_filename}")
