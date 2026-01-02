
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Helper functions to wrap pandas_ta indicators for backtesting.py
def MFI(high, low, close, volume, length=14):
    """Calculates Money Flow Index."""
    return ta.mfi(pd.Series(high), pd.Series(low), pd.Series(close), pd.Series(volume), length=length).values

def MACD(close, fast=12, slow=26, signal=9):
    """Calculates MACD and returns its components."""
    macd = ta.macd(pd.Series(close), fast=fast, slow=slow, signal=signal)
    return macd.values.T  # Transpose to get each column as a separate array

class MarketCipherBScalping(Strategy):
    """
    This strategy is a proxy for the Market Cipher B scalping strategy,
    using standard indicators as substitutes for the proprietary Cipher B.

    Indicators:
    - Money Flow: Money Flow Index (MFI)
    - Momentum Waves: MACD Histogram
    - Trend: Higher timeframe VWAP (Volume-Weighted Average Price)
    """
    # --- Strategy Parameters ---
    # Indicator settings
    mfi_length = 14
    mfi_ob = 80
    mfi_os = 20
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    # Risk management
    tp_pct = 0.015  # 1.5% take profit
    sl_pct = 0.0075 # 0.75% stop loss

    def init(self):
        """Initialize the strategy and its indicators."""
        # --- Proxy for Money Flow (MFI) ---
        self.mfi = self.I(MFI, self.data.High, self.data.Low,
                          self.data.Close, self.data.Volume,
                          length=self.mfi_length)

        # --- Proxy for Momentum Waves (MACD) ---
        macd_output = self.I(MACD, self.data.Close,
                             fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        self.macd_hist = macd_output[1]  # MACDh is the 2nd column (histogram)

        # --- Higher Timeframe Trend (VWAP on 1H) ---
        # This approach is not ideal for optimization but works for a single run.
        df = self.data.df.copy()
        df.index = pd.to_datetime(df.index)

        htf_df = df.resample('1H').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

        htf_df['vwap'] = ta.vwap(htf_df.High, htf_df.Low, htf_df.Close, htf_df.Volume)

        df['htf_vwap'] = df.index.floor('H').map(htf_df['vwap'])
        df.bfill(inplace=True)
        df.ffill(inplace=True)

        self.htf_vwap = self.I(lambda: df['htf_vwap'].values, name="HTF_VWAP")

    def next(self):
        """Define the trading logic for the next bar."""
        price = self.data.Close[-1]

        # --- Trend Condition ---
        is_bullish_trend = price > self.htf_vwap[-1]
        is_bearish_trend = price < self.htf_vwap[-1]

        # --- Money Flow Signal (Proxy for Cipher B dots) ---
        long_money_flow_signal = crossover(self.mfi, self.mfi_os)
        short_money_flow_signal = crossover(self.mfi_ob, self.mfi)

        # --- Momentum Signal (Proxy for Cipher B waves) ---
        is_green_momentum = self.macd_hist[-1] > 0
        is_red_momentum = self.macd_hist[-1] < 0

        # --- Entry Logic ---
        if not self.position:
            if is_bullish_trend and long_money_flow_signal and is_green_momentum:
                sl = price * (1 - self.sl_pct)
                tp = price * (1 + self.tp_pct)
                self.buy(sl=sl, tp=tp)

            elif is_bearish_trend and short_money_flow_signal and is_red_momentum:
                sl = price * (1 + self.sl_pct)
                tp = price * (1 - self.tp_pct)
                self.sell(sl=sl, tp=tp)

if __name__ == '__main__':
    import json
    import os

    data_path = 'data/BTC-USD-15m.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data without usecols, will sanitize names next
    data = pd.read_csv(data_path)
    data.columns = [c.strip().title() for c in data.columns]

    # Keep only the necessary columns
    data = data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data.set_index('Datetime', inplace=True)

    bt = Backtest(data, MarketCipherBScalping, cash=100000, commission=.002)
    stats = bt.run()

    print(stats)

    os.makedirs('results', exist_ok=True)

    stats_dict = dict(stats)
    stats_dict.pop('_strategy', None)
    stats_dict.pop('_equity_curve', None)
    stats_dict.pop('_trades', None)

    with open('results/temp_result.json', 'w') as f:
        json.dump({k: str(v) for k, v in stats_dict.items()}, f, indent=4)

    try:
        bt.plot(filename='results/market_cipher_b_scalping.html', open_browser=False)
    except Exception as e:
        print(f"Could not generate plot: {e}")
