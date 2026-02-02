"""
Vumanchu Trend Continuation Strategy
=====================================
A trend-following strategy that uses a combination of EMA ribbons for trend direction
and the VuManChu Cipher B indicator for entry confirmation and momentum.

Entry Rules:
- Long:
    - Higher timeframe trend is bullish (Price > 4H 200 EMA).
    - 15m EMAs are stacked for an uptrend (21 > 50 > 200).
    - Volume is above its moving average.
    - VuManChu Money Flow (rsimfi) is rising.
    - VuManChu gives a buy signal (green dot).
- Short:
    - Higher timeframe trend is bearish (Price < 4H 200 EMA).
    - 15m EMAs are stacked for a downtrend (21 < 50 < 200).
    - Volume is above its moving average.
    - VuManChu Money Flow (rsimfi) is falling.
    - VuManChu gives a sell signal (red dot).

Exit Rules:
- Stop Loss: 2 * ATR
- Take Profit: 3 * ATR
- Early Exit: A counter-signal from the VuManChu indicator.
"""
import os
import sys
import json
import pandas as pd
import pandas_ta as ta

from backtesting import Backtest, Strategy

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b


def preprocess_data(df: pd.DataFrame, atr_period=14, volume_ma_period=20) -> pd.DataFrame:
    """
    Applies all necessary indicators to the DataFrame for the strategy.
    """
    df.columns = [c.capitalize() for c in df.columns]

    # -- VuManChu Cipher B --
    df = cipher_b(df)
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)

    # -- Primary Timeframe EMAs --
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)

    # -- Higher Timeframe (HTF) Trend Filter --
    # Resample to 4H, calculate EMA, and then map back to the original timeframe
    df_4h = df.resample('4H').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_4h['EMA_200_4H'] = ta.ema(df_4h.Close, length=200)

    # Map the 4H EMA to the original DataFrame's index
    # Normalize index to the start of the 4H period
    df['EMA_200_4H'] = df.index.floor('4H').map(df_4h['EMA_200_4H'])
    df['EMA_200_4H'].ffill(inplace=True)

    # -- ATR for Risk Management --
    df.ta.atr(length=atr_period, append=True)

    # -- Volume Confirmation --
    df['Volume_MA'] = df['Volume'].rolling(window=volume_ma_period).mean()

    df.dropna(inplace=True)

    return df


class VumanchuTrendContinuation(Strategy):
    # -- Strategy Parameters --
    atr_multiplier_sl = 2.0
    atr_multiplier_tp = 3.0

    def init(self):
        # -- Pre-calculated Indicators --
        self.ema_21 = self.I(lambda: self.data.df['EMA_21'], name='EMA_21')
        self.ema_50 = self.I(lambda: self.data.df['EMA_50'], name='EMA_50')
        self.ema_200 = self.I(lambda: self.data.df['EMA_200'], name='EMA_200')
        self.ema_200_4h = self.I(lambda: self.data.df['EMA_200_4H'], name='EMA_200_4H')
        self.atr = self.I(lambda: self.data.df[self.data.df.columns.str.startswith('ATRr_')].iloc[:, 0], name='ATR')
        self.volume_ma = self.I(lambda: self.data.df['Volume_MA'], name='Volume_MA')

        # -- VuManChu Indicators --
        self.buy_signal = self.I(lambda: self.data.df['buy_signal'], name='buy_signal')
        self.sell_signal = self.I(lambda: self.data.df['sell_signal'], name='sell_signal')
        self.rsimfi = self.I(lambda: self.data.df['rsimfi'], name='rsimfi')

    def next(self):
        price = self.data.Close[-1]
        atr_val = self.atr[-1]

        # -- Exit Logic --
        if self.position:
            if self.position.is_long and self.sell_signal[-1] == 1:
                self.position.close()
            elif self.position.is_short and self.buy_signal[-1] == 1:
                self.position.close()

        # -- Entry Logic --
        if not self.position:
            # Long Conditions
            is_htf_uptrend = price > self.ema_200_4h[-1]
            is_ema_stacked_up = self.ema_21[-1] > self.ema_50[-1] > self.ema_200[-1]
            is_volume_confirmed = self.data.Volume[-1] > self.volume_ma[-1]
            is_money_flow_up = self.rsimfi[-1] > self.rsimfi[-2]

            if is_htf_uptrend and is_ema_stacked_up and is_volume_confirmed and is_money_flow_up and self.buy_signal[-1] == 1:
                sl = price - self.atr_multiplier_sl * atr_val
                tp = price + self.atr_multiplier_tp * atr_val
                self.buy(sl=sl, tp=tp)

            # Short Conditions
            is_htf_downtrend = price < self.ema_200_4h[-1]
            is_ema_stacked_down = self.ema_21[-1] < self.ema_50[-1] < self.ema_200[-1]
            is_money_flow_down = self.rsimfi[-1] < self.rsimfi[-2]

            if is_htf_downtrend and is_ema_stacked_down and is_volume_confirmed and is_money_flow_down and self.sell_signal[-1] == 1:
                sl = price + self.atr_multiplier_sl * atr_val
                tp = price - self.atr_multiplier_tp * atr_val
                self.sell(sl=sl, tp=tp)


if __name__ == '__main__':
    # -- Backtest Configuration --
    data_path = 'data/BTC-USD-15m.csv'

    try:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    # -- Preprocessing --
    data = preprocess_data(df)

    # -- Run Backtest --
    bt = Backtest(data, VumanchuTrendContinuation, cash=100_000, commission=.002)
    stats = bt.run()

    print("\n" + "="*80)
    print("Vumanchu Trend Continuation Strategy Results")
    print("="*80)
    print(stats)

    # -- Save Results --
    os.makedirs('results', exist_ok=True)

    # Sanitize stats for JSON output
    stats_serializable = {k: (str(v) if isinstance(v, (pd.Timestamp, pd.Timedelta)) else v)
                          for k, v in stats.items() if k not in ['_strategy', '_equity_curve', '_trades']}

    with open('results/temp_result.json', 'w') as f:
        json.dump(stats_serializable, f, indent=4)

    plot_filename = 'results/vumanchu_trend_continuation.html'
    bt.plot(filename=plot_filename, open_browser=False)

    print(f"\nBacktest results saved to results/temp_result.json")
    print(f"Plot saved to {plot_filename}")
    print("="*80 + "\n")
