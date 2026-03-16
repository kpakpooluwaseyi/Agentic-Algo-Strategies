import sys
import os
from pathlib import Path
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest
from backtesting.lib import crossover

# Add root to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.strategies.base import MoonDevStrategy

class VuManchuCipherBSuperTrend(MoonDevStrategy):
    # Optimizable parameters
    risk_pct = 1.0 
    st_period = 10
    st_multiplier = 3.0
    atr_period = 14
    atr_stop_mult = 2.0
    atr_target_mult = 4.0 # Extended Target for trend riding
    
    def init(self):
        # Data Series
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        
        # 1. SuperTrend
        # pandas_ta.supertrend returns multiple columns (SUPERT_, SUPERTk_, SUPERTd_, SUPERTl_)
        # Usually returns 'SUPERT_10_3.0' as the trend line.
        st_df = ta.supertrend(high, low, close, length=self.st_period, multiplier=self.st_multiplier)
        # We need the trend line column. It's usually the first one or named after params.
        # Let's assume it's the first column for the line, and second for direction?
        # Actually, let's just take the first column which is the SuperTrend line.
        if st_df is not None and not st_df.empty:
            st_line = st_df.iloc[:, 0]
            # Direction is usually the second column (1 for bull, -1 for bear)
            # st_dir = st_df.iloc[:, 1]
            self.st = self.I(lambda: st_line.to_numpy())
        
        # 2. ATR
        self.atr = self.I(ta.atr, high, low, close, length=self.atr_period)
        
        # 3. WaveTrend (Cipher B)
        ap = ta.hlc3(high, low, close)
        n1 = 10
        n2 = 21
        esa = ta.ema(ap, length=n1)
        d = ta.ema((ap - esa).abs(), length=n1)
        ci = (ap - esa) / (0.015 * d)
        tci = ta.ema(ci, length=n2)
        wt1_series = tci
        wt2_series = ta.sma(wt1_series, length=4)
        
        self.wt1 = self.I(lambda: wt1_series.to_numpy())
        self.wt2 = self.I(lambda: wt2_series.to_numpy())
        
        # 4. MFI
        mfi_series = ta.mfi(high, low, close, volume, length=60)
        self.mfi = self.I(lambda: mfi_series.to_numpy())
        
    def next(self):
        if len(self.data) < 100: return
            
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        # Filters
        # SuperTrend Filter: Price > ST Line is Bullish? 
        # Usually yes. If Price closed above ST line, it's Uptrend.
        # Or check direction column if I saved it.
        # Simple check: Close > ST usually implies Bullish in standard calculation.
        trend_bullish = price > self.st[-1]
        trend_bearish = price < self.st[-1]
        
        # Signals
        bull_cross = crossover(self.wt1, self.wt2)
        bear_cross = crossover(self.wt2, self.wt1)
        
        # Relaxed Oversold/Overbought (Avoid extremas only)
        # Logic: Don't buy if ALREADY oversaturated (e.g. > 60)
        # Thesis: wt2 < 60 for Long. wt2 > -60 for Short.
        # This prevents buying the absolute top blow-off.
        
        valid_long_zone = self.wt2[-1] < 60
        valid_short_zone = self.wt2[-1] > -60
        
        # Money Flow (Aggressive)
        green_money = self.mfi[-1] > 30 
        red_money = self.mfi[-1] < 70
        
        # Entry Logic
        if trend_bullish and bull_cross and valid_long_zone and green_money:
            sl_price = price - (self.atr_stop_mult * atr)
            tp_price = price + (self.atr_target_mult * atr)
            self.buy(size=0.99, sl=sl_price, tp=tp_price)
            
        elif trend_bearish and bear_cross and valid_short_zone and red_money:
            sl_price = price + (self.atr_stop_mult * atr)
            tp_price = price - (self.atr_target_mult * atr)
            self.sell(size=0.99, sl=sl_price, tp=tp_price)

if __name__ == "__main__":
    # Load data
    data_path = "data/BTC-USD-15m.csv"
    if os.path.exists(data_path):
        data = pd.read_csv(data_path, parse_dates=True, index_col="datetime")
        data.columns = data.columns.str.strip()
        data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        
        print(f"Data Loaded: {len(data)} rows")
        
        bt = Backtest(data, VuManchuCipherBSuperTrend, cash=1_000_000, commission=.002)
        stats = bt.run()
        print("\n--- BACKTEST RESULTS (CYCLE 4) ---")
        print(stats)
        print(f"DEBUG_METRICS: Sharpe={stats['Sharpe Ratio']:.2f}, Return={stats['Return [%]']:.2f}, Trades={stats['# Trades']}")
    else:
        print(f"Data file not found: {data_path}")
