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

class VuManchuCipherBADX(MoonDevStrategy):
    # Optimizable parameters
    risk_pct = 1.0 
    ema_period = 50
    adx_period = 14
    adx_threshold = 20
    atr_period = 14
    atr_stop_mult = 1.5
    atr_target_mult = 2.5
    
    def init(self):
        # Data Series
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        
        # 1. EMA Trend
        self.ema = self.I(ta.ema, close, length=self.ema_period)
        
        # 2. ADX
        # pandas_ta.adx returns ADX_14, DMP_14, DMN_14 usually
        adx_df = ta.adx(high, low, close, length=self.adx_period)
        if adx_df is not None and not adx_df.empty:
            adx_line = adx_df.iloc[:, 0] # Assume ADX is first column
            self.adx = self.I(lambda: adx_line.to_numpy())
        
        # 3. ATR
        self.atr = self.I(ta.atr, high, low, close, length=self.atr_period)
        
        # 4. WaveTrend
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
        
        # 5. MFI
        mfi_series = ta.mfi(high, low, close, volume, length=60)
        self.mfi = self.I(lambda: mfi_series.to_numpy())
        
    def next(self):
        if len(self.data) < max(100, self.ema_period, self.adx_period): return
            
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        # Filters
        trend_bullish = price > self.ema[-1]
        trend_bearish = price < self.ema[-1]
        
        strong_trend = self.adx[-1] > self.adx_threshold
        
        # Signals
        bull_cross = crossover(self.wt1, self.wt2)
        bear_cross = crossover(self.wt2, self.wt1)
        
        # Overbought/Oversold (Relaxed slightly from -50/50 to -45/45)
        # Thesis: Pullback is mandatory!
        wt_pullback_long = self.wt2[-1] < -45
        wt_pullback_short = self.wt2[-1] > 45
        
        # Money Flow
        green_money = self.mfi[-1] > 30 
        red_money = self.mfi[-1] < 70
        
        # Entry Logic
        if trend_bullish and strong_trend and bull_cross and wt_pullback_long and green_money:
            sl_price = price - (self.atr_stop_mult * atr)
            tp_price = price + (self.atr_target_mult * atr)
            self.buy(size=0.99, sl=sl_price, tp=tp_price)
            
        elif trend_bearish and strong_trend and bear_cross and wt_pullback_short and red_money:
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
        
        bt = Backtest(data, VuManchuCipherBADX, cash=1_000_000, commission=.002)
        stats = bt.run()
        print("\n--- BACKTEST RESULTS (CYCLE 5) ---")
        print(stats)
        print(f"DEBUG_METRICS: Sharpe={stats['Sharpe Ratio']:.2f}, Return={stats['Return [%]']:.2f}, Trades={stats['# Trades']}")
    else:
        print(f"Data file not found: {data_path}")
