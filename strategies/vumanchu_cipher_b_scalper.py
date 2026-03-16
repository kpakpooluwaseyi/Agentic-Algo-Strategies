import sys
import os
from pathlib import Path
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest
from backtesting.lib import crossover

# Add root to python path to allow importing src
sys.path.append(str(Path(__file__).parent.parent))

from src.strategies.base import MoonDevStrategy

class VuManchuCipherBScalper(MoonDevStrategy):
    # Optimizable parameters
    risk_pct = 1.0
    ema_period = 200
    take_profit_pct = 0.015
    stop_loss_pct = 0.010
    time_exit_bars = 12
    
    def init(self):
        # Prepare Data Series for pandas_ta
        # We need to construct Series from the wrapper arrays to ensure ta works
        # Note: self.data.Close is an accessor, constructing Series copies the data.
        # Since init runs once, this is efficient enough.
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        
        # 1. EMA Trend Filter
        self.ema = self.I(ta.ema, close, length=self.ema_period)
        
        # 2. WaveTrend (VuManchu Cipher B Logic)
        # wt1, wt2 calculation
        # AP = (H+L+C)/3
        # We can calculate AP directly
        ap = ta.hlc3(high, low, close)
        
        # We define a helper to calculate WT so we can pass it to self.I
        # Actually, standard practice: calculate series, then register result with self.I
        
        # ESA = EMA(AP, 10)
        n1 = 10
        n2 = 21
        
        esa = ta.ema(ap, length=n1)
        d = ta.ema((ap - esa).abs(), length=n1)
        ci = (ap - esa) / (0.015 * d)
        tci = ta.ema(ci, length=n2)
        
        wt1_series = tci
        wt2_series = ta.sma(wt1_series, length=4)
        
        # Register for plotting/access
        self.wt1 = self.I(lambda: wt1_series.to_numpy())
        self.wt2 = self.I(lambda: wt2_series.to_numpy())
        
        # 3. MFI (Money Flow)
        # MFI requires H, L, C, V
        mfi_series = ta.mfi(high, low, close, volume, length=60)
        # Normalized MFI - The thesis mentioned MFI > 0 (Green). 
        # Standard MFI is 0-100. Center is 50.
        # Maybe Thesis meant "Money Flow Oscillator"?
        # Given "Cipher B" context, it often uses a specific Money Flow algo.
        # We will use standard MFI > 50 as bullish proxy per previous thought.
        self.mfi = self.I(lambda: mfi_series.to_numpy())
        
    def next(self):
        # Trading logic per bar
        
        # Filters
        price = self.data.Close[-1]
        trend_bullish = price > self.ema[-1]
        trend_bearish = price < self.ema[-1]
        
        # Signals
        # Bullish Crossover: wt1 crosses above wt2
        bull_cross = crossover(self.wt1, self.wt2)
        # Bearish Crossover: wt1 crosses below wt2
        bear_cross = crossover(self.wt2, self.wt1)
        
        # Oversold/Overbought Conditions (at time of crossover)
        # Using previous value or current? 
        # Crossover implies current is crossed. So check current or previous level.
        # Usually checking if the 'cross' happened in the zone.
        # Using self.wt2[-1] is fine.
        oversold = self.wt2[-1] < -60 # Deep oversold
        overbought = self.wt2[-1] > 60 # Deep overbought
        # Thesis said 50/ -50. Adjusting to 60 for better signal quality as "Scalper".
        # Let's stick to Thesis: -50 / 50.
        oversold = self.wt2[-1] < -50
        overbought = self.wt2[-1] > 50

        # Money Flow
        green_money = self.mfi[-1] > 50
        red_money = self.mfi[-1] < 50
        
        # Entry Logic
        if trend_bullish and bull_cross and oversold and green_money:
            self.buy(size=0.99, sl=price * (1 - self.stop_loss_pct), tp=price * (1 + self.take_profit_pct))
            
        elif trend_bearish and bear_cross and overbought and red_money:
             self.sell(size=0.99, sl=price * (1 + self.stop_loss_pct), tp=price * (1 - self.take_profit_pct))
             
        # Time Exit
        for trade in self.trades:
            # Simple bar count check
            if (len(self.data) - trade.entry_bar) > self.time_exit_bars:
                trade.close()

if __name__ == "__main__":
    # Load data
    data_path = "data/BTC-USD-15m.csv"
    if os.path.exists(data_path):
        data = pd.read_csv(data_path, parse_dates=True, index_col="datetime")
        
        # CLEANUP DATA
        # 1. Strip whitespace from columns
        data.columns = data.columns.str.strip()
        
        # 2. Rename to Title Case (Open, High...)
        data = data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # 3. Filter to strict columns + Volume
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # 4. Drop NaNs
        data = data.dropna()
        
        print(f"Data Loaded: {len(data)} rows")
        print(data.head())
        
        # Run backtest with high cash to avoid fractional issues
        bt = Backtest(data, VuManchuCipherBScalper, cash=1_000_000, commission=.002)
        stats = bt.run()
        print("\n--- BACKTEST RESULTS ---")
        print(stats)
        # Extract Key Metrics for Auditor
        print(f"DEBUG_METRICS: Sharpe={stats['Sharpe Ratio']:.2f}, Return={stats['Return [%]']:.2f}, Trades={stats['# Trades']}")
        
    else:
        print(f"Data file not found: {data_path}")
