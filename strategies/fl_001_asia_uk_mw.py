import pandas as pd
import numpy as np
from backtesting import Backtest
import sys
import os

# Add the project root to sys.path to import MoonDevStrategy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.strategies.base import MoonDevStrategy

class FL001AsiaUKMW(MoonDevStrategy):
    """
    FL-001: Single Session M / W Formation (Asia → UK Day Trade)
    
    Rules:
    - Asia Range: 8:00 PM - 2:00 AM NY (20:00 - 02:00)
    - Asia Range Filter: Height < 2% of Low
    - Trigger: M or W formation entirely within Asia range
    - Entry: 15m candle after Peak 2 (UK Session: 2:00 AM - 5:00 AM NY)
    - SL: At Peak 1 (Anchor)
    - TP: Opposite Asia Range boundary
    - Mandatory Close: 8:00 AM NY
    """
    
    # Strategy Parameters
    timezone_offset = -5  # NY offset from UTC (adjust if data is UTC)
    max_asia_range_pct = 2.0
    peak_window = 3 # 15m candles to define a local peak
    
    def init(self):
        super().init()
        self.asia_high = 0
        self.asia_low = 0
        self.asia_range_ok = False
        
        # State for sequence tracking
        self.peaks = [] # List of (timestamp, type, price) where type is 'high' or 'low'
        self.formation_started = False
        self.peak1 = None
        self.peak2 = None
        
    def next(self):
        # Current time in NY
        current_time = self.data.index[-1]
        ny_time = current_time + pd.Timedelta(hours=self.timezone_offset)
        ny_hour = ny_time.hour
        ny_minute = ny_time.minute
        
        # 1. Mandatory Exit at 8:00 AM NY
        if ny_hour >= 8 and len(self.trades) > 0:
            for trade in self.trades:
                trade.close()
            return

        # 2. Reset daily state at Start of Asia (8:00 PM NY)
        if ny_hour == 20 and ny_minute == 0:
            self.asia_high = self.data.High[-1]
            self.asia_low = self.data.Low[-1]
            self.peaks = []
            self.formation_started = False
            self.peak1 = None
            self.peak2 = None
            self.asia_range_ok = False
            return

        # 3. Asia Range Tracking (8:00 PM - 2:00 AM NY)
        if (ny_hour >= 20 or ny_hour < 2):
            self.asia_high = max(self.asia_high, self.data.High[-1])
            self.asia_low = min(self.asia_low, self.data.Low[-1])
            
            # Detect local peaks within Asia
            # W: Low1 -> High -> Low2 (Low2 > Low1)
            # M: High1 -> Low -> High2 (High2 < High1)
            
            # Simple peak detection using a rolling window
            if len(self.data) > self.peak_window * 2:
                # Check for Low
                if self.data.Low[-self.peak_window] == min(self.data.Low[-self.peak_window*2:]):
                    self.peaks.append((ny_time, 'low', self.data.Low[-self.peak_window]))
                # Check for High
                if self.data.High[-self.peak_window] == max(self.data.High[-self.peak_window*2:]):
                    self.peaks.append((ny_time, 'high', self.data.High[-self.peak_window]))
            return

        # 4. Entry Selection (2:00 AM - 5:00 AM NY)
        if ny_hour >= 2 and ny_hour < 5:
            # Check Asia Range Filter
            asia_range_pct = ((self.asia_high - self.asia_low) / self.asia_low) * 100
            if ny_hour == 2 and ny_minute == 0:
                print(f"{ny_time}: Asia Range: {self.asia_high - self.asia_low:.2f} ({asia_range_pct:.2f}%), Peaks: {len(self.peaks)}")

            if asia_range_pct > self.max_asia_range_pct:
                return # Day off
            
            # Search for M or W formations in peaks
            ws = []
            for i in range(len(self.peaks) - 2):
                p1, p_mid, p2 = self.peaks[i], self.peaks[i+1], self.peaks[i+2]
                if p1[1] == 'low' and p_mid[1] == 'high' and p2[1] == 'low':
                    if p2[2] > p1[2]: # Higher Low
                        ws.append((p1, p2))
            
            ms = []
            for i in range(len(self.peaks) - 2):
                p1, p_mid, p2 = self.peaks[i], self.peaks[i+1], self.peaks[i+2]
                if p1[1] == 'high' and p_mid[1] == 'low' and p2[1] == 'high':
                    if p2[2] < p1[2]: # Lower High
                        ms.append((p1, p2))

            if not self.position:
                if ws:
                    print(f"{ny_time}: Potential W found")
                    last_w = ws[-1]
                    sl, tp, entry_price = last_w[0][2], self.asia_high, self.data.Close[-1]
                    if sl < entry_price < tp:
                        self.buy(sl=sl, tp=tp, size=0.1) # 10% of equity
                        print(f"{ny_time}: BUY W at {entry_price}")
                elif ms:
                    print(f"{ny_time}: Potential M found")
                    last_m = ms[-1]
                    sl, tp, entry_price = last_m[0][2], self.asia_low, self.data.Close[-1]
                    if tp < entry_price < sl:
                        self.sell(sl=sl, tp=tp, size=0.1) # 10% of equity
                        print(f"{ny_time}: SELL M at {entry_price}")

if __name__ == '__main__':
    # Test with sample data or existing dataset
    import pandas as pd
    import os
    
    # Path to sample data
    data_path = '/Users/kpakpo/RBI_Swarm/moon-dev-ai-agents-for-trading/data/crypto/BTCUSD_15m.csv'
    
    if os.path.exists(data_path):
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Handle column names (ensure Open, High, Low, Close exist)
        data.columns = [c.capitalize() for c in data.columns]
        
        bt = Backtest(data, FL001AsiaUKMW, cash=10000, commission=.002)
        stats = bt.run()
        print(stats)
        bt.plot(filename='results/fl_001_backtest.html')
    else:
        print(f"Data file not found at {data_path}")
