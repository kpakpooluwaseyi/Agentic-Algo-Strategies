"""
VuManchu Cipher B Strategy
==========================
A backtesting strategy using the VuManchu Cipher B indicator.

Buy Signal: WaveTrend cross up while oversold (wt1 & wt2 <= -53)
Sell Signal: WaveTrend cross down while overbought (wt1 & wt2 >= 53)
"""

from backtesting import Strategy
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.indicators.vumanchu import cipher_b


def preprocess_data(df, **params):
    """Apply VuManchu Cipher B indicators to the dataframe."""
    from src.indicators.vumanchu import cipher_b
    df = cipher_b(df)
    # Convert boolean signals to int for backtesting.py compatibility
    df['buy_signal'] = df['buy_signal'].astype(int)
    df['sell_signal'] = df['sell_signal'].astype(int)
    return df


class VuManchuCipherB(Strategy):
    """
    VuManchu Cipher B Trading Strategy
    
    Entry (Long):
    - WaveTrend cross up while both wt1 and wt2 are below oversold level (-53)
    - CONFLUENCE: Money Flow (rsimfi) must be positive (> 0)
    
    Exit:
    - WaveTrend cross down while both wt1 and wt2 are above overbought level (53)
    - CONFLUENCE: Money Flow (rsimfi) must be negative (< 0)
    - Or stop loss / take profit
    """
    
    # Optimizable parameters
    stop_loss_pct = 0.03      # 3% stop loss
    take_profit_pct = 0.06    # 6% take profit (2:1 RR)
    
    def init(self):
        # Use self.I() to properly wrap signals and indicators for backtesting.py
        self.buy_sig = self.I(lambda: self.data.buy_signal, name='buy_signal')
        self.sell_sig = self.I(lambda: self.data.sell_signal, name='sell_signal')
        self.mf = self.I(lambda: self.data.rsimfi, name='money_flow')
    
    def next(self):
        # Skip warmup period  
        if len(self.data) < 65: # MFI needs 60 + SMMA warmup
            return
        
        current_price = self.data.Close[-1]
        
        # Entry logic
        if not self.position:
            # Buy on Cipher B buy signal + Positive Money Flow
            if self.buy_sig[-1] == 1 and self.mf[-1] > 0:
                sl = current_price * (1 - self.stop_loss_pct)
                tp = current_price * (1 + self.take_profit_pct)
                self.buy(sl=sl, tp=tp)
        
        # Exit logic
        else:
            # Exit on bearish signal or Negative Money Flow
            if self.sell_sig[-1] == 1 or self.mf[-1] < 0:
                self.position.close()


# Standalone testing
if __name__ == '__main__':
    from backtesting import Backtest
    
    # Load data
    try:
        df = pd.read_csv('data/BTC_1h.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("No data file found. Using sample data generation...")
        # Generate simple sample data
        dates = pd.date_range('2023-01-01', periods=1000, freq='1h')
        np.random.seed(42)
        price = 20000 + np.cumsum(np.random.randn(1000) * 100)
        df = pd.DataFrame({
            'Open': price,
            'High': price + np.random.rand(1000) * 200,
            'Low': price - np.random.rand(1000) * 200,
            'Close': price + np.random.randn(1000) * 50,
            'Volume': np.random.rand(1000) * 1000000
        }, index=dates)
    
    # Preprocess
    df = preprocess_data(df)
    
    # Drop NaN rows
    df = df.dropna()
    
    # Run backtest
    bt = Backtest(df, VuManchuCipherB, cash=100000, commission=0.001)
    stats = bt.run()
    
    print("\n=== VuManchu Cipher B Strategy Results ===")
    print(stats)
    
    # Save plot
    bt.plot(filename='results/plots/vumanchu_cipher_b.html', open_browser=False)
    print("\nPlot saved to results/plots/vumanchu_cipher_b.html")
