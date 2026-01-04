"""
Strategy Template for Backtesting Pipeline
==========================================

Instructions:
1. Rename this file to your_strategy_name.py (use underscores, lowercase)
2. Rename the class to YourStrategyName (CamelCase)
3. Implement your trading logic in next()
4. Drop the file in strategies/ folder
5. The local_runner will automatically pick it up and test it

The pipeline will:
- Run your strategy on 6 BTC timeframes (4h, 1h, 15m, 5m, 1m)
- Perform Walk-Forward Analysis (WFA) with 30% out-of-sample data
- Perform Walk-Forward Optimization (WFO) if WFA fails
- Add results to the leaderboard
"""

from backtesting import Strategy
import numpy as np

# Optional: preprocessing function (runs before backtesting)
def preprocess_data(df, **params):
    """
    Add any custom indicators or columns to the dataframe.
    This runs once before the backtest starts.
    
    Args:
        df: DataFrame with columns: Open, High, Low, Close, Volume
        **params: Any parameters passed from optimization
    
    Returns:
        df: Modified DataFrame with new columns
    """
    df = df.copy()
    
    # Example: Add a simple moving average
    # df['SMA_20'] = df['Close'].rolling(20).mean()
    
    # Example: Add RSI
    # delta = df['Close'].diff()
    # gain = delta.where(delta > 0, 0).rolling(14).mean()
    # loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    # df['RSI'] = 100 - (100 / (1 + gain / loss))
    
    return df


class MyStrategyTemplate(Strategy):
    """
    Your strategy description here.
    
    Entry conditions:
    - Describe when you buy
    
    Exit conditions:
    - Describe when you sell
    """
    
    # ===== OPTIMIZABLE PARAMETERS =====
    # These can be optimized by the WFO process
    # Syntax: param_name = default_value
    
    lookback_period = 20      # Example: period for indicators
    stop_loss_pct = 0.02      # Example: 2% stop loss
    take_profit_pct = 0.04    # Example: 4% take profit
    
    def init(self):
        """
        Initialize indicators here.
        Called once at the start of the backtest.
        
        Use self.I() to create indicators that will be plotted.
        """
        # Example: Create a simple moving average indicator
        # self.sma = self.I(lambda x: x.rolling(self.lookback_period).mean(), self.data.Close)
        
        # Example: Use preprocessed data
        # if hasattr(self.data, 'SMA_20'):
        #     self.sma = self.data.SMA_20
        
        pass  # Remove this and add your initialization
    
    def next(self):
        """
        Main trading logic. Called on each new bar.
        
        Available data:
        - self.data.Open, .High, .Low, .Close, .Volume (current and historical)
        - self.data.Close[-1] = current close, self.data.Close[-2] = previous close
        - self.position = current position (truthy if in a trade)
        - self.position.size = position size (+ for long, - for short)
        
        Available actions:
        - self.buy(size=1.0, sl=None, tp=None)  # Open long
        - self.sell(size=1.0, sl=None, tp=None) # Open short
        - self.position.close()                  # Close current position
        """
        
        # ===== YOUR ENTRY LOGIC HERE =====
        # Example long entry:
        # if not self.position:
        #     if self.data.Close[-1] > self.sma[-1]:
        #         self.buy(
        #             sl=self.data.Close[-1] * (1 - self.stop_loss_pct),
        #             tp=self.data.Close[-1] * (1 + self.take_profit_pct)
        #         )
        
        # ===== YOUR EXIT LOGIC HERE =====
        # Example exit:
        # if self.position:
        #     if self.data.Close[-1] < self.sma[-1]:
        #         self.position.close()
        
        pass  # Remove this and add your trading logic


# ===== STANDALONE MODE =====
# This allows you to test the strategy directly: python your_strategy.py
if __name__ == '__main__':
    import pandas as pd
    from backtesting import Backtest
    
    # Load sample data (uses BTC 1h data if available)
    try:
        df = pd.read_csv('data/BTC_1h.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("No data file found. Create data/BTC_1h.csv or modify the path.")
        exit(1)
    
    # Preprocess if needed
    df = preprocess_data(df)
    
    # Run backtest
    bt = Backtest(df, MyStrategyTemplate, cash=100000, commission=0.001)
    stats = bt.run()
    print(stats)
    
    # Show plot
    bt.plot()
