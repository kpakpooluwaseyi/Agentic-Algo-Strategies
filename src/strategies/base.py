"""
🌙 Moon Dev's Base Strategy Class
================================
The standard contract for all trading strategies in the Moon Dev swarm.
Ensures consistency, safety, and performance.

All agent-generated strategies MUST inherit from MoonDevStrategy.
"""

from backtesting import Strategy
from abc import abstractmethod
import logging

class MoonDevStrategy(Strategy):
    """
    Standard base class for all strategies.
    Inherits from backtesting.py Strategy for core mechanics.
    
    Subclasses MUST implement:
        - init(): Initialize indicators
        - next(): Main trading logic per bar
    """
    
    # Standard parameters every strategy should have
    risk_pct = 1.0  # Risk 1% of equity per trade
    
    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def init(self):
        """
        Initialize indicators. 
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def next(self):
        """
        Main trading logic per bar.
        Must be implemented by subclasses.
        """
        pass

    def get_position_size(self):
        """
        Universal position sizing helper.
        Calculates size based on risk_pct and current equity.
        """
        return 0.1 # Placeholder for 10% of equity
