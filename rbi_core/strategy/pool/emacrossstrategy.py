from typing import Optional, Deque
from collections import deque
import math
from rbi_core.strategy.base import BaseStrategy, Signal

class EMACrossStrategy(BaseStrategy):
    def