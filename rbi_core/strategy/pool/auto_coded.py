from rbi_core.strategy.base import BaseStrategy
from rbi_core.domain.signal import Signal
from collections import deque
from statistics import mean

class Adaptive