class BidAskEMACrossStrategy(BaseStrategy):
    def __init__(self, fast_period: int = 12, slow_period: int = 26):
        super().__init__()
        self.fast_period = fast