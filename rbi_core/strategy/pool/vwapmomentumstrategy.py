class VWAPMomentumStrategy(BaseStrategy):
    def __init__(self, window: int = 15, volume_surge: float = 1.5, momentum_threshold: float = 0.001):
        super().__init__()
        self.window = window
        self.volume_surge = volume_surge
        self.momentum_threshold = momentum_threshold
        self.price_volume_products: deque = deque(maxlen=window)
        self.volumes: deque = deque(maxlen=window)
        self.prices: deque = deque(maxlen=window)
        self.last_price: Optional[float] = None
        
    def reset(self) -> None:
        self.price_volume_products.clear()
        self.volumes.clear()
        self.prices.clear()
        self.last_price = None
        
    def on_tick(self, tick_data) -> Optional[Signal]:
        price = tick_data.price
        volume = tick_data.volume
        
        self.price_volume_products.append(price * volume)
        self.volumes.append(volume)
        self.prices.append(price)
        
        if len(self.prices) < self.window:
            self.last_price = price
            return None
            
        total_volume = sum(self.volumes)
        if total_volume == 0:
            self.last_price = price