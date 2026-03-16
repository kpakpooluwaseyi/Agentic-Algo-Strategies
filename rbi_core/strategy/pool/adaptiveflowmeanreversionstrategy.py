from rbi_core.strategy.base import BaseStrategy
from rbi_core.signal import Signal
from collections import deque
from typing import Dict, Any

class AdaptiveFlowMeanReversionStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, atr_multiplier: float = 1.5, 
                 flow_threshold: float = 2.0, max_spread_pct: float = 0.002,
                 cooldown_ticks: int = 5):
        super().__init__()
        self.lookback = lookback
        self.atr_multiplier = atr_multiplier
        self.flow_threshold = flow_threshold
        self.max_spread_pct = max_spread_pct
        self.cooldown_ticks = cooldown_ticks
        
        self.prices = deque(maxlen=lookback)
        self.volumes = deque(maxlen=lookback)
        self.atr_history = deque(maxlen=lookback)
        self.flow_history = deque(maxlen=lookback)
        self.mid_history = deque(maxlen=lookback)
        
        self.cumulative_flow = 0.0
        self.vwap_numerator = 0.0
        self.vwap_denominator = 0.0
        self.position = 0
        self.last_signal_tick = 0
        self.tick_count = 0
        
    def reset(self):
        self.prices.clear()
        self.volumes.clear()
        self.atr_history.clear()
        self.flow_history.clear()
        self.mid_history.clear()
        self.cumulative_flow = 0.0
        self.vwap_numerator = 0.0
        self.vwap_denominator = 0.0
        self.position = 0
        self.last_signal_tick = 0
        self.tick_count = 0
        
    def _update_vwap(self, price: float, volume: float):
        self.vwap_numerator += price * volume
        self.vwap_denominator += volume
        if len(self.prices) == self.lookback:
            removed_price = self.prices[0]
            removed_vol = self.volumes[0]
            self.vwap_numerator -= removed_price * removed_vol
            self.vwap_denominator -= removed_vol
        if self.vwap_denominator > 0:
            return self.vwap_numerator / self.vwap_denominator
        return price
        
    def _calculate_flow_imbalance(self, bid: float, ask: float, volume: float, 
                                  price: float) -> float:
        mid = (bid + ask) / 2
        if not self.mid_history:
            self.mid_history.append(mid)
            return 0.0
            
        last_mid = self.mid_history[-1]
        tick_direction = 1 if price > last_mid else -1 if price < last_mid else 0
        
        spread = ask - bid
        if spread <= 0:
            return 0.0
            
        micro_impact = ((price - mid) / spread) * volume
        delta_flow = tick_direction * volume + micro_impact
        
        self.cumulative_flow = 0.9 * self.cumulative_flow + 0.1 * delta_flow
        self.mid_history.append(mid)
        
        return self.cumulative_flow
        
    def _calculate_adaptive_bands(self, current_atr: float) -> tuple:
        if self.vwap_denominator <= 0 or len(self.atr_history) < 5:
            return None, None
            
        vwap = self.vwap_numerator / self.vwap_denominator
        
        atr_list = list(self.atr_history)
        atr_mean = sum(atr_list) / len(atr_list)
        atr_std = (sum((a - atr_mean) ** 2 for a in atr_list) / len(atr_list)) ** 0.5
        
        volatility_regime = current_atr / atr_mean if atr_mean > 0 else 1.0
        adaptive_width = self.atr_multiplier * (1 + 0.5 * (volatility_regime - 1))
        
        upper = vwap + current_atr * adaptive_width
        lower = vwap - current_atr * adaptive_width
        
        return lower, upper
        
    def on_tick(self, tick_data: Dict[str, Any]) -> Signal:
        price = float(tick_data['price'])
        volume = float(tick_data['volume'])
        timestamp = int(tick_data['timestamp'])
        atr = float(tick_data['atr'])
        bid = float(tick_data['bid'])
        ask = float(tick_data['ask'])
        
        self.tick_count += 1
        
        spread_pct = (ask - bid) / ((ask + bid) / 2) if (ask + bid) > 0 else 0
        if spread_pct > self.max_spread_pct:
            return Signal(action='HOLD', confidence=0.0, 
                         meta={'filter': 'wide_spread', 'spread': spread_pct})
        
        self.prices.append(price)
        self.volumes.append(volume)
        self.atr_history.append(atr)
        
        vwap = self._update_vwap(price, volume)
        flow = self._calculate_flow_imbalance(bid, ask, volume, price)
        self.flow_history.append(flow)
        
        meta = {
            'vwap': vwap,
            'flow': flow,
            'position': self.position,
            'tick': self.tick_count
        }
        
        if len(self.prices) < self.lookback // 2:
            meta['status'] = 'warming_up'
            return Signal(action='HOLD', confidence=0.0, meta=meta)
        
        lower_band, upper_band = self._calculate_adaptive_bands(atr)
        if lower_band is None:
            return Signal(action='HOLD', confidence=0.0, meta=meta)
            
        meta['bands'] = (lower_band, upper_band)
        
        if self.tick_count - self.last_signal_tick < self.cooldown_ticks:
            meta['cooldown'] = True
            return Signal(action='HOLD', confidence=0.0, meta=meta)
        
        flow_list = list(self.flow_history)
        flow_mean = sum(flow_list) / len(flow_list)
        flow_std = (sum((f - flow_mean) ** 2 for f in flow_list) / len(flow_list)) ** 0.5
        flow_zscore = (flow - flow_mean) / flow_std if flow_std > 0 else 0
        
        meta['flow_z