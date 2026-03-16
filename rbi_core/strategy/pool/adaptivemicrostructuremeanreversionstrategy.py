from rbi_core.strategy.base import BaseStrategy, Signal
from collections import deque
import math
from typing import Dict, Any
from datetime import datetime


class AdaptiveMicrostructureMeanReversionStrategy(BaseStrategy):
    """
    Novel strategy combining ATR-based volatility channels with bid-ask microstructure
    imbalance and volume-weighted mean reversion scoring.
    """
    
    def __init__(self, 
                 lookback: int = 20,
                 atr_channel_mult: float = 1.8,
                 volume_percentile_threshold: float = 75.0,
                 microstructure_threshold: float = 0.65,
                 cooldown_ticks: int = 5):
        super().__init__()
        self.lookback = lookback
        self.atr_channel_mult = atr_channel_mult
        self.volume_percentile_threshold = volume_percentile_threshold / 100.0
        self.microstructure_threshold = microstructure_threshold
        self.cooldown_ticks = cooldown_ticks
        
        # State management
        self.price_buffer = deque(maxlen=lookback)
        self.volume_buffer = deque(maxlen=lookback)
        self.atr_buffer = deque(maxlen=lookback)
        self.microstructure_buffer = deque(maxlen=lookback)
        self.tick_counter = 0
        self.last_signal_tick = -cooldown_ticks
        self.session_vwap_num = 0.0
        self.session_vwap_den = 0.0
        self.prev_price = None
        self.price_velocity = 0.0
        
    def reset(self):
        """Reset all stateful components for fresh strategy run."""
        self.price_buffer.clear()
        self.volume_buffer.clear()
        self.atr_buffer.clear()
        self.microstructure_buffer.clear()
        self.tick_counter = 0
        self.last_signal_tick = -self.cooldown_ticks
        self.session_vwap_num = 0.0
        self.session_vwap_den = 0.0
        self.prev_price = None
        self.price_velocity = 0.0
        
    def _update_vwap(self, typical_price: float, volume: float) -> float:
        """Calculate volume-weighted average price cumulatively."""
        self.session_vwap_num += typical_price * volume
        self.session_vwap_den += volume
        if self.session_vwap_den > 0:
            return self.session_vwap_num / self.session_vwap_den
        return typical_price
        
    def _calculate_microstructure_pressure(self, bid: float, ask: float, price: float, 
                                          volume: float) -> float:
        """
        Calculate order flow imbalance based on position within spread and volume.
        Returns 0.0 (strong sell pressure) to 1.0 (strong buy pressure).
        """
        spread = ask - bid
        if spread <= 0 or price <= 0:
            return 0.5
            
        # Position in spread: 0 = at bid, 1 = at ask
        spread_position = (price - bid) / spread
        
        # Volume-weighted adjustment: large volume near bid/ask indicates urgency
        if len(self.volume_buffer) >= 5:
            avg_vol = sum(self.volume_buffer) / len(self.volume_buffer)
            if avg_vol > 0:
                volume_intensity = min(volume / avg_vol, 3.0)
                # Shift spread position based on volume (high volume at extremes = stronger signal)
                if spread_position < 0.3:
                    spread_position -= (0.1 * volume_intensity)  # More bearish
                elif spread_position > 0.7:
                    spread_position += (0.1 * volume_intensity)  # More bullish
                    
        return max(0.0, min(1.0, spread_position))
        
    def _calculate_dynamic_percentile(self, current: float, history: deque) -> float:
        """Calculate percentile rank of current value in historical distribution."""
        if len(history) < 5:
            return 0.5
        sorted_hist = sorted(history)
        count = sum(1 for x in sorted_hist if x <= current)
        return count / len(sorted_hist)
        
    def _calculate_volatility_regime(self) -> str:
        """Determine if volatility is expanding, contracting, or stable."""
        if len(self.atr_buffer) < 10:
            return "neutral"
        recent = sum(list(self.atr_buffer)[-5:]) / 5
        previous = sum(list(self.atr_buffer)[-10:-5]) / 5
        if previous == 0:
            return "neutral"
        ratio = recent / previous
        if ratio > 1.2:
            return "expanding"
        elif ratio < 0.8:
            return "contracting"
        return "stable"
        
    def on_tick(self, tick_data: Dict[str, Any]) -> Signal:
        price = float(tick_data.get('price', 0.0))
        volume = float(tick_data.get('volume', 0.0))
        timestamp = tick_data.get('timestamp')
        atr = float(tick_data.get('atr', 0.0))
        bid = float(tick_data.get('bid', price))
        ask = float(tick_data.get('ask', price))
        
        if price <= 0 or atr <= 0:
            return Signal(action='HOLD', confidence=0.0, meta={'error': 'invalid_tick_data'})
            
        # Update velocity
        if self.prev_price is not None:
            self.price_velocity = price - self.prev_price
        self.prev_price = price
        
        # Update buffers
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
        self.atr_buffer.append(atr)
        
        typical_price = (bid + ask + price) / 3.0
        vwap = self._update_vwap(typical_price, volume)
        micro_pressure = self._calculate_microstructure_pressure(bid, ask, price, volume)
        self.microstructure_buffer.append(micro_pressure)
        
        self.tick_counter += 1
        
        # Minimum data check
        if len(self.price_buffer) < self.lookback // 2:
            return Signal(action='HOLD', confidence=0.0, meta={'status': 'initializing'})
            
        # Calculate ATR bands
        sma = sum(self.price_buffer) / len(self.price_buffer)
        upper_band = sma + (atr * self.atr_channel_mult)
        lower_band = sma - (atr * self.atr_channel_mult)
        
        # Volume percentile
        vol_percentile = self._calculate_dynamic_percentile(volume, self.volume_buffer)
        
        # Mean reversion score: distance from VWAP normalized by ATR
        vwap_deviation = (price - vwap) / atr
        mr_score = math.tanh(abs(vwap_deviation) / 2)  # 0 to 1, higher = more extreme
        
        # Volatility regime
        vol_regime = self._calculate_volatility_regime()
        
        # Signal logic
        action = 'HOLD'
        confidence = 0.0
        meta = {
            'vwap_dev': vwap_deviation,
            'micro_pressure': micro_pressure,
            'vol_pct': vol_percentile,
            'vol_regime': vol_regime,
            'mr_score': mr_score
        }
        
        # Cooldown check
        if (self.tick_counter - self.last_signal_tick) < self.cooldown_ticks:
            meta['cooldown'] = True
            return Signal(action='HOLD', confidence=0.0, meta=meta)
            
        # BUY Signal: Price below lower band + Extreme selling pressure fading (micro_pressure < 0.3) 
        # + High volume (capitulation) + Mean reversion opportunity
        if (price < lower_band and 
            micro_pressure < (1 - self.microstructure_threshold) and
            vol_percentile > self.volume_percentile_threshold and
            vwap_deviation < -1.5):
            
            # Confidence based on how extreme the deviation is and microstructure confirmation
            deviation_factor = min(abs(vwap_deviation) / 3.0, 1.0)
            pressure_factor = (0.5 - micro_pressure) * 2  # 0 to 1, higher = more extreme pressure
            confidence = 0.5 + (deviation_factor * 0.3) + (pressure_factor * 0.2)
            
            action = 'BUY'
            meta['trigger'] = 'capitulation_reversal'
            meta['band_distance'] = (lower_band - price) / atr
            self.last_signal_tick = self.tick_counter
            
        # SELL Signal: Price above upper band + Extreme buying pressure fading + High volume (euphoria)
        elif (price > upper_band and 
              micro_pressure > self.microstructure_threshold and
              vol_percentile > self.volume_percentile_threshold and
              vwap_deviation > 1.5):
              
            deviation_factor = min(vwap_deviation / 3.0, 1.0)
            pressure_factor = (micro_pressure - 0.5) * 2
            confidence = 0.5 + (deviation_factor * 0.3) + (pressure_factor * 0.2)
            
            action = 'SELL'
            meta['trigger'] = 'euphoria_reversal'
            meta['band_distance'] = (price - upper_band) / atr
            self.last_signal_tick = self.tick_counter
            
        # Volatility breakout fade (secondary signal)
        elif (vol_regime == 'expanding' and 
              abs(vwap_deviation) > 2.0 and 
              len(self.microstructure_buffer) >= 3):
              
            # Check for divergence: price extreme but microstructure weakening
            recent_pressure = list(self.microstructure_buffer)[-3:]
            if vwap_deviation > 2.0 and recent_pressure[-1] < recent_pressure[0]:
                # Bullish exhaustion
                confidence = mr_score * 0.7
                action = 'SELL'
                meta['trigger'] = 'volatility_fade_exhaustion'
                self.last_signal_tick = self.tick_counter
            elif vwap_deviation < -2.0 and recent_pressure[-1] > recent_pressure[0]:
                # Bearish exhaustion  
                confidence = mr_score * 0.7
                action = 'BUY'
                meta['trigger'] = 'volatility_fade_capitulation'
                self.last_signal_tick = self.tick_counter
                
        return Signal(action=action, confidence=min(confidence, 1.0), meta=meta)