from rbi_core.strategy.base import BaseStrategy
from typing import Dict, Any, List, Optional
from collections import deque
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    action: str
    confidence: float
    meta: Dict[str, Any]


class MicrostructureMomentumStrategy(BaseStrategy):
    """
    Adaptive Microstructure Momentum Strategy combining Order Flow Imbalance (OFI),
    ATR-based volatility regime detection, and Bid-Ask pressure analysis.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        atr_regime_window: int = 10,
        ofi_smoothing: int = 5,
        spread_pressure_threshold: float = 0.3,
        vol_filter_mult: float = 2.0,
        max_hold_ticks: int = 100
    ):
        super().__init__()
        self.lookback = lookback
        self.atr_regime_window = atr_regime_window
        self.ofi_smoothing = ofi_smoothing
        self.spread_pressure_threshold = spread_pressure_threshold
        self.vol_filter_mult = vol_filter_mult
        self.max_hold_ticks = max_hold_ticks
        self.reset()
    
    def reset(self) -> None:
        """Reset all stateful variables for strategy reuse."""
        # Circular buffers for tick history
        self.price_history: deque = deque(maxlen=self.lookback)
        self.volume_history: deque = deque(maxlen=self.lookback)
        self.atr_history: deque = deque(maxlen=self.atr_regime_window)
        self.ofi_history: deque = deque(maxlen=self.ofi_smoothing)
        self.spread_history: deque = deque(maxlen=self.lookback)
        self.timestamp_history: deque = deque(maxlen=self.lookback)
        
        # Previous tick state
        self.prev_bid: Optional[float] = None
        self.prev_ask: Optional[float] = None
        self.prev_price: Optional[float] = None
        self.prev_volume: Optional[float] = None
        
        # Indicator state
        self.ema_fast: Optional[float] = None
        self.ema_slow: Optional[float] = None
        self.cumulative_ofi: float = 0.0
        self.volatility_regime: float = 1.0
        
        # Position state
        self.position: int = 0
        self.entry_price: float = 0.0
        self.entry_tick: int = 0
        self.current_tick: int = 0
        
        # Adaptive thresholds
        self.ofi_std: float = 1.0
        self.mean_volume: float = 0.0
    
    def on_tick(self, tick_data: Dict[str, Any]) -> Signal:
        """
        Process incoming tick data and generate trading signal.
        
        Args:
            tick_data: Dictionary with keys 'price', 'volume', 'timestamp', 'atr', 'bid', 'ask'
        """
        # Extract tick components
        price = float(tick_data['price'])
        volume = float(tick_data['volume'])
        timestamp = float(tick_data['timestamp'])
        atr = float(tick_data['atr'])
        bid = float(tick_data['bid'])
        ask = float(tick_data['ask'])
        
        # Update histories
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.atr_history.append(atr)
        self.timestamp_history.append(timestamp)
        
        # Calculate spread and mid
        spread = ask - bid
        mid = (bid + ask) / 2
        self.spread_history.append(spread)
        
        # Calculate Order Flow Imbalance (OFI)
        ofi = self._calculate_ofi(bid, ask, volume, spread)
        self.ofi_history.append(ofi)
        self.cumulative_ofi += ofi
        
        # Update adaptive statistics
        self._update_statistics()
        
        # Update EMAs for trend detection
        self._update_emas(price)
        
        # Calculate microstructure components
        spread_pressure = self._calculate_spread_pressure(price, bid, ask, spread)
        momentum = self._calculate_atr_normalized_momentum(price, atr)
        vol_regime = self._calculate_volatility_regime(atr)
        
        # Check for signal generation conditions
        signal = self._generate_signal_logic(
            price=price,
            ofi=ofi,
            spread_pressure=spread_pressure,
            momentum=momentum,
            vol_regime=vol_regime,
            volume=volume,
            timestamp=timestamp
        )
        
        # Update previous values
        self.prev_bid = bid
        self.prev_ask = ask
        self.prev_price = price
        self.prev_volume = volume
        self.current_tick += 1
        
        return signal
    
    def _calculate_ofi(self, bid: float, ask: float, volume: float, spread: float) -> float:
        """Calculate Order Flow Imbalance based on bid/ask changes."""
        if self.prev_bid is None or self.prev_ask is None or spread <= 0:
            return 0.0
        
        # Price changes at bid and ask
        delta_bid = bid - self.prev_bid
        delta_ask = ask - self.prev_ask
        
        # Signed volume based on aggressor side
        if delta_bid > 0 and delta_ask >= 0:
            # Buying pressure at bid
            signed_volume = volume
        elif delta_ask < 0 and delta_bid <= 0:
            # Selling pressure at ask
            signed_volume = -volume
        else:
            # Mixed or no change
            signed_volume = 0.0
        
        # Normalize by spread and ATR
        current_atr = self.atr_history[-1] if self.atr_history else spread
        if current_atr > 0:
            normalized_ofi = signed_volume * spread / current_atr
        else:
            normalized_ofi = signed_volume
        
        return normalized_ofi
    
    def _update_statistics(self) -> None:
        """Update running statistics for adaptive thresholds."""
        if len(self.ofi_history) >= 5:
            # Calculate rolling std of OFI
            mean_ofi = sum(self.ofi_history) / len(self.ofi_history)
            variance = sum((x - mean_ofi) ** 2 for x in self.ofi_history) / len(self.ofi_history)
            self.ofi_std = math.sqrt(variance) if variance > 0 else 1.0
        
        if len(self.volume_history) > 0:
            self.mean_volume = sum(self.volume_history) / len(self.volume_history)
    
    def _update_emas(self, price: float) -> None:
        """Update exponential moving averages."""
        alpha_fast = 2 / (5 + 1)  # 5-period EMA
        alpha_slow = 2 / (self.lookback + 1)
        
        if self.ema_fast is None:
            self.ema_fast = price
            self.ema_slow = price
        else:
            self.ema_fast = alpha_fast * price + (1 - alpha_fast) * self.ema_fast
            self.ema_slow = alpha_slow * price + (1 - alpha_slow) * self.ema_slow
    
    def _calculate_spread_pressure(self, price: float, bid: float, ask: float, spread: float) -> float:
        """
        Calculate position of price within spread and proximity to extremes.
        Returns value in [-1, 1] where 1 = strong buying (near ask), -1 = strong selling (near bid)
        """
        if spread <= 0:
            return 0.0
        
        # Location within spread
        relative_position = (price - bid) / spread
        
        # Adjust for historical spread context
        if len(self.spread_history) > 5:
            avg_spread = sum(self.spread_history) / len(self.spread_history)
            spread_ratio = spread / avg_spread if avg_spread > 0 else 1.0
            
            # Compress signal when spreads widen (uncertainty)
            if spread_ratio > 1.5:
                relative_position = 0.5 + (relative_position - 0.5) / spread_ratio
        
        return 2 * (relative_position - 0.5)  # Scale to [-1, 1]
    
    def _calculate_atr_normalized_momentum(self, price: float, current_atr: float) -> float:
        """Calculate price momentum normalized by ATR."""
        if len(self.price_history) < 5 or current_atr <= 0:
            return 0.0
        
        # Multi-period momentum
        short_change = (price - self.price_history[-5]) / current_atr
        long_change = (price - self.price_history[0]) / (current_atr * 2) if len(self.price_history) == self.lookback else 0
        
        # Weighted combination
        return 0.7 * short_change + 0.3 * long_change
    
    def _calculate_volatility_regime(self, current_atr: float) -> float:
        """Determine if volatility is expanding or contracting."""
        if len(self.atr_history) < self.atr_regime_window:
            return 1.0
        
        atr_mean = sum(self.atr_history) / len(self.atr_history)
        if atr_mean <= 0:
            return 1.0
        
        ratio = current_atr / atr_mean
        self.volatility_regime = ratio
        return ratio
    
    def _generate_signal_logic(
        self,
        price: float,
        ofi: float,
        spread_pressure: float,
        momentum: float,
        vol_regime: float,
        volume: float,
        timestamp: float
    ) -> Signal:
        """Generate final trading signal based on composite indicators."""
        
        # Meta data collection
        meta = {
            'tick': self.current_tick,
            'ofi': round(ofi, 6),
            'spread_pressure': round(spread_pressure, 4),
            'momentum': round(momentum, 4),
            'vol_regime': round(vol_regime, 2),
            'position': self.position
        }
        
        # Insufficient data check
        if len(self.price_history) < self.lookback:
            return Signal('HOLD', 0.0, {**meta, 'reason': 'warming_up'})
        
        # Volatility filter - avoid trading in extreme volatility unless OFI confirms
        if vol_regime > self.vol_filter_mult and abs(ofi) < self.ofi_std:
            return Signal('HOLD', 0.0, {**meta, 'reason': 'volatility_filter'})
        
        # Calculate composite microstructure score
        # OFI component (trend of recent OFI)
        ofi_trend = sum(self.ofi_history) / len(self.ofi_history) if self.ofi_history else 0
        ofi_score = ofi_trend / (self.ofi_std + 1e-9)  # Z-score normalization
        
        # Trend alignment
        trend_direction = 1 if self.ema_fast > self.ema_slow else -1 if self.ema_fast < self.ema_slow else 0
        
        # Composite score: [-inf, +inf]
        # Weights: OFI (35%), Momentum (25%), Spread Pressure (25%), Trend (15%)
        composite_score = (
            0.35 * ofi_score +
            0.25 * momentum +
            0.25 * spread_pressure +
            0.15 * trend_direction
        )
        
        # Volume confirmation factor
        volume_ratio = volume / (self.mean_volume + 1e-9) if self.mean_volume > 0 else 1.0
        volume_confirm = 1.0 if volume_ratio > 1.2 else 0.8 if volume_ratio > 0.8 else 0.6
        
        # Dynamic threshold based on volatility regime
        base_threshold = 0.5
        adaptive_threshold = base_threshold * (1 + (vol_regime - 1) * 0.2)
        
        # Signal generation
        if composite_score > adaptive_threshold:
            confidence = min(0.99, (composite_score / (adaptive_threshold * 2)) * volume_confirm)
            
            # Check for position flip or new entry
            if self.position <= 0:
                self.position = 1
                self.entry_price = price
                self.entry_tick = self.current_tick
                return Signal('BUY', confidence, {**meta, 'score': composite_score, 'type': 'microstructure_breakout'})
            else:
                # Hold existing long
                return Signal('HOLD', 0.0, {**meta, 'hold_reason': 'already_long'})
                
        elif composite_score < -adaptive_threshold:
            confidence = min(0.99, (abs(composite_score) / (adaptive_threshold * 2)) * volume_confirm)
            
            if self.position >= 0:
                self.position = -1
                self.entry_price = price
                self.entry_tick = self.current_tick
                return Signal('SELL', confidence, {**meta, 'score': composite_score, 'type': 'microstructure_breakdown'})
            else:
                return Signal('HOLD', 0.0, {**meta, 'hold_reason': 'already_short'})
        
        else:
            # Check time-based exit for existing positions
            if self.position != 0 and (self.current_tick - self.entry_tick) > self.max_hold_ticks:
                prev_pos = self.position
                self.position = 0
                return Signal('SELL' if prev_pos == 1 else 'BUY', 0.5, {**meta, 'reason': 'time_exit'})
            
            return Signal('HOLD', 0.0, {**meta, 'score': composite_score, 'threshold': adaptive_threshold})