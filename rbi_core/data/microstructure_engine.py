"""rbi_core/data/microstructure_engine.py — Real-time Microstructure Intelligence Engine.

Processes raw tick + l2Book data into three derived signals:
1. CVD (Cumulative Volume Delta) — net buying pressure over rolling windows
2. OFI (Order Flow Imbalance) — ratio of aggressive buys to sells
3. Regime — real-time market state classification (trending/ranging/volatile/illiquid)
4. ATR — rolling Wilder Average True Range from tick-aggregated bars

Sits between the WS feed and the strategy combiner, enriching every tick dict
with microstructure fields before downstream consumption.
"""
import threading
import time
import math
from collections import deque
from typing import Callable, Optional
from enum import Enum
from rbi_core.data.atr_calculator import ATRCalculator


class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    ILLIQUID = "illiquid"


class MicrostructureEngine:
    """
    Stateful engine that accumulates trade-level order flow and book snapshots
    to compute real-time microstructure signals.

    Thread-safe: ingests ticks from WS thread, state read by combiner thread.
    """

    def __init__(
        self,
        cvd_window: int = 500,            # ticks for rolling CVD
        ofi_window: int = 200,            # ticks for rolling OFI
        volatility_window: int = 300,     # ticks for regime volatility estimate
        regime_update_interval: int = 50, # re-classify regime every N ticks
        illiquidity_spread_pct: float = 0.15,  # spread > 15bps = illiquid
        trending_threshold: float = 0.65,  # |OFI| > this = trending
        volatile_threshold: float = 2.0,   # realized vol > N * baseline = volatile
        atr_bar_ticks: int = 50,           # ticks per ATR bar
        atr_period: int = 14,              # Wilder ATR lookback in bars
    ):
        self.cvd_window = cvd_window
        self.ofi_window = ofi_window
        self.volatility_window = volatility_window
        self.regime_update_interval = regime_update_interval
        self.illiquidity_spread_pct = illiquidity_spread_pct
        self.trending_threshold = trending_threshold
        self.volatile_threshold = volatile_threshold

        # Per-symbol ATR calculators (critical: can't mix BTC $67k + ETH $2k in one bar)
        self._atr_bar_ticks = atr_bar_ticks
        self._atr_period = atr_period
        self._atr_calcs: dict[str, ATRCalculator] = {}

        # Rolling buffers
        self._signed_volumes: deque = deque(maxlen=cvd_window)    # +vol for buy, -vol for sell
        self._trade_sides: deque = deque(maxlen=ofi_window)       # +1 buy, -1 sell
        self._price_returns: deque = deque(maxlen=volatility_window)
        self._spreads: deque = deque(maxlen=ofi_window)

        # State (thread-safe via lock)
        self._lock = threading.Lock()
        self._cvd: float = 0.0
        self._ofi: float = 0.0                    # -1.0 to +1.0
        self._regime: MarketRegime = MarketRegime.RANGING
        self._regime_weight: float = 1.0           # 0.0 to 2.0 (for combiner)
        self._last_price: float = 0.0
        self._tick_count: int = 0
        self._baseline_vol: float = 0.0            # calibrated baseline volatility

        # Book state
        self._best_bid: float = 0.0
        self._best_ask: float = 0.0
        self._bid_depth: float = 0.0               # total bid size at top N levels
        self._ask_depth: float = 0.0               # total ask size at top N levels

    # === PUBLIC API ===

    def process_tick(self, tick: dict) -> dict:
        """
        Enrich a tick dict with microstructure fields. Called from WS callback.

        Adds to tick:
            'cvd': float           — cumulative volume delta
            'ofi': float           — order flow imbalance (-1 to +1)
            'regime': str          — market regime label
            'regime_weight': float — regime weight for combiner (0.0 to 2.0)
            'bid_depth': float     — aggregated bid depth
            'ask_depth': float     — aggregated ask depth

        Returns:
            The enriched tick dict (mutated in-place).
        """
        price = tick.get('price', 0.0)
        volume = tick.get('volume', 0.0)
        bid = tick.get('bid', 0.0)
        ask = tick.get('ask', 0.0)

        # Feed per-symbol ATR calculator (outside main lock — ATR has its own lock)
        symbol = tick.get('symbol', 'DEFAULT')
        if symbol not in self._atr_calcs:
            self._atr_calcs[symbol] = ATRCalculator(
                bar_ticks=self._atr_bar_ticks, atr_period=self._atr_period
            )
        self._atr_calcs[symbol].process_tick(tick)

        # Determine trade side from price vs mid
        side = self._infer_side(price, bid, ask)

        with self._lock:
            # Update CVD
            signed_vol = volume * side
            self._signed_volumes.append(signed_vol)
            self._cvd = sum(self._signed_volumes)

            # Update OFI
            self._trade_sides.append(side)
            total = len(self._trade_sides)
            if total > 0:
                buys = sum(1 for s in self._trade_sides if s > 0)
                self._ofi = (2.0 * buys / total) - 1.0  # Normalize to [-1, +1]
            else:
                self._ofi = 0.0

            # Update price returns for volatility
            if self._last_price > 0 and price > 0:
                ret = math.log(price / self._last_price)
                self._price_returns.append(ret)
            self._last_price = price

            # Update spread tracking
            if bid > 0 and ask > 0:
                spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
                self._spreads.append(spread_pct)
                self._best_bid = bid
                self._best_ask = ask

            self._tick_count += 1

            # Periodic regime reclassification
            if self._tick_count % self.regime_update_interval == 0:
                self._classify_regime()

            # Enrich tick with live per-symbol ATR
            sym = tick.get('symbol', 'DEFAULT')
            atr_calc = self._atr_calcs.get(sym)
            live_atr = atr_calc.atr if atr_calc else 0.0
            tick['atr'] = live_atr if live_atr > 0 else tick.get('atr', 0.0)

            # Enrich tick with microstructure signals
            tick['cvd'] = self._cvd
            tick['ofi'] = self._ofi
            tick['regime'] = self._regime.value
            tick['regime_weight'] = self._regime_weight
            tick['bid_depth'] = self._bid_depth
            tick['ask_depth'] = self._ask_depth

        return tick

    def update_book(self, book_data: dict) -> None:
        """
        Update depth state from l2Book snapshot. Called separately from trade processing.

        Args:
            book_data: {'levels': [[{px, sz}, ...], [{px, sz}, ...]]}
        """
        levels = book_data.get('levels', [])
        if len(levels) < 2:
            return

        with self._lock:
            bids = levels[0]
            asks = levels[1]
            # Aggregate top-5 bid/ask depth
            self._bid_depth = sum(float(b.get('sz', 0)) for b in bids[:5])
            self._ask_depth = sum(float(a.get('sz', 0)) for a in asks[:5])

    @property
    def regime(self) -> MarketRegime:
        with self._lock:
            return self._regime

    @property
    def regime_weight(self) -> float:
        with self._lock:
            return self._regime_weight

    @property
    def cvd(self) -> float:
        with self._lock:
            return self._cvd

    @property
    def ofi(self) -> float:
        with self._lock:
            return self._ofi

    @property
    def atr(self) -> float:
        """ATR of the first (primary) symbol. Use atr_for(symbol) for specific."""
        if self._atr_calcs:
            primary = next(iter(self._atr_calcs.values()))
            return primary.atr
        return 0.0

    def atr_for(self, symbol: str) -> float:
        """Get ATR for a specific symbol."""
        calc = self._atr_calcs.get(symbol)
        return calc.atr if calc else 0.0

    def reset(self) -> None:
        with self._lock:
            self._signed_volumes.clear()
            self._trade_sides.clear()
            self._price_returns.clear()
            self._spreads.clear()
            self._cvd = 0.0
            self._ofi = 0.0
            self._regime = MarketRegime.RANGING
            self._regime_weight = 1.0
            self._last_price = 0.0
            self._tick_count = 0
            self._baseline_vol = 0.0
            for calc in self._atr_calcs.values():
                calc.reset()

    # === PRIVATE ===

    def _infer_side(self, price: float, bid: float, ask: float) -> float:
        """
        Infer trade aggressor side using Lee-Ready algorithm (simplified).
        Trade at or above midpoint = buyer-initiated (+1).
        Trade below midpoint = seller-initiated (-1).
        """
        if bid <= 0 or ask <= 0:
            return 1.0  # Default to buy if no book data yet
        mid = (bid + ask) / 2.0
        if price >= mid:
            return 1.0   # Buyer-initiated
        else:
            return -1.0   # Seller-initiated

    def _classify_regime(self) -> None:
        """
        Classify current market regime from microstructure state.

        Priority logic:
        1. ILLIQUID: wide spreads → max caution
        2. VOLATILE: high realized vol → reduce exposure
        3. TRENDING: strong directional OFI → lean into it
        4. RANGING: default → mean-reversion friendly
        """
        # 1. Check illiquidity
        if len(self._spreads) > 10:
            avg_spread = sum(self._spreads) / len(self._spreads)
            if avg_spread > self.illiquidity_spread_pct:
                self._regime = MarketRegime.ILLIQUID
                self._regime_weight = 0.2  # Almost halt trading
                return

        # 2. Check volatility
        if len(self._price_returns) > 50:
            realized_vol = self._compute_realized_vol()
            if self._baseline_vol == 0:
                self._baseline_vol = realized_vol  # Calibrate on first pass
            elif self._baseline_vol > 0 and realized_vol > self.volatile_threshold * self._baseline_vol:
                self._regime = MarketRegime.VOLATILE
                self._regime_weight = 0.5  # Reduce but don't halt
                return
            # Update baseline with EMA
            self._baseline_vol = 0.95 * self._baseline_vol + 0.05 * realized_vol

        # 3. Check trending via OFI
        if abs(self._ofi) > self.trending_threshold:
            self._regime = MarketRegime.TRENDING
            self._regime_weight = 1.5  # Increase exposure for directional flow
            return

        # 4. Default: ranging
        self._regime = MarketRegime.RANGING
        self._regime_weight = 1.0

    def _compute_realized_vol(self) -> float:
        """Annualized realized volatility from tick-level returns."""
        if len(self._price_returns) < 20:
            return 0.0
        returns = list(self._price_returns)
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance) if variance > 0 else 0.0
