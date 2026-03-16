# RBI BMAD Implementation Plan v2 — LLM-Executable Specification

> **Scope:** Scaffolding + all 13 adversarial review fixes.
> **Root:** `/Users/kpakpo/RBI_Swarm/moon-dev-ai-agents-for-trading/rbi_core/`
> **Python:** 3.11+ (use `venv` at project root if needed)
> **Zero external infra:** No Docker, no cloud, no paid APIs.

---

## PHASE 0: Directory Scaffolding

**Goal:** Create the full directory tree and empty `__init__.py` files.

### 0.1 — Create directories

Run from project root (`/Users/kpakpo/RBI_Swarm/moon-dev-ai-agents-for-trading/`):

```bash
mkdir -p rbi_core/data/collectors
mkdir -p rbi_core/strategy/pool
mkdir -p rbi_core/ai
mkdir -p rbi_core/execution
mkdir -p rbi_core/swarm_integration
mkdir -p rbi_core/dashboard
```

### 0.2 — Create `__init__.py` files

```bash
touch rbi_core/__init__.py
touch rbi_core/data/__init__.py
touch rbi_core/data/collectors/__init__.py
touch rbi_core/strategy/__init__.py
touch rbi_core/strategy/pool/__init__.py
touch rbi_core/ai/__init__.py
touch rbi_core/execution/__init__.py
touch rbi_core/swarm_integration/__init__.py
touch rbi_core/dashboard/__init__.py
```

### 0.3 — Create all module files (empty stubs)

```bash
touch rbi_core/data/collectors/hyperliquid_ws.py
touch rbi_core/data/collectors/buffer_mgr.py
touch rbi_core/strategy/base.py
touch rbi_core/strategy/combiner.py
touch rbi_core/ai/corrective_agent.py
touch rbi_core/ai/swarm_sync.py
touch rbi_core/execution/risk.py
touch rbi_core/execution/dispatcher.py
touch rbi_core/swarm_integration/nanobot_trigger.py
touch rbi_core/swarm_integration/picoclaw_ingest.py
touch rbi_core/swarm_integration/nanoclaw_monitor.py
touch rbi_core/dashboard/run.py
```

### 0.4 — Verification

```bash
find rbi_core/ -type f | sort
# Expected: 21 files (9 __init__.py + 12 module files)
```

---

## PHASE 1: Strategy Base ABC (Fix #12)

**File:** `rbi_core/strategy/base.py`

**Why first:** Every subsequent phase depends on this contract. The combiner, pool strategies, and RL agent all reference `strategy.name`, `strategy.on_tick()`, and `strategy.current_confidence`.

### 1.1 — Write the ABC

```python
"""rbi_core/strategy/base.py — Abstract base class for all strategies."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Signal:
    """Standardized signal output from any strategy."""
    action: str          # "BUY", "SELL", or "HOLD"
    confidence: float    # 0.0 to 1.0
    meta: dict = field(default_factory=dict)  # strategy-specific metadata


class BaseStrategy(ABC):
    """
    Contract for all strategies in the pool.
    Every strategy MUST subclass this and implement on_tick().
    """

    def __init__(self, name: str):
        self.name: str = name
        self.current_confidence: float = 0.0
        self._enabled: bool = True

    @abstractmethod
    def on_tick(self, tick_data: dict) -> Optional[Signal]:
        """
        Process a single tick of market data.

        Args:
            tick_data: dict with at minimum:
                - 'price': float
                - 'volume': float
                - 'timestamp': float (unix epoch)
                - 'atr': float (current ATR value)
                - 'bid': float
                - 'ask': float

        Returns:
            Signal if the strategy has an actionable opinion, else None.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state. Called when RL agent prunes this strategy."""
        ...

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    @property
    def is_enabled(self) -> bool:
        return self._enabled
```

### 1.2 — Write one example strategy stub

**File:** `rbi_core/strategy/pool/ema_scalp.py`

```python
"""rbi_core/strategy/pool/ema_scalp.py — Example EMA Scalp strategy."""
from rbi_core.strategy.base import BaseStrategy, Signal
from typing import Optional
from collections import deque


class EMAScalp(BaseStrategy):
    """Simple EMA crossover scalper. Placeholder for real logic."""

    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        super().__init__(name="EMA_Scalp")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._prices: deque = deque(maxlen=slow_period + 1)

    def on_tick(self, tick_data: dict) -> Optional[Signal]:
        self._prices.append(tick_data['price'])
        if len(self._prices) < self.slow_period:
            return None
        # Placeholder: real EMA math goes here
        self.current_confidence = 0.0
        return None

    def reset(self) -> None:
        self._prices.clear()
        self.current_confidence = 0.0
```

### 1.3 — Verification

```bash
cd /Users/kpakpo/RBI_Swarm/moon-dev-ai-agents-for-trading
python -c "from rbi_core.strategy.base import BaseStrategy, Signal; print('ABC OK')"
python -c "from rbi_core.strategy.pool.ema_scalp import EMAScalp; s = EMAScalp(); print(f'{s.name} OK')"
```

---

## PHASE 2: Data Layer — SQLite WAL + Ring Buffer (Fix #7, #10, #13)

**Files:** `rbi_core/data/collectors/buffer_mgr.py`

### 2.1 — Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Storage engine | SQLite WAL mode | Zero read contention, 50K+ rows/sec writes, SQL queries for combiner |
| In-memory buffer | `collections.deque` ring buffer | Batch inserts every 500ms or 100 ticks (whichever first) |
| State snapshots | Atomic `.tmp` rename (keep original pattern) | Correct for small state blobs (orderbook, params) |
| Polymarket collector | **REMOVED** from local | PicoClaw Research on Dell handles this exclusively via webhook |

### 2.2 — Write `buffer_mgr.py`

```python
"""rbi_core/data/collectors/buffer_mgr.py — SQLite WAL tick buffer + atomic state writer."""
import sqlite3
import threading
import tempfile
import os
import time
from collections import deque
from typing import Optional


class TickBuffer:
    """
    High-throughput tick data storage using SQLite WAL mode.
    Batches writes from an in-memory ring buffer.
    Thread-safe: multiple readers (combiner, RL agent) never block writers.
    """

    def __init__(self, db_path: str, flush_interval_ms: int = 500, flush_batch_size: int = 100):
        self.db_path = db_path
        self.flush_interval_s = flush_interval_ms / 1000.0
        self.flush_batch_size = flush_batch_size
        self._buffer: deque = deque()
        self._lock = threading.Lock()
        self._running = False
        self._flush_thread: Optional[threading.Thread] = None

        # Initialize DB with WAL mode
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")  # Safe with WAL, faster writes
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                volume REAL NOT NULL,
                bid REAL,
                ask REAL,
                atr REAL,
                raw_json TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_ts ON ticks(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_symbol ON ticks(symbol)")
        conn.commit()
        conn.close()

    def append_tick(self, tick: dict) -> None:
        """Add tick to in-memory ring buffer. Non-blocking."""
        with self._lock:
            self._buffer.append(tick)

    def start(self) -> None:
        """Start the background flush thread."""
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def stop(self) -> None:
        """Stop flush thread and drain remaining buffer."""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
        self._flush_to_db()  # Final drain

    def _flush_loop(self) -> None:
        while self._running:
            time.sleep(self.flush_interval_s)
            self._flush_to_db()

    def _flush_to_db(self) -> None:
        # Snapshot the buffer under lock, then write outside lock
        with self._lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            self._buffer.clear()

        conn = sqlite3.connect(self.db_path)
        try:
            conn.executemany(
                "INSERT INTO ticks (timestamp, symbol, price, volume, bid, ask, atr, raw_json) "
                "VALUES (:timestamp, :symbol, :price, :volume, :bid, :ask, :atr, :raw_json)",
                batch
            )
            conn.commit()
        finally:
            conn.close()

    def query_recent(self, symbol: str, limit: int = 500) -> list:
        """Read recent ticks. Safe to call from any thread (WAL readers never block)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM ticks WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                (symbol, limit)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


class AtomicStateWriter:
    """
    Atomic file writer for small state blobs (orderbook snapshot, active params).
    Uses the .tmp rename pattern. NOT for high-throughput tick data.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._lock = threading.Lock()

    def write(self, content: str) -> None:
        with self._lock:
            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(self.filepath) or '.')
            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(content)
                os.replace(tmp_path, self.filepath)
            except Exception:
                # Clean up temp file on failure
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

    def read(self) -> Optional[str]:
        if not os.path.exists(self.filepath):
            return None
        with open(self.filepath, 'r') as f:
            return f.read()
```

### 2.3 — Write `hyperliquid_ws.py` stub

```python
"""rbi_core/data/collectors/hyperliquid_ws.py — Hyperliquid WebSocket tick feed."""
import json
import threading
import time
from typing import Callable, Optional

# NOTE: websockets or websocket-client required. Add to requirements.
# Reference: existing src/nice_funcs_hl.py for REST API patterns.

# Retry decorator (shared utility)
def with_retry(max_retries: int = 3, backoff: float = 1.5):
    """Exponential backoff retry decorator for network calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(backoff ** attempt)
            return None
        return wrapper
    return decorator


class HyperliquidWSFeed:
    """
    Connects to Hyperliquid WebSocket and pushes ticks to a TickBuffer.
    Auto-reconnects on disconnect using with_retry.
    """

    WS_URL = "wss://api.hyperliquid.xyz/ws"

    def __init__(self, symbols: list[str], on_tick: Callable[[dict], None]):
        """
        Args:
            symbols: List of symbols to subscribe (e.g., ["BTC", "ETH"])
            on_tick: Callback invoked with each tick dict. Typically TickBuffer.append_tick.
        """
        self.symbols = symbols
        self.on_tick = on_tick
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start WebSocket listener in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run_loop(self) -> None:
        """Main reconnection loop. Subclass or fill in with websocket-client logic."""
        while self._running:
            try:
                self._connect_and_listen()
            except Exception as e:
                print(f"[HLP_WS] Connection error: {e}. Reconnecting in 3s...")
                time.sleep(3.0)

    def _connect_and_listen(self) -> None:
        """
        TODO: Implement with websocket-client library.
        Subscribe to l2Book and trades channels for self.symbols.
        Parse each message into tick dict format:
            {'timestamp': float, 'symbol': str, 'price': float,
             'volume': float, 'bid': float, 'ask': float, 'atr': float, 'raw_json': str}
        Call self.on_tick(tick_dict) for each.
        """
        raise NotImplementedError("Implement WebSocket connection logic")
```

### 2.4 — Verification

```bash
python -c "from rbi_core.data.collectors.buffer_mgr import TickBuffer, AtomicStateWriter; print('Buffer OK')"
python -c "from rbi_core.data.collectors.hyperliquid_ws import HyperliquidWSFeed; print('WS OK')"
```

---

## PHASE 3: Strategy Combiner — Conflict Resolution + Thread Cap (Fix #5, #6)

**File:** `rbi_core/strategy/combiner.py`

### 3.1 — Design decisions

- **Conflict resolution:** Confidence-weighted net direction. Sum of `confidence * (+1 for BUY, -1 for SELL)`. Execute net direction only if absolute sum exceeds a configurable threshold (default: 0.3).
- **Thread cap:** `min(len(strategies), os.cpu_count() or 4)`

### 3.2 — Write `combiner.py`

```python
"""rbi_core/strategy/combiner.py — Multi-strategy concurrent evaluator with conflict resolution."""
import os
import concurrent.futures
from typing import Optional
from rbi_core.strategy.base import BaseStrategy, Signal


class CombinedSignal:
    """Aggregated, conflict-resolved output from the combiner."""
    def __init__(self, action: str, net_confidence: float, contributing_strategies: list[str]):
        self.action = action                           # "BUY", "SELL", or "HOLD"
        self.net_confidence = net_confidence
        self.contributing_strategies = contributing_strategies


class StrategyCombiner:
    """
    Runs enabled strategies in parallel on each tick.
    Resolves BUY/SELL conflicts via confidence-weighted voting.
    """

    DIRECTION_MAP = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}

    def __init__(
        self,
        strategies: list[BaseStrategy],
        consensus_threshold: float = 0.3,
        regime_weight: float = 1.0,
    ):
        """
        Args:
            strategies: Instantiated strategy objects (must subclass BaseStrategy).
            consensus_threshold: Minimum |net_score| to produce a non-HOLD signal.
            regime_weight: Multiplier from PicoClaw regime score (0.0 to 2.0).
                           Updated externally via set_regime_weight().
        """
        self.strategies = strategies
        self.consensus_threshold = consensus_threshold
        self.regime_weight = regime_weight
        max_workers = min(len(strategies), os.cpu_count() or 4)
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def set_regime_weight(self, weight: float) -> None:
        """Called by picoclaw_ingest when a new regime score arrives."""
        self.regime_weight = max(0.0, min(2.0, weight))  # Clamp [0, 2]

    def evaluate_tick(self, tick_data: dict) -> Optional[CombinedSignal]:
        """
        Push tick to all enabled strategies in parallel.
        Returns CombinedSignal if consensus is reached, else None.
        """
        enabled = [s for s in self.strategies if s.is_enabled]
        if not enabled:
            return None

        # Dispatch all strategies
        future_to_strat = {
            self._pool.submit(s.on_tick, tick_data): s
            for s in enabled
        }

        signals: list[tuple[str, float, str]] = []  # (action, confidence, strategy_name)
        for future in concurrent.futures.as_completed(future_to_strat):
            strat = future_to_strat[future]
            try:
                result: Optional[Signal] = future.result(timeout=2.0)
                if result and result.action != "HOLD":
                    signals.append((result.action, result.confidence, strat.name))
            except Exception as e:
                print(f"[Combiner] Error in {strat.name}: {e}")

        if not signals:
            return None

        # Confidence-weighted voting
        net_score = 0.0
        for action, confidence, _ in signals:
            direction = self.DIRECTION_MAP.get(action, 0.0)
            net_score += direction * confidence * self.regime_weight

        if abs(net_score) < self.consensus_threshold:
            return None  # No consensus — HOLD

        final_action = "BUY" if net_score > 0 else "SELL"
        contributors = [name for action, _, name in signals if self.DIRECTION_MAP.get(action, 0) * (1 if final_action == "BUY" else -1) > 0]

        return CombinedSignal(
            action=final_action,
            net_confidence=abs(net_score),
            contributing_strategies=contributors,
        )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
```

### 3.3 — Verification

```bash
python -c "
from rbi_core.strategy.combiner import StrategyCombiner, CombinedSignal
from rbi_core.strategy.pool.ema_scalp import EMAScalp
c = StrategyCombiner([EMAScalp()])
result = c.evaluate_tick({'price': 100, 'volume': 1, 'timestamp': 0, 'atr': 1, 'bid': 99.9, 'ask': 100.1})
print(f'Combiner OK, result={result}')
"
```

---

## PHASE 4: Corrective AI — Rewrite RL Agent (Fix #1, #2)

**File:** `rbi_core/ai/corrective_agent.py`

### 4.1 — Design decisions for fixes

| Original flaw | Fix applied |
|---|---|
| Symmetric gradient inversion (oscillation) | EMA-smoothed gradients + revert-to-best on consecutive negative rewards |
| Kill-switch reward cliff (-100 flat) | Continuous breach-severity penalty + proximity penalty |
| No parameter history | Maintain `_best_params` snapshot, revert on degradation |

### 4.2 — Write `corrective_agent.py`

```python
"""rbi_core/ai/corrective_agent.py — RL parameter mutation with Sortino optimization."""
import copy
import numpy as np
from typing import Optional


class CorrectiveRLAgent:
    """
    Iteratively mutates strategy/execution parameters based on live PnL feedback.
    Reward function: continuous Sortino-based with proximity penalty near kill-switch.

    Key safety mechanisms:
    - EMA-smoothed gradients prevent oscillation.
    - Revert-to-best on consecutive negative rewards.
    - Hard clip bounds on all parameters.
    """

    PARAM_BOUNDS = {
        'atr_multiplier':            (0.5, 3.0),
        'max_holding_time_minutes':  (5, 480),
        'vol_threshold':             (100, 50000),
        'kill_switch_pnl_pct':       (-0.10, -0.01),
    }

    def __init__(self, learning_rate: float = 0.05, ema_alpha: float = 0.3,
                 max_consecutive_negatives: int = 3):
        self.learning_rate = learning_rate
        self.ema_alpha = ema_alpha  # Smoothing factor for gradient EMA
        self.max_consecutive_negatives = max_consecutive_negatives

        self.active_params: dict[str, float] = {
            'atr_multiplier': 1.5,
            'max_holding_time_minutes': 60,
            'vol_threshold': 1000,
            'kill_switch_pnl_pct': -0.05,
        }

        # Safety state
        self._best_params: dict[str, float] = copy.deepcopy(self.active_params)
        self._best_reward: float = float('-inf')
        self._consecutive_negatives: int = 0
        self._grad_ema: dict[str, float] = {k: 0.0 for k in self.active_params}
        self.swarm_degraded: bool = False  # Set True if Dell sync fails

    def step(self, current_sortino: float, current_pnl: float,
             raw_gradients: dict[str, float]) -> dict[str, float]:
        """
        Main RL step. Called after each trade batch or time interval.

        Args:
            current_sortino: Rolling Sortino ratio of recent trades.
            current_pnl: Current PnL as fraction of account (e.g., -0.03 = -3%).
            raw_gradients: Dict of param_name -> raw gradient estimate.

        Returns:
            Updated active_params dict.
        """
        reward = self._calculate_reward(current_sortino, current_pnl)

        # Track consecutive negatives
        if reward < 0:
            self._consecutive_negatives += 1
        else:
            self._consecutive_negatives = 0

        # Revert-to-best on sustained degradation
        if self._consecutive_negatives >= self.max_consecutive_negatives:
            self.active_params = copy.deepcopy(self._best_params)
            self._consecutive_negatives = 0
            self._grad_ema = {k: 0.0 for k in self.active_params}  # Reset EMA
            return copy.deepcopy(self.active_params)

        # Update best snapshot
        if reward > self._best_reward:
            self._best_reward = reward
            self._best_params = copy.deepcopy(self.active_params)

        # EMA-smooth gradients to prevent oscillation
        for param, raw_grad in raw_gradients.items():
            if param not in self._grad_ema:
                continue
            self._grad_ema[param] = (
                self.ema_alpha * raw_grad +
                (1 - self.ema_alpha) * self._grad_ema[param]
            )

        # Apply smoothed gradients scaled by reward magnitude
        reward_scale = np.clip(reward / 10.0, -1.0, 1.0)  # Normalize
        for param in self.active_params:
            if param in self._grad_ema:
                self.active_params[param] += (
                    self._grad_ema[param] * self.learning_rate * reward_scale
                )

        self._clip_parameters()
        return copy.deepcopy(self.active_params)

    def _calculate_reward(self, sortino: float, pnl: float) -> float:
        """
        Continuous reward function with kill-switch proximity penalty.
        No cliff — severity scales smoothly.
        """
        kill_pct = self.active_params['kill_switch_pnl_pct']

        # If breached: scale penalty by severity of breach
        if pnl < kill_pct:
            breach_severity = (kill_pct - pnl) / abs(kill_pct)
            return -100.0 * (1.0 + breach_severity)

        # Proximity penalty: penalize being CLOSE to kill-switch even if not breached
        if kill_pct != 0:
            proximity_ratio = max(0.0, 1.0 - (pnl / kill_pct))  # 0=far, 1=at threshold
            proximity_penalty = proximity_ratio * 5.0
        else:
            proximity_penalty = 0.0

        return (sortino * 10.0) - proximity_penalty

    def _clip_parameters(self) -> None:
        for param, (lo, hi) in self.PARAM_BOUNDS.items():
            if param in self.active_params:
                self.active_params[param] = float(np.clip(self.active_params[param], lo, hi))

    def load_weights_from_swarm(self, swarm_params: dict[str, float]) -> None:
        """
        Apply externally computed parameter bounds from Nanobot WFA.
        Only updates bounds if swarm is not degraded.
        """
        if self.swarm_degraded:
            return
        # Merge: swarm provides updated bounds, we apply them
        for param, value in swarm_params.items():
            if param in self.PARAM_BOUNDS:
                lo, hi = self.PARAM_BOUNDS[param]
                self.active_params[param] = float(np.clip(value, lo, hi))
        self._best_params = copy.deepcopy(self.active_params)
        self._best_reward = float('-inf')  # Reset best after external update
```

### 4.3 — Verification

```bash
python -c "
from rbi_core.ai.corrective_agent import CorrectiveRLAgent
agent = CorrectiveRLAgent()
# Normal step
p = agent.step(1.5, -0.02, {'atr_multiplier': 0.1, 'vol_threshold': 10})
print(f'Step 1 OK: atr_mult={p[\"atr_multiplier\"]:.3f}')
# Simulate 3 consecutive negatives to trigger revert
for _ in range(3):
    p = agent.step(-2.0, -0.08, {'atr_multiplier': -0.5})
print(f'Revert OK: atr_mult={p[\"atr_multiplier\"]:.3f} (should be 1.500)')
print('RL Agent OK')
"
```

---

## PHASE 5: Execution & Risk — Dynamic Equity + Missing Import (Fix #3, #4)

**File:** `rbi_core/execution/risk.py`

### 5.1 — Write `risk.py`

```python
"""rbi_core/execution/risk.py — Dynamic position sizing + execution queue."""
import threading
import queue
import time
from typing import Optional


class PortfolioExecutionQueue:
    """
    Thread-safe execution queue.
    Position sizing uses LIVE account equity, not static initial value.
    """

    def __init__(self, initial_equity: float, max_concurrent: int = 5,
                 risk_per_trade_pct: float = 0.01):
        """
        Args:
            initial_equity: Starting account size in USD.
            max_concurrent: Max simultaneous open positions.
            risk_per_trade_pct: Fraction of equity risked per trade (0.01 = 1%).
        """
        self.trade_queue: queue.Queue = queue.Queue()
        self._equity: float = initial_equity
        self._equity_lock = threading.Lock()
        self.active_positions: dict[str, dict] = {}
        self._positions_lock = threading.Lock()
        self.max_concurrent = max_concurrent
        self.risk_per_trade_pct = risk_per_trade_pct
        self.halted: bool = False  # NanoClaw sets this

    @property
    def equity(self) -> float:
        with self._equity_lock:
            return self._equity

    def update_equity(self, new_equity: float) -> None:
        """Called after each trade close or periodic balance fetch."""
        with self._equity_lock:
            self._equity = new_equity

    def halt(self) -> None:
        """Emergency halt. Called by NanoClaw watchdog."""
        self.halted = True

    def resume(self) -> None:
        self.halted = False

    def calculate_position_size(self, current_atr: float, atr_multiplier: float) -> float:
        """
        Dynamic position sizing: risk_per_trade_pct of CURRENT equity.
        Returns size in units (contracts/coins).
        """
        risk_amount = self.equity * self.risk_per_trade_pct
        stop_distance = current_atr * atr_multiplier
        if stop_distance <= 0:
            return 0.0
        return risk_amount / stop_distance

    def submit_signal(self, signal: dict, params: dict,
                      market_data: dict) -> bool:
        """
        Validate and queue a trade signal.

        Args:
            signal: {'action': 'BUY'|'SELL', 'strategy': str, ...}
            params: Active RL params (must contain 'atr_multiplier').
            market_data: {'price': float, 'atr': float, ...}

        Returns:
            True if queued, False if rejected.
        """
        if self.halted:
            return False

        with self._positions_lock:
            if len(self.active_positions) >= self.max_concurrent:
                return False

            pos_size = self.calculate_position_size(
                market_data['atr'],
                params['atr_multiplier'],
            )
            if pos_size <= 0:
                return False

            order = {
                'action': signal['action'],
                'size': pos_size,
                'strategy': signal.get('strategy', 'unknown'),
                'price': market_data['price'],
                'timestamp': time.time(),
            }
            self.trade_queue.put(order)
            self.active_positions[f"{signal['strategy']}_{time.time()}"] = order
            return True

    def close_position(self, position_key: str, realized_pnl: float) -> None:
        """Remove position and update equity."""
        with self._positions_lock:
            self.active_positions.pop(position_key, None)
        self.update_equity(self.equity + realized_pnl)
```

### 5.2 — Write `dispatcher.py` stub

```python
"""rbi_core/execution/dispatcher.py — Order dispatch to exchange API (mock/live)."""
import threading
import time
from typing import Optional


class OrderDispatcher:
    """
    Consumes orders from PortfolioExecutionQueue.trade_queue.
    In mock mode: simulates fills. In live mode: hits exchange API.
    """

    def __init__(self, execution_queue, mode: str = "mock"):
        """
        Args:
            execution_queue: PortfolioExecutionQueue instance.
            mode: "mock" for paper trading, "live" for real execution.
        """
        self.queue = execution_queue
        self.mode = mode
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _dispatch_loop(self) -> None:
        while self._running:
            try:
                order = self.queue.trade_queue.get(timeout=1.0)
                if self.queue.halted:
                    print(f"[Dispatcher] HALTED — dropping order: {order}")
                    continue
                self._execute(order)
            except Exception:
                pass  # queue.Empty on timeout

    def _execute(self, order: dict) -> None:
        if self.mode == "mock":
            print(f"[MockExec] {order['action']} {order['size']:.4f} @ {order['price']:.2f} "
                  f"({order['strategy']})")
        else:
            # TODO: Implement live exchange API call
            raise NotImplementedError("Live execution not implemented")
```

### 5.3 — Verification

```bash
python -c "
from rbi_core.execution.risk import PortfolioExecutionQueue
q = PortfolioExecutionQueue(100.0)
assert q.equity == 100.0
q.update_equity(110.0)
assert q.equity == 110.0
size = q.calculate_position_size(current_atr=2.0, atr_multiplier=1.5)
assert size > 0
print(f'Risk OK: equity=110, pos_size={size:.4f}')
# Test halt
q.halt()
result = q.submit_signal({'action':'BUY','strategy':'test'}, {'atr_multiplier':1.5}, {'price':100,'atr':2})
assert result == False
print('Halt guard OK')
"
```

---

## PHASE 6: Swarm Integration — Heartbeat, Pull-Sync, Ingest (Fix #8, #11)

### 6.1 — `swarm_sync.py` — Pull-based epoch-versioned weight sync

**File:** `rbi_core/ai/swarm_sync.py`

```python
"""rbi_core/ai/swarm_sync.py — Async pull-based weight sync from Dell Nanobot."""
import json
import threading
import time
from typing import Callable, Optional
import urllib.request
import urllib.error


class SwarmWeightSync:
    """
    Periodically polls Dell Nanobot for updated RL parameter bounds.
    Uses epoch-versioned polling: only fetches if Dell has a newer epoch.
    Gracefully degrades on failure (sets swarm_degraded flag on RL agent).
    """

    def __init__(
        self,
        nanobot_url: str,           # e.g., "http://192.168.1.50:8080"
        on_weights_updated: Callable[[dict], None],  # CorrectiveRLAgent.load_weights_from_swarm
        on_degraded: Callable[[bool], None],          # sets agent.swarm_degraded
        poll_interval_s: float = 60.0,
        staleness_threshold_s: float = 300.0,  # 5 minutes
    ):
        self.nanobot_url = nanobot_url.rstrip('/')
        self.on_weights_updated = on_weights_updated
        self.on_degraded = on_degraded
        self.poll_interval_s = poll_interval_s
        self.staleness_threshold_s = staleness_threshold_s
        self._current_epoch: int = 0
        self._last_success_ts: float = time.time()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._poll_once()
            except Exception as e:
                print(f"[SwarmSync] Poll error: {e}")
            self._check_staleness()
            time.sleep(self.poll_interval_s)

    def _poll_once(self) -> None:
        url = f"{self.nanobot_url}/weights/latest?since_epoch={self._current_epoch}"
        try:
            req = urllib.request.Request(url, method='GET')
            req.add_header('Accept', 'application/json')
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"[SwarmSync] Network error: {e}")
            return

        new_epoch = data.get('epoch', 0)
        if new_epoch > self._current_epoch:
            weights = data.get('params', {})
            self.on_weights_updated(weights)
            self._current_epoch = new_epoch
            self._last_success_ts = time.time()
            self.on_degraded(False)  # Clear degraded flag
            print(f"[SwarmSync] Updated to epoch {new_epoch}")

    def _check_staleness(self) -> None:
        elapsed = time.time() - self._last_success_ts
        if elapsed > self.staleness_threshold_s:
            self.on_degraded(True)
            print(f"[SwarmSync] WARNING: Stale for {elapsed:.0f}s. Swarm degraded.")
```

### 6.2 — `nanobot_trigger.py` — SSH-based backtest trigger

**File:** `rbi_core/swarm_integration/nanobot_trigger.py`

```python
"""rbi_core/swarm_integration/nanobot_trigger.py — Trigger batch backtests on Dell via SSH."""
import subprocess
import json
from typing import Optional


class NanobotTrigger:
    """
    Sends backtest jobs to Dell Nanobot instance via SSH.
    Dell runs WFA and publishes updated weights to its HTTP endpoint.
    """

    def __init__(self, dell_host: str, dell_user: str, dell_ssh_key: str,
                 remote_script: str = "~/nanobot/run_wfa.py"):
        """
        Args:
            dell_host: IP or hostname of Dell WSL2 (e.g., "192.168.1.50")
            dell_user: SSH username on Dell
            dell_ssh_key: Path to SSH private key
            remote_script: Path to WFA runner script on Dell
        """
        self.dell_host = dell_host
        self.dell_user = dell_user
        self.dell_ssh_key = dell_ssh_key
        self.remote_script = remote_script

    def trigger_wfa(self, strategy_name: str, params_json: str,
                    data_range: str = "30d") -> Optional[str]:
        """
        Trigger Walk-Forward Analysis on Dell.

        Args:
            strategy_name: Name of strategy to optimize.
            params_json: JSON string of current parameter bounds.
            data_range: Lookback period for WFA (e.g., "30d", "90d").

        Returns:
            Job ID string if successful, None on failure.
        """
        cmd = [
            "ssh", "-i", self.dell_ssh_key,
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no",
            f"{self.dell_user}@{self.dell_host}",
            f"python3 {self.remote_script} "
            f"--strategy {strategy_name} "
            f"--params '{params_json}' "
            f"--range {data_range} "
            f"--output-endpoint /weights/latest"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                output = result.stdout.strip()
                print(f"[NanobotTrigger] WFA triggered: {output}")
                return output
            else:
                print(f"[NanobotTrigger] SSH error: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print("[NanobotTrigger] SSH timeout")
            return None
```

### 6.3 — `picoclaw_ingest.py` — Webhook listener for regime scores

**File:** `rbi_core/swarm_integration/picoclaw_ingest.py`

```python
"""rbi_core/swarm_integration/picoclaw_ingest.py — HTTP listener for PicoClaw regime scores."""
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable, Optional


class RegimeHandler(BaseHTTPRequestHandler):
    """Handles POST /regime with JSON body: {"score": 0-100, "label": "trending|ranging|volatile"}"""

    callback: Optional[Callable] = None  # Set by PicoClawIngest

    def do_POST(self):
        if self.path != '/regime':
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            score = float(data.get('score', 50))
            # Normalize: score 0-100 → weight 0.0-2.0
            # 50 = neutral (1.0), 100 = max aggression (2.0), 0 = full defense (0.0)
            weight = score / 50.0
            if self.callback:
                self.callback(weight)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        except (json.JSONDecodeError, ValueError) as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f'{{"error":"{e}"}}'.encode())

    def log_message(self, format, *args):
        pass  # Suppress default logging


class PicoClawIngest:
    """
    Runs a lightweight HTTP server to receive regime score webhooks
    from PicoClaw Research agent on the Dell.
    """

    def __init__(self, port: int, on_regime_update: Callable[[float], None]):
        """
        Args:
            port: Port to listen on (e.g., 9090).
            on_regime_update: Callback with new regime weight. Typically combiner.set_regime_weight.
        """
        self.port = port
        self.on_regime_update = on_regime_update
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        RegimeHandler.callback = self.on_regime_update
        self._server = HTTPServer(('0.0.0.0', self.port), RegimeHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[PicoClawIngest] Listening on :{self.port}/regime")

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
```

### 6.4 — Verification

```bash
python -c "from rbi_core.ai.swarm_sync import SwarmWeightSync; print('SwarmSync OK')"
python -c "from rbi_core.swarm_integration.nanobot_trigger import NanobotTrigger; print('NanobotTrigger OK')"
python -c "from rbi_core.swarm_integration.picoclaw_ingest import PicoClawIngest; print('PicoClawIngest OK')"
```

---

## PHASE 7: NanoClaw Watchdog Daemon (Fix #9 — Refined)

**File:** `rbi_core/swarm_integration/nanoclaw_monitor.py`

### 7.1 — Design

NanoClaw is **NOT** an LLM agent. It is a deterministic watchdog daemon:
- Runs as a local thread on the Mac.
- Monitors: websocket liveness, tick staleness, spread collapse, PnL threshold.
- On anomaly: calls `dispatcher_queue.halt()` directly. Zero LLM inference latency.
- Independent of the main event loop — survives combiner hangs.

```python
"""rbi_core/swarm_integration/nanoclaw_monitor.py — Deterministic local kill-switch watchdog."""
import threading
import time
from typing import Callable, Optional


class NanoClawMonitor:
    """
    Lightweight local watchdog daemon. No LLM. Pure threshold checks.
    Independently monitors system health and triggers emergency halt.

    Monitors:
    1. Tick staleness (no new tick for N seconds = feed dead)
    2. Spread collapse (ask - bid < threshold = flash crash / no liquidity)
    3. PnL breach (account drawdown exceeds limit)
    4. Heartbeat to Dell swarm (optional secondary check)
    """

    def __init__(
        self,
        halt_callback: Callable[[], None],           # PortfolioExecutionQueue.halt
        get_last_tick_ts: Callable[[], float],        # Returns timestamp of latest tick
        get_current_spread: Callable[[], float],      # Returns current bid-ask spread
        get_current_pnl: Callable[[], float],         # Returns current PnL as fraction
        check_interval_s: float = 1.0,                # Check every 1 second
        tick_stale_threshold_s: float = 10.0,         # No tick for 10s = stale
        min_spread: float = 0.0001,                   # Spread below this = danger
        pnl_kill_threshold: float = -0.08,            # -8% account drawdown = halt
    ):
        self.halt_callback = halt_callback
        self.get_last_tick_ts = get_last_tick_ts
        self.get_current_spread = get_current_spread
        self.get_current_pnl = get_current_pnl
        self.check_interval_s = check_interval_s
        self.tick_stale_threshold_s = tick_stale_threshold_s
        self.min_spread = min_spread
        self.pnl_kill_threshold = pnl_kill_threshold
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._halt_triggered = False

    def start(self) -> None:
        self._running = True
        self._halt_triggered = False
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("[NanoClaw] Watchdog started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        print("[NanoClaw] Watchdog stopped")

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                self._check_all()
            except Exception as e:
                # NanoClaw must NEVER crash. Log and continue.
                print(f"[NanoClaw] Internal error (non-fatal): {e}")
            time.sleep(self.check_interval_s)

    def _check_all(self) -> None:
        if self._halt_triggered:
            return  # Already halted, don't spam

        # Check 1: Tick staleness
        last_tick = self.get_last_tick_ts()
        tick_age = time.time() - last_tick
        if tick_age > self.tick_stale_threshold_s:
            self._trigger_halt(f"TICK STALE: No tick for {tick_age:.1f}s")
            return

        # Check 2: Spread collapse
        spread = self.get_current_spread()
        if spread < self.min_spread:
            self._trigger_halt(f"SPREAD COLLAPSE: spread={spread:.6f}")
            return

        # Check 3: PnL breach
        pnl = self.get_current_pnl()
        if pnl < self.pnl_kill_threshold:
            self._trigger_halt(f"PNL BREACH: pnl={pnl:.4f} < kill={self.pnl_kill_threshold}")
            return

    def _trigger_halt(self, reason: str) -> None:
        print(f"[NanoClaw] *** EMERGENCY HALT *** Reason: {reason}")
        self._halt_triggered = True
        self.halt_callback()
```

### 7.2 — Verification

```bash
python -c "
import time
from rbi_core.swarm_integration.nanoclaw_monitor import NanoClawMonitor

halted = [False]
def mock_halt(): halted[0] = True
last_ts = [time.time()]

nc = NanoClawMonitor(
    halt_callback=mock_halt,
    get_last_tick_ts=lambda: last_ts[0],
    get_current_spread=lambda: 0.01,
    get_current_pnl=lambda: -0.02,
    check_interval_s=0.1,
    tick_stale_threshold_s=0.5,
)
nc.start()
time.sleep(1.0)  # Tick goes stale after 0.5s
nc.stop()
assert halted[0] == True, 'Halt should have triggered on stale tick'
print('NanoClaw OK: halt triggered on stale tick')
"
```

---

## PHASE 8: Dashboard Orchestrator

**File:** `rbi_core/dashboard/run.py`

### 8.1 — Write the orchestrator

```python
"""rbi_core/dashboard/run.py — All-in-One orchestrator wiring all components."""
import os
import sys
import time
import signal

# Ensure rbi_core is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rbi_core.data.collectors.buffer_mgr import TickBuffer
from rbi_core.data.collectors.hyperliquid_ws import HyperliquidWSFeed
from rbi_core.strategy.combiner import StrategyCombiner
from rbi_core.strategy.pool.ema_scalp import EMAScalp
from rbi_core.ai.corrective_agent import CorrectiveRLAgent
from rbi_core.ai.swarm_sync import SwarmWeightSync
from rbi_core.execution.risk import PortfolioExecutionQueue
from rbi_core.execution.dispatcher import OrderDispatcher
from rbi_core.swarm_integration.picoclaw_ingest import PicoClawIngest
from rbi_core.swarm_integration.nanoclaw_monitor import NanoClawMonitor


# === CONFIGURATION ===
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ticks.db')
SYMBOLS = ["BTC", "ETH"]
INITIAL_EQUITY = 100.0
NANOBOT_URL = os.environ.get("NANOBOT_URL", "http://192.168.1.50:8080")
PICOCLAW_PORT = int(os.environ.get("PICOCLAW_PORT", "9090"))
EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "mock")  # "mock" or "live"


def main():
    print("=" * 60)
    print("  RBI CORE — All-in-One Orchestrator")
    print("=" * 60)

    # --- 1. Data Layer ---
    tick_buffer = TickBuffer(DB_PATH)
    tick_buffer.start()

    ws_feed = HyperliquidWSFeed(
        symbols=SYMBOLS,
        on_tick=tick_buffer.append_tick,
    )
    # ws_feed.start()  # Uncomment when WebSocket is implemented

    # --- 2. Strategy Layer ---
    strategies = [
        EMAScalp(),
        # Add more strategy instances here
    ]
    combiner = StrategyCombiner(strategies)

    # --- 3. AI Layer ---
    rl_agent = CorrectiveRLAgent()

    swarm_sync = SwarmWeightSync(
        nanobot_url=NANOBOT_URL,
        on_weights_updated=rl_agent.load_weights_from_swarm,
        on_degraded=lambda degraded: setattr(rl_agent, 'swarm_degraded', degraded),
    )
    swarm_sync.start()

    # --- 4. Execution Layer ---
    exec_queue = PortfolioExecutionQueue(INITIAL_EQUITY)
    dispatcher = OrderDispatcher(exec_queue, mode=EXECUTION_MODE)
    dispatcher.start()

    # --- 5. Swarm Integration ---
    picoclaw = PicoClawIngest(
        port=PICOCLAW_PORT,
        on_regime_update=combiner.set_regime_weight,
    )
    picoclaw.start()

    # Track state for NanoClaw callbacks
    _last_tick_ts = [time.time()]
    _current_spread = [0.01]
    _current_pnl = [0.0]

    def _on_tick_for_nanoclaw(tick: dict):
        """Wrapper: feeds tick buffer AND updates NanoClaw health trackers."""
        tick_buffer.append_tick(tick)
        _last_tick_ts[0] = time.time()
        bid = tick.get('bid', 0)
        ask = tick.get('ask', 0)
        if bid and ask:
            _current_spread[0] = ask - bid

    nanoclaw = NanoClawMonitor(
        halt_callback=exec_queue.halt,
        get_last_tick_ts=lambda: _last_tick_ts[0],
        get_current_spread=lambda: _current_spread[0],
        get_current_pnl=lambda: _current_pnl[0],
    )
    nanoclaw.start()

    # --- 6. Main Loop ---
    def shutdown(signum=None, frame=None):
        print("\n[Orchestrator] Shutting down...")
        nanoclaw.stop()
        picoclaw.stop()
        swarm_sync.stop()
        dispatcher.stop()
        # ws_feed.stop()
        tick_buffer.stop()
        combiner.shutdown()
        print("[Orchestrator] Clean shutdown complete.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("[Orchestrator] All systems online. Ctrl+C to stop.")
    print(f"  Execution mode: {EXECUTION_MODE}")
    print(f"  Nanobot URL: {NANOBOT_URL}")
    print(f"  PicoClaw port: {PICOCLAW_PORT}")
    print(f"  NanoClaw watchdog: active")

    # Placeholder event loop — TODO: replace with real tick processing
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
```

### 8.2 — Verification

```bash
python -c "
import sys, os
sys.path.insert(0, '/Users/kpakpo/RBI_Swarm/moon-dev-ai-agents-for-trading')
from rbi_core.dashboard.run import main
print('Orchestrator imports OK')
"
```

---

## FINAL VERIFICATION CHECKLIST

Run all of these sequentially after all phases are implemented:

```bash
cd /Users/kpakpo/RBI_Swarm/moon-dev-ai-agents-for-trading

# 1. Structure check — expect 21 files
echo "=== STRUCTURE ===" && find rbi_core/ -type f | sort && echo ""

# 2. Syntax check — all files must compile cleanly
echo "=== SYNTAX ===" && find rbi_core/ -name "*.py" -exec python -m py_compile {} \; && echo "All files compile OK"

# 3. Import chain — full dependency tree
echo "=== IMPORTS ===" && python -c "
from rbi_core.strategy.base import BaseStrategy, Signal
from rbi_core.strategy.pool.ema_scalp import EMAScalp
from rbi_core.strategy.combiner import StrategyCombiner, CombinedSignal
from rbi_core.data.collectors.buffer_mgr import TickBuffer, AtomicStateWriter
from rbi_core.data.collectors.hyperliquid_ws import HyperliquidWSFeed
from rbi_core.ai.corrective_agent import CorrectiveRLAgent
from rbi_core.ai.swarm_sync import SwarmWeightSync
from rbi_core.execution.risk import PortfolioExecutionQueue
from rbi_core.execution.dispatcher import OrderDispatcher
from rbi_core.swarm_integration.nanobot_trigger import NanobotTrigger
from rbi_core.swarm_integration.picoclaw_ingest import PicoClawIngest
from rbi_core.swarm_integration.nanoclaw_monitor import NanoClawMonitor
print('All 12 modules import successfully')
"

# 4. Functional smoke tests
echo "=== SMOKE TESTS ===" && python -c "
import time
from rbi_core.ai.corrective_agent import CorrectiveRLAgent
from rbi_core.execution.risk import PortfolioExecutionQueue
from rbi_core.strategy.combiner import StrategyCombiner
from rbi_core.strategy.pool.ema_scalp import EMAScalp
from rbi_core.swarm_integration.nanoclaw_monitor import NanoClawMonitor

# Test 1: RL agent revert-to-best
agent = CorrectiveRLAgent()
for _ in range(3):
    agent.step(-2.0, -0.08, {'atr_multiplier': -0.5})
assert agent.active_params['atr_multiplier'] == 1.5, 'Revert failed'
print('  RL revert-to-best: PASS')

# Test 2: Dynamic equity
q = PortfolioExecutionQueue(100.0)
q.update_equity(150.0)
assert q.equity == 150.0
print('  Dynamic equity: PASS')

# Test 3: Halt guard
q.halt()
ok = q.submit_signal({'action':'BUY','strategy':'t'}, {'atr_multiplier':1.5}, {'price':100,'atr':2})
assert ok == False
print('  Halt guard: PASS')

# Test 4: NanoClaw stale tick detection
halted = [False]
nc = NanoClawMonitor(
    halt_callback=lambda: halted.__setitem__(0, True),
    get_last_tick_ts=lambda: time.time() - 20,  # 20s stale
    get_current_spread=lambda: 0.01,
    get_current_pnl=lambda: 0.0,
    check_interval_s=0.1, tick_stale_threshold_s=5.0,
)
nc.start(); time.sleep(0.5); nc.stop()
assert halted[0] == True
print('  NanoClaw stale detection: PASS')

print('All smoke tests PASSED')
"
```

**The plan is complete if and only if ALL 4 verification sections above produce zero errors.**

---

## DEPENDENCY MAP

```mermaid
graph TD
    A["hyperliquid_ws.py"] -->|ticks| B["buffer_mgr.py (TickBuffer)"]
    B -->|query_recent| C["combiner.py"]
    C -->|CombinedSignal| D["risk.py (PortfolioExecutionQueue)"]
    D -->|orders| E["dispatcher.py"]
    F["corrective_agent.py"] -->|active_params| D
    G["swarm_sync.py"] -->|weights| F
    H["picoclaw_ingest.py"] -->|regime_weight| C
    I["nanoclaw_monitor.py"] -->|halt()| D
    J["nanobot_trigger.py"] -.->|SSH trigger| G
```
