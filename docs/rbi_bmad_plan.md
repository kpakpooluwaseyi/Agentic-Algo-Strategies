# BMAD Master Plan: 48h RBI Zero-Budget Architecture

## 1. Background (B)
We are replacing a $1497 premium trading architecture (Moondev Quant Elite, HLP Data Layer, 20+ specialized AI agents) with a customized, zero-budget, 100% open-source local application. The target is moving a $100 account to $800 within 48 hours (50% probability), or defining a pathway to 49%+ sustainable quarterly growth. Time is locked at <60 hours. 

The edge is defined by:
1. **Tick & Liquidation Data processing** (superior to standard OHLC).
2. **Strategy Combiner** (multi-timeframe, multi-asset concurrent testing).
3. **Corrective AI (Ernest Chan methodology)**: An RL agent that dynamically mutates trading parameters (session, volume, liquidity, OI, order flow, session time, kill switches, limits, PnL, trend, long/short) to optimize Sharpe/Sortino ratios, rather than pure directional price prediction.

## 2. Mission (M)
Build and deploy a unified, thread-safe, modular, and near-autonomous Python/TypeScript orchestration dashboard that tightly bundles data collection, the multi-strategy Combiner, the Corrective AI RL engine, and execution logic.

**User Stories (RBI Alignment):**
*   **Research (R):** As an analyst, I want the system to continuously capture tick, liquidation, Hyperliquid, and Polymarket data to a memory-mapped CSV buffer to feed the combiner.
*   **Backtest (B):** As a quant, I want the Combiner to concurrently test 10+ core strategy archetypes across data subsets identically to live execution to avoid look-ahead bias.
*   **Implement/Execution (I):** As an execution engine, I want the Corrective AI (RL loop) to actively monitor live PnL, adjust ATR-based risk, scale position sizes, and prune failing strategies without manual intervention.

## 3. Architecture (A)

### Structural Guidelines
*   **Thread Safety:** Zero data races via Strict locks/Queues (`queue.Queue`, `threading.Lock`, `asyncio`).
*   **I/O:** Atomic CSV writes (write to `.tmp` then rename) for memory caching.
*   **Resiliency:** Advanced retry decorators for API rate limits and WebSockets disconnects.
*   **Risk:** Global ATR-based dynamic position sizing, portfolio kill switches.

### Directory Tree
```text
rbi_core/
├── data/
│   ├── collectors/
│   │   ├── hyperliquid_ws.py  # WebSocket feed, tick data, LQS
│   │   ├── polymarket_api.py  # Prediction market sentiment sweeps
│   │   └── buffer_mgr.py      # Thread-safe atomic CSV writers
├── strategy/
│   ├── base.py                # Abstract strategy interface
│   ├── combiner.py            # Multi-strategy concurrent evaluator
│   └── pool/                  # Implementations (EMA_Scalp, RSI_Div, MACD_AIR, etc.)
├── ai/
│   ├── rl_env.py              # Gym-like environment for state representation
│   └── corrective_agent.py    # Parameter mutation logic (Sharpe/Sortino optimization)
├── execution/
│   ├── risk.py                # ATR dynamic sizing, exposure limits
│   └── dispatcher.py          # Trade queue and exchange APIs (mock/live)
└── dashboard/
    └── run.py                 # "All-in-One" Orchestrator / Dashboard
```

## 4. Design & Core Execution Code (D)

### 4.1. Data Layer Scaffolding (Unified Interfaces & Safety)
This layer handles the high-throughput incoming data using Queues and locks.

```python
# rbi_core/data/buffer_mgr.py
import threading
import tempfile
import os
import shutil
import time

class AtomicCSVWriter:
    """Thread-safe, atomic file writer to prevent mid-read corruption by the combiners"""
    def __init__(self, filename):
        self.filename = filename
        self.lock = threading.Lock()

    def append_data(self, row_str):
        with self.lock:
            # Atomic append: write to tmp, concat to main
            # For pure append, strict lock is usually enough, but for state updates, atomic rename is preferred.
            with open(self.filename, 'a') as f:
                f.write(row_str + '\n')
            
    def write_state(self, state_str):
        """Atomic write for things like current orderbook or active parameters"""
        with self.lock:
            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(self.filename))
            with os.fdopen(fd, 'w') as f:
                f.write(state_str)
            os.replace(tmp_path, self.filename) # Atomic swap

# Example API Retry Decorator
def with_retry(max_retries=3, backoff=1.5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == max_retries - 1:
                        raise e
                    time.sleep(backoff ** i)
            return None
        return wrapper
    return decorator
```

### 4.2. Strategy Combiner
Evaluates multiple strategies concurrently.

```python
# rbi_core/strategy/combiner.py
import concurrent.futures

class StrategyCombiner:
    """Runs multiple strategies across different data subsets continuously"""
    def __init__(self, active_strategies):
        self.strategies = active_strategies  # List of instantiated strategy classes
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(active_strategies))

    def evaluate_tick(self, latest_data):
        """
        Pushes tick data to all strategies in parallel.
        Returns a aggregated list of recommended trade signals.
        """
        futures = {}
        for strat in self.strategies:
            futures[self.pool.submit(strat.on_tick, latest_data)] = strat
            
        signals = []
        for future in concurrent.futures.as_completed(futures):
            strat = futures[future]
            try:
                res = future.result()
                if res:
                   signals.append({
                       'strategy': strat.name,
                       'signal': res,
                       'confidence': strat.current_confidence
                   })
            except Exception as e:
                # Log cleanly, don't crash combiner
                print(f"Combiner Error in {strat.name}: {e}")
        return signals
```

### 4.3. Corrective AI & RL Engine (Ernest Chan Approach)
Mutates parameters stringently focusing on risk-adjusted returns (Sharpe/Sortino), NOT just accuracy.

```python
# rbi_core/ai/corrective_agent.py
import numpy as np

class CorrectiveRLAgent:
    """
    Iteratively mutates strategy parameters based on execution feedback.
    Reward function heavily penalizes drawdown to maximize Sortino.
    """
    def __init__(self):
        # Parameters we are permitted to mutate: Limits, ATR multiplier, session times, etc.
        self.active_params = {
            'atr_multiplier': 1.5,
            'max_holding_time_minutes': 60,
            'vol_threshold': 1000,
            'kill_switch_pnl_pct': -0.05
        }
        self.learning_rate = 0.05

    def step(self, current_sortino, current_pnl, action_space_gradients):
        """
        Receives the current state (Sharpe/Sortino) and adjusts active params.
        """
        reward = self._calculate_reward(current_sortino, current_pnl)
        
        # Simple policy gradient / evolutionary step update to active_params
        # If reward is negative, we revert or step randomly; if positive, we push gradients
        for param, grad in action_space_gradients.items():
            if reward > 0:
                self.active_params[param] += grad * self.learning_rate
            else:
                self.active_params[param] -= grad * self.learning_rate  # Revert / penalize direction
                
        # Constrain parameters
        self._clip_parameters()
        return self.active_params
        
    def _calculate_reward(self, sortino, pnl):
        # Heavy penalty for negative expectancy
        if pnl < self.active_params['kill_switch_pnl_pct']:
            return -100.0
        return sortino * 10.0  # Reward is strongly tied to Sortino
        
    def _clip_parameters(self):
        self.active_params['atr_multiplier'] = np.clip(self.active_params['atr_multiplier'], 0.5, 3.0)
        self.active_params['kill_switch_pnl_pct'] = np.clip(self.active_params['kill_switch_pnl_pct'], -0.1, -0.01)
```

### 4.4. Risk & Execution
Thread-safe execution queue that applies the global ATR dynamic sizing just before order transmission.

```python
# rbi_core/execution/risk.py
import queue

class PortfolioExecutionQueue:
    def __init__(self, account_size):
        self.trade_queue = queue.Queue()
        self.account_size = account_size
        self.active_positions = {}
        self.global_lock = threading.Lock()

    def calculate_position_size(self, current_price, current_atr, param_multiplier):
        """Dynamic position sizing using ATR and Corrective AI's multiplier"""
        risk_per_trade = self.account_size * 0.01  # Risk 1% of account
        stop_distance = current_atr * param_multiplier
        if stop_distance == 0:
            return 0
        return risk_per_trade / stop_distance

    def submit_signal(self, signal, params, latest_market_data):
        with self.global_lock:
            # Apply global limits and risk filter
            if len(self.active_positions) >= 5: # Max 5 concurrent trades
                return False
                
            pos_size = self.calculate_position_size(
                latest_market_data['price'], 
                latest_market_data['atr'], 
                params['atr_multiplier']
            )
            
            if pos_size > 0:
                validated_order = {
                    'action': signal['action'], # BUY/SELL
                    'size': pos_size,
                    'strategy': signal['strategy'],
                    'timestamp': time.time()
                }
                self.trade_queue.put(validated_order)
                return True
        return False
```
