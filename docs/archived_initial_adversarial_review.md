# Archived Adversarial Review (v1.0)
*This document contains the critical findings identified before the RBI Swarm scaffolding implementation.*

---

## 🔴 CRITICAL LOGIC FLAWS

### 1. Corrective RL Agent: Gradient Oscillation Trap
The `step()` method applied a **symmetric inversion** on negative reward (`param -= grad * lr`). This created a destructive oscillation loop.
**Fix (Implemented):** Replaced with EMA smoothing, revert-to-best logic, and parameter clipping.

### 2. `_calculate_reward()`: Kill-Switch Reward Masking
The reward previously cliffed at -100.0 regardless of Sortino.
**Fix (Implemented):** Use a continuous penalty function that scales with breach severity and penalizes proximity to the kill-switch.

### 3. Missing `import threading` in `risk.py`
Corrected to include `import threading`.

### 4. `account_size` Is Static
Corrected to use a dynamic equity property with locking.

---

## 🟡 LOGIC GAPS & EDGE CASES

### 5. Strategy Combiner: No Conflict Resolution
Fixed by adding a consensus/voting layer with confidence-weighted aggregation.

### 6. `ThreadPoolExecutor` Over-subscription
Capped workers at `os.cpu_count()`.

### 7. CSV Append Bottleneck
Replaced with SQLite WAL mode in `buffer_mgr.py` for high-throughput concurrency.

### 8. No Mac↔Dell Heartbeat
Added `swarm_degraded` flag and staleness detection in `swarm_sync.py`.

### 9. NanoClaw Placement Ambiguity
Resolved: NanoClaw is a local-only deterministic daemon in `nanoclaw_monitor.py`.

---

## 🟢 EFFICIENCY IMPROVEMENTS

### 10. SQLite WAL Migration
Implemented across the data layer.

### 11. Pull-based Weight Sync
Implemented in `swarm_sync.py`.

### 12. `strategy/base.py` ABC
Defined and verified across all strategy pool implementations.
