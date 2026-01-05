# 🌙 Architecture Upgrade Walkthrough

## Summary
The Moon Dev Swarm has been upgraded from a "Script Soup" architecture to a modular, high-performance system.

---

## ✅ Phase 1: The Contract (Standardization)

### New File: `src/strategies/base.py`
```python
from backtesting import Strategy
from abc import abstractmethod

class MoonDevStrategy(Strategy):
    @abstractmethod
    def init(self): pass
    @abstractmethod
    def next(self): pass
```

### Impact
- `research_feeder.py`: Now instructs Jules to inherit from `MoonDevStrategy`
- `pr_gatekeeper.py`: Now rejects non-compliant code
- `walk_forward.py`: Recognizes both `Strategy` and `MoonDevStrategy`

---

## ✅ Phase 2: The Overseer (Orchestration)

### New File: `overseer.py`
Unified async entry point for the swarm.

**Usage:**
```bash
python3 overseer.py              # Run all agents
python3 overseer.py --runner     # Run only local_runner
python3 overseer.py --gatekeeper # Run only pr_gatekeeper
```

**Logs:** `logs/overseer.log`

---

## ✅ Phase 3: The Runner (Performance)

### Modified: `local_runner.py`
`execute_strategy()` now uses **in-memory execution** via `WalkForwardAnalyzer._run_backtest()` instead of `subprocess.run()`.

**Before:**
```python
result = subprocess.run([sys.executable, str(strategy_file)], ...)
```

**After:**
```python
results = self.wfa_analyzer._run_backtest(strategy_class, data, ...)
```

**Expected Speedup:** ~10x (eliminates Python interpreter startup overhead per test)

---

## Deferred Items
- `ProcessPoolExecutor` for parallel backtesting
- SQLite migration for leaderboard

---

## Rollback
If issues arise:
```bash
git checkout v1.0-stable
```
Or restore from `backups/20260105_stable_baseline/`
