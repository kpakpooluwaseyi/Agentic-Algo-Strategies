# Principal Architecture Audit: Moon Dev Agent Swarm

## 1. Executive Summary
**Current Vibe**: "Script Soup". The existing architecture is a collection of loosely coupled Python scripts (`local_runner.py`, `pr_gatekeeper.py`, `research_feeder.py`) that communicate primarily through the file system (logs, CSVs) and GitHub (Issues/PRs). While the "GitOps" approach (Agents interacting via PRs) is excellent for observability and human-in-the-loop safety, the execution layer is brittle, synchronous, and computationally inefficient.

**Future Vibe**: "Event-Driven Swarm". A unified, asynchronous system where agents are specialized worker nodes managed by an "Overseer". Friction is removed by replacing polling loops with event queues, and cost is optimized by caching LLM reasoning and using data-efficient formats.

---

## 2. Comprehensive Code Audit

### 🔴 Critical Inefficiencies
1.  **Synchronous Blocking Loops**:
    *   **File**: `local_runner.py`, `pr_gatekeeper.py`
    *   **Issue**: Heavy use of `time.sleep()` in `while True` loops. This blocks the entire process while waiting, preventing parallel execution of lightweight tasks (like checking prices while running a backtest).
    *   **Fix**: Migrate to `asyncio` with `await asyncio.sleep()`.

2.  **Expensive Subprocess Management**:
    *   **File**: `local_runner.py`
    *   **Issue**: `subprocess.run([sys.executable...])` is used to execute strategies. This spins up a full new Python interpreter for *every single backtest*. This is extremely CPU and memory heavy.
    *   **Fix**: Dynamically import strategy classes using `importlib` and run them within the main process (or a persistent worker pool) using `ProcessPoolExecutor`.

3.  **Data Serialization Friction**:
    *   **File**: `results/leaderboard.csv`
    *   **Issue**: Repeatedly reading/writing huge CSV files for every result update is O(N) and blocks I/O.
    *   **Fix**: Use SQLite for the leaderboard. It allows concurrent reads/writes, SQL querying for analytics, and is a single file.

4.  **Implicit Dependencies**:
    *   **File**: `requirements.txt`
    *   **Issue**: Contains unpinned or duplicate dependencies (`scipy` listed twice). `Backtesting` is capitalized (non-standard).
    *   **Fix**: Use `poetry` or `pip-tools` to lock dependencies and ensure reproducible environments.

### 🟡 Agent Redundancy & Logic Gaps
1.  **The "Strategy" Disconnect**:
    *   Strategies in `strategies/` are raw Python files. Authentication of their logic happens via "grep" (in `pr_gatekeeper`'s prompt) rather than static analysis.
    *   **Risk**: A strategy could define valid logic but fail at runtime due to API mismatches, wasting a full backtest cycle.
    *   **Fix**: Introduce a strictly typed `BaseStrategy` class (using Pydantic) that all generated strategies *must* inherit from. This allows static verification of the interface before execution.

2.  **Redundant Data Fetching**:
    *   `research_feeder.py` and potentially trading agents might be fetching overlapping data strings.
    *   **Fix**: A centralized `DataManager` agent that caches OHLCV and sentiment data, serving it to other agents.

---

## 3. Agentic Roadmap to "Frictionless Autonomy"

### Phase 1: The "Overseer" Upgrade (Week 1)
Replace the multiple terminal tabs with a single entry point.
- **Action**: Create `swarm_manager.py`.
- **Tech**: `asyncio` + `Supervisor`.
- **Function**:
    - Spawns `PrGatekeeper`, `ResearchFeeder`, and `LocalRunner` as async tasks.
    - Monitors their health (restarts if crashed).
    - Centralized logging (no more `tail -f`ing 5 different files).

### Phase 2: The "Typed" Strategy Standard (Week 1-2)
Enforce a contract for what a strategy is.
- **Action**: Define `src/strategies/base.py`.
- **Tech**: `pydantic` + `abc` (Abstract Base Classes).
- **Rule**: Agents must output code that inherits from `MoonDevStrategy`.
- **Benefit**: Zero-cost "compile time" checking of generated strategies.

### Phase 3: The "Hot-Swap" Execution Engine (Week 2)
Stop starting new Python processes for every backtest.
- **Action**: Refactor `local_runner.py`.
- **Tech**: `importlib.reload` + `ProcessPoolExecutor`.
- **Flow**:
    1. Detect new strategy file.
    2. Hot-load the module in memory.
    3. Dispatch to a warm worker process.
    4. Return results via `Queue`.
- **Speedup**: Estimated 10-20x faster backtesting throughput.

---

## 4. Triple-Check Internal Audit

### Audit 1: Logic Check
*   **Critique**: Is `asyncio` overkill?
*   **Verdict**: Not for I/O bound tasks (API calls, DB writes), but backtesting is CPU bound.
*   **Correction**: We must use `asyncio` for the *Manager* (Orchestration) but `multiprocessing` for the *Runner* (Backtesting). Mixing them incorrectly is a common trap. **Revised Plan**: The `Overseer` stays async, but delegates actual backtests to a `ProcessPool`.

### Audit 2: Cost Analysis
*   **Critique**: Will this save money?
*   **Verdict**: Yes. By catching syntax/interface errors in the `PrGatekeeper` using Type Checking (free) instead of running a full backtest cycle (expensive compute), we save resources. Also, optimizing the `research_feeder` to not just "re-read" files but store vector embeddings means fewer large context calls to Gemini.
*   **Correction**: Add a "Vector Memory" requirement to the roadmap to stop re-reading PDF tokens.

### Audit 3: Breaking Risks
*   **Critique**: Will migrating to SQLite break `leaderboard.csv` workflows?
*   **Verdict**: Yes, the user likely opens that CSV in Excel.
*   **Correction**: Keep `leaderboard.csv` as a *read-only export* generated from the SQLite DB, rather than the primary data store. This maintains backward compatibility for the user's manual workflow.

---

## 5. Final Recommendations (The "Quick Wins")

1.  **Immediate**: Run `pip-compile` to clean `requirements.txt`.
2.  **High Impact**: Refactor `pr_gatekeeper.py` to use `ast.parse` for syntax checking *before* asking Gemini to audit logic. This saves tokens on broken code.
3.  **Stability**: Implement a `StrategyValidator` class that loads a strategy and checks for `init` and `next` methods before `local_runner` attempts to execute it.
