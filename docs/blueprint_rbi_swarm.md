# RBI Swarm Architecture Blueprint: 48h Zero-Budget Local & Agentic Framework

## 1. Background (B)
We are executing a $0 budget, 48-hour challenge to build an institutional-grade, modular trading architecture (RBI). The goal is hyper-efficient capital scaling using Tick Data, a Strategy Combiner, and a Corrective AI (Ernest Chan RL approach). 

Simultaneously, we have active agentic instances on a Dell WSL2 server—**Nanobot** and **PicoClaw** (Research and Execution). The objective is to hybridize the local ultra-fast RBI architecture (Mac) with the distributed Dell swarm to unlock maximum computational leverage without cloud costs, preventing resource bottlenecks on the local machine while providing deep analytical and safety edge.

## 2. Mission (M)
Design the complete scaffolding for the RBI architecture while establishing a clear API/RPC boundary between the local trading node and the Dell Swarm.

**Key Objectives:**
- **Local Machine (Mac - High Frequency):** Hosts the latency-critical components: Data Buffer, Strategy Combiner, Risk Engine, Fast RL Inference, and Order Dispatcher.
- **Dell WSL2 (Swarm - Heavy Compute & Async):**
  - **Nanobot:** Offloads heavy parallel backtesting and deep Walk-Forward Analysis (WFA) to continuously train and update the Corrective AI weights, storing them for the Mac to fetch.
  - **PicoClaw (Research):** Asynchronously parses Polymarket, social sentiment, and news, piping a "Market Regime" score (0-100) via webhook to the local Combiner to adjust strategy weights.
  - **PicoClaw (Executor):** Handles non-latency-sensitive execution tasks (e.g., executing delta-neutral hedges, low-timeframe grid bot rebalancing).
- **NanoClaw (Local Watchdog Daemon — NOT an LLM Agent):** A deterministic Python watchdog thread running **exclusively on the Mac**. It monitors websocket tick staleness, bid-ask spread collapse, and PnL breach thresholds. On anomaly detection, it bypasses all standard logic and calls `dispatcher.halt()` directly. Zero LLM inference latency. Independent of the main event loop — survives combiner hangs. PicoClaw (Executor) on the Dell serves as a redundant secondary kill-switch via direct exchange API cancel-all if the Mac goes dark.

## 3. Architecture (A)
### 3.1 Component Topology
1. **Data Layer (`buffer_mgr.py`)**: Subscribes to Hyperliquid websockets. Uses simple internal Queues and `.tmp` rename atomics for threads to read without race conditions.
2. **Strategy Combiner (`combiner.py`)**: Reads the data streams and dynamically weights incoming signals from `EMA_Scalp`, `RSI_Div`, etc., adjusting them based on PicoClaw Research's regime broadcasts.
3. **Corrective AI RL Node (`corrective_agent.py`)**: Local inference clips parameters (e.g. ATR). Periodically fetches new optimal bounds compiled asynchronously by **Nanobot** on the Dell.
4. **Execution & Risk (`risk.py`, `dispatcher.py`)**: Queues orders dynamically sized, guarded by the **NanoClaw** continuous health checks.

### 3.2 Thread-Safety & Resilience Boundaries
- Communication between Local Mac and Dell WSL2 relies on lightweight APIs/Webhooks to reduce tightly-coupled crashes.
- Swarm failures (Nanobot goes down) should gracefully degrade the RBI framework (falling back to the last known good Corrective AI parameters).

## 4. Design & Scaffolding (D)
The codebase will be modularized to cleanly separate the Local Engine from Swarm connectivity.

### Directory Structure Blueprint
```text
rbi_core/
├── data/
│   ├── collectors/
│   │   ├── hyperliquid_ws.py  # Local tick-data continuous websocket
│   │   └── buffer_mgr.py      # Thread-safe atomic CSV buffers
├── strategy/
│   ├── combiner.py            # Multithreaded signal fusion
│   └── pool/                  # Core strategy implementations (EMA, MACD, etc.)
├── ai/
│   ├── corrective_agent.py    # Local fast-inference parameter mutant
│   └── swarm_sync.py          # Asynchronously fetches WFA weights from Dell Nanobot
├── execution/
│   ├── risk.py                # Local ATR sizing & exposure limits queue
│   └── dispatcher.py          # Local exchange API mock/live interactions
├── swarm_integration/         # Dell Communication Scaffolds
│   ├── nanobot_trigger.py     # Invokes extensive backtests on Dell WSL2
│   ├── picoclaw_ingest.py     # Webhook listener for PicoClaw JSON sentiment
│   └── nanoclaw_monitor.py    # Dedicated thread for instant kill-switch logic (Health & Flash crashes)
└── dashboard/
    └── run.py                 # "All-in-One" multi-threaded Orchestrator
```

### Next Steps for Implementation
1. Ensure the user confirms the structural layout and logic mapping for the swarms (especially the NanoClaw scope).
2. Scaffold all nested empty directories and files using simple `mkdir` and `touch`.
3. Add base boilerplate classes to `swarm_integration/` to outline the connection mechanics to the Dell node.
