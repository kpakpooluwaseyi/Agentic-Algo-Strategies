# Optimal Multi-Machine Workflow: RBI Swarm + Nexus BT

This strategy leverages the **Mac M1 (8GB)** for low-latency live execution and the **Dell Latitude (16GB)** for continuous data research and heavy computation.

---

## 💻 Machine Roles

### 1. Mac M1 (The Sniper)
- **Primary Task:** `rbi_core` Live Execution / Mock Paper Trading.
- **Resource Priority:** CPU/RAM headroom for high-frequency tick processing.
- **Dependency:** Pulls "Optimal Weights" from the Dell via `swarm_sync.py`.
- **Operating Window:** 24/7 (Live Markets).

### 2. Dell Latitude (The Librarian & General)
- **Primary Task:** Data ingestion, Nexus Backtesting, and Agentic Swarm (PicoClaw/Nanobot).
- **Resource Priority:** Parallel processing and disk/memory management.
- **Operating Window:** 24/7 (Background Research).

---

## 🔄 The 24/7 Continuous Pipeline

| Phase | Timeframe | Machine | Activity |
| :--- | :--- | :--- | :--- |
| **Inference/Execution** | 24/7 | **Mac** | `rbi_core` runs strategies using parameters provided by the Dell. |
| **Data Harvesting** | 24/7 | **Dell** | Async fetchers gather OHLCV and volume data across 1,140+ symbols into the Nexus library. |
| **Shadow Backtesting** | On-Demand | **Dell** | When you change a line in a strategy rewrite, it's pushed to the Dell for a 1-minute "Smoke Backtest" to ensure no logic breakage. |
| **Deep Optimization** | Weekend | **Dell** | **Nexus BT + Optuna** runs heavy Bayesian search to find the "Perfect Window" for the coming week. |
| **WFA Validation** | Mon-Fri (Night) | **Dell** | **Nanobot** runs Walk-Forward Analysis on the previous day's data to nudge the RL agent weights. |

---

## 🛠 Optimal Resource Allocation (Dell 16GB)

To prevent the Dell from locking up, partition the 16GB RAM as follows:

1. **Base OS/WSL2:** 2GB.
2. **PicoClaw Research (Async):** 2GB (Monitoring news/sentiment).
3. **Nanobot WFA (Rolling):** 4GB (Background parameter nudging).
4. **Nexus BT Pool:** 8GB (Reserved for on-demand strategy sweeps).

### Recommended Optimization Thresholds
- **CPU:** Cap Nexus BT workers to `N-1` cores (e.g., 3 out of 4) to ensure PicoClaw doesn't drop news updates during a backtest.
- **I/O:** Use the Dell's SSD for the Nexus data library. Do not store the large historical databases on a network drive or the Mac.

---

## 🚀 The "Week Ahead" Workflow

### Friday Night (The Hand-off)
1. Mac stops `live` trading.
2. Dell launches **Nexus BT Full Sweep** across all rewrite strategies.
3. Goal: Find which assets and parameters are "Heat Mapping" best for the current trend.

### Sunday Afternoon (The Calibration)
1. Review Nexus stats (Sharpe/Sortino).
2. Export the `absolute_params.csv`.
3. Update the `CorrectiveRLAgent` weights via the Dell's Nanobot endpoint.

### Monday Morning (The Deployment)
1. Mac pulls fresh weights from Dell.
2. Mac starts in `mock` mode for the first 2 hours of the session.
3. If Microstructure (CVD/OFI) confirms the Dell's findings, flip to `live`.
