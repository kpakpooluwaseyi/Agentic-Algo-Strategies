# RBI Swarm — Project Summary & Handoff Document
> Generated: 2026-02-25 | For use in a new chat session to resume work.

---

## Project Overview
The RBI (Research-Backtest-Implement) Swarm is a zero-budget, modular trading architecture running on a Mac, with distributed AI agents (Nanobot, PicoClaw) on a Dell WSL2 machine for heavy compute. It processes live tick data from Hyperliquid, runs multiple concurrent strategies through a signal combiner, uses an RL agent for parameter optimization, and includes a deterministic kill-switch watchdog (NanoClaw).

**Codebase**: `/Users/kpakpo/RBI_Swarm/moon-dev-ai-agents-for-trading/rbi_core/`

---

## Architecture
```
HLP WebSocket → Microstructure Engine (CVD/OFI/Regime) → Tick Buffer (SQLite WAL)
                          ↓                                        ↓
                Strategy Combiner ← PicoClaw Regime Webhook    RL Agent ← SwarmSync (Dell Nanobot)
                          ↓
                Risk Queue (dynamic equity, halt guard) ← NanoClaw Watchdog
                          ↓
                Order Dispatcher (mock/live)
```

## File Map (17 source files, 1,835 lines)

### Data Layer
| File | Lines | Purpose |
|---|---|---|
| `data/collectors/hyperliquid_ws.py` | ~230 | WebSocket feed (trades + l2Book), auto-reconnect, certifi SSL fix, on_book callback |
| `data/collectors/buffer_mgr.py` | ~135 | SQLite WAL tick buffer with batch flush + AtomicStateWriter |
| `data/microstructure_engine.py` | ~225 | **Microstructure Intelligence Engine** — CVD, OFI (Lee-Ready), regime classifier (trending/ranging/volatile/illiquid), depth aggregation |

### Strategy Layer
| File | Lines | Purpose |
|---|---|---|
| `strategy/base.py` | ~65 | Abstract BaseStrategy + Signal dataclass |
| `strategy/combiner.py` | ~100 | Confidence-weighted voting, capped ThreadPool, regime_weight support |
| `strategy/pool/ema_scalp.py` | ~30 | EMA crossover stub |
| `strategy/pool/rsi_divergence.py` | ~105 | RSI divergence reversal (tick-to-bar) |
| `strategy/pool/vwap_mean_reversion.py` | ~90 | Running VWAP deviation signals |
| `strategy/pool/bollinger_scalp.py` | ~85 | BB band-touch with z-score confidence |

### AI Layer
| File | Lines | Purpose |
|---|---|---|
| `ai/corrective_agent.py` | ~130 | RL parameter mutation: EMA-smoothed gradients, continuous reward (no cliff), revert-to-best, parameter clipping |
| `ai/swarm_sync.py` | ~85 | Pull-based epoch-versioned weight sync from Dell Nanobot + staleness detection |

### Execution Layer
| File | Lines | Purpose |
|---|---|---|
| `execution/risk.py` | ~100 | Dynamic equity position sizing, halt guard, trade queue |
| `execution/dispatcher.py` | ~55 | Mock/live order dispatch, halt-aware |

### Swarm Integration
| File | Lines | Purpose |
|---|---|---|
| `swarm_integration/nanoclaw_monitor.py` | ~85 | Deterministic local kill-switch: tick staleness, spread collapse, PnL breach |
| `swarm_integration/picoclaw_ingest.py` | ~75 | HTTP webhook listener for PicoClaw regime scores |
| `swarm_integration/nanobot_trigger.py` | ~65 | SSH WFA trigger to Dell |

### Orchestrator
| File | Lines | Purpose |
|---|---|---|
| `dashboard/run.py` | ~165 | Wires all components, heartbeat with regime/OFI/CVD display |

### Test
| File | Purpose |
|---|---|
| `tests/test_integration.py` | End-to-end live test (WS → buffer → combiner → risk → mock dispatch) |

---

## Verification Status

| Test | Result |
|---|---|
| All files compile (`py_compile`) | ✅ Pass |
| Full import chain (12 modules) | ✅ Pass |
| RL revert-to-best | ✅ Pass |
| Dynamic equity update | ✅ Pass |
| Halt guard blocks orders | ✅ Pass |
| NanoClaw stale tick detection | ✅ Pass |
| WebSocket live test (232 ticks, 15s) | ✅ Pass |
| Integration test (129 ticks, DB persistence) | ✅ Pass |
| Microstructure Engine live test (67 ticks, CVD/OFI/regime) | ✅ Pass |

---

## Key Design Decisions

1. **SQLite WAL over CSV** — WAL gives non-blocking reads for concurrent combiner/RL access, indexed queries, and crash safety.
2. **NanoClaw is local-only, deterministic** — Not an LLM agent. Pure threshold checks in a daemon thread. Survives main loop hangs.
3. **Microstructure Engine** — Lee-Ready trade classification, CVD accumulation, OFI rolling ratio, regime classifier with priority logic (illiquid > volatile > trending > ranging). Regime weight auto-feeds combiner.
4. **PicoClaw regime score is secondary** — Microstructure engine provides primary regime signal locally. PicoClaw webhook is additive (Polymarket sentiment).
5. **RL safety triple-lock** — EMA-smoothed gradients prevent oscillation, continuous reward function (no kill-switch cliff), revert-to-best on 3 consecutive negatives.

---

## What Works Now

- Run `python3 rbi_core/dashboard/run.py` (from project root, with venv activated)
- Connects to Hyperliquid WebSocket, receives live BTC+ETH ticks
- Enriches ticks with CVD, OFI, and regime classification
- Runs 4 strategies in parallel through combiner
- Routes consensus signals to mock execution queue
- NanoClaw monitors health, PicoClaw listens on port 9090
- Ctrl+C for graceful shutdown

---

## What Needs To Be Done Next (Priority Order)

### P0 — Critical Before Real Money
1. **Position Lifecycle Coupling** — Update `risk.py` to return `position_key` from `submit_signal` and track it in `run.py` via `_trade_to_position` map. This is critical for matching closed trade PnL back to the execution queue state.
2. **ATR computation from live data** — Compute ATR from aggregated candle data; system currently uses a 0.01 fallback.
3. **Trade lifecycle tracking** — Connect realized PnL from closed trades to `corrective_agent.step()` to enable RL learning loop.

### P1 — Operational & Logic
4. **Reward Proximity Fix** — Correct the `proximity_ratio` calculation in `corrective_agent.py` for negative `kill_switch_pnl_pct` bounds.
5. **Test Discoverability** — Add `pytest.mark.integration` to `test_integration.py` and implement environment variable guards (`RBI_RUN_LIVE_INTEGRATION`).
6. **Requirement Parity** — Add `websocket-client` and `certifi` to `requirements.txt`.
7. **Dell Nanobot/Pancake integration** — Finalize the FastAPI endpoints on the Dell machine.


### P2 — Enhancement
7. **Replay/backtest mode** — Add `replay_mode` to `HyperliquidWSFeed` that reads from SQLite DB for deterministic replay.
8. **Unit tests** — Individual module tests for RL agent edge cases, combiner conflict resolution, microstructure regime boundaries.
9. **Live exchange API** — Implement `dispatcher.py` live mode with Hyperliquid execution API.
10. **More strategies** — Port top-performing strategies from `src/strategies/` into `pool/`.

---

## Dependencies
- Python 3.11+
- `websocket-client`, `certifi`, `numpy` (in venv)
- `pandas`, `pandas_ta` (for strategy development, not yet in rbi_core)
- SSH access to Dell WSL2 (for Nanobot/PicoClaw integration)

## Environment Variables
- `NANOBOT_URL` — Dell Nanobot endpoint (default: `http://192.168.1.50:8080`)
- `PICOCLAW_PORT` — PicoClaw webhook listen port (default: `9090`)
- `EXECUTION_MODE` — `"mock"` or `"live"` (default: `mock`)

---

## Key Source Documents
- `docs/rbi_bmad_plan.md` — Original BMAD master plan
- `docs/blueprint_rbi_swarm.md` — Refined architecture blueprint with NanoClaw/PicoClaw scoping
- `docs/rbi_implementation_plan_v2.md` — Phase-by-phase LLM-executable implementation plan
