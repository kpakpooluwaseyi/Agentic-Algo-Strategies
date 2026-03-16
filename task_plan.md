# RBI System Architecture Plan

## Background
We need to replace a $1497 premium stack (Moondev Quant Elite, HLP Data Layer, AI Agents, Snipers) with a $0 budget, open-source, local solution within 48 hours. Target is $100 -> $800 or 49%+ sustainable quarterly growth.

## Mission
Deploy a multi-asset RL-driven trading system and dashboard using the RBI (Research, Backtest, Implement) methodology. The core edge relies on Tick Data, a Strategy Combiner for multi-asset testing, and an RL-based Corrective AI optimization engine.

## Phases

### Phase 1: Planning and Architecture (Current)
- [ ] Define BMAD master plan (Background, Mission, Architecture, Design)
- [ ] Detail Epics and User Stories mapping to RBI
- [ ] Outline Directory Tree for the new architecture
- [ ] Code the interface scaffolding and initial RL logic

### Phase 2: Data Layer Scaffolding
- [ ] Unified interfaces for Webhook/API/REST/WS handlers
- [ ] Thread-safe collector for Hyperliquid data
- [ ] Thread-safe Polymarket sweep integration
- [ ] Tick data collection and handling
- [ ] Atmoic CSV writes for memory caching

### Phase 3: Multi-Asset Strategy Combiner
- [ ] Stub execution classes (EMA Scalp, Ribbon, Markov, MACD+EMA+AIR, RSI Div, Breakout, Adaptive RSI, Momentum, Mean Reversion, Trend Following)
- [ ] Combiner function for concurrent testing across data subsets

### Phase 4: Corrective AI & RL Engine
- [ ] Orchestration loop for RL Agent
- [ ] Logic to mutate variables (OI, volume, PnL limits, session time) to optimize Sharpe/Sortino

### Phase 5: Risk & Execution
- [ ] ATR-based dynamic risk allocator
- [ ] Portfolio-level limiters
- [ ] Holding time calculators
- [ ] Thread-safe trade execution queue

## Known Constraints
- Thread safety, zero data races, 24/7 parallel execution.
- Atomic CSV writes, advanced API retry/caching.
- Dense execution, no placeholders in critical paths.
