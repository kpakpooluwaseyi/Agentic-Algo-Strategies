"""rbi_core/dashboard/run.py — All-in-One orchestrator wiring all components.

Wires: WS feed → MicrostructureEngine (CVD/OFI/ATR/Regime) → TickBuffer
       → StrategyCombiner → TradeManager → RL Agent → MockDispatcher
       + NanoClaw watchdog + PicoClaw ingest + SwarmSync + StrategyMetrics
       + DarwinianForge (optional)
"""
import os
import sys
import time
import signal as signal_mod

# Ensure rbi_core is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rbi_core.data.collectors.buffer_mgr import TickBuffer
from rbi_core.data.collectors.hyperliquid_ws import HyperliquidWSFeed
from rbi_core.data.microstructure_engine import MicrostructureEngine
from rbi_core.data.strategy_metrics import StrategyMetrics
from rbi_core.strategy.combiner import StrategyCombiner
from rbi_core.strategy.pool.ema_scalp import EMAScalp
from rbi_core.strategy.pool.rsi_divergence import RSIDivergence
from rbi_core.strategy.pool.vwap_mean_reversion import VWAPMeanReversion
from rbi_core.strategy.pool.bollinger_scalp import BollingerScalp
from rbi_core.ai.corrective_agent import CorrectiveRLAgent
from rbi_core.ai.swarm_sync import SwarmWeightSync
from rbi_core.execution.risk import PortfolioExecutionQueue
from rbi_core.execution.dispatcher import OrderDispatcher
from rbi_core.execution.trade_manager import TradeManager
from rbi_core.swarm_integration.picoclaw_ingest import PicoClawIngest
from rbi_core.swarm_integration.nanoclaw_monitor import NanoClawMonitor


# === CONFIGURATION ===
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ticks.db')
SYMBOLS = ["BTC", "ETH"]
INITIAL_EQUITY = 100.0
NANOBOT_URL = os.environ.get("NANOBOT_URL", "http://192.168.1.50:8080")
PICOCLAW_PORT = int(os.environ.get("PICOCLAW_PORT", "9090"))
EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "mock")  # "mock" or "live"
FORGE_ENABLED = os.environ.get("FORGE_ENABLED", "0") == "1"


def main():
    print("=" * 60)
    print("  RBI CORE — All-in-One Orchestrator")
    print("=" * 60)

    # --- 1. Data Layer ---
    tick_buffer = TickBuffer(DB_PATH)
    tick_buffer.start()

    # --- 2. Microstructure Intelligence Engine (CVD + OFI + ATR + Regime) ---
    micro_engine = MicrostructureEngine(atr_bar_ticks=20, atr_period=5)

    # --- 3. Strategy Layer ---
    strategies = [
        EMAScalp(),
        RSIDivergence(bar_ticks=50),
        VWAPMeanReversion(min_ticks_for_vwap=100, entry_deviation_pct=0.3),
        BollingerScalp(bar_ticks=30, period=20),
    ]
    combiner = StrategyCombiner(strategies, consensus_threshold=0.2)

    # --- 4. AI Layer ---
    rl_agent = CorrectiveRLAgent()

    swarm_sync = SwarmWeightSync(
        nanobot_url=NANOBOT_URL,
        on_weights_updated=rl_agent.load_weights_from_swarm,
        on_degraded=lambda degraded: setattr(rl_agent, 'swarm_degraded', degraded),
    )
    swarm_sync.start()

    # --- 5. Execution Layer ---
    exec_queue = PortfolioExecutionQueue(INITIAL_EQUITY)
    dispatcher = OrderDispatcher(exec_queue, mode=EXECUTION_MODE)
    dispatcher.start()

    # --- 6. Strategy Metrics (persistent SQLite) ---
    strat_metrics = StrategyMetrics(DB_PATH)

    # Trade lifecycle mapping: TradeManager trade_id -> exec queue position key
    _trade_to_position: dict[str, str] = {}

    # --- 7. Trade Manager + RL Wiring ---
    def _on_trade_closed(result: dict):
        """Callback: trade closed → update metrics → feed RL agent."""
        strategy = result['strategy']
        regime = micro_engine.regime.value
        pnl = result['realized_pnl']

        # 1. Update persistent strategy metrics
        strat_metrics.record_trade(strategy, regime, pnl)

        # 2. Release execution slot and update equity.
        # Fallback keeps legacy behavior if mapping is missing.
        position_key = _trade_to_position.pop(result['trade_id'], None)
        if position_key:
            exec_queue.close_position(position_key, pnl)
        else:
            exec_queue.update_equity(exec_queue.equity + pnl)

        # 3. Feed RL agent: compute raw gradients from PnL direction
        sortino = trade_manager.rolling_sortino
        pnl_frac = trade_manager.total_pnl / max(INITIAL_EQUITY, 0.01)
        raw_grads = {
            'atr_multiplier': 0.1 if pnl > 0 else -0.1,
            'vol_threshold': 10.0 if pnl > 0 else -10.0,
            'max_holding_time_minutes': 5.0 if result.get('exit_reason') == 'max_hold' else 0.0,
        }
        new_params = rl_agent.step(sortino, pnl_frac, raw_grads)

        # 4. Update NanoClaw PnL tracker
        _current_pnl[0] = pnl_frac

        print(f"[TradeClose] {result['action']} {strategy} "
              f"pnl={pnl:+.4f} reason={result['exit_reason']} "
              f"sortino={sortino:.3f} atr_mult={new_params['atr_multiplier']:.3f}")

    trade_manager = TradeManager(on_trade_closed=_on_trade_closed)

    # --- 8. Swarm Integration ---
    picoclaw = PicoClawIngest(
        port=PICOCLAW_PORT,
        on_regime_update=combiner.set_regime_weight,
    )
    picoclaw.start()

    # Track state for NanoClaw callbacks and metrics
    _last_tick_ts = [time.time()]
    _current_spread = [0.01]
    _current_pnl = [0.0]
    _tick_count = [0]
    _signal_count = [0]

    # Enrichment state (Innovation #2)
    _vwap_cum_pv = [0.0]     # cumulative price * volume
    _vwap_cum_vol = [0.0]    # cumulative volume
    _spread_history = []     # rolling spread for z-score
    _atr_history = []        # rolling ATR for percentile
    _daily_bar_count = [0]

    def _get_session(ts):
        """Determine trading session from unix timestamp."""
        import datetime as _dt
        hour = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).hour
        if 0 <= hour < 8:
            return "asia"
        elif 8 <= hour < 13:
            return "london"
        elif 13 <= hour < 16:
            return "overlap_london_ny"
        elif 16 <= hour < 22:
            return "newyork"
        return "asia"

    def _on_tick_pipeline(tick: dict):
        """Full tick pipeline: microstructure → enrich → buffer → combiner → trade mgr → risk."""
        # 1. Enrich tick with microstructure signals (CVD, OFI, ATR, regime)
        tick = micro_engine.process_tick(tick)

        # 2. Auto-update combiner regime weight from microstructure
        combiner.set_regime_weight(tick.get('regime_weight', 1.0))

        # ── Innovation #2: Enriched tick_data fields ──
        bid = tick.get('bid', 0)
        ask = tick.get('ask', 0)
        price = tick.get('price', 0)
        volume = tick.get('volume', 0)
        ts = tick.get('timestamp', time.time())

        # Session
        tick['session'] = _get_session(ts)

        # Spread & spread z-score
        spread = (ask - bid) if (bid and ask) else 0.0
        tick['spread'] = spread
        _spread_history.append(spread)
        if len(_spread_history) > 100:
            _spread_history.pop(0)
        if len(_spread_history) >= 10:
            mean_s = sum(_spread_history) / len(_spread_history)
            std_s = (sum((s - mean_s)**2 for s in _spread_history) / len(_spread_history)) ** 0.5
            tick['spread_zscore'] = (spread - mean_s) / std_s if std_s > 0 else 0.0
        else:
            tick['spread_zscore'] = 0.0

        # VWAP
        _vwap_cum_pv[0] += price * volume
        _vwap_cum_vol[0] += volume
        tick['vwap'] = _vwap_cum_pv[0] / _vwap_cum_vol[0] if _vwap_cum_vol[0] > 0 else price

        # Tick imbalance (use OFI from microstructure as proxy)
        tick['tick_imbalance'] = tick.get('ofi', 0.0)

        # Regime (string label from microstructure engine)
        tick['regime'] = micro_engine.regime.value if hasattr(micro_engine, 'regime') else 'unknown'

        # ATR percentile
        current_atr = tick.get('atr', 0.0)
        if current_atr > 0:
            _atr_history.append(current_atr)
            if len(_atr_history) > 100:
                _atr_history.pop(0)
            rank = sum(1 for a in _atr_history if a <= current_atr)
            tick['atr_percentile'] = (rank / len(_atr_history)) * 100.0
        else:
            tick['atr_percentile'] = 50.0

        # Daily bar count
        _daily_bar_count[0] += 1
        tick['daily_bar_count'] = _daily_bar_count[0]

        # Position count & daily PnL
        tick['position_count'] = len(trade_manager.open_trades)
        tick['daily_pnl'] = trade_manager.total_pnl / max(INITIAL_EQUITY, 0.01)

        tick_buffer.append_tick(tick)
        _last_tick_ts[0] = time.time()
        _tick_count[0] += 1

        if bid and ask:
            _current_spread[0] = ask - bid


        # 3. Check open trade exits (stop-loss, max-hold)
        trade_manager.check_exits(tick)

        # 4. Route through strategy combiner
        combined = combiner.evaluate_tick(tick)
        if combined:
            _signal_count[0] += 1
            current_atr = tick.get('atr', 0.0)

            # Only trade if ATR is warmed up (> 0)
            if current_atr > 0:
                signal_dict = {
                    'action': combined.action,
                    'strategy': ','.join(combined.contributing_strategies),
                }
                position_key = exec_queue.submit_signal(
                    signal_dict,
                    rl_agent.active_params,
                    {'price': tick['price'], 'atr': current_atr},
                )
                if position_key:
                    # Open trade in TradeManager for lifecycle tracking
                    trade_id = trade_manager.open_trade(
                        action=combined.action,
                        entry_price=tick['price'],
                        size=exec_queue.calculate_position_size(
                            current_atr, rl_agent.active_params['atr_multiplier']
                        ),
                        strategy=','.join(combined.contributing_strategies),
                        entry_atr=current_atr,
                        atr_multiplier=rl_agent.active_params['atr_multiplier'],
                        max_holding_minutes=rl_agent.active_params['max_holding_time_minutes'],
                    )
                    _trade_to_position[trade_id] = position_key

    # --- 9. WebSocket (tick + book callbacks) ---
    ws_feed = HyperliquidWSFeed(
        symbols=SYMBOLS,
        on_tick=_on_tick_pipeline,
        on_book=micro_engine.update_book,
    )
    ws_feed.start()

    # --- 10. NanoClaw Watchdog ---
    nanoclaw = NanoClawMonitor(
        halt_callback=exec_queue.halt,
        get_last_tick_ts=lambda: _last_tick_ts[0],
        get_current_spread=lambda: _current_spread[0],
        get_current_pnl=lambda: _current_pnl[0],
        tick_stale_threshold_s=60.0,  # WS has natural gaps during book-only periods
    )
    nanoclaw.start()

    # --- 11. Darwinian Forge (optional) ---
    forge = None
    if FORGE_ENABLED:
        try:
            from rbi_core.ai.darwinian_forge import DarwinianForge
            pool_dir = os.path.join(os.path.dirname(__file__), '..', 'strategy', 'pool')
            forge = DarwinianForge(
                pool_dir=pool_dir,
                combiner=combiner,
                microstructure_engine=micro_engine,
                strategy_metrics=strat_metrics,
                nanobot_trigger=None,  # Wire when Dell endpoint is live
            )
            forge.start()
            print("[Orchestrator] Darwinian Forge: ACTIVE")
        except Exception as e:
            print(f"[Orchestrator] Forge disabled: {e}")

    # --- 12. Shutdown + Main Loop ---
    def shutdown(signum=None, frame=None):
        print(f"\n[Orchestrator] Shutting down... "
              f"({_tick_count[0]} ticks, {_signal_count[0]} signals, "
              f"{trade_manager.trade_count} trades)")
        ws_feed.stop()
        nanoclaw.stop()
        picoclaw.stop()
        swarm_sync.stop()
        if forge:
            forge.stop()
        dispatcher.stop()
        tick_buffer.stop()
        combiner.shutdown()
        print(f"[Orchestrator] Final PnL: {trade_manager.total_pnl:+.4f} "
              f"Sortino: {trade_manager.rolling_sortino:.3f} "
              f"Winrate: {trade_manager.winrate:.1%}")
        print("[Orchestrator] Clean shutdown complete.")
        sys.exit(0)

    signal_mod.signal(signal_mod.SIGINT, shutdown)
    signal_mod.signal(signal_mod.SIGTERM, shutdown)

    print("[Orchestrator] All systems online. Ctrl+C to stop.")
    print(f"  Symbols:         {SYMBOLS}")
    print(f"  Strategies:      {[s.name for s in strategies]}")
    print(f"  Execution mode:  {EXECUTION_MODE}")
    print(f"  Nanobot URL:     {NANOBOT_URL}")
    print(f"  PicoClaw port:   {PICOCLAW_PORT}")
    print(f"  NanoClaw:        active")
    print(f"  ATR warmup:      {micro_engine._atr_period} bars × "
          f"{micro_engine._atr_bar_ticks} ticks = "
          f"{micro_engine._atr_period * micro_engine._atr_bar_ticks} ticks/symbol")
    print(f"  Forge:           {'active' if FORGE_ENABLED else 'disabled (FORGE_ENABLED=1 to enable)'}")

    start_receiver(metrics_ref=strat_metrics)
    try:
        while True:
            time.sleep(10.0)
            if _tick_count[0] > 0:
                btc_atr = micro_engine.atr_for('BTC')
                eth_atr = micro_engine.atr_for('ETH')
                print(f"[Heartbeat] ticks={_tick_count[0]} signals={_signal_count[0]} "
                      f"trades={trade_manager.trade_count} open={len(trade_manager.open_trades)} "
                      f"equity={exec_queue.equity:.2f} halted={exec_queue.halted} "
                      f"regime={micro_engine.regime.value} ofi={micro_engine.ofi:.3f} "
                      f"cvd={micro_engine.cvd:.1f} "
                      f"atr_btc={btc_atr:.2f} atr_eth={eth_atr:.4f} "
                      f"sortino={trade_manager.rolling_sortino:.3f}")
    except KeyboardInterrupt:
        shutdown()

# Add near the bottom of dashboard/run.py (after orchestrator wiring)

from flask import Flask, request, jsonify
import os
import glob

app = Flask(__name__)

# Module-level ref, set by start_receiver()
_strat_metrics_ref = None

@app.route('/new_strategy', methods=['POST'])
def receive_strategy():
    data = request.json
    filename = data.get('filename')
    code = data.get('code')
    source = data.get('source', 'unknown')
    mode = data.get('mode', 'genesis')
    cycle = data.get('cycle', 0)
    ts = data.get('timestamp', 'unknown')
    
    if not filename or not code:
        return jsonify({"error": "Missing filename or code"}), 400
    
    # Prepend evolution metadata as a header
    header = (
        f'"""\n'
        f'RBI SWARM AUTO-GENERATED STRATEGY\n'
        f'Source: {source}\n'
        f'Mode:   {mode.upper()}\n'
        f'Cycle:  {cycle}\n'
        f'Date:   {ts}\n'
        f'"""\n\n'
    )
    full_code = header + code

    pool_dir = "rbi_core/strategy/pool"
    os.makedirs(pool_dir, exist_ok=True)
    file_path = os.path.join(pool_dir, filename)
    
    with open(file_path, "w") as f:
        f.write(full_code)
    
    print(f"[RECEIVER] Saved {mode.upper()} strategy: {file_path} (Cycle {cycle})")
    return jsonify({"status": "received", "filename": filename}), 200



@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return top and bottom strategies by cumulative PnL for breeder feedback."""
    if _strat_metrics_ref is None:
        return jsonify({"top": [], "bottom": [], "regime": "unknown"}), 200
    try:
        top = _strat_metrics_ref.get_top_strategies(limit=5)
        # Bottom 5: query all, sort ascending
        all_metrics = _strat_metrics_ref.get_metrics()
        # Aggregate by strategy name
        by_strat = {}
        for m in all_metrics:
            name = m['strategy']
            if name not in by_strat:
                by_strat[name] = {"strategy": name, "cumulative_pnl": 0, "trade_count": 0,
                                  "winrate": 0, "max_drawdown": 0, "sortino": 0, "regimes": []}
            by_strat[name]["cumulative_pnl"] += m.get("cumulative_pnl", 0)
            by_strat[name]["trade_count"] += m.get("trade_count", 0)
            by_strat[name]["max_drawdown"] = min(by_strat[name]["max_drawdown"], m.get("max_drawdown", 0))
            by_strat[name]["regimes"].append(m.get("regime", "unknown"))
            by_strat[name]["sortino"] = max(by_strat[name]["sortino"], m.get("sortino", 0))
            tc = by_strat[name]["trade_count"]
            wc = sum(1 for mm in all_metrics if mm["strategy"] == name and mm.get("winrate", 0) > 0.5)
            by_strat[name]["winrate"] = m.get("winrate", 0)

        sorted_all = sorted(by_strat.values(), key=lambda x: x["cumulative_pnl"])
        bottom = sorted_all[:5]
        return jsonify({"top": top, "bottom": bottom}), 200
    except Exception as e:
        return jsonify({"error": str(e), "top": [], "bottom": []}), 200


@app.route('/strategy_source/<name>', methods=['GET'])
def get_strategy_source(name):
    """Return the source code of a strategy file from the pool."""
    pool_dir = "rbi_core/strategy/pool"
    # Sanitize: only allow alphanumeric + underscore
    safe_name = "".join(c for c in name if c.isalnum() or c == '_')
    file_path = os.path.join(pool_dir, f"{safe_name}.py")
    if not os.path.isfile(file_path):
        # Try case-insensitive match
        matches = glob.glob(os.path.join(pool_dir, f"{safe_name}*.py"))
        if matches:
            file_path = matches[0]
        else:
            return jsonify({"error": f"Strategy {safe_name} not found"}), 404
    try:
        with open(file_path, "r") as f:
            code = f.read()
        return jsonify({"strategy": safe_name, "code": code}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Start Flask in background thread
def start_receiver(metrics_ref=None):
    global _strat_metrics_ref
    _strat_metrics_ref = metrics_ref
    from threading import Thread
    Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 9091, "debug": False, "use_reloader": False}).start()

# Call this after your existing main() setup (now moved inside main)

if __name__ == "__main__":
    main()

