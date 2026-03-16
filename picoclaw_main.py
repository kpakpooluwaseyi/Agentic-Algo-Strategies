#!/usr/bin/env python3
"""picoclaw_main.py — PicoClaw worker for the RBI Swarm.
Roles: breeder | backtester | coder | monitor
Comms:
  /new_strategy  → POST http://<mac_ip>:9091   (Flask receiver)
  /regime        → POST http://<mac_ip>:9090   (PicoClawIngest)
  /health        → GET  http://<mac_ip>:9090   (PicoClawIngest)
LLM  : OpenRouter (OPENROUTER_API_KEY + model via PICOCLAW_MODEL)
"""
import argparse, json, logging, os, signal, subprocess, sys
import threading, time
from datetime import datetime, timezone
from typing import Optional

import litellm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Port constants ─────────────────────────────────────────────────────────────
# Flask receiver (new strategies from breeders)
STRATEGY_PORT = int(os.getenv("MAC_STRATEGY_PORT", "9091"))
# PicoClawIngest (regime scores + health)
REGIME_PORT   = int(os.getenv("MAC_REGIME_PORT",   "9090"))

# ── Config ────────────────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("PICOCLAW_MODEL", "openai/gpt-4o-mini")
INTERVALS = {"breeder": 600, "backtester": 900, "coder": 600, "monitor": 120}
_shutdown = threading.Event()

def _sig(s, _): logging.info("Signal %s — stopping.", s); _shutdown.set()
signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True

# ── Util ──────────────────────────────────────────────────────────────────────
def _init_log(iid):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(d, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(d, f"picoclaw_{iid}.log"), encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])

def _llm(prompt, mx=4096):
    """Call LLM via LiteLLM (OpenRouter default). Returns generated text or None."""
    api_key  = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        logging.error("OPENROUTER_API_KEY / OPENAI_API_KEY not set — cannot call LLM.")
        return None
    
    # Robustness: ensure OpenRouter models have the 'openrouter/' prefix if missing
    model = LLM_MODEL
    if "openrouter.ai" in base_url and not model.startswith("openrouter/") and not model.startswith("openai/"):
        model = f"openrouter/{model}"

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=mx, temperature=0.8,
            api_key=api_key,
            base_url=base_url,
        )
        return response.choices[0].message.content

    except Exception as e:
        logging.error("LLM fail: %s", e)
        return None


def _make_session() -> requests.Session:
    """Return a requests.Session with automatic retry + backoff."""
    retry = Retry(
        total=3,
        backoff_factor=1.5,       # waits 0, 1.5, 3.0 s between attempts
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
        raise_on_status=False,
    )
    s = requests.Session()
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

_SESSION = _make_session()

def _post(mac, path, data):
    """POST JSON to Mac orchestrator. Routes to correct port per path."""
    port = STRATEGY_PORT if path.startswith("/new_strategy") else REGIME_PORT
    url  = f"http://{mac}:{port}{path}"
    try:
        r = _SESSION.post(url, json=data, timeout=15)
        logging.info("POST %s → HTTP %s", url, r.status_code)
        return r.ok
    except requests.exceptions.ConnectionError as e:
        logging.error("POST %s — ConnectionError (is %s:%d reachable?): %s", url, mac, port, e)
    except requests.exceptions.Timeout:
        logging.error("POST %s — Timed out after 15 s", url)
    except Exception as e:
        logging.error("POST %s — Unexpected error: %s", url, e)
    return False

def _extract_cls(code):
    for ln in code.splitlines():
        if ln.strip().startswith("class ") and "BaseStrategy" in ln:
            return ln.split("class ")[1].split("(")[0].strip()
    return None

# ── BREEDER ───────────────────────────────────────────────────────────────────
# Innovation #1: Backtest Feedback Loop
# Innovation #2: Enriched Tick Data (communicated to LLM via prompt)
# Innovation #3: Mutation of Survivors (alternating genesis/mutation cycles)

_breeder_cycle = 0  # Module-level counter for alternating genesis/mutation

# Enriched tick_data fields (Innovation #2) — described in the prompt so the
# LLM generates strategies that USE these fields.
_TICK_FIELDS_DOC = (
    "tick_data dict contains:\n"
    "  price (float), volume (float), timestamp (float, unix epoch),\n"
    "  atr (float), bid (float), ask (float),\n"
    "  session (str: 'asia'|'london'|'newyork'|'overlap_london_ny'),\n"
    "  spread (float: ask-bid), spread_zscore (float: spread vs rolling mean),\n"
    "  vwap (float: volume-weighted avg price), tick_imbalance (float: buy-sell pressure),\n"
    "  regime (str: 'trending'|'ranging'|'volatile'|'unknown'),\n"
    "  atr_percentile (float 0-100: volatility rank vs last 100 bars),\n"
    "  daily_bar_count (int: ticks since session open),\n"
    "  position_count (int: currently open positions),\n"
    "  daily_pnl (float: realized PnL today).\n"
    "Not all fields are guaranteed; use .get() with defaults.\n"
)

_BASE_INSTRUCTIONS = (
    "Subclass BaseStrategy. Import: from rbi_core.strategy.base import BaseStrategy, Signal.\n"
    "Implement on_tick(self, tick_data) -> Optional[Signal] and reset(self) -> None.\n"
    "Return Signal(action='BUY'|'SELL'|'HOLD', confidence=0.0-1.0, meta={}).\n"
    "Use ONLY Python standard library + the tick_data fields above.\n"
    "Include position sizing logic (use confidence to scale size).\n"
    "Add a safety kill-switch: if daily_pnl < -0.02 (2% loss), always return HOLD.\n"
    "No placeholders. No markdown fences. Output ONLY valid Python code.\n"
)


def _fetch_metrics(mac):
    """Innovation #1: Fetch top/bottom strategies from Mac's /metrics endpoint."""
    try:
        url = f"http://{mac}:{STRATEGY_PORT}/metrics"
        r = _SESSION.get(url, timeout=10)
        if r.ok:
            return r.json()
    except Exception as e:
        logging.warning("[breeder] Could not fetch metrics: %s", e)
    return {"top": [], "bottom": []}


def _fetch_strategy_source(mac, strategy_name):
    """Innovation #3: Fetch source code of a winning strategy for mutation."""
    try:
        safe = "".join(c for c in strategy_name if c.isalnum() or c == '_').lower()
        url = f"http://{mac}:{STRATEGY_PORT}/strategy_source/{safe}"
        r = _SESSION.get(url, timeout=10)
        if r.ok:
            data = r.json()
            return data.get("code", "")
    except Exception as e:
        logging.warning("[breeder] Could not fetch strategy source for %s: %s", strategy_name, e)
    return ""


def _format_metrics_context(metrics):
    """Format metrics into a human-readable context block for the LLM prompt."""
    lines = []
    top = metrics.get("top", [])
    bottom = metrics.get("bottom", [])
    if top:
        lines.append("=== TOP 5 PERFORMING STRATEGIES (learn from these) ===")
        for i, s in enumerate(top[:5], 1):
            lines.append(
                f"{i}. {s.get('strategy','?')} — PnL: {s.get('cumulative_pnl',0):+.4f}, "
                f"Trades: {s.get('trade_count',0)}, MaxDD: {s.get('max_drawdown',0):.4f}"
            )
    if bottom:
        lines.append("\n=== BOTTOM 5 STRATEGIES (avoid these patterns) ===")
        for i, s in enumerate(bottom[:5], 1):
            lines.append(
                f"{i}. {s.get('strategy','?')} — PnL: {s.get('cumulative_pnl',0):+.4f}, "
                f"Trades: {s.get('trade_count',0)}, MaxDD: {s.get('max_drawdown',0):.4f}"
            )
    return "\n".join(lines) if lines else "No historical metrics available yet."


def _breeder(iid, mac):
    global _breeder_cycle
    _breeder_cycle += 1
    is_mutation = (_breeder_cycle % 2 == 0)  # Even cycles = mutate, odd = genesis
    mode_label = "MUTATION" if is_mutation else "GENESIS"
    logging.info("[breeder] Cycle %d — Mode: %s", _breeder_cycle, mode_label)

    # ── Innovation #1: Fetch performance feedback ──
    metrics = _fetch_metrics(mac)
    metrics_context = _format_metrics_context(metrics)

    if is_mutation:
        # ── Innovation #3: Mutation of Survivors ──
        top = metrics.get("top", [])
        if not top:
            logging.info("[breeder] No top strategies yet, falling back to GENESIS mode")
            is_mutation = False
        else:
            # Pick the best performer and fetch its source code
            winner = top[0]
            winner_name = winner.get("strategy", "")
            winner_code = _fetch_strategy_source(mac, winner_name)
            if not winner_code:
                logging.info("[breeder] Could not fetch source for %s, falling back to GENESIS", winner_name)
                is_mutation = False
            else:
                # Decide what to mutate based on cycle number
                mutation_targets = [
                    "entry logic (change the indicators or thresholds used to enter trades)",
                    "exit logic (improve stop-loss, take-profit, or trailing stop rules)",
                    "risk management (add position sizing, daily PnL limits, or regime filters)",
                    "regime adaptation (make the strategy behave differently in trending vs ranging markets)",
                    "session filtering (restrict trading to specific sessions like London-NY overlap)",
                ]
                import random
                mutation_focus = random.choice(mutation_targets)

                prompt = (
                    f"You are an elite quantitative strategist.\n\n"
                    f"PERFORMANCE CONTEXT:\n{metrics_context}\n\n"
                    f"TICK DATA AVAILABLE:\n{_TICK_FIELDS_DOC}\n"
                    f"WINNING STRATEGY TO MUTATE (PnL: {winner.get('cumulative_pnl', 0):+.4f}):\n"
                    f"```\n{winner_code}\n```\n\n"
                    f"TASK: Create a MUTATED version of {winner_name} that improves its {mutation_focus}.\n"
                    f"RULES:\n"
                    f"- Keep the core logic that made it a winner\n"
                    f"- Change ONLY the {mutation_focus}\n"
                    f"- Use the enriched tick_data fields (session, regime, spread, daily_pnl, etc)\n"
                    f"{_BASE_INSTRUCTIONS}"
                )
                logging.info("[breeder] Mutating %s — focus: %s", winner_name, mutation_focus)

    if not is_mutation:
        # ── GENESIS mode: generate a brand-new strategy ──
        prompt = (
            f"You are an elite quantitative strategist.\n\n"
            f"PERFORMANCE CONTEXT:\n{metrics_context}\n\n"
            f"TICK DATA AVAILABLE:\n{_TICK_FIELDS_DOC}\n"
            f"TASK: Generate 1 unique Python trading strategy that:\n"
            f"- Combines strengths from the top performers above (if any)\n"
            f"- Avoids failure patterns from the bottom performers above (if any)\n"
            f"- Uses at least 2 of the enriched tick_data fields (session, regime, spread, etc)\n"
            f"- Implements a novel approach NOT seen in the top/bottom lists\n"
            f"{_BASE_INSTRUCTIONS}"
        )

    raw = _llm(prompt, 6000)
    if not raw:
        return
    code = raw.replace("```python", "").replace("```", "").strip()
    cls = _extract_cls(code) or f"auto_{iid}_{_breeder_cycle}"
    if "class " not in code:
        logging.warning("[breeder] LLM output did not contain a valid class definition")
        return
    _post(mac, "/new_strategy", {
        "filename": f"{cls.lower()}.py", "code": code,
        "source": f"picoclaw/{iid}/breeder/{mode_label.lower()}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cycle": _breeder_cycle,
        "mode": mode_label,
    })
    logging.info("[breeder] Sent %s (%d chars) — mode: %s", cls, len(code), mode_label)


# ── BACKTESTER ────────────────────────────────────────────────────────────────
def _backtester(iid, mac):
    host = os.getenv("DELL_HOST", "192.168.1.50")
    user = os.getenv("DELL_USER", "kpakpo")
    key = os.getenv("DELL_SSH_KEY", os.path.expanduser("~/.ssh/id_rsa"))
    script = os.getenv("NANOBOT_WFA_SCRIPT", "~/nanobot/run_wfa.py")
    logging.info("[backtester] Triggering WFA on %s@%s…", user, host)
    cmd = ["ssh","-i",key,"-o","ConnectTimeout=10","-o","StrictHostKeyChecking=no",
           f"{user}@{host}", f"python3 {script} --strategy all --range 30d --output-endpoint /weights/latest"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode == 0:
            out = res.stdout.strip()
            try: d = json.loads(out)
            except json.JSONDecodeError: d = {"raw": out[:500]}
            d["source"] = f"picoclaw/{iid}/backtester"
            _post(mac, "/regime", d)
        else: logging.error("[backtester] SSH err: %s", res.stderr[:300])
    except subprocess.TimeoutExpired: logging.error("[backtester] SSH timeout")

# ── CODER ─────────────────────────────────────────────────────────────────────
def _coder(iid, mac):
    logging.info("[coder] Generating strategy…")
    prompt = ("Write ONE complete Python trading strategy subclassing BaseStrategy "
        "from rbi_core.strategy.base. Implement on_tick and reset. "
        "tick_data keys: price, volume, timestamp, atr, bid, ask. "
        "Return Signal(action='BUY'|'SELL'|'HOLD', confidence=0-1, meta={{}}). "
        "Use novel indicator combos, proper state mgmt. Standard lib only. "
        "No placeholders. Output ONLY Python code, no markdown fences.")
    code = _llm(prompt, 4096)
    if not code: return
    code = code.replace("```python","").replace("```","").strip()
    cls = _extract_cls(code) or "auto_coded"
    _post(mac, "/new_strategy", {"filename": f"{cls.lower()}.py", "code": code,
        "source": f"picoclaw/{iid}/coder", "timestamp": datetime.now(timezone.utc).isoformat()})
    logging.info("[coder] Sent %s (%d chars)", cls, len(code))

# ── MONITOR ───────────────────────────────────────────────────────────────────
def _monitor(iid, mac):
    logging.info("[monitor] Health check…")
    m = {"instance_id": iid, "timestamp": datetime.now(timezone.utc).isoformat()}
    # Health endpoint lives on the PicoClawIngest server (port 9090)
    health_url = f"http://{mac}:{REGIME_PORT}/health"
    try:
        m["mac_ok"] = _SESSION.get(health_url, timeout=5).ok
    except Exception as e:
        logging.warning("[monitor] Mac health check failed (%s): %s", health_url, e)
        m["mac_ok"] = False
    host = os.getenv("DELL_HOST", "localhost")
    user = os.getenv("DELL_USER", os.getenv("USER", "user"))
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=3", "-o", "StrictHostKeyChecking=no",
             f"{user}@{host}", "echo ok"],
            capture_output=True, text=True, timeout=10,
        )
        m["dell_ok"] = r.returncode == 0
    except Exception: m["dell_ok"] = False
    bad = [k for k in ("mac_ok", "dell_ok") if not m.get(k)]
    if bad:
        m.update(anomalies=bad, score=20, label="anomaly")
        _post(mac, "/regime", m)      # routed to 9090
        logging.warning("[monitor] Anomalies: %s", bad)
    else:
        logging.info("[monitor] All nominal.")

# ── Dispatch & Main ──────────────────────────────────────────────────────────
ROLES = {"breeder": _breeder, "backtester": _backtester, "coder": _coder, "monitor": _monitor}

BANNER = """
╔══════════════════════════════════════════════════╗
║  🦀 PicoClaw Worker v1.2 — RBI Swarm            ║
║  LLM: OpenRouter via LiteLLM + retry            ║
╠══════════════════════════════════════════════════╣
║  Instance    : {iid:<35s}║
║  Role        : {role:<35s}║
║  Mac IP      : {mac:<35s}║
║  Strat port  : {sp:<35s}║
║  Regime port : {rp:<35s}║
║  Model       : {model:<35s}║
║  PID         : {pid:<35s}║
║  Interval    : {iv:<35s}║
╚══════════════════════════════════════════════════╝"""

def main():
    p = argparse.ArgumentParser(description="PicoClaw Worker — RBI Swarm agent")
    p.add_argument("--instance_id", required=True, help="e.g. extra_0")
    p.add_argument("--role", required=True, choices=list(ROLES), help="breeder|backtester|coder|monitor")
    p.add_argument("--mac_ip", default=os.getenv("MAC_IP", "100.74.67.56"), help="Mac orchestrator IP")
    a = p.parse_args()
    _init_log(a.instance_id)
    iv = INTERVALS.get(a.role, 300)
    print(BANNER.format(
        iid=a.instance_id, role=a.role, mac=a.mac_ip,
        sp=str(STRATEGY_PORT), rp=str(REGIME_PORT),
        model=LLM_MODEL, pid=str(os.getpid()), iv=f"{iv}s",
    ))
    logging.info(
        "Started: id=%s role=%s mac=%s strategy_port=%d regime_port=%d model=%s",
        a.instance_id, a.role, a.mac_ip, STRATEGY_PORT, REGIME_PORT, LLM_MODEL,
    )
    fn = ROLES[a.role]
    while not _shutdown.is_set():
        try:
            fn(a.instance_id, a.mac_ip)
        except Exception as e:
            logging.exception("Cycle error: %s", e)
        for _ in range(iv):
            if _shutdown.is_set():
                break
            time.sleep(1)
    logging.info("Worker %s [%s] stopped.", a.instance_id, a.role)

if __name__ == "__main__":
    main()
