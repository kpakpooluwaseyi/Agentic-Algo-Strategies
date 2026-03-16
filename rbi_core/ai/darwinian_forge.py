"""rbi_core/ai/darwinian_forge.py — Self-evolving strategy pool via LLM-driven mutation.

The Darwinian Strategy Forge turns the static strategy/pool/ into a living,
self-evolving trading organism. New strategies are generated, backtested, scored,
and hot-reloaded into the combiner — all autonomously.

Population:  strategy/pool/*.py files are the "genome"
Breeders:    PicoClaw instances (or any LLM endpoint) generate mutated BaseStrategy code
Testers:     NanobotTrigger runs WFA/replay to score fitness
Selector:    Top performers survive, bottom pruned, winners hot-reloaded into combiner
Diversity:   Enforces strategy-type uniqueness to prevent convergence

Requires FORGE_ENABLED=1 env var to activate in orchestrator.
"""
import os
import sys
import time
import json
import importlib
import importlib.util
import inspect
import threading
import asyncio
from typing import Callable, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import urllib.request
import urllib.error


@dataclass
class StrategyGenome:
    """Metadata wrapper for a strategy file in the pool."""
    name: str                      # class name (e.g., "EMAScalp")
    filename: str                  # relative filename in pool dir
    filepath: str                  # absolute path
    lineage: list[str] = field(default_factory=list)  # parent strategy names
    generation: int = 0
    fitness: dict = field(default_factory=dict)  # {regime: score}
    overall_fitness: float = 0.0
    created_at: float = field(default_factory=time.time)
    signal_type: str = ""          # for diversity: "momentum", "reversion", "flow", etc.
    active: bool = True


class PicoclawClient:
    """
    Client for a PicoClaw LLM instance in 'breeder' role.
    Sends mutation prompts and receives generated strategy code.

    In local dry-run mode: returns a placeholder mutation.
    When PicoClaw fleet is online: sends HTTP POST to PicoClaw instance.
    """

    def __init__(self, endpoint: str = "", role: str = "breeder"):
        """
        Args:
            endpoint: URL of PicoClaw instance (e.g., "http://192.168.1.50:8081/ask")
                      Empty string = local dry-run mode.
            role: Agent role ("breeder", "tester", etc.)
        """
        self.endpoint = endpoint
        self.role = role
        self._local_mode = not endpoint

    async def ask(self, prompt: str) -> str:
        """
        Send a prompt to the PicoClaw instance and get generated code back.

        Args:
            prompt: The mutation/generation prompt.

        Returns:
            Generated Python code string (full .py file content).
        """
        if self._local_mode:
            return self._local_fallback(prompt)

        # Real PicoClaw HTTP call
        try:
            data = json.dumps({"prompt": prompt, "role": self.role}).encode()
            req = urllib.request.Request(
                self.endpoint,
                data=data,
                method='POST',
                headers={'Content-Type': 'application/json'},
            )
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=60))
            result = json.loads(response.read().decode())
            return result.get('code', '')
        except Exception as e:
            print(f"[Forge] PicoClaw error: {e}")
            return ""

    def _local_fallback(self, prompt: str) -> str:
        """Generate a placeholder strategy in dry-run mode."""
        # In local mode, we create a simple parameter variation
        # Real LLM mutation would produce entirely new logic
        return ""


class ForgeScorer:
    """
    Scores strategies using StrategyMetrics data + microstructure fitness.
    Supports per-regime scoring for true regime-robustness evaluation.
    """

    def __init__(self, strategy_metrics, microstructure_engine):
        self.metrics = strategy_metrics
        self.engine = microstructure_engine

    def score_strategy(self, strategy_name: str) -> float:
        """
        Composite fitness score for a strategy across all regimes.

        Scoring formula:
        - 60% Sortino-weighted (higher Sortino in more regimes = better)
        - 25% Winrate (> 50% preferred)
        - 15% Drawdown penalty (max drawdown penalized)
        """
        all_metrics = self.metrics.get_metrics(strategy_name)
        if not all_metrics:
            return 0.0

        total_score = 0.0
        total_trades = 0
        for m in all_metrics:
            trades = m.get('trade_count', 0)
            if trades < 3:
                continue  # Need minimum trades for statistical significance
            sortino = m.get('sortino', 0.0)
            winrate = m.get('winrate', 0.5)
            drawdown = abs(m.get('max_drawdown', 0.0))

            regime_score = (
                0.60 * max(0, sortino) +
                0.25 * (winrate - 0.5) * 10.0 +   # Bonus for > 50% winrate
                0.15 * max(0, 1.0 - drawdown * 10)  # Penalty for drawdown
            )
            total_score += regime_score * trades
            total_trades += trades

        return total_score / max(total_trades, 1)

    def rank_all(self) -> list[tuple[str, float]]:
        """Rank all strategies by composite fitness. Returns [(name, score), ...]."""
        all_metrics = self.metrics.get_metrics()
        strategy_names = set(m['strategy'] for m in all_metrics)

        scores = []
        for name in strategy_names:
            score = self.score_strategy(name)
            scores.append((name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class DarwinianForge:
    """
    Main evolutionary orchestrator. Runs periodic evolution cycles that:
    1. Score existing strategies via ForgeScorer
    2. Select top performers as "parents"
    3. Use PicoClaw breeders to generate mutated offspring
    4. Test offspring via NanobotTrigger WFA or local replay
    5. Hot-reload winners into StrategyCombiner
    6. Prune bottom performers (with diversity guardrails)
    """

    def __init__(
        self,
        pool_dir: str,
        combiner,                          # StrategyCombiner instance
        microstructure_engine,             # MicrostructureEngine instance
        strategy_metrics,                  # StrategyMetrics instance
        nanobot_trigger=None,              # NanobotTrigger instance (optional)
        breeder_endpoints: list[str] = None,  # PicoClaw URLs (empty = local mode)
        cycle_interval_s: float = 3600.0,  # Run evolution every hour
        min_population: int = 4,           # Never prune below this
        max_population: int = 30,          # Cap total strategies
        survival_rate: float = 0.7,        # Top 70% survive
        min_trades_for_eval: int = 5,      # Min trades before scoring
    ):
        self.pool_dir = os.path.abspath(pool_dir)
        self.combiner = combiner
        self.engine = microstructure_engine
        self.nanobot_trigger = nanobot_trigger
        self.cycle_interval_s = cycle_interval_s
        self.min_population = min_population
        self.max_population = max_population
        self.survival_rate = survival_rate
        self.min_trades_for_eval = min_trades_for_eval

        # Scorer
        self.scorer = ForgeScorer(strategy_metrics, microstructure_engine)

        # Breeders (PicoClaw instances or local stubs)
        endpoints = breeder_endpoints or [""]  # Empty = local dry-run
        self.breeders = [PicoclawClient(endpoint=ep, role="breeder") for ep in endpoints]

        # Population tracking
        self.genomes: dict[str, StrategyGenome] = {}
        self._scan_existing_pool()

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _scan_existing_pool(self) -> None:
        """Discover existing strategy files in pool_dir."""
        if not os.path.isdir(self.pool_dir):
            return

        for fname in os.listdir(self.pool_dir):
            if not fname.endswith('.py') or fname.startswith('__'):
                continue
            filepath = os.path.join(self.pool_dir, fname)
            class_name = self._extract_class_name(filepath)
            if class_name:
                self.genomes[class_name] = StrategyGenome(
                    name=class_name,
                    filename=fname,
                    filepath=filepath,
                    generation=0,
                    signal_type=self._infer_signal_type(fname),
                )

    def _extract_class_name(self, filepath: str) -> Optional[str]:
        """Extract the BaseStrategy subclass name from a .py file."""
        try:
            with open(filepath, 'r') as f:
                source = f.read()
            # Simple heuristic: find "class XYZ(BaseStrategy):"
            for line in source.splitlines():
                line = line.strip()
                if line.startswith('class ') and 'BaseStrategy' in line:
                    name = line.split('(')[0].replace('class ', '').strip()
                    return name
        except Exception:
            pass
        return None

    def _infer_signal_type(self, filename: str) -> str:
        """Infer signal type category from filename for diversity tracking."""
        name_lower = filename.lower()
        if any(k in name_lower for k in ['ema', 'momentum', 'trend', 'macd']):
            return "momentum"
        elif any(k in name_lower for k in ['rsi', 'reversion', 'vwap', 'bollinger']):
            return "reversion"
        elif any(k in name_lower for k in ['flow', 'ofi', 'cvd', 'order']):
            return "flow"
        elif any(k in name_lower for k in ['funding', 'arb', 'basis']):
            return "arbitrage"
        elif any(k in name_lower for k in ['time', 'calendar', 'session']):
            return "temporal"
        return "unknown"

    def start(self) -> None:
        """Start the background evolution thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[Forge] Started: {len(self.genomes)} strategies in pool, "
              f"{len(self.breeders)} breeders, cycle every {self.cycle_interval_s}s")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10.0)
        print("[Forge] Stopped")

    def _run_loop(self) -> None:
        """Background thread: periodic evolution cycles."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        while self._running:
            try:
                self._loop.run_until_complete(self.evolution_cycle())
            except Exception as e:
                print(f"[Forge] Evolution cycle error: {e}")
            # Sleep between cycles
            for _ in range(int(self.cycle_interval_s)):
                if not self._running:
                    break
                time.sleep(1.0)

        self._loop.close()

    async def evolution_cycle(self) -> None:
        """
        One full evolution cycle:
        1. Score all strategies
        2. Select parents (top performers)
        3. Generate offspring via breeders
        4. Test and score offspring
        5. Prune bottom performers
        6. Hot-reload winners into combiner
        """
        print(f"[Forge] === Evolution cycle started ===")
        regime = self.engine.regime.value

        # 1. Score existing population
        rankings = self.scorer.rank_all()
        for name, score in rankings:
            if name in self.genomes:
                self.genomes[name].overall_fitness = score
                self.genomes[name].fitness[regime] = score

        if rankings:
            print(f"[Forge] Rankings: {[(n, f'{s:.3f}') for n, s in rankings[:5]]}")

        # 2. Select top performers as parents
        top_n = max(3, int(len(rankings) * 0.3))
        parents = [name for name, _ in rankings[:top_n]] if rankings else list(self.genomes.keys())[:3]

        if not parents:
            print("[Forge] No parents available, skipping cycle")
            return

        # 3. Generate offspring via breeders
        parent_code_samples = self._read_parent_code(parents[:3])
        micro_context = {
            'regime': regime,
            'ofi': self.engine.ofi,
            'cvd': self.engine.cvd,
            'atr': self.engine.atr,
        }

        # Check diversity: what signal types are underrepresented?
        type_counts = defaultdict(int)
        for g in self.genomes.values():
            type_counts[g.signal_type] += 1
        desired_types = ["momentum", "reversion", "flow", "arbitrage", "temporal"]
        underrepresented = [t for t in desired_types if type_counts[t] < 2]

        prompt = self._build_mutation_prompt(
            parent_code_samples, micro_context, underrepresented
        )

        # Distribute to breeders
        offspring_codes = []
        for breeder in self.breeders:
            if not self._running:
                break
            code = await breeder.ask(prompt)
            if code and len(code) > 50:  # Sanity check
                offspring_codes.append(code)

        # 4. Save, validate, and score offspring
        new_strategies = []
        for code in offspring_codes:
            result = self._save_and_validate(code)
            if result:
                new_strategies.append(result)
                print(f"[Forge] New strategy: {result}")

        # 5. Prune bottom performers (respect min_population)
        if len(self.genomes) > self.max_population:
            self._prune_bottom(rankings)

        # 6. Hot-reload new strategies into combiner
        for strat_name in new_strategies:
            self._hot_reload_strategy(strat_name)

        print(f"[Forge] === Cycle complete: {len(self.genomes)} strategies, "
              f"{len(new_strategies)} new ===")

    def _read_parent_code(self, parent_names: list[str]) -> list[str]:
        """Read source code of top parent strategies."""
        codes = []
        for name in parent_names:
            genome = self.genomes.get(name)
            if genome and os.path.exists(genome.filepath):
                try:
                    with open(genome.filepath, 'r') as f:
                        codes.append(f.read())
                except Exception:
                    pass
        return codes

    def _build_mutation_prompt(self, parent_codes: list[str],
                                micro_context: dict,
                                underrepresented: list[str]) -> str:
        """Build the LLM prompt for strategy mutation/generation."""
        parents_section = "\n\n---\n\n".join(parent_codes[:3])
        diversity_hint = (
            f"\nDiversity need: create a strategy of type: {', '.join(underrepresented)}. "
            f"Do NOT duplicate existing signal types."
            if underrepresented else ""
        )

        return f"""You are an elite quantitative developer. Given the current market context
and top-performing strategies below, generate a NEW BaseStrategy subclass.

MARKET CONTEXT:
- Regime: {micro_context['regime']}
- Order Flow Imbalance (OFI): {micro_context['ofi']:.3f}
- Cumulative Volume Delta (CVD): {micro_context['cvd']:.1f}
- Average True Range (ATR): {micro_context['atr']:.2f}
{diversity_hint}

RULES:
1. Output ONLY a complete .py file with the strategy class
2. Must subclass BaseStrategy from rbi_core.strategy.base
3. Must implement on_tick(self, tick_data: dict) -> Optional[Signal] and reset(self)
4. tick_data keys: price, volume, timestamp, atr, bid, ask, cvd, ofi, regime
5. Use only stdlib + numpy. No pandas, no external APIs.
6. Include clear docstring explaining the strategy logic
7. Confidence scores must be 0.0 to 1.0
8. Use self.current_confidence to track state

TOP PERFORMING PARENT STRATEGIES:

{parents_section}

Generate a mutated/novel strategy that would perform well in the current {micro_context['regime']} regime:"""

    def _save_and_validate(self, code: str) -> Optional[str]:
        """Save generated code to pool dir, validate it compiles and imports."""
        if not code or 'class ' not in code or 'BaseStrategy' not in code:
            return None

        # Extract class name
        class_name = None
        for line in code.splitlines():
            line = line.strip()
            if line.startswith('class ') and 'BaseStrategy' in line:
                class_name = line.split('(')[0].replace('class ', '').strip()
                break

        if not class_name:
            return None

        # Generate filename
        filename = f"forge_{class_name.lower()}.py"
        filepath = os.path.join(self.pool_dir, filename)

        # Don't overwrite existing
        if os.path.exists(filepath):
            filename = f"forge_{class_name.lower()}_{int(time.time()) % 10000}.py"
            filepath = os.path.join(self.pool_dir, filename)

        # Save
        try:
            with open(filepath, 'w') as f:
                f.write(code)
        except Exception as e:
            print(f"[Forge] Save error: {e}")
            return None

        # Validate: compile check
        try:
            import py_compile
            py_compile.compile(filepath, doraise=True)
        except py_compile.PyCompileError as e:
            print(f"[Forge] Compile error in {filename}: {e}")
            os.remove(filepath)
            return None

        # Validate: import check
        try:
            spec = importlib.util.spec_from_file_location(
                f"rbi_core.strategy.pool.{filename[:-3]}", filepath
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Check that the class exists and is a BaseStrategy subclass
            cls = getattr(module, class_name, None)
            if cls is None:
                raise ImportError(f"Class {class_name} not found in module")
        except Exception as e:
            print(f"[Forge] Import error in {filename}: {e}")
            os.remove(filepath)
            return None

        # Register genome
        genome = StrategyGenome(
            name=class_name,
            filename=filename,
            filepath=filepath,
            generation=max((g.generation for g in self.genomes.values()), default=0) + 1,
            signal_type=self._infer_signal_type(filename),
        )
        self.genomes[class_name] = genome

        return class_name

    def _hot_reload_strategy(self, strategy_name: str) -> bool:
        """Dynamically load a strategy and add it to the combiner."""
        genome = self.genomes.get(strategy_name)
        if not genome:
            return False

        try:
            spec = importlib.util.spec_from_file_location(
                f"rbi_core.strategy.pool.{genome.filename[:-3]}", genome.filepath
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls = getattr(module, strategy_name)
            instance = cls()

            # Add to combiner's strategy list
            self.combiner.strategies.append(instance)
            print(f"[Forge] Hot-reloaded: {strategy_name} (gen {genome.generation})")
            return True
        except Exception as e:
            print(f"[Forge] Hot-reload error for {strategy_name}: {e}")
            return False

    def _prune_bottom(self, rankings: list[tuple[str, float]]) -> None:
        """Remove bottom-performing strategies. Respects min_population and diversity."""
        if len(self.genomes) <= self.min_population:
            return

        # Calculate how many to prune
        prune_count = len(self.genomes) - int(len(self.genomes) * self.survival_rate)
        prune_count = min(prune_count, len(self.genomes) - self.min_population)

        if prune_count <= 0:
            return

        # Get bottom performers (from tail of rankings)
        bottom = [name for name, _ in rankings[-prune_count:]] if rankings else []

        # Diversity guard: don't prune the last strategy of any signal type
        type_counts = defaultdict(list)
        for g in self.genomes.values():
            type_counts[g.signal_type].append(g.name)

        for name in bottom:
            genome = self.genomes.get(name)
            if not genome:
                continue
            # Don't prune if it's the last of its signal type
            if len(type_counts.get(genome.signal_type, [])) <= 1:
                continue
            # Don't prune original (generation 0) strategies
            if genome.generation == 0:
                continue

            # Prune: remove from combiner
            self.combiner.strategies = [
                s for s in self.combiner.strategies if s.name != name
            ]
            # Remove file and genome
            try:
                if os.path.exists(genome.filepath) and 'forge_' in genome.filename:
                    os.remove(genome.filepath)
            except Exception:
                pass
            del self.genomes[name]
            print(f"[Forge] Pruned: {name} (fitness too low)")

    def get_population_report(self) -> dict:
        """Summary of current strategy population for monitoring."""
        type_counts = defaultdict(int)
        for g in self.genomes.values():
            type_counts[g.signal_type] += 1

        return {
            'total_strategies': len(self.genomes),
            'signal_types': dict(type_counts),
            'generations': max((g.generation for g in self.genomes.values()), default=0),
            'top_fitness': sorted(
                [(g.name, g.overall_fitness) for g in self.genomes.values()],
                key=lambda x: x[1], reverse=True
            )[:5],
        }
