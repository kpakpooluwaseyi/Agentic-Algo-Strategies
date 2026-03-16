"""
Auditor Agent
=============
Role: Chief Risk Officer (CRO)
Task: Validate strategies against the Final Audit Checklist.
Scores strategies (0-10) and provides detailed critique.
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd

# Add root to path to import walk_forward
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from walk_forward import WalkForwardAnalyzer
except ImportError:
    # Handling for when run from different CWD
    sys.path.append(str(Path.cwd()))
    from walk_forward import WalkForwardAnalyzer

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient

AUDITOR_SYS_PROMPT = """You are the Chief Risk Officer (CRO) at a quantitative trading firm.
Your job is to reject strategies that are overfit, risky, or mathematically unsound.

You will be given:
1. Strategy Code
2. Walk-Forward Analysis (WFA) Results (In-Sample vs Out-of-Sample)
3. The "Final Audit Checklist" metrics.

Your Task:
1. Calculate a Score (0-10) based on the Checklist.
2. If Score < 8: Provide specific "Pivot Instructions" to fix the flaws.
3. If Score >= 8: APPROVE the strategy.

Checklist Criteria:
- Overfitting: OOS Return must be positive and > 50% of IS Return. (Critical)
- Bias: No look-ahead functions (e.g. shift(-1)).
- Friction: Must use commission >= 0.1%.
- Significance: Trade count > 30 (for this rapid loop).
- Risk: Max Drawdown < 20%.

Output Format:
Must follow the 'Audit Feedback' markdown template exactly.
"""

class AuditorAgent(BaseAgent):
    def __init__(self, verbose: bool = False):
        super().__init__('auditor', verbose)
        self.llm = LLMClient()
        self.strategies_dir = Path('strategies')
        self.feedback_file = Path('audit_feedback.md')
        self.data_path = Path('data/BTC-USD-15m.csv') # Default for now

    def run(self, strategy_file: str = None) -> bool:
        """
        Run the audit cycle.
        
        Args:
            strategy_file: Name of strategy file (e.g. 'my_strategy.py'). 
                           If None, finds latest modified file in strategies/.
        """
        self.log_action("START", "Starting audit cycle")
        
        # 1. Identify Target Strategy
        if strategy_file:
            target_path = self.strategies_dir / strategy_file
        else:
            # Find latest modified python file in strategies/
            py_files = list(self.strategies_dir.glob("*.py"))
            if not py_files:
                self.log_action("ERROR", "No strategy files found to audit.")
                return False
            target_path = max(py_files, key=lambda f: f.stat().st_mtime)
        
        strategy_name = target_path.stem
        self.log_action("TARGET", f"Auditing {strategy_name}...")
        
        # 2. Run Technical Audit (WFA)
        try:
            wfa_results = self._run_wfa(strategy_name)
        except Exception as e:
            self.log_action("ERROR", f"WFA failed: {e}")
            wfa_results = {"error": str(e)}
        
        # 3. Read Strategy Code for Logic Audit
        strategy_code = self._read_file(target_path)
        
        # 4. Generate Critique
        full_prompt = self._construct_prompt(strategy_name, strategy_code, wfa_results)
        
        model = self.get_model()
        self.log_action("THINK", f"Generating critique using {model}...")
        
        critique = self.llm.generate(
            model=model,
            prompt=full_prompt,
            system_instruction=AUDITOR_SYS_PROMPT
        )
        
        if not critique:
            self.log_action("ERROR", "Failed to generate critique")
            return False
            
        # 5. Save Output
        self._write_file(self.feedback_file, critique)
        self.log_action("COMPLETE", f"Audit feedback written to {self.feedback_file}")
        
        # Parse score to return bool success/fail
        if "Score: 8" in critique or "Score: 9" in critique or "Score: 10" in critique:
            self.log_action("RESULT", "Strategy PASSED audit! ✅")
            return True
        else:
            self.log_action("RESULT", "Strategy FAILED audit. ❌")
            return False

    def _run_wfa(self, strategy_name: str) -> dict:
        """Run Walk-Forward Analysis using existing infrastructure."""
        if not self.data_path.exists():
             raise FileNotFoundError(f"Data file not found: {self.data_path}")
             
        # Load data
        data = pd.read_csv(self.data_path, parse_dates=True, index_col="datetime")
        
        # Initialize WFA
        # Note: WalkForwardAnalyzer usually expects class, but run_single_split finds it by name in strategies/
        # We need to make sure walk_forward.py can find it.
        # It looks in `strategies/`. Our file is there.
        
        wfa = WalkForwardAnalyzer()
        results = wfa.run_single_split(strategy_name, data)
        return results

    def _construct_prompt(self, name: str, code: str, wfa: dict) -> str:
        prompt = f"AUDIT TARGET: {name}\n\n"
        prompt += f"--- STRATEGY CODE ---\n{code}\n\n"
        prompt += f"--- TECHNICAL METRICS (WFA 70/30 SPLIT) ---\n"
        prompt += json.dumps(wfa, indent=2)
        return prompt

if __name__ == "__main__":
    agent = AuditorAgent(verbose=True)
    agent.run()
