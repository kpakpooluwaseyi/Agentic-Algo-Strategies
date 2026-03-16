"""
Developer Agent
===============
Role: Senior Quant Developer
Task: Convert Research Thesis into robust, backtestable Python code.
"""

from pathlib import Path
import re
import subprocess
import logging
from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient

DEV_SYS_PROMPT = """You are a Senior Quantitative Developer at a top-tier HFT firm.
Your task is to implement trading strategies in Python using the `backtesting.py` framework.

Reference Architecture:
1. All strategies MUST inherit from `MoonDevStrategy` (provided in environment).
2. Use `pandas_ta` for standard indicators.
3. Use `self.I` wrapper for all indicator calculations in `init()`.
4. Implement entry/exit logic in `next()`.

Key Requirements:
- ROBUSTNESS: Handle NaN values, data gaps, and edge cases.
- SPEED: Vectorize calculations where possible (inside `init`).
- SAFETY: No infinite loops, no file system writes, no external network calls.
- SYNTAX: Valid Python 3.12+.

You will be given:
1. A Research Thesis describing the strategy.
2. (Optional) Auditor Feedback from previous failed attempts.

Output:
Return ONLY the complete Python code block. No markdown fencing, no conversational text.
"""

STRATEGY_TEMPLATE = """
from src.strategies.base import MoonDevStrategy
from backtesting.lib import crossover
import pandas_ta as ta

class {ClassName}(MoonDevStrategy):
    # Optimizable parameters
    risk_pct = 1.0
    
    def init(self):
        # Precompute indicators here using self.I
        # e.g. self.sma = self.I(ta.sma, self.data.Close, length=20)
        pass
        
    def next(self):
        # Trading logic per bar
        # Use self.data.Close[-1], self.sma[-1] etc.
        pass

if __name__ == "__main__":
    from backtesting import Backtest
    import pandas as pd
    from src.strategies.base import MoonDevStrategy
    import os
    
    # Load data
    data_path = "data/BTC-USD-15m.csv"
    if os.path.exists(data_path):
        data = pd.read_csv(data_path, parse_dates=True, index_col="datetime")
        
        # Run backtest
        bt = Backtest(data, {ClassName}, cash=10000, commission=.002)
        stats = bt.run()
        print(stats)
        try:
             bt.plot()
        except:
             pass
    else:
        print(f"Data file not found: {data_path}")
"""

class DeveloperAgent(BaseAgent):
    def __init__(self, verbose: bool = False):
        super().__init__('developer', verbose)
        self.llm = LLMClient()
        self.thesis_file = Path('research_thesis.md')
        self.feedback_file = Path('audit_feedback.md')
        self.strategies_dir = Path('strategies')

    def run(self) -> bool:
        """
        Run the development cycle.
        """
        self.log_action("START", "Starting development cycle")
        
        # 1. Gather Context
        if not self.thesis_file.exists():
            self.log_action("ERROR", "No research thesis found.")
            return False
            
        context = [f"--- RESEARCH THESIS ---\n{self._read_file(self.thesis_file)}"]
        
        if self.feedback_file.exists():
            feedback = self._read_file(self.feedback_file)
            if "status: failed" in feedback.lower():
                self.log_action("CONTEXT", "Found previous Auditor feedback. Fixing...")
                context.append(f"--- FAILED AUDIT FEEDBACK ---\n{feedback}")

        full_prompt = "\n\n".join(context)
        
        # 2. Select Model (Primary or Fallback)
        # Check if we are in a retry loop (simple approximation: if feedback exists, maybe use fallback?)
        use_fallback = False # specific logic can be added
        model = self.get_model(use_fallback)
        
        self.log_action("THINK", f"Generating code using {model}...")
        
        # 3. Generate Code
        code = self.llm.generate(
            model=model,
            prompt=full_prompt + "\n\nImplement the above thesis. Use this template:\n" + STRATEGY_TEMPLATE,
            system_instruction=DEV_SYS_PROMPT
        )
        
        if not code:
            self.log_action("ERROR", "Failed to generate code")
            return False

        # Clean code (remove markdown fences if present)
        code = self._clean_code(code)
        
        # 4. Save and Verify
        # Extract class name to name the file
        class_match = re.search(r'class\s+(\w+)\s*\(', code)
        if class_match:
            class_name = class_match.group(1)
            # CamelCase to snake_case for filename
            filename = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower() + ".py"
        else:
            filename = "generated_strategy.py"
            class_name = "GeneratedStrategy" # Default if regex fails but code might still handle it?
        
        # Ensure the template {ClassName} is replaced if the LLM didn't use the template exactly
        # Actually LLM should generate the full code. 
        # But if we want to enforce the main block, we typically append it.
        # But the prompt asks LLM to use the template.
        # If LLM returns the code with the main block, good.
        # If not, we might need to append it.
        # For now, simplistic approach: Trust LLM to follow "Use this template"
        
        file_path = self.strategies_dir / filename
        self._write_file(file_path, code)
        
        # 5. Syntax Check
        if self._check_syntax(file_path):
            self.log_action("COMPLETE", f"Strategy written to {file_path}")
            return True
        else:
            self.log_action("ERROR", "Generated code failed syntax check")
            return False

    def _clean_code(self, text: str) -> str:
        """Remove markdown code blocks."""
        if "```" in text:
            pattern = r"```(?:python)?\n(.*?)```"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1)
        return text

    def _check_syntax(self, file_path: Path) -> bool:
        """Verify Python syntax."""
        try:
            subprocess.check_call(
                ["python3", "-m", "py_compile", str(file_path)],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE
            )
            return True
        except subprocess.CalledProcessError:
            return False

if __name__ == "__main__":
    agent = DeveloperAgent(verbose=True)
    agent.run()
