"""
Researcher Agent
================
Role: Lead Quantitative Research Strategist
Task: Identify market inefficiencies and generate trading theses.
"""

from pathlib import Path
from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
import re

RESEARCH_SYS_PROMPT = """You are a Senior Quantitative Researcher at a top-tier HFT firm.
Your mandate is to identify specific, mathematically definable market inefficiencies.

Rules:
1. NO vagueness. "Wait for a pullback" is unacceptable. "Price retraces to 0.5 Fib of last 4H impulse" is clear.
2. Focus on quantifiable edges: Order flow imbalance, statistical arbitrage, volatility regimes, specific indicator confluences.
3. Output MUST follow the "Research Thesis" markdown format exactly.
4. If providing Python formulas, ensure they are compatible with pandas/numpy.

You will be given:
1. Current Market Context (if available)
2. Auditor Feedback from previous iterations (if available)
3. Research notes or raw data.

Your Goal: Produce a high-probability alpha thesis that a Developer Agent can code without ambiguity.
"""

class ResearcherAgent(BaseAgent):
    def __init__(self, verbose: bool = False):
        super().__init__('researcher', verbose)
        self.llm = LLMClient()
        self.thesis_file = Path('research_thesis.md')
        self.feedback_file = Path('audit_feedback.md')
        self.inputs_dir = Path('research_inputs')

    def run(self, specific_input: str = None) -> bool:
        """
        Run the research cycle.
        
        Args:
            specific_input: Optional raw text to analyze (overrides file scanning)
        """
        self.log_action("START", "Starting research cycle")
        
        # 1. Gather Context
        context = []
        
        # Check for feedback (Loop iteration)
        if self.feedback_file.exists():
            feedback = self._read_file(self.feedback_file)
            if "status: failed" in feedback.lower():
                self.log_action("CONTEXT", "Found previous Auditor feedback. Pivoting...")
                context.append(f"--- PREVIOUS AUDIT FEEDBACK ---\n{feedback}")
            else:
                 pass
        
        # Check for inputs
        if specific_input:
             context.append(f"--- RAW RESEARCH ASSIGNMENT ---\n{specific_input}")
        else:
            idea_files = list(self.inputs_dir.glob("*.txt"))
            if idea_files:
                seed_data = self._read_file(idea_files[0])
                context.append(f"--- RESEARCH SEED DATA ({idea_files[0].name}) ---\n{seed_data[:5000]}...") 
        
        if not context:
            context.append("No specific external input. Generate a novel alpha based on current market microstructure principles (e.g. Mean Reversion in high vol regimes).")

        full_prompt = "\n\n".join(context)
        
        # 2. Select Model with Automatic Fallback
        use_fallback = False
        model = self.get_model(use_fallback)
        
        # Check if model provider is available in LLMClient
        if "claude" in model.lower() and not self.llm.anthropic_available:
             self.log_action("WARNING", f"Primary model {model} unavailable (No Anthropic Key). Switching to fallback.")
             use_fallback = True
             model = self.get_model(use_fallback)

        if "gemini" in model.lower() and not self.llm.gemini_available:
             self.log_action("ERROR", f"Model {model} unavailable (No Gemini Key). Cannot proceed.")
             return False
        
        self.log_action("THINK", f"Generating thesis using {model}...")
        
        # 3. Generate
        thesis = self.llm.generate(
            model=model,
            prompt=full_prompt,
            system_instruction=RESEARCH_SYS_PROMPT
        )
        
        if not thesis:
            self.log_action("ERROR", "Failed to generate thesis")
            return False

        # 4. Save Output
        self._write_file(self.thesis_file, thesis)
        self.log_action("COMPLETE", f"Thesis written to {self.thesis_file}")
        
        return True

if __name__ == "__main__":
    agent = ResearcherAgent(verbose=True)
    agent.run()
