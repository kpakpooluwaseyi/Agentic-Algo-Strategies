"""
Test Autonomous Quant Loop
==========================
Integration tests for the Researcher -> Developer -> Auditor loop.
Mocks LLM interactions to verify logic flow without API costs.
"""

import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.researcher_agent import ResearcherAgent
from src.agents.developer_agent import DeveloperAgent
from src.agents.auditor_agent import AuditorAgent

def test_researcher_agent(tmp_path):
    # Setup
    os.chdir(tmp_path)
    (tmp_path / "research_inputs").mkdir()
    
    with patch('src.agents.researcher_agent.LLMClient') as MockClient:
        mock_llm = MockClient.return_value
        mock_llm.generate.return_value = "# Research Thesis\n\nMarket Inefficiency: Buy when RSI < 30."
        
        agent = ResearcherAgent(verbose=True)
        # Mock inputs
        with patch.object(Path, "glob", return_value=[]): # No files found, uses blue sky
             success = agent.run()
    
    assert success
    assert (tmp_path / "research_thesis.md").exists()
    assert "RSI < 30" in (tmp_path / "research_thesis.md").read_text()

def test_developer_agent(tmp_path):
    # Setup
    os.chdir(tmp_path)
    (tmp_path / "strategies").mkdir()
    (tmp_path / "research_thesis.md").write_text("# Thesis\nStrategy")
    
    # Mock code generation
    mock_code = """
from src.strategies.base import MoonDevStrategy
class MockStrategy(MoonDevStrategy):
    def init(self): pass
    def next(self): pass
if __name__ == "__main__": pass
"""
    
    with patch('src.agents.developer_agent.LLMClient') as MockClient:
        mock_llm = MockClient.return_value
        mock_llm.generate.return_value = mock_code
        
        agent = DeveloperAgent(verbose=True)
        # Mock simple syntax check
        with patch.object(agent, '_check_syntax', return_value=True):
            success = agent.run()
        
    assert success
    assert (tmp_path / "strategies/mock_strategy.py").exists()

def test_auditor_agent(tmp_path):
    # Setup
    os.chdir(tmp_path)
    (tmp_path / "strategies").mkdir()
    strategy_file = tmp_path / "strategies/mock_strategy.py"
    strategy_file.write_text("class MockStrategy: pass")
    
    # Mock WFA
    with patch('src.agents.auditor_agent.WalkForwardAnalyzer') as MockWFA:
        wfa_instance = MockWFA.return_value
        wfa_instance.run_single_split.return_value = {"roi": 0.1, "sharpe": 2.0}
        
        with patch('src.agents.auditor_agent.LLMClient') as MockClient:
            mock_llm = MockClient.return_value
            # Mock LLM critique (PASS)
            mock_llm.generate.return_value = "# Audit Feedback\n\nScore: 9\nStatus: PASSED"
            
            agent = AuditorAgent(verbose=True)
            # Mock data existence check (since data path won't exist in tmp)
            with patch.object(Path, 'exists', return_value=True):
                 # We need to ensure _run_wfa doesn't fail on pd.read_csv if run for real
                 # WFA mocked above, so pd.read_csv call inside _run_wfa is bypassed?
                 # No, _run_wfa calls pd.read_csv BEFORE calling wfa.run_single_split.
                 # So we must mock pd.read_csv too? 
                 # Or mock _run_wfa entirely?
                 with patch.object(agent, '_run_wfa', return_value={"roi": 0.1}):
                      success = agent.run()
            
    assert success
    assert (tmp_path / "audit_feedback.md").exists()
    assert "Score: 9" in (tmp_path / "audit_feedback.md").read_text()

def test_auditor_agent_fail(tmp_path):
    # Setup
    os.chdir(tmp_path)
    (tmp_path / "strategies").mkdir()
    strategy_file = tmp_path / "strategies/fail_strategy.py"
    strategy_file.write_text("class FailStrategy: pass")
    
    with patch('src.agents.auditor_agent.LLMClient') as MockClient:
        mock_llm = MockClient.return_value
        # Mock LLM critique (FAIL)
        mock_llm.generate.return_value = "# Audit Feedback\n\nScore: 4\nStatus: FAILED"
        
        agent = AuditorAgent(verbose=True)
        with patch.object(Path, 'exists', return_value=True):
             with patch.object(agent, '_run_wfa', return_value={"roi": -0.1}):
                  success = agent.run()
            
    assert not success
    assert (tmp_path / "audit_feedback.md").exists()
    assert "Score: 4" in (tmp_path / "audit_feedback.md").read_text()
