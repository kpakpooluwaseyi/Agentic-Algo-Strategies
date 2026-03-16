"""
Base Agent Class
================
Abstract base class for all autonomous agents in the Quant Factory.
Standardizes model selection, logging, and execution interface.
"""

from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
from typing import Optional, Any, Dict
from src.agents.model_registry import MODELS, MAX_RETRIES

# Setup base logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class BaseAgent(ABC):
    """
    Abstract base class for autonomous agents.
    """
    
    def __init__(self, agent_type: str, verbose: bool = False):
        """
        Initialize the agent.
        
        Args:
            agent_type: One of 'researcher', 'developer', 'auditor'
            verbose: Enable debug logging
        """
        if agent_type not in MODELS:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        self.agent_type = agent_type
        self.verbose = verbose
        self.logger = logging.getLogger(f"Agent.{agent_type.capitalize()}")
        
        if verbose:
            self.logger.setLevel(logging.DEBUG)
            
        # Load model config
        self.model_config = MODELS[agent_type]
        self.max_retries = MAX_RETRIES.get(agent_type, 2)
        
        # Workspace paths
        self.root_dir = Path(__file__).parent.parent.parent
        self.results_dir = self.root_dir / 'results' / 'autonomous_loop'
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def get_model(self, use_fallback: bool = False) -> str:
        """Get the model string for this agent."""
        model_key = 'fallback' if use_fallback else 'primary'
        return self.model_config[model_key]

    def log_action(self, action: str, details: str):
        """Structured logging of agent actions."""
        self.logger.info(f"[{action}] {details}")

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        Main execution method. Must be implemented by subclasses.
        """
        pass
    
    def _read_file(self, file_path: Path) -> str:
        """Helper to safely read text files."""
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            raise

    def _write_file(self, file_path: Path, content: str):
        """Helper to safely write text files."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            self.logger.debug(f"Wrote to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to write file {file_path}: {e}")
            raise