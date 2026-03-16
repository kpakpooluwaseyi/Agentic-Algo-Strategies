"""
Model Registry for Autonomous Quant Factory
===========================================
Centralized configuration for agent model assignments.
Maps specific agents to their primary and fallback models capabilities.
Updated for Feb 2026 Standards.
"""

from typing import Dict, TypedDict

class ModelConfig(TypedDict):
    primary: str
    fallback: str

# Model Capability Map
# --------------------
# researcher: Alpha discovery, high reasoning
# developer: Code generation, high throughput
# auditor: Risk assessment, high reasoning

MODELS: Dict[str, ModelConfig] = {
    "researcher": {
        "primary": "claude-4-5-opus-20260201",  # Highest reasoning
        "fallback": "gemini-3.0-pro-exp",       # Strong reasoning fallback
    },
    "developer": {
        "primary": "gemini-3.0-pro-001",        # User requested 'Gemini 3 Pro (Low)'
        "fallback": "claude-4-5-sonnet-20260201", # Stronger reasoning for complex strategies
    },
    "auditor": {
        "primary": "claude-4-5-sonnet-20260201", # Balanced reasoning & efficient
        "fallback": "gemini-3.0-pro-exp",        # Strong capabilities fallback
    }
}

# Retry configurations
MAX_RETRIES = {
    "researcher": 2,
    "developer": 3, 
    "auditor": 2
}
