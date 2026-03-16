"""
LLM Client Wrapper
==================
Abstracts interactions with Gemini and Claude APIs.
Handles authentication, model routing, and error handling.
"""

import os
import logging
import time
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Logger
logger = logging.getLogger("LLMClient")

class LLMClient:
    """
    Unified client for LLM interactions.
    Supports Google Generative AI (Gemini) and Anthropic (Claude).
    """
    
    def __init__(self):
        self._init_gemini()
        self._init_anthropic()

    def _init_gemini(self):
        try:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_available = True
                logger.info("✅ Gemini API initialized")
            else:
                self.gemini_available = False
                logger.warning("⚠️ Gemini API key not found")
        except ImportError:
            self.gemini_available = False
            logger.warning("⚠️ google-generativeai package not found")

    def _init_anthropic(self):
        try:
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.anthropic_client = Anthropic(api_key=api_key)
                self.anthropic_available = True
                logger.info("✅ Anthropic API initialized")
            else:
                self.anthropic_available = False
                logger.warning("⚠️ Anthropic API key not found")
        except ImportError:
            self.anthropic_available = False
            logger.warning("⚠️ anthropic package not found")

    def generate(self, model: str, prompt: str, system_instruction: str = None, temp: float = 0.7) -> Optional[str]:
        """
        Generate content using the specified model.
        
        Args:
            model: Model identifier (e.g. 'gemini-1.5-pro', 'claude-3-opus')
            prompt: User prompt
            system_instruction: System instruction (supported by both)
            temp: Temperature
            
        Returns:
            Generated text or None on failure
        """
        if "gemini" in model.lower():
            return self._generate_gemini(model, prompt, system_instruction, temp)
        elif "claude" in model.lower():
            return self._generate_claude(model, prompt, system_instruction, temp)
        else:
            logger.error(f"❌ Unknown model provider for: {model}")
            return None

    def _generate_gemini(self, model_name: str, prompt: str, system: str, temp: float) -> Optional[str]:
        if not self.gemini_available:
            logger.error("❌ Gemini API not available")
            return None
            
        try:
            import google.generativeai as genai
            
            # Map robust model names to API names if needed
            # For now assuming registry has correct API names or aliases
            # But the user registry had 'gemini-2.0-flash-exp'
            # We trust the registry unless mapping is needed
            
            # Configure generation config
            config = genai.types.GenerationConfig(
                temperature=temp,
                max_output_tokens=8192000 if 'flash' in model_name else 8192
            )
            
            # Gemini Python SDK handles system prompts differently based on version
            # Using standard model.generate_content with request options
            
            # Note: 2.5/Flash/Pro support system instructions in instantiation
            model = genai.GenerativeModel(model_name, system_instruction=system)
            
            response = model.generate_content(prompt, generation_config=config)
            return response.text
            
        except Exception as e:
            logger.error(f"❌ Gemini generation failed ({model_name}): {e}")
            return None

    def _generate_claude(self, model_name: str, prompt: str, system: str, temp: float) -> Optional[str]:
        if not self.anthropic_available:
            logger.error("❌ Anthropic API not available")
            return None
            
        try:
            # Map registry model names to API names
            # Registry: claude-3-opus-20240229
            # API: claude-3-opus-20240229
            
            message = self.anthropic_client.messages.create(
                model=model_name,
                max_tokens=4096,
                temperature=temp,
                system=system,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"❌ Claude generation failed ({model_name}): {e}")
            return None
