"""
AI PM Exploration Toolkit - Configuration and Environment Settings
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = WORKSPACE_ROOT / "data"

@dataclass
class ToolkitConfig:
    """Toolkit runtime configuration"""
    user_name: str = "Abhinav"
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY", None)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    default_model: str = os.getenv("OPENROUTER_DEFAULT_MODEL", "openai/gpt-4o-mini")
    app_version: str = "2.5.0-pm"
    toolkit_name: str = "Abhinav's AI PM Exploration Toolkit"
    
def ensure_directories():
    """Ensure data and export directories exist"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (WORKSPACE_ROOT / "exports").mkdir(parents=True, exist_ok=True)

ensure_directories()
