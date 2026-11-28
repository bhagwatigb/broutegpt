import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


DEFAULT_CACHE_FILE = ".free_models_cache.json"

PROVIDER_RATE_INFO = {
    "ollama": ("Unlimited (Local)", "100% Free (Local)"),
    "gemini": ("~300-900 RPH (Flash)", "Free tier (Pro=0 quota)"),
    "groq": ("~1,800 RPH (30 RPM)", "Free tier (RPD capped)"),
    "openrouter": ("~1,200 RPH (50 RPD)", "100% Free (:free)"),
    "nvidia": ("~2,400 RPH (40 RPM)", "Free dev credits"),
    "huggingface": ("~100-300 RPH", "Free monthly quota"),
}


def _redact_secrets(text: object) -> str:
    return re.sub(
        r"([?&](?:key|api_key|apikey|token)=)[^&\s'\"]+",
        r"\1REDACTED",
        str(text),
        flags=re.IGNORECASE,
    )


def _load_env_file(env_path: str = ".env") -> None:
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path)
        return
    path = Path(env_path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _is_valid_key(val: Optional[str]) -> bool:
    return bool(val and val.strip() and not val.strip().startswith("your_"))


def get_or_prompt_api_key(key_name: str, display_name: str, required: bool = False) -> Optional[str]:
    api_key = os.environ.get(key_name)
    if api_key and api_key.strip() and not api_key.startswith("your_"):
        return api_key.strip()
    if not required:
        return None
    if not sys.stdin.isatty():
        print(f"Error: {display_name} not found in environment and running non-interactively.")
        sys.exit(1)
    print(f"Warning: {display_name} not found in .env.")
    user_input = input(f"Please enter your {display_name}: ").strip()
    if not user_input:
        sys.exit(f"Error: {display_name} cannot be empty.")
    return user_input


def load_all_api_keys(env_path: str = ".env") -> Dict[str, str]:
    _load_env_file(env_path)
    loaded = {}
    for key_name, label in [
        ("GEMINI_API_KEY", "Gemini API Key"),
        ("GROQ_API_KEY", "Groq API Key"),
        ("OPENROUTER_API_KEY", "OpenRouter API Key"),
        ("NVIDIA_API_KEY", "NVIDIA API Key"),
        ("HF_API_KEY", "Hugging Face API Key"),
    ]:
        val = get_or_prompt_api_key(key_name, label, required=False)
        if val:
            loaded[key_name] = val
    return loaded

