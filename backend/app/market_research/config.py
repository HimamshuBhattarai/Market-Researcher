import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# Research budget knobs
MAX_QUESTIONS = 6
RESULTS_PER_QUERY = 3
MAX_FINDINGS_PER_QUESTION = 3

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_api_key: str
    llm_model: str
    llm_base_url: str
    search_api_key: str
    max_chars_per_source: int


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"Missing environment variable {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()

    if provider == "openrouter":
        key = _require("OPENROUTER_API_KEY")
        model = _require("OPENROUTER_MODEL")
        base_url = OPENROUTER_BASE_URL
        max_chars = 12000
    elif provider == "ollama":
        key = "ollama"  # the client requires a non-empty key; Ollama ignores it
        model = _require("OLLAMA_MODEL")
        base_url = os.getenv("OLLAMA_BASE_URL", "").strip() or DEFAULT_OLLAMA_URL
        max_chars = 2500  # local models often run with small context windows
    else:
        raise ConfigError(
            f"LLM_PROVIDER must be 'openrouter' or 'ollama', got '{provider}'."
        )

    return Settings(
        llm_provider=provider,
        llm_api_key=key,
        llm_model=model,
        llm_base_url=base_url,
        search_api_key=_require("SEARCH_API_KEY"),
        max_chars_per_source=max_chars,
    )