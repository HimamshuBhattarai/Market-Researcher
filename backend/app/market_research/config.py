import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# Research budget knobs
MAX_QUESTIONS = 6
RESULTS_PER_QUERY = 2
MAX_FINDINGS_PER_QUESTION = 3

DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_api_key: str
    llm_model: str
    llm_base_url: str
    llm_json_mode: str
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
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    json_mode = os.getenv("LLM_JSON_MODE", "auto").strip().lower()
    if json_mode not in {"auto", "on", "off"}:
        json_mode = "auto"
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    search_key = tavily_key or _require("SEARCH_API_KEY")

    if provider == "groq":
        key = _require("groq")
        model = os.getenv("GROQ_MODEL", "").strip() or "openai/gpt-oss-20b"
        base_url = GROQ_BASE_URL
        max_chars = 2000
    elif provider == "ollama":
        key = "ollama"  # the client requires a non-empty key; Ollama ignores it
        model = _require("OLLAMA_MODEL")
        base_url = os.getenv("OLLAMA_BASE_URL", "").strip() or DEFAULT_OLLAMA_URL
        max_chars = 2500  # local models often run with small context windows
    else:
        raise ConfigError(
            f"LLM_PROVIDER must be 'groq' or 'ollama', got '{provider}'."
        )

    return Settings(
        llm_provider=provider,
        llm_api_key=key,
        llm_model=model,
        llm_base_url=base_url,
        llm_json_mode=json_mode,
        search_api_key=search_key,
        max_chars_per_source=max_chars,
    )