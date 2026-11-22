import market_research.config as config

import pytest


@pytest.fixture(autouse=True)
def clear_settings_cache():
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_missing_groq_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("groq", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)

    with pytest.raises(config.ConfigError):
        config.get_settings()


def test_groq_settings(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("groq", "test-groq-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-search-key")

    settings = config.get_settings()

    assert settings.llm_provider == "groq"
    assert settings.llm_api_key == "test-groq-key"
    assert settings.search_api_key == "test-search-key"


def test_groq_modern_key_and_tavily_primary(monkeypatch):
    config.get_settings.cache_clear()
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-modern-groq-key")
    monkeypatch.setenv("groq", "test-old-groq-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("SEARCH_API_KEY", "test-fallback-key")

    settings = config.get_settings()

    assert settings.llm_api_key == "test-modern-groq-key"
    assert settings.search_api_key == "test-tavily-key"


def test_old_groq_and_search_aliases(monkeypatch):
    config.get_settings.cache_clear()
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("groq", "test-old-groq-key")
    monkeypatch.setenv("SEARCH_API_KEY", "test-fallback-key")

    settings = config.get_settings()

    assert settings.llm_api_key == "test-old-groq-key"
    assert settings.search_api_key == "test-fallback-key"


def test_ollama_settings(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    monkeypatch.setenv("TAVILY_API_KEY", "test-search-key")

    settings = config.get_settings()

    assert settings.llm_provider == "ollama"
    assert settings.llm_api_key == "ollama"
    assert settings.llm_model == "test-model"
