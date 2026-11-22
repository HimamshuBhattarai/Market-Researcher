import sys

import requests

from openai import OpenAI

from market_research.config import get_settings

settings = get_settings()
print(f"Provider: {settings.llm_provider} | Model: {settings.llm_model}")
failures = 0


def check(name, fn):
    global failures
    print(f"\n=== {name} ===")
    try:
        fn()
        print(f"PASS: {name}")
    except Exception as e:
        failures += 1
        print(f"FAIL: {name}\n  {type(e).__name__}: {e}")


openai_client = (
    OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    if settings.llm_provider in {"groq", "ollama"}
    else None
)


def llm_plain():
    r = openai_client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=10,
    )
    print("Reply:", repr(r.choices[0].message.content))


def llm_json_mode():
    r = openai_client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "user",
                "content": 'Return JSON: {"status": "ok", "n": 3}',
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=50,
    )
    print("Reply:", repr(r.choices[0].message.content))


def tavily_search():
    r = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": settings.search_api_key,
            "query": "food delivery market Nepal",
            "search_depth": "basic",
            "max_results": 2,
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=90,
    )
    print("HTTP status:", r.status_code)
    r.raise_for_status()
    body = r.json()
    print("Top-level keys:", list(body.keys()))
    results = body.get("results", [])
    print("Number of results:", len(results or []))
    if not results:
        raise RuntimeError("No results returned")
    first = results[0]
    print("Result keys:", list(first.keys()))
    print("URL:", first.get("url"))
    print("Title:", first.get("title"))
    print("Content length:", len(first.get("content") or ""))


check("LLM plain call", llm_plain)
check("LLM JSON mode", llm_json_mode)
check("Tavily search + content", tavily_search)

print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILED'}")
sys.exit(1 if failures else 0)