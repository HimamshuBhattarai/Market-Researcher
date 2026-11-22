from dataclasses import dataclass

import requests

from market_research.config import get_settings

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class SearchError(RuntimeError):
    pass


@dataclass
class SearchResult:
    url: str
    title: str
    text: str
    published: str | None


def search_web(query: str, limit: int = 3) -> list[SearchResult]:
    s = get_settings()
    try:
        r = requests.post(
            TAVILY_SEARCH_URL,
            json={
                "api_key": s.search_api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": limit,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=90,
        )
    except requests.RequestException as e:
        raise SearchError(f"Search request failed: {e}") from e

    if r.status_code != 200:
        raise SearchError(f"Tavily returned {r.status_code}: {r.text[:200]}")

    results: list[SearchResult] = []
    for item in r.json().get("results", []):
        url = item.get("url")
        text = (item.get("content") or "").strip()
        if not url or not text:
            continue
        results.append(
            SearchResult(
                url=url,
                title=item.get("title") or url,
                text=text[: s.max_chars_per_source],
                published=item.get("published_date"),
            )
        )
    return results