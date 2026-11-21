from dataclasses import dataclass

import requests

from market_research.config import get_settings

FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v2/search"


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
            FIRECRAWL_SEARCH_URL,
            headers={"Authorization": f"Bearer {s.search_api_key}"},
            json={
                "query": query,
                "limit": limit,
                "scrapeOptions": {"formats": ["markdown"]},
            },
            timeout=90,
        )
    except requests.RequestException as e:
        raise SearchError(f"Search request failed: {e}") from e

    if r.status_code != 200:
        raise SearchError(f"Firecrawl returned {r.status_code}: {r.text[:200]}")

    data = r.json().get("data") or {}
    items = data.get("web", []) if isinstance(data, dict) else data

    results: list[SearchResult] = []
    for item in items or []:
        url = item.get("url")
        meta = item.get("metadata") or {}
        text = (item.get("markdown") or item.get("description") or "").strip()
        if not url or not text:
            continue
        published = (
            meta.get("publishedTime")
            or meta.get("publishedDate")
            or meta.get("article:published_time")
        )
        results.append(
            SearchResult(
                url=url,
                title=item.get("title") or meta.get("title") or url,
                text=text[: s.max_chars_per_source],
                published=str(published) if published else None,
            )
        )
    return results