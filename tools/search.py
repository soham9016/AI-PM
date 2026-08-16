"""Web search tool — provider-agnostic interface.

The concrete provider is selected via config.SEARCH_PROVIDER, not
hardcoded here or in any agent file. Agents only ever import and call
the `web_search` tool below; none of them import a provider client
directly, so swapping providers (Tavily was acquired by Nebius in Feb
2026; Google sued SerpAPI in Dec 2025 — providers change) means adding
one function + one config value, not touching agents/researcher.py,
agents/primary_research.py, or agents/competitive_audit.py.

Every provider function returns the same normalized shape: a list of
{title, url, content, score} dicts.
"""

from langchain_core.tools import tool

from config import SEARCH_PROVIDER, TAVILY_API_KEY
from utils.retry import with_retry

_tavily_client = None


def _tavily_search(query: str, max_results: int) -> list[dict]:
    global _tavily_client
    if _tavily_client is None:
        if not TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY is not set (check your .env file)")
        from tavily import TavilyClient

        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

    response = _tavily_client.search(query=query, max_results=max_results, search_depth="basic")
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "score": item.get("score", 0.0),
        }
        for item in response.get("results", [])
    ]


_PROVIDERS = {
    "tavily": _tavily_search,
}


@tool
@with_retry(max_attempts=3)
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via the configured provider and return a list of results.

    Each result dict has: title, url, content (snippet), score.
    """
    provider_fn = _PROVIDERS.get(SEARCH_PROVIDER)
    if provider_fn is None:
        raise RuntimeError(
            f"Unknown SEARCH_PROVIDER {SEARCH_PROVIDER!r} — add an implementation to "
            f"tools/search.py's _PROVIDERS (known: {sorted(_PROVIDERS)})"
        )
    return provider_fn(query, max_results)
