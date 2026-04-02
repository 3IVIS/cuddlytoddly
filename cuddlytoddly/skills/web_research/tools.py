# skills/web_research/tools.py

import re
from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)

# Maximum characters returned by fetch_url before truncation.
# Keeps executor prompt sizes manageable.
_MAX_FETCH_CHARS = 8_000


def _web_search(args: dict) -> str:
    """
    Search the web using DuckDuckGo (no API key required).

    Requires:  pip install duckduckgo-search
    """
    query       = args.get("query", "").strip()
    max_results = int(args.get("max_results", 5))

    if not query:
        return "ERROR: query is required"

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return (
            "ERROR: duckduckgo-search is not installed. "
            "Run: pip install duckduckgo-search"
        )

    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"Title: {r.get('title', '')}\n"
                    f"URL:   {r.get('href', '')}\n"
                    f"Snippet: {r.get('body', '')}"
                )

        if not results:
            return f"No results found for: {query}"

        logger.info("[WEB_SEARCH] Query: %r → %d result(s)", query, len(results))
        return "\n\n---\n\n".join(results)

    except Exception as e:
        logger.error("[WEB_SEARCH] Failed: %s", e)
        return f"ERROR: web search failed — {e}"


def _fetch_url(args: dict) -> str:
    """
    Fetch a URL and return its content as cleaned plain text.

    Requires:  pip install requests beautifulsoup4
    HTML tags, scripts, and style blocks are stripped.
    Content is truncated to _MAX_FETCH_CHARS to keep prompts manageable.
    """
    url = args.get("url", "").strip()
    if not url:
        return "ERROR: url is required"

    try:
        import requests
    except ImportError:
        return "ERROR: requests is not installed. Run: pip install requests"

    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; cuddlytoddly/1.0)"},
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")

        # Return plain-text responses directly
        if "text/plain" in content_type:
            text = resp.text
        else:
            # Try BeautifulSoup for HTML; fall back to regex stripping
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                # Remove boilerplate elements
                for tag in soup(["script", "style", "nav", "footer",
                                 "header", "aside", "form"]):
                    tag.decompose()
                text = soup.get_text(separator="\n")
            except ImportError:
                # Regex fallback — less clean but no extra dependency
                text = re.sub(r"<[^>]+>", " ", resp.text)

        # Collapse whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = text.strip()

        if len(text) > _MAX_FETCH_CHARS:
            text = text[:_MAX_FETCH_CHARS] + f"\n\n…[truncated — {len(text)} chars total]"

        logger.info("[FETCH_URL] %s → %d chars", url, len(text))
        return text

    except Exception as e:
        logger.error("[FETCH_URL] Failed for %s: %s", url, e)
        return f"ERROR: could not fetch {url} — {e}"


TOOLS = {
    "web_search": {
        "description": (
            "Search the web for current information and return titles, URLs, and snippets. "
            "Use for salary data, market research, company info, news, or any real-world fact. "
            "Args: query (required), max_results (optional, default 5)."
        ),
        "input_schema": {
            "query":       "string",
            "max_results": "integer (optional, default 5)",
        },
        "fn": _web_search,
    },
    "fetch_url": {
        "description": (
            "Fetch the content of a URL and return it as cleaned plain text. "
            "Use after web_search to read the full content of a promising result. "
            "Content is truncated to 8000 characters."
        ),
        "input_schema": {
            "url": "string",
        },
        "fn": _fetch_url,
    },
}
