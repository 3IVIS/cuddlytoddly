# skills/web_research/tools.py

import re

from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)

# Maximum characters returned by fetch_url before truncation.
_MAX_FETCH_CHARS = 8_000

# Placeholder values the clarification node uses for missing context.
# Queries containing only these tokens after stripping noise cannot be
# answered by any search engine and should not be attempted.
_PLACEHOLDER_TOKENS = frozenset(
    {
        "unknown",
        "n/a",
        "not specified",
        "not provided",
        "none",
        "unspecified",
        "tbd",
        "?",
    }
)


def _sanitise_query(raw: str) -> tuple[str, list[str]]:
    """
    Remove placeholder tokens from a search query.

    Returns (clean_query, removed_tokens).  If nothing useful remains
    after removal the caller should abort the search rather than fire
    a nonsensical query.

    Examples
    --------
    "average salary for job title unknown"
        → ("average salary for job title", ["unknown"])
    "key achievements for current salary unknown and job title unknown"
        → ("key achievements for current salary and job title", ["unknown"])
    "software engineer salaries"
        → ("software engineer salaries", [])
    """
    removed = []
    tokens = raw.split()
    cleaned = []
    for tok in tokens:
        # Strip punctuation from both ends before comparing
        bare = tok.strip(".,;:\"'()[]").lower()
        if bare in _PLACEHOLDER_TOKENS:
            removed.append(tok)
        else:
            cleaned.append(tok)

    clean_query = " ".join(cleaned).strip()
    # Collapse runs of whitespace left by removed tokens
    clean_query = re.sub(r"\s{2,}", " ", clean_query)
    return clean_query, removed


def _web_search(args: dict) -> str:
    """
    Search the web using DuckDuckGo (no API key required).

    Placeholder tokens such as "unknown" are stripped from the query
    before the search fires.  If the cleaned query is too short to be
    meaningful the call returns an informative message rather than
    submitting a nonsensical query.

    Requires:  pip install duckduckgo-search
    """
    raw_query = args.get("query", "").strip()
    max_results = int(args.get("max_results", 5))

    if not raw_query:
        return "ERROR: query is required"

    # ── Fix 1: sanitise placeholder tokens ───────────────────────────────────
    query, removed = _sanitise_query(raw_query)

    if removed:
        logger.info(
            "[WEB_SEARCH] Stripped placeholder token(s) from query: %s → %r",
            removed,
            query,
        )

    # If nothing substantive remains, abort rather than search for noise
    meaningful_words = [w for w in query.split() if len(w) > 2]
    if len(meaningful_words) < 2:
        return (
            f"SEARCH SKIPPED: query '{raw_query}' contained only placeholder "
            f"values ({removed}) with no specific searchable terms. "
            "Use your own knowledge to answer this task, or request more "
            "specific information from the user via the clarification node."
        )

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "ERROR: duckduckgo-search is not installed. Run: pip install duckduckgo-search"

    import time

    last_error = None
    for attempt in range(3):
        if attempt > 0:
            time.sleep(2**attempt)  # 2s, 4s backoff on retry
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        f"Title: {r.get('title', '')}\n"
                        f"URL:   {r.get('href', '')}\n"
                        f"Snippet: {r.get('body', '')}"
                    )
            if results:
                logger.info(
                    "[WEB_SEARCH] Query: %r → %d result(s) (attempt %d)",
                    query,
                    len(results),
                    attempt + 1,
                )
                return "\n\n---\n\n".join(results)
            logger.warning("[WEB_SEARCH] No results on attempt %d for: %r", attempt + 1, query)
            last_error = f"No results found for: {query}"
        except Exception as e:
            logger.error("[WEB_SEARCH] Attempt %d failed: %s", attempt + 1, e)
            last_error = f"ERROR: web search failed — {e}"

    return last_error or f"No results found for: {query}"


def _fetch_url(args: dict) -> str:
    """
    Fetch a URL and return its content as cleaned plain text.

    Requires:  pip install requests beautifulsoup4
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

        if "text/plain" in content_type:
            text = resp.text
        else:
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                    tag.decompose()
                text = soup.get_text(separator="\n")
            except ImportError:
                text = re.sub(r"<[^>]+>", " ", resp.text)

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
            "Do NOT include placeholder values like 'unknown' in queries — strip them and search "
            "for the general concept instead. "
            "Args: query (required), max_results (optional, default 5)."
        ),
        "input_schema": {
            "query": "string",
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
