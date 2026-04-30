# skills/web_research/tools.py

import random
import re

from toddly.infra.logging import get_logger

logger = get_logger(__name__)

# Maximum characters returned by fetch_url before truncation.
# trafilatura extracts only the main article body, so 8 000 chars of its output
# is far denser than 8 000 chars of raw get_text() output.  The executor's
# max_tool_result_chars setting applies a second cap before the result reaches
# the model; this limit exists only to prevent extremely large pages from
# consuming the whole context window.
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

# Common English stop words stripped before relevance scoring.
# Keeping this list short and focused on function words that carry no
# topic-specific meaning prevents false negatives on queries that are
# phrased differently from the result text.
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "of",
        "in",
        "on",
        "to",
        "with",
        "by",
        "at",
        "from",
        "how",
        "what",
        "why",
        "when",
        "where",
        "is",
        "are",
        "do",
        "does",
        "can",
        "could",
        "would",
        "should",
        "will",
        "be",
        "been",
        "have",
        "has",
        "had",
        "not",
        "no",
        "vs",
        "versus",
        "using",
        "via",
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


def _query_signal_words(query: str) -> frozenset:
    """
    Return the meaningful (non-stop-word) lowercase tokens from a search query.

    Only tokens longer than two characters and not in ``_STOP_WORDS`` are
    returned.  These are the terms we expect to appear somewhere in any result
    that is genuinely about the query topic.
    """
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return frozenset(t for t in tokens if t not in _STOP_WORDS and len(t) > 2)


def _results_are_relevant(query: str, results_text: str, min_hits: int = 2) -> bool:
    """
    Return True if at least ``min_hits`` query signal words appear somewhere
    in the combined result titles and snippets (case-insensitive).

    This is an intentionally loose, fast check — it only catches clearly
    off-topic results, such as a page about Python's ``@`` decorator symbol
    being returned for a query about Python package management.

    ``min_hits=2`` means a single shared token (e.g. "python") is not enough
    to declare relevance; at least two distinct signal words must appear.
    When the query has only one signal word, a single hit is required instead
    (``min(min_hits, len(signal_words))`` handles this gracefully).

    No ML or external calls are made — the check is a plain token-overlap
    over lowercase text, so it adds negligible latency.
    """
    signal_words = _query_signal_words(query)
    if not signal_words:
        # Cannot judge relevance — assume the results are fine.
        return True

    haystack = results_text.lower()
    hits = sum(1 for w in signal_words if w in haystack)
    threshold = min(min_hits, len(signal_words))
    return hits >= threshold


# ── Search-engine configuration (set by configure() in _init_system) ─────────
# Defaults to DuckDuckGo so the skill works out-of-the-box with no API keys.
# Call configure(cfg) once per run to switch to Google Custom Search.

_SEARCH_ENGINE: str = "duckduckgo"  # "duckduckgo" | "google"
_GOOGLE_API_KEY: str = ""
_GOOGLE_CX: str = ""  # Custom Search Engine ID
_MAX_RESULTS: int = 5


def configure(cfg: dict) -> None:
    """
    Apply [web_research] config settings to this module.

    Must be called once per run from _init_system, after SkillLoader has
    registered the tools and before the orchestrator loop starts.

    Parameters
    ----------
    cfg : dict
        The parsed [web_research] config section (e.g. from
        get_web_research_cfg(full_cfg)).  Keys:
          search_engine   "duckduckgo" (default) or "google"
          google_api_key  required when search_engine = "google"
          google_cx       Custom Search Engine ID, required for Google
          max_results     number of results to request (default 5)
    """
    global _SEARCH_ENGINE, _GOOGLE_API_KEY, _GOOGLE_CX, _MAX_RESULTS

    engine = cfg.get("search_engine", "duckduckgo").lower().strip()
    if engine not in ("duckduckgo", "google"):
        logger.warning(
            "[WEB_SEARCH] Unknown search_engine %r in config — falling back to duckduckgo",
            engine,
        )
        engine = "duckduckgo"

    _SEARCH_ENGINE = engine
    _GOOGLE_API_KEY = cfg.get("google_api_key", "").strip()
    _GOOGLE_CX = cfg.get("google_cx", "").strip()
    _MAX_RESULTS = int(cfg.get("max_results", 5))

    if engine == "google":
        missing = [
            k for k, v in [("google_api_key", _GOOGLE_API_KEY), ("google_cx", _GOOGLE_CX)] if not v
        ]
        if missing:
            logger.error(
                "[WEB_SEARCH] search_engine=google but missing config key(s): %s — "
                "searches will fail until these are set in [web_research]",
                ", ".join(missing),
            )
        else:
            logger.info("[WEB_SEARCH] Configured to use Google Custom Search (cx=%s)", _GOOGLE_CX)
    else:
        logger.info("[WEB_SEARCH] Configured to use DuckDuckGo")


def _web_search(args: dict) -> str:
    """
    Dispatch to the configured search backend (DuckDuckGo or Google).

    The backend is selected by configure() at startup via [web_research]
    search_engine in config.toml.  Defaults to DuckDuckGo so the skill
    works out-of-the-box with no API keys.
    """
    if _SEARCH_ENGINE == "google":
        return _web_search_google(args)
    return _web_search_duckduckgo(args)


def _web_search_duckduckgo(args: dict) -> str:
    """
    Search the web using DuckDuckGo (no API key required).

    Placeholder tokens such as "unknown" are stripped from the query
    before the search fires.  If the cleaned query is too short to be
    meaningful the call returns an informative message rather than
    submitting a nonsensical query.

    After results are returned, a lightweight relevance check confirms
    that the result titles and snippets share at least two signal words
    with the query.  Results that fail this check are treated as errors
    so the executor marks them as failed tool calls.

    Requires:  pip install duckduckgo-search
    """
    raw_query = args.get("query", "").strip()
    max_results = _MAX_RESULTS

    if not raw_query:
        return "ERROR: query is required"

    query, removed = _sanitise_query(raw_query)

    if removed:
        logger.info(
            "[WEB_SEARCH] Stripped placeholder token(s) from query: %s → %r",
            removed,
            query,
        )

    meaningful_words = [w for w in query.split() if len(w) > 2]
    if len(meaningful_words) < 2:
        return (
            f"SEARCH SKIPPED: query {raw_query!r} contained only placeholder "
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
    for attempt in range(5):
        if attempt > 0:
            sleep_secs = 5 * (2 ** (attempt - 1)) + random.uniform(0, 2)
            logger.info(
                "[WEB_SEARCH] Backing off %.1fs before attempt %d for: %r",
                sleep_secs,
                attempt + 1,
                query,
            )
            time.sleep(sleep_secs)
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(
                    query,
                    max_results=max_results,
                    region="wt-wt",
                ):
                    results.append(
                        f"Title: {r.get('title', '')}\n"
                        f"URL:   {r.get('href', '')}\n"
                        f"Snippet: {r.get('body', '')}"
                    )
            if results:
                combined = "\n\n---\n\n".join(results)

                if not _results_are_relevant(query, combined):
                    first_title = results[0].split("\n")[0] if results else "(unknown)"
                    logger.warning(
                        "[WEB_SEARCH] Query %r → %d result(s) but content appears "
                        "irrelevant (top result: %s)",
                        query,
                        len(results),
                        first_title,
                    )
                    last_error = (
                        f'ERROR: results for "{query}" appear unrelated to the query '
                        f"(top result: {first_title}). "
                        "Try a different search query with more specific or different keywords."
                    )
                    if attempt == 0:
                        logger.info("[WEB_SEARCH] Relevance failure on attempt 1 — retrying once")
                        continue
                    break

                logger.info(
                    "[WEB_SEARCH] Query: %r → %d result(s) (attempt %d)",
                    query,
                    len(results),
                    attempt + 1,
                )
                return combined

            logger.warning("[WEB_SEARCH] No results on attempt %d for: %r", attempt + 1, query)
            last_error = f'ERROR: no results found for "{query}" (attempt {attempt + 1}/5)'
        except Exception as e:
            logger.error("[WEB_SEARCH] Attempt %d failed: %s", attempt + 1, e)
            last_error = f"ERROR: web search failed — {e}"

    return (
        last_error or f'ERROR: no results found for "{query}"'
    ) + " — try a different search query with different keywords or synonyms."


def _web_search_google(args: dict) -> str:
    """
    Search the web using Google Custom Search API.

    Requires a Google API key and a Custom Search Engine ID (cx), both
    set via [web_research] google_api_key and google_cx in config.toml.
    The free tier allows 100 queries/day; paid tiers are available from
    Google Cloud Console.

    Results are formatted identically to the DuckDuckGo backend so the
    rest of the executor pipeline is unaffected by which backend is active.

    Requires:  pip install requests
    API docs:  https://developers.google.com/custom-search/v1/overview
    """
    raw_query = args.get("query", "").strip()
    max_results = min(_MAX_RESULTS, 10)  # Google CSE caps at 10 per request

    if not raw_query:
        return "ERROR: query is required"

    if not _GOOGLE_API_KEY or not _GOOGLE_CX:
        return (
            "ERROR: Google Custom Search is not configured. "
            "Set google_api_key and google_cx under [web_research] in config.toml."
        )

    query, removed = _sanitise_query(raw_query)

    if removed:
        logger.info(
            "[WEB_SEARCH] Stripped placeholder token(s) from query: %s → %r",
            removed,
            query,
        )

    meaningful_words = [w for w in query.split() if len(w) > 2]
    if len(meaningful_words) < 2:
        return (
            f"SEARCH SKIPPED: query {raw_query!r} contained only placeholder "
            f"values ({removed}) with no specific searchable terms. "
            "Use your own knowledge to answer this task, or request more "
            "specific information from the user via the clarification node."
        )

    try:
        import requests
    except ImportError:
        return "ERROR: requests is not installed. Run: pip install requests"

    import time

    last_error = None
    for attempt in range(3):
        if attempt > 0:
            sleep_secs = 2 * attempt + random.uniform(0, 1)
            logger.info(
                "[WEB_SEARCH] Google: backing off %.1fs before attempt %d for: %r",
                sleep_secs,
                attempt + 1,
                query,
            )
            time.sleep(sleep_secs)
        try:
            resp = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": _GOOGLE_API_KEY,
                    "cx": _GOOGLE_CX,
                    "q": query,
                    "num": max_results,
                },
                timeout=15,
            )

            if resp.status_code == 429:
                last_error = "ERROR: Google Custom Search rate limit exceeded — try again later"
                continue

            if resp.status_code == 403:
                return (
                    "ERROR: Google Custom Search API key is invalid or the daily quota "
                    "has been exceeded. Check your API key and billing in Google Cloud Console."
                )

            resp.raise_for_status()
            data = resp.json()

            items = data.get("items", [])
            if not items:
                logger.warning(
                    "[WEB_SEARCH] Google: no results on attempt %d for: %r", attempt + 1, query
                )
                last_error = (
                    f'ERROR: no results found for "{query}" (Google, attempt {attempt + 1}/3)'
                )
                continue

            results = [
                f"Title: {item.get('title', '')}\n"
                f"URL:   {item.get('link', '')}\n"
                f"Snippet: {item.get('snippet', '')}"
                for item in items
            ]
            combined = "\n\n---\n\n".join(results)

            if not _results_are_relevant(query, combined):
                first_title = results[0].split("\n")[0] if results else "(unknown)"
                logger.warning(
                    "[WEB_SEARCH] Google query %r → %d result(s) but content appears "
                    "irrelevant (top result: %s)",
                    query,
                    len(results),
                    first_title,
                )
                last_error = (
                    f'ERROR: results for "{query}" appear unrelated to the query '
                    f"(top result: {first_title}). "
                    "Try a different search query with more specific or different keywords."
                )
                if attempt == 0:
                    logger.info(
                        "[WEB_SEARCH] Google: relevance failure on attempt 1 — retrying once"
                    )
                    continue
                break

            logger.info(
                "[WEB_SEARCH] Google query: %r → %d result(s) (attempt %d)",
                query,
                len(results),
                attempt + 1,
            )
            return combined

        except Exception as e:
            logger.error("[WEB_SEARCH] Google attempt %d failed: %s", attempt + 1, e)
            last_error = f"ERROR: Google web search failed — {e}"

    return (
        last_error or f'ERROR: no results found for "{query}" via Google'
    ) + " — try a different search query with different keywords or synonyms."


def _fetch_url(args: dict) -> str:
    """
    Fetch a URL and return its content as cleaned plain text.

    Extraction strategy (in order):
      1. trafilatura  — extracts only the main article body, discarding nav,
                        ads, sidebars, cookie banners, and other boilerplate.
                        Produces the most information-dense result per character.
      2. BeautifulSoup — strips common structural tags then calls get_text().
                         Used when trafilatura returns nothing (e.g. JS-heavy
                         pages, paywalls, or unusual document structures).
      3. Regex strip   — last resort when neither library is installed.

    Requires:  pip install requests trafilatura
    Optional:  pip install beautifulsoup4   (fallback)
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
            text = _extract_main_text(resp.text, url)

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


def _extract_main_text(html: str, url: str = "") -> str:
    """
    Extract the main readable text from an HTML page.

    Tries trafilatura first (main-content extraction), falls back to
    BeautifulSoup get_text(), then to a plain regex tag-strip.
    """
    # ── 1. trafilatura: purpose-built main-content extractor ─────────────────
    # extract() returns None when it cannot identify a main-content region
    # (e.g. pure JS apps, login walls).  We treat that as a failure and fall
    # through to BeautifulSoup so the caller still gets something useful.
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html,
            include_tables=True,
            include_links=False,
            no_fallback=False,
        )
        if extracted:
            logger.debug("[FETCH_URL] trafilatura extracted %d chars from %s", len(extracted), url)
            return extracted
        logger.debug("[FETCH_URL] trafilatura returned empty result for %s — using fallback", url)
    except Exception as e:
        logger.debug("[FETCH_URL] trafilatura failed for %s (%s) — using fallback", url, e)

    # ── 2. BeautifulSoup: strip structural/boilerplate tags, call get_text() ─
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        return soup.get_text(separator="\n")
    except ImportError:
        pass

    # ── 3. Regex: last resort ─────────────────────────────────────────────────
    return re.sub(r"<[^>]+>", " ", html)


TOOLS = {
    "web_search": {
        "description": (
            "Search the web for current information and return titles, URLs, and snippets. "
            "Use for salary data, market research, company info, news, or any real-world fact. "
            "Do NOT include placeholder values like 'unknown' in queries — strip them and search "
            "for the general concept instead. "
            "Args: query (required)."
        ),
        "input_schema": {
            "query": "string",
        },
        "fn": _web_search,
    },
    "fetch_url": {
        "description": (
            "Fetch the content of a URL and return it as cleaned plain text. "
            "Uses trafilatura to extract only the main article body, discarding "
            "navigation, ads, and boilerplate — result is information-dense. "
            "Use after web_search to read the full content of a promising result. "
            "Content is truncated to 8000 characters."
        ),
        "input_schema": {
            "url": "string",
        },
        "fn": _fetch_url,
    },
}
