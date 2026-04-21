# web_research

## Description
Search the web and fetch page content. Use for tasks that require current information, salary data, market research, company research, news, or any fact that cannot be reliably answered from training knowledge alone.

## When to use
- The task requires current or real-world data (salaries, prices, statistics, news, company info)
- The task involves researching a specific topic, product, person, or organisation
- Upstream context contains unknowns that a web search could resolve

## Tools
- `web_search`: Search the web and return a list of results (title, URL, snippet) for a query
- `fetch_url`: Fetch the content of a URL and return cleaned plain text

## Expected output format
Summarise findings as structured text. Cite the source URL for each key fact. Do not reproduce large blocks of verbatim text.
