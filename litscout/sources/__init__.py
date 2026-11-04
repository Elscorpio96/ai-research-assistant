"""Search providers. Each turns a query into a list of normalised Paper records."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ..config import Settings
from ..models import Paper
from .arxiv import search_arxiv
from .crossref import search_crossref
from .demo import search_demo
from .openalex import search_openalex

PROVIDERS = {
    "openalex": ("OpenAlex", search_openalex),
    "arxiv": ("arXiv", search_arxiv),
    "crossref": ("Crossref", search_crossref),
    "demo": ("Demo corpus (offline)", search_demo),
}


@dataclass
class SourceResult:
    source: str
    papers: list[Paper]
    error: str | None = None


class SourceError(Exception):
    """A provider failed in a way worth showing to the user."""


def run_search(
    settings: Settings,
    query: str,
    sources: list[str],
    limit: int,
    year_from: int | None,
    year_to: int | None,
    client: httpx.Client | None = None,
) -> list[SourceResult]:
    """Query each source independently: one failing source never loses the others' results."""
    own = client is None
    headers = {"User-Agent": "LitScout/1.0 (https://github.com/Elscorpio96/ai-research-assistant)"}
    client = client or httpx.Client(timeout=settings.http_timeout, headers=headers, follow_redirects=True)
    out = []
    try:
        for name in sources:
            if name not in PROVIDERS or name not in settings.sources:
                out.append(SourceResult(name, [], "Source is not enabled"))
                continue
            fn = PROVIDERS[name][1]
            try:
                papers = fn(client, query, limit=limit, year_from=year_from, year_to=year_to, mailto=settings.contact_email)
                out.append(SourceResult(name, papers))
            except httpx.TimeoutException:
                out.append(SourceResult(name, [], "Timed out — the service may be slow or unreachable"))
            except httpx.HTTPStatusError as e:
                out.append(SourceResult(name, [], f"Service returned HTTP {e.response.status_code}"))
            except httpx.HTTPError as e:
                out.append(SourceResult(name, [], f"Network error: {type(e).__name__}"))
            except (SourceError, ValueError, KeyError) as e:
                out.append(SourceResult(name, [], f"Could not read the response: {e}"))
    finally:
        if own:
            client.close()
    return out
