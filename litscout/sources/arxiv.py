from __future__ import annotations

import re

import httpx
from defusedxml import ElementTree

from ..models import Paper

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def parse(xml: str) -> list[Paper]:
    root = ElementTree.fromstring(xml)
    papers = []
    for e in root.findall("a:entry", NS):

        def text(tag: str, e=e) -> str:
            node = e.find(tag, NS)
            return node.text.strip() if node is not None and node.text else ""

        published = text("a:published")
        abs_url = text("a:id")
        papers.append(
            Paper(
                title=text("a:title"),
                abstract=text("a:summary"),
                authors=[n.text.strip() for n in e.findall("a:author/a:name", NS) if n.text],
                year=int(published[:4]) if re.match(r"\d{4}", published) else None,
                venue=text("arxiv:journal_ref") or "arXiv preprint",
                doi=text("arxiv:doi"),
                url=abs_url,
                source="arxiv",
                source_id=abs_url.rsplit("/", 1)[-1],
            )
        )
    return [p for p in papers if p.title]


def _query(q: str) -> str:
    """Turn free text into an arXiv query: every word must appear in title or abstract."""
    words = re.findall(r"[A-Za-z0-9\-]+", q)
    return " AND ".join(f"all:{w}" for w in words[:12]) or "all:*"


def search_arxiv(
    client: httpx.Client, query: str, *, limit: int, year_from: int | None, year_to: int | None, mailto: str = ""
) -> list[Paper]:
    params = {"search_query": _query(query), "start": 0, "max_results": min(limit, 100), "sortBy": "relevance"}
    r = client.get(API, params=params)
    r.raise_for_status()
    papers = parse(r.text)
    # The arXiv API has no date filter on relevance search, so filter here
    return [p for p in papers if (not year_from or (p.year or 0) >= year_from) and (not year_to or (p.year or 9999) <= year_to)]
