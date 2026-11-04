from __future__ import annotations

import httpx

from ..models import Paper

API = "https://api.crossref.org/works"


def parse(data: dict) -> list[Paper]:
    papers = []
    for it in data.get("message", {}).get("items", []):
        parts = ((it.get("issued") or {}).get("date-parts") or [[None]])[0]
        authors = [" ".join(x for x in (a.get("given"), a.get("family")) if x) for a in it.get("author", [])]
        papers.append(
            Paper(
                title=(it.get("title") or [""])[0],
                abstract=it.get("abstract", ""),
                authors=[a for a in authors if a],
                year=parts[0] if parts and isinstance(parts[0], int) else None,
                venue=(it.get("container-title") or [""])[0],
                doi=it.get("DOI", ""),
                url=it.get("URL", ""),
                source="crossref",
                source_id=it.get("DOI", ""),
                citations=it.get("is-referenced-by-count"),
            )
        )
    return [p for p in papers if p.title]


def search_crossref(
    client: httpx.Client, query: str, *, limit: int, year_from: int | None, year_to: int | None, mailto: str = ""
) -> list[Paper]:
    filters = ["type:journal-article,type:proceedings-article,type:posted-content"]
    if year_from:
        filters.append(f"from-pub-date:{year_from}")
    if year_to:
        filters.append(f"until-pub-date:{year_to}")
    params = {
        "query.bibliographic": query,
        "rows": min(limit, 100),
        "filter": ",".join(filters),
        "select": "DOI,title,author,issued,container-title,abstract,URL,is-referenced-by-count",
    }
    if mailto:
        params["mailto"] = mailto
    r = client.get(API, params=params)
    r.raise_for_status()
    return parse(r.json())
