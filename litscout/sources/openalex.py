from __future__ import annotations

import httpx

from ..models import Paper

API = "https://api.openalex.org/works"


def rebuild_abstract(inverted: dict[str, list[int]] | None) -> str:
    """OpenAlex ships abstracts as an inverted index {word: [positions]}; put the words back in order."""
    if not inverted:
        return ""
    slots: dict[int, str] = {}
    for word, positions in inverted.items():
        for p in positions:
            slots[p] = word
    return " ".join(slots[i] for i in sorted(slots))


def parse(data: dict) -> list[Paper]:
    papers = []
    for w in data.get("results", []):
        loc = (w.get("primary_location") or {}).get("source") or {}
        papers.append(
            Paper(
                title=w.get("title") or w.get("display_name") or "",
                abstract=rebuild_abstract(w.get("abstract_inverted_index")),
                authors=[a["author"]["display_name"] for a in w.get("authorships", []) if a.get("author", {}).get("display_name")],
                year=w.get("publication_year"),
                venue=loc.get("display_name") or "",
                doi=w.get("doi") or "",
                url=w.get("doi") or w.get("id") or "",
                source="openalex",
                source_id=w.get("id", ""),
                citations=w.get("cited_by_count"),
            )
        )
    return [p for p in papers if p.title]


def search_openalex(
    client: httpx.Client, query: str, *, limit: int, year_from: int | None, year_to: int | None, mailto: str = ""
) -> list[Paper]:
    filters = ["type:article|preprint|review|book-chapter"]
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if year_to:
        filters.append(f"to_publication_date:{year_to}-12-31")
    params = {"search": query, "per-page": min(limit, 100), "filter": ",".join(filters)}
    if mailto:
        params["mailto"] = mailto
    r = client.get(API, params=params)
    r.raise_for_status()
    return parse(r.json())
