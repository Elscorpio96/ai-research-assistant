"""Offline provider: BM25 search over a bundled sample corpus.

The corpus is FICTIONAL — invented titles, authors and abstracts written for demonstrations and tests.
It has no DOIs and must never be cited.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

import httpx

from ..models import Paper
from ..text import BM25


@lru_cache
def corpus() -> list[Paper]:
    raw = json.loads(resources.files("litscout.data").joinpath("demo_corpus.json").read_text(encoding="utf-8"))
    return [Paper(**{**r, "venue": r.get("venue", "") + " [demo]"}, source="demo", source_id=f"demo-{i + 1}") for i, r in enumerate(raw)]


def search_demo(
    client: httpx.Client | None, query: str, *, limit: int, year_from: int | None, year_to: int | None, mailto: str = ""
) -> list[Paper]:
    docs = corpus()
    scores = BM25([f"{p.title} {p.title} {p.abstract}" for p in docs]).scores(query)
    ranked = sorted(zip(scores, docs, strict=True), key=lambda x: -x[0])
    floor = 0.1 * (ranked[0][0] if ranked else 0)  # drop weak, single-word matches
    return [
        p for s, p in ranked if s > floor and (not year_from or (p.year or 0) >= year_from) and (not year_to or (p.year or 9999) <= year_to)
    ][:limit]
