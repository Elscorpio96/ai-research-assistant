"""Builds a ready-made sample project from the fictional demo corpus (for demos, screenshots and tests)."""

from __future__ import annotations

from .db import Store
from .sources.demo import search_demo
from .synthesis import extractive_review

QUESTION = "How effective are machine learning techniques at detecting SQL injection attacks, and what limits their use in practice?"
QUERIES = [
    "machine learning SQL injection detection",
    "SQL injection prevention web application",
    "deep learning injection detector evaluation",
]
OFF_TOPIC = {
    "Learning Analytics Dashboards": "Wrong population or context",
    "Teaching Secure Database Programming": "Wrong study type",
    "Parameterised Queries in Practice": "Not relevant to the research question",
    "Security Threats to Computer-Based Examination": "Not relevant to the research question",
}


def seed_demo_project(store: Store, *, screen: bool = True, review: bool = True) -> int:
    pid = store.create_project(
        "ML-based SQL injection detection (demo)",
        QUESTION,
        inclusion="Primary studies or reviews that build or evaluate SQL injection detection or prevention; 2018 onwards.",
        exclusion="Studies not about SQL injection; non-technical studies; studies without an evaluation.",
    )
    for q in QUERIES:
        store.add_search(pid, q, ["demo"], 2018, None, {"demo": search_demo(None, q, limit=25, year_from=2018, year_to=None)}, {})
    if screen:
        for i, p in enumerate(store.papers(pid)):
            reason = next((r for t, r in OFF_TOPIC.items() if p["title"].startswith(t)), None)
            if reason:
                store.screen(pid, p["id"], "excluded", reason)
            elif i < 30:
                store.screen(pid, p["id"], "included")
    if screen and review:
        r = extractive_review(dict(store.project(pid)), store.papers(pid, "included"), store.prisma(pid), ["Demo corpus (offline)"])
        store.save_review(pid, "extractive", "", r)
    return pid
