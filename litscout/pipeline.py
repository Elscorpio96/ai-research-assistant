"""The review-drafting pipeline as a LangGraph state graph.

    retrieve ──► draft ──► verify ──► (pass, or out of attempts) ──► finalise ──► END
                   ▲          │
                   └──────────┘  (citations failed checks: redraft with feedback)

* retrieve  selects the included papers most relevant to the research question (hybrid BM25 + vector search when an
            embedder is configured), capped at `max_sources` so the prompt stays within the model's context.
* draft     runs the LLM drafter (or the offline extractive drafter).
* verify    reads the citation checks. If the LLM invented sources or left paragraphs uncited, the graph loops back to
            `draft` with concrete feedback, up to `max_attempts` drafts in total.
* finalise  keeps the best draft seen (fewest problems) and records a trace of every step for auditability.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import synthesis

Retriever = Callable[[str, int], list[dict]]  # (query, k) -> ranked papers


class ReviewState(TypedDict, total=False):
    project: dict
    papers: list[dict]
    method: str  # "llm" | "extractive"
    prisma: dict
    sources_used: list[str]
    max_sources: int
    max_attempts: int
    context: list[dict]
    attempts: int
    feedback: str
    review: dict
    best: dict
    trace: Annotated[list[str], operator.add]


def problems(review: dict) -> int:
    c = review.get("checks") or {}
    return len(c.get("invented_citations", [])) + c.get("uncited_paragraphs", 0)


def build_graph(complete: Callable | None = None, retriever: Retriever | None = None):
    def retrieve(state: ReviewState) -> dict:
        papers, cap = state["papers"], state.get("max_sources", 40)
        if len(papers) <= cap:
            return {"context": papers, "trace": [f"retrieve: using all {len(papers)} included papers"]}
        if retriever:
            ranked = retriever(state["project"]["question"], cap)
            how = "hybrid relevance"
        else:
            ranked = sorted(papers, key=lambda p: -p.get("relevance", 0))[:cap]
            how = "stored relevance"
        return {"context": ranked, "trace": [f"retrieve: selected top {len(ranked)} of {len(papers)} included papers by {how}"]}

    def draft(state: ReviewState) -> dict:
        attempt = state.get("attempts", 0) + 1
        if state["method"] == "llm":
            if complete is None:
                raise ValueError("No LLM is configured.")
            review = synthesis.llm_review(state["project"], state["context"], complete, state.get("feedback", ""))
        else:
            review = synthesis.extractive_review(state["project"], state["context"], state.get("prisma", {}), state.get("sources_used", []))
        best = state.get("best")
        if best is None or problems(review) < problems(best):
            best = review
        return {"review": review, "best": best, "attempts": attempt, "trace": [f"draft: attempt {attempt} ({review['method']})"]}

    def verify(state: ReviewState) -> dict:
        c = state["review"].get("checks") or {}
        bad, uncited = c.get("invented_citations", []), c.get("uncited_paragraphs", 0)
        if not bad and not uncited:
            return {"feedback": "", "trace": ["verify: all citations valid"]}
        notes = []
        if bad:
            notes.append(f"- You cited sources that do not exist: {synthesis.cite_list(bad)}. Cite only the numbered sources provided.")
        if uncited:
            notes.append(f"- {uncited} paragraph(s) had no citation. Every factual paragraph must cite at least one numbered source.")
        return {"feedback": "\n".join(notes), "trace": [f"verify: {len(bad)} invented citation(s), {uncited} uncited paragraph(s)"]}

    def route(state: ReviewState) -> str:
        if state.get("feedback") and state["attempts"] < state.get("max_attempts", 2):
            return "draft"
        return "finalise"

    def finalise(state: ReviewState) -> dict:
        review = dict(state["best"])
        trace = [*state.get("trace", []), f"finalise: kept draft with {problems(review)} remaining problem(s)"]
        review["pipeline"] = {"engine": "langgraph", "attempts": state["attempts"], "sources": len(state["context"]), "trace": trace}
        return {"review": review, "trace": [trace[-1]]}

    g = StateGraph(ReviewState)
    g.add_node("retrieve", retrieve)
    g.add_node("draft", draft)
    g.add_node("verify", verify)
    g.add_node("finalise", finalise)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", route, {"draft": "draft", "finalise": "finalise"})
    g.add_edge("finalise", END)
    return g.compile()


def run_review(
    project: dict,
    papers: list[dict],
    method: str,
    *,
    complete: Callable | None = None,
    retriever: Retriever | None = None,
    prisma: dict | None = None,
    sources_used: list[str] | None = None,
    max_sources: int = 40,
    max_attempts: int = 2,
) -> dict[str, Any]:
    if len(papers) < 2:
        raise ValueError("Include at least two papers before drafting a review.")
    state = build_graph(complete, retriever).invoke(
        {
            "project": project,
            "papers": papers,
            "method": method,
            "prisma": prisma or {},
            "sources_used": sources_used or [],
            "max_sources": max(2, max_sources),
            "max_attempts": max(1, max_attempts),
            "trace": [],
        }
    )
    return state["review"]
