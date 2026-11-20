import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from litscout.app import create_app
from litscout.config import Settings
from litscout.db import Store
from litscout.embeddings import EmbeddingError, OpenAICompatibleEmbedder, make_embedder, rrf
from litscout.models import Paper
from litscout.pipeline import run_review

# A tiny "semantic" embedder: synonyms map to the same dimension, so it understands what BM25 cannot.
CONCEPTS = {
    "sqli": 0,
    "injection": 0,
    "sql": 0,
    "malicious": 1,
    "attack": 1,
    "attacks": 1,
    "detect": 2,
    "detection": 2,
    "detected": 2,
    "classifier": 2,
    "crop": 3,
    "maize": 3,
    "yield": 3,
    "farmers": 3,
}


class FakeEmbedder:
    name = "fake-embed"

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail

    def embed(self, texts):
        if self.fail:
            raise EmbeddingError("down")
        self.calls.append(list(texts))
        out = []
        for t in texts:
            v = [0.0] * 5
            for w in t.lower().replace(".", " ").replace("?", " ").split():
                v[CONCEPTS.get(w, 4)] += 1.0 if w in CONCEPTS else 0.05
            out.append(v)
        return out


PAPERS = [
    Paper(
        title="Catching SQLi with a lightweight classifier", abstract="We flag malicious payloads in web logs.", source="demo", year=2022
    ),
    Paper(title="Maize yield forecasting for smallholder farmers", abstract="Crop models from satellite data.", source="demo", year=2021),
    Paper(title="Payload logging in web servers", abstract="An overview of web server logs.", source="demo", year=2020),
]


def seeded(embedder=None) -> tuple[Store, int]:
    store = Store(":memory:", embedder=embedder)
    pid = store.create_project("Review", "How are SQL injection attacks detected?")
    store.add_search(pid, "sql injection detection", ["demo"], None, None, {"demo": PAPERS}, {})
    return store, pid


def test_rrf_rewards_items_ranked_well_by_both_lists():
    fused = rrf([3.0, 2.0, 1.0], [1.0, 3.0, 2.0])
    assert fused.index(max(fused)) == 1  # 2nd in one list, 1st in the other beats 1st + 3rd


def test_rrf_ignores_items_a_list_did_not_retrieve():
    assert rrf([0.0, 0.0], [0.2, 0.9]) == [1 / 62, 1 / 61]


def test_bm25_alone_misses_the_paraphrased_paper_but_hybrid_finds_it():
    store, pid = seeded()
    sqli = next(h for h in store.hybrid_search(pid, "SQL injection attacks detected", k=3) if "SQLi" in h["title"])
    assert sqli["bm25"] == 0  # no shared keywords, so BM25 gives it nothing
    store.embedder = FakeEmbedder()
    hits = store.hybrid_search(pid, "SQL injection attacks detected", k=3)
    assert hits[0]["title"].startswith("Catching SQLi")
    assert hits[0]["similarity"] > hits[1]["similarity"]


def test_rerank_uses_hybrid_relevance_when_embeddings_are_on():
    store, pid = seeded(FakeEmbedder())
    top = store.papers(pid, sort="relevance")[0]
    assert top["title"].startswith("Catching SQLi")
    assert top["relevance"] == 1.0


def test_embeddings_are_cached_per_paper_and_model():
    emb = FakeEmbedder()
    store, pid = seeded(emb)
    calls_after_seed = len(emb.calls)
    store.hybrid_search(pid, "maize", k=2)
    store.hybrid_search(pid, "sqli", k=2)
    new_doc_calls = [c for c in emb.calls[calls_after_seed:] if len(c) > 1]
    assert new_doc_calls == []  # only the queries were embedded again
    rows = store.conn.execute("SELECT COUNT(*), MIN(dim) FROM embeddings").fetchone()
    assert tuple(rows) == (3, 5)


def test_embeddings_are_deleted_with_their_project():
    store, pid = seeded(FakeEmbedder())
    store.delete_project(pid)
    assert store.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0


def test_embedding_outage_falls_back_to_bm25():
    store, pid = seeded(FakeEmbedder(fail=True))
    hits = store.hybrid_search(pid, "web server logs", k=1)
    assert hits[0]["similarity"] is None
    assert hits[0]["title"].startswith("Payload logging")


@respx.mock
def test_openai_compatible_embedder_batches_and_orders_results():
    route = respx.post("http://localhost:11434/v1/embeddings").mock(
        side_effect=lambda req: httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": [float(i), 1.0]} for i in reversed(range(len(json.loads(req.content)["input"])))]},
        )
    )
    emb = OpenAICompatibleEmbedder("nomic-embed-text", "http://localhost:11434/v1", batch=2)
    vecs = emb.embed(["a", "b", "c"])
    assert vecs == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
    assert route.call_count == 2


@respx.mock
def test_embedder_errors_do_not_leak_the_api_key():
    respx.post("https://api.openai.com/v1/embeddings").mock(return_value=httpx.Response(401, text="bad key sk-secret"))
    emb = OpenAICompatibleEmbedder("text-embedding-3-small", "https://api.openai.com/v1", api_key="sk-secret")
    with pytest.raises(EmbeddingError) as e:
        emb.embed(["x"])
    assert "sk-secret" not in str(e.value) and "401" in str(e.value)


def test_make_embedder_respects_settings():
    assert make_embedder(Settings(_env_file=None)) is None
    emb = make_embedder(Settings(embed_provider="openai", llm_provider="openai", llm_api_key="k", _env_file=None))
    assert emb.name == "text-embedding-3-small" and emb.api_key == "k"


def test_retrieve_api_reports_mode():
    store, pid = seeded()
    client = TestClient(create_app(Settings(secret_key="s", _env_file=None), store, embedder=FakeEmbedder()))
    body = client.get(f"/api/projects/{pid}/retrieve", params={"q": "SQL injection detection", "k": 2}).json()
    assert body["mode"] == "hybrid"
    assert len(body["results"]) == 2 and body["results"][0]["title"].startswith("Catching SQLi")
    assert client.get(f"/api/projects/{pid}/retrieve", params={"q": "x", "status": "bogus"}).status_code == 400


# ------------------------------------------------------------------ LangGraph pipeline
def two_papers():
    return [
        {"id": 1, "title": "Paper one", "abstract": "We found detection accuracy of 95% on SQL injection payloads.", "authors": ["Ann Lee"],
         "year": 2021, "venue": "J", "doi": "", "url": "", "source": "openalex", "relevance": 0.4},
        {"id": 2, "title": "Paper two", "abstract": "Our classifier reduced false positives in web logs.", "authors": ["Bo Kim"],
         "year": 2022, "venue": "J", "doi": "", "url": "", "source": "openalex", "relevance": 0.9},
    ]  # fmt: skip


PROJECT = {"title": "T", "question": "How is SQL injection detected?"}


def llm_reply(cite: str) -> str:
    return json.dumps(
        {
            "themes": [{"label": "Detection", "papers": [1, 2]}],
            "sections": [{"heading": "Introduction", "paragraphs": [f"Detection works well {cite}."]}],
        }
    )


def test_pipeline_redrafts_when_the_model_invents_a_citation():
    prompts = []
    replies = iter([llm_reply("[1, 99]"), llm_reply("[1, 2]")])

    def complete(system, prompt):
        prompts.append(prompt)
        return next(replies), "fake-model"

    review = run_review(PROJECT, two_papers(), "llm", complete=complete)
    assert review["pipeline"]["attempts"] == 2
    assert "[99]" in prompts[1] and "do not exist" in prompts[1]
    assert review["checks"]["invented_citations"] == []
    assert not any("do not exist" in w for w in review["warnings"])
    assert review["pipeline"]["engine"] == "langgraph"


def test_pipeline_stops_after_max_attempts_and_keeps_the_best_draft():
    replies = iter([llm_reply("[7]"), llm_reply("[1, 8, 9]")])
    review = run_review(PROJECT, two_papers(), "llm", complete=lambda s, p: (next(replies), "m"), max_attempts=2)
    assert review["pipeline"]["attempts"] == 2
    assert review["checks"]["invented_citations"] == [7]  # one problem beats two
    assert "remaining problem" in review["pipeline"]["trace"][-1]


def test_pipeline_retrieve_caps_sources_using_the_retriever():
    seen = {}

    def retriever(q, k):
        seen["args"] = (q, k)
        return two_papers()[1:] + two_papers()[:1]

    papers = two_papers() + [{**two_papers()[0], "id": 3, "title": "Paper three"}]
    review = run_review(PROJECT, papers, "extractive", retriever=retriever, max_sources=2)
    assert seen["args"] == (PROJECT["question"], 2)
    assert review["pipeline"]["sources"] == 2 and len(review["references"]) == 2
    assert review["pipeline"]["trace"][0].startswith("retrieve: selected top 2 of 3")


def test_pipeline_extractive_runs_once_without_an_llm():
    review = run_review(PROJECT, two_papers(), "extractive")
    assert review["method"] == "extractive" and review["pipeline"]["attempts"] == 1


def test_pipeline_rejects_too_few_papers():
    with pytest.raises(ValueError):
        run_review(PROJECT, two_papers()[:1], "extractive")
