import pytest

from litscout.models import Paper, normalise_title
from litscout.text import BM25, kmeans, stem, tfidf_vectors


def test_stemming_merges_variants():
    assert stem("detection") == stem("detect") == "detect"
    assert stem("attacks") == stem("attack")
    assert stem("studies") == "study"


def test_bm25_ranks_relevant_documents_first():
    docs = ["SQL injection detection with neural networks", "Teaching mathematics in primary schools", "Injection attacks on web forms"]
    s = BM25(docs).scores("SQL injection")
    assert s[0] > s[2] > s[1] == 0


def test_kmeans_separates_two_topics_deterministically():
    docs = [f"sql injection attack detection database query {i}" for i in range(4)] + [
        f"student examination teacher school test {i}" for i in range(4)
    ]
    vecs, _ = tfidf_vectors(docs)
    labels = kmeans(vecs, 2)
    assert labels == kmeans(vecs, 2)
    assert len(set(labels[:4])) == 1 and len(set(labels[4:])) == 1 and labels[0] != labels[4]


def test_title_normalisation_for_dedupe():
    assert normalise_title("Détection of  SQL-Injection: A Survey!") == normalise_title("detection of sql injection a survey")


def P(title, doi="", year=2022, abstract="A sufficiently long abstract about injection attacks and detection."):
    return Paper(title=title, doi=doi, year=year, abstract=abstract, authors=["Ada Example"], source="test")


def test_duplicates_are_removed_within_a_search_and_across_searches(store):
    pid = store.create_project("T", "How are injection attacks detected?")
    c = store.add_search(
        pid,
        "q1",
        ["a", "b"],
        None,
        None,
        {
            "a": [P("Paper One", doi="10.1/ONE"), P("Paper Two")],
            "b": [P("paper one (preprint)", doi="https://doi.org/10.1/one"), P("PAPER TWO!")],
        },
        {},
    )
    assert c == {"identified": 4, "duplicates": 2, "added": 2}
    c2 = store.add_search(pid, "q2", ["a"], None, None, {"a": [P("Paper Two"), P("Paper Three")]}, {})
    assert c2 == {"identified": 2, "duplicates": 1, "added": 1}
    assert store.prisma(pid)["identified"] == 6 and store.prisma(pid)["duplicates"] == 3


def test_same_title_different_year_is_not_a_duplicate(store):
    pid = store.create_project("T", "A question long enough")
    store.add_search(pid, "q", ["a"], None, None, {"a": [P("Annual Report", year=2021), P("Annual Report", year=2022)]}, {})
    assert len(store.papers(pid)) == 2


def test_screening_rules_and_prisma(store):
    pid = store.create_project("T", "How are injection attacks detected?")
    store.add_search(pid, "q", ["a"], None, None, {"a": [P("One"), P("Two"), P("Three")]}, {})
    a, b, c = (p["id"] for p in store.papers(pid, sort="title"))
    with pytest.raises(ValueError, match="reason"):
        store.screen(pid, a, "excluded", "  ")
    with pytest.raises(ValueError):
        store.screen(pid, a, "maybe")
    store.screen(pid, a, "excluded", "Wrong study type")
    store.screen(pid, b, "included", notes="Key paper")
    assert store.paper(pid, b)["notes"] == "Key paper"
    store.screen(pid, b, "included")  # notes preserved when not given
    assert store.paper(pid, b)["notes"] == "Key paper"
    assert not store.screen(pid, 9999, "included")
    pr = store.prisma(pid)
    assert (pr["screened"], pr["included"], pr["excluded"], pr["pending"]) == (3, 1, 1, 1)
    assert pr["reasons"] == [{"reason": "Wrong study type", "n": 1}]
    store.screen(pid, a, "included", "ignored reason")
    assert store.paper(pid, a)["reason"] == ""


def test_relevance_is_normalised_and_follows_the_question(store):
    pid = store.create_project("T", "neural network detection of SQL injection")
    store.add_search(
        pid,
        "SQL injection",
        ["a"],
        None,
        None,
        {
            "a": [
                P("Neural network SQL injection detection", abstract="Neural network detects SQL injection."),
                P("School timetabling", abstract="Timetables for schools and teachers."),
            ]
        },
        {},
    )
    top, bottom = store.papers(pid)
    assert top["relevance"] == 1.0 and top["title"].startswith("Neural") and bottom["relevance"] == 0


def test_deleting_a_project_removes_its_papers(store):
    pid = store.create_project("T", "A question long enough")
    store.add_search(pid, "q", ["a"], None, None, {"a": [P("One")]}, {})
    store.delete_project(pid)
    assert store.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
