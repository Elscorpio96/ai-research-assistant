import re

import pytest

from litscout import citations
from litscout.db import Store
from litscout.demo_project import seed_demo_project
from litscout.synthesis import CITE_RE, extractive_review, llm_review, numbered, to_markdown, validate_citations


def paper(authors, title="A Title.", year=2021, venue="Journal of Things", doi="10.1/x", **kw):
    return {"id": 1, "authors": authors, "title": title, "year": year, "venue": venue, "doi": doi, "url": "", "abstract": "", **kw}


@pytest.mark.parametrize(
    "authors,expected",
    [
        (["Ada King Lovelace"], "Lovelace, A. K. (2021). A Title. *Journal of Things*. https://doi.org/10.1/x"),
        (["Ada Lovelace", "Alan Turing"], "Lovelace, A., & Turing, A. (2021). A Title. *Journal of Things*. https://doi.org/10.1/x"),
        (["Turing, Alan Mathison"], "Turing, A. M. (2021). A Title. *Journal of Things*. https://doi.org/10.1/x"),
    ],
)
def test_apa(authors, expected):
    assert citations.apa(paper(authors)) == expected


def test_apa_without_authors_or_year():
    assert citations.apa(paper([], year=None, doi="")) == "A Title. (n.d.). *Journal of Things*."


def test_ieee_uses_et_al_after_six_authors():
    assert citations.ieee(paper([f"Ann Person{i}" for i in range(7)]), 3).startswith("[3] A. Person0 et al., “A Title,”")
    assert citations.ieee(paper(["Ada Lovelace", "Alan Turing"]), 1).startswith("[1] A. Lovelace and A. Turing,")


def test_bibtex_keys_are_unique_and_special_characters_escaped():
    ps = [paper(["José Núñez"], title="Fast & Safe Queries"), paper(["José Núñez"], title="Fast & Safe Queries")]
    bib = citations.bibtex(ps)
    keys = re.findall(r"@\w+\{([^,]+),", bib)
    assert keys == ["nunez2021fast", "nunez2021fastb"]
    assert r"Fast \& Safe" in bib


def test_validate_citations_removes_invented_sources_and_expands_ranges():
    text, removed = validate_citations("A [1]. B [2, 9]. C [3–5]. D [42].", {1, 2, 3, 4, 5})
    assert text == "A [1]. B [2]. C [3, 4, 5]. D."
    assert removed == [9, 42]


@pytest.fixture
def demo():
    store = Store(":memory:")
    pid = seed_demo_project(store, review=False)
    return store, pid


def test_extractive_review_cites_only_included_papers_and_every_theme_paragraph_is_cited(demo):
    store, pid = demo
    papers = store.papers(pid, "included")
    r = extractive_review(dict(store.project(pid)), papers, store.prisma(pid), ["Demo corpus"])
    valid = {ref["n"] for ref in r["references"]}
    assert valid == set(range(1, len(papers) + 1))
    for s in r["sections"]:
        for p in s["paragraphs"]:
            for m in CITE_RE.finditer(p):
                assert {int(x) for x in m.group(1).split(",")} <= valid
    theme_sections = [s for s in r["sections"] if s["heading"].startswith("Theme")]
    assert len(theme_sections) >= 2 and all(CITE_RE.search(s["paragraphs"][0]) for s in theme_sections)
    assert sorted(n for t in r["themes"] for n in t["papers"]) == sorted(valid)  # every paper in exactly one theme
    assert any("FICTIONAL" in w for w in r["warnings"])


def test_extractive_review_quotes_abstract_text_verbatim(demo):
    store, pid = demo
    papers = {p["id"]: p for p in store.papers(pid, "included")}
    r = extractive_review(dict(store.project(pid)), list(papers.values()), store.prisma(pid), [])
    for row in r["table"]:
        ref = next(x for x in r["references"] if x["n"] == row["n"])
        abstract = papers[ref["paper_id"]]["abstract"]
        core = row["finding"].replace("The authors", "We").replace(" they ", " we ").rstrip(".")
        assert core[:40] in abstract


def test_review_needs_two_papers(demo):
    store, pid = demo
    with pytest.raises(ValueError, match="at least two"):
        extractive_review(dict(store.project(pid)), store.papers(pid, "included")[:1], {}, [])


def test_llm_review_strips_hallucinated_citations_and_flags_uncited_text(demo):
    store, pid = demo
    papers = store.papers(pid, "included")
    seen = {}

    def fake(system, prompt):
        seen["prompt"] = prompt
        return (
            'Here you go:\n{"themes": [{"label": "Detectors", "papers": [1, 2, 77]}], "sections": ['
            '{"heading": "Introduction", "paragraphs": ["Detection is studied widely [1, 2]."]},'
            '{"heading": "Detectors", "paragraphs": ["Accuracy is high [2] but one study claims 100% [99].", "An uncited claim."]},'
            '{"heading": "Conclusion", "paragraphs": ["More work is needed."]}]}'
        ), "fake-model-1"

    r = llm_review(dict(store.project(pid)), papers, fake)
    assert "[1] " in seen["prompt"] and "Never invent sources" not in seen["prompt"]  # rules are in the system prompt
    body = r["sections"][1]["paragraphs"][0]
    assert "[99]" not in body and body.endswith("100%.")
    assert r["themes"][0]["papers"] == [1, 2]
    assert any("[99]" in w for w in r["warnings"]) and any("no citation" in w for w in r["warnings"])
    assert r["method"] == "llm" and r["model"] == "fake-model-1"


def test_llm_review_rejects_non_json(demo):
    store, pid = demo
    with pytest.raises(ValueError, match="JSON"):
        llm_review(dict(store.project(pid)), store.papers(pid, "included"), lambda s, p: ("Sorry, I cannot.", "m"))


def test_markdown_export_contains_references_in_both_styles(demo):
    store, pid = demo
    proj = dict(store.project(pid))
    r = extractive_review(proj, store.papers(pid, "included"), store.prisma(pid), [])
    apa = to_markdown(proj, r, "apa")
    ieee = to_markdown(proj, r, "ieee")
    n = len(r["references"])
    assert f"\n{n}. " in apa and f"[{n}] " in ieee and "## Summary of included studies" in apa


def test_numbering_is_chronological():
    ps = numbered([paper(["B Bee"], year=2024), paper(["A Ay"], year=2019), paper(["C Cee"], year=None)])
    assert [p["year"] for p in ps] == [2019, 2024, None]
