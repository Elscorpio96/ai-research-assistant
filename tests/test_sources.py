import json

import httpx
import pytest
import respx
from defusedxml import EntitiesForbidden

from litscout.sources import run_search
from litscout.sources.arxiv import parse as parse_arxiv
from litscout.sources.crossref import parse as parse_crossref
from litscout.sources.demo import corpus, search_demo
from litscout.sources.openalex import parse as parse_openalex
from litscout.sources.openalex import rebuild_abstract

from .conftest import fixture_text


def test_openalex_abstract_is_rebuilt_in_word_order():
    assert rebuild_abstract({"world": [1], "hello": [0], "again": [3], "hello,": [2]}) == "hello world hello, again"
    assert rebuild_abstract(None) == ""


def test_openalex_parse_normalises_doi_and_handles_missing_fields():
    a, b = parse_openalex(json.loads(fixture_text("openalex.json")))
    assert a.doi == "10.1234/example.001"
    assert a.abstract == "We detect attacks quickly. Few missed attacks remain."
    assert a.authors == ["Ada Example", "Bola Sample"] and a.venue == "Sample Journal" and a.citations == 12
    assert b.doi == "" and b.abstract == "" and b.venue == "" and b.year == 2020


def test_crossref_parse_strips_jats_and_tolerates_missing_dates():
    a, b = parse_crossref(json.loads(fixture_text("crossref.json")))
    assert a.abstract == "We detect attacks quickly. Few missed attacks remain."
    assert a.title == "Sample Work on Injection Detection"  # whitespace collapsed
    assert a.year == 2023
    assert b.year is None and b.authors == ["Solo"]


def test_same_paper_from_two_sources_has_same_dedupe_key():
    oa = parse_openalex(json.loads(fixture_text("openalex.json")))[0]
    cr = parse_crossref(json.loads(fixture_text("crossref.json")))[0]
    assert oa.dedupe_key == cr.dedupe_key == "doi:10.1234/example.001"


def test_arxiv_parse():
    new, old = parse_arxiv(fixture_text("arxiv.xml"))
    assert new.title == "Preprint Sample on Detector Robustness"
    assert new.year == 2024 and new.venue == "arXiv preprint" and new.source_id == "2401.00001v1"
    assert old.doi == "10.9999/older.sample" and old.venue == "Sample Proceedings 2018"


def test_arxiv_parser_rejects_entity_expansion():
    bomb = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;">]><feed>&b;</feed>'
    with pytest.raises(EntitiesForbidden):
        parse_arxiv(bomb)


@respx.mock
def test_one_failing_source_does_not_lose_the_others(settings):
    respx.get("https://api.openalex.org/works").mock(return_value=httpx.Response(200, json=json.loads(fixture_text("openalex.json"))))
    respx.get("https://export.arxiv.org/api/query").mock(side_effect=httpx.ConnectTimeout("slow"))
    respx.get("https://api.crossref.org/works").mock(return_value=httpx.Response(503))
    out = {r.source: r for r in run_search(settings, "injection", ["openalex", "arxiv", "crossref"], 10, None, None)}
    assert len(out["openalex"].papers) == 2 and out["openalex"].error is None
    assert "Timed out" in out["arxiv"].error
    assert "HTTP 503" in out["crossref"].error


@respx.mock
def test_year_filters_are_sent_to_apis(settings):
    route = respx.get("https://api.openalex.org/works").mock(return_value=httpx.Response(200, json={"results": []}))
    run_search(settings, "x y", ["openalex"], 10, 2019, 2022)
    assert "from_publication_date:2019-01-01" in route.calls[0].request.url.params["filter"]
    assert "to_publication_date:2022-12-31" in route.calls[0].request.url.params["filter"]


@respx.mock
def test_arxiv_results_are_filtered_by_year_locally(settings):
    respx.get("https://export.arxiv.org/api/query").mock(return_value=httpx.Response(200, text=fixture_text("arxiv.xml")))
    [r] = run_search(settings, "detectors", ["arxiv"], 10, 2020, None)
    assert [p.year for p in r.papers] == [2024]


def test_disabled_source_is_reported(store):
    from litscout.config import Settings

    s = Settings(sources_enabled="demo", _env_file=None)
    [r] = run_search(s, "x", ["openalex"], 5, None, None)
    assert r.error == "Source is not enabled"


def test_demo_corpus_is_marked_and_searchable():
    assert all(p.source == "demo" and "[demo]" in p.venue and not p.doi for p in corpus())
    top = search_demo(None, "adversarial evasion of deep learning detectors", limit=3, year_from=None, year_to=None)
    assert top[0].title.startswith("Adversarial Examples")
    assert all(p.year >= 2023 for p in search_demo(None, "SQL injection", limit=50, year_from=2023, year_to=None))
