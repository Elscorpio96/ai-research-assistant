import httpx
import respx
from fastapi.testclient import TestClient

from litscout.app import create_app
from litscout.config import Settings
from litscout.db import Store
from litscout.demo_project import seed_demo_project
from litscout.llm import LLMError, make_completer


def new_project(client, title="My review", question="How are SQL injection attacks detected?"):
    r = client.post("/projects", data={"title": title, "question": question}, follow_redirects=False)
    assert r.status_code == 303
    return int(r.headers["location"].split("/")[2].split("?")[0])


def test_full_workflow_through_the_ui(client, store):
    pid = new_project(client)
    r = client.post(f"/projects/{pid}/search", data={"query": "SQL injection detection", "sources": ["demo"], "limit": "25"})
    assert r.status_code == 200 and "new (0 duplicates skipped)" in r.text
    papers = store.papers(pid)
    assert len(papers) >= 5 and papers[0]["relevance"] == 1.0
    for p in papers[:4]:
        client.post(f"/projects/{pid}/papers/{p['id']}/screen", data={"status": "included"})
    r = client.post(f"/projects/{pid}/papers/{papers[4]['id']}/screen", data={"status": "excluded", "reason": "Wrong study type"})
    assert r.status_code == 200
    assert store.prisma(pid)["included"] == 4
    r = client.post(f"/projects/{pid}/review", data={"method": "extractive"})
    assert "Draft generated from 4 included papers" in r.text and "References" in r.text
    md = client.get(f"/projects/{pid}/export/review.md")
    assert md.headers["content-disposition"].startswith("attachment") and "## References" in md.text
    assert "@" in client.get(f"/projects/{pid}/export/references.bib").text
    csv = client.get(f"/projects/{pid}/export/screening.csv").text
    assert csv.startswith("status,reason,relevance") and "Wrong study type" in csv


def test_searching_twice_reports_duplicates(client):
    pid = new_project(client)
    client.post(f"/projects/{pid}/search", data={"query": "SQL injection", "sources": ["demo"]})
    r = client.post(f"/projects/{pid}/search", data={"query": "SQL injection", "sources": ["demo"]})
    assert ", 0 new (" in r.text


def test_validation(client):
    assert client.post("/projects", data={"title": "x", "question": "short"}).status_code == 422
    pid = new_project(client)
    assert "Select at least one source" in client.post(f"/projects/{pid}/search", data={"query": "abc"}).text
    assert (
        client.post(
            f"/projects/{pid}/search", data={"query": "abc", "sources": ["demo"], "year_from": "2025", "year_to": "2020"}
        ).status_code
        == 400
    )
    assert client.post(f"/projects/{pid}/search", data={"query": "abc", "sources": ["demo"], "year_from": "20x5"}).status_code == 400
    assert client.post(f"/projects/{pid}/review", data={"method": "extractive"}).status_code == 400  # nothing included
    assert client.post(f"/projects/{pid}/review", data={"method": "llm"}).status_code == 400  # no LLM configured
    assert client.get("/projects/999").status_code == 404


def test_exclusion_needs_a_reason(client, store):
    pid = new_project(client)
    client.post(f"/projects/{pid}/search", data={"query": "SQL injection", "sources": ["demo"]})
    pap = store.papers(pid)[0]
    r = client.post(f"/projects/{pid}/papers/{pap['id']}/screen", data={"status": "excluded"})
    assert r.status_code == 400


def test_redirect_after_screening_stays_inside_the_project(client, store):
    pid = new_project(client)
    client.post(f"/projects/{pid}/search", data={"query": "SQL injection", "sources": ["demo"]})
    pap = store.papers(pid)[0]
    r = client.post(
        f"/projects/{pid}/papers/{pap['id']}/screen", data={"status": "included", "next": "https://evil.example/"}, follow_redirects=False
    )
    assert r.headers["location"] == f"/projects/{pid}/screen"


def test_user_content_is_escaped(client):
    pid = new_project(client, title="<script>alert(1)</script>", question="Is <b>bold</b> escaped in the question?")
    page = client.get(f"/projects/{pid}").text
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    assert "<b>bold</b>" not in page


def test_cross_site_posts_are_blocked(client):
    r = client.post("/projects", data={"title": "t", "question": "a long enough question"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    ok = client.post(
        "/projects",
        data={"title": "t", "question": "a long enough question"},
        headers={"Origin": "http://testserver"},
        follow_redirects=False,
    )
    assert ok.status_code == 303


def test_security_headers(client):
    h = client.get("/").headers
    assert "script-src 'self'" in h["content-security-policy"] and h["x-frame-options"] == "DENY"


def test_password_protection():
    app = create_app(Settings(access_password="correct horse", secret_key="k", _env_file=None), Store(":memory:"))
    c = TestClient(app)
    assert c.get("/", follow_redirects=False).headers["location"] == "/login"
    assert c.get("/api/projects").status_code == 401
    assert c.get("/static/app.css").status_code == 200 and c.get("/healthz").status_code == 200
    assert c.post("/login", data={"password": "wrong"}).status_code == 401
    r = c.post("/login", data={"password": "correct horse"}, follow_redirects=False)
    assert "httponly" in r.headers["set-cookie"].lower() and "samesite=strict" in r.headers["set-cookie"].lower()
    assert c.get("/api/projects").status_code == 200
    c.post("/logout")
    assert c.get("/api/projects").status_code == 401


def test_json_api(client, store):
    pid = client.post("/api/projects", json={"title": "API", "question": "How are SQL injection attacks detected?"}).json()["id"]
    r = client.post(f"/api/projects/{pid}/search", json={"query": "SQL injection detection", "sources": ["demo"], "limit": 10})
    assert r.status_code == 200 and r.json()["added"] > 0 and r.json()["per_source"]["demo"] > 0
    papers = client.get(f"/api/projects/{pid}/papers?status=pending").json()
    for p in papers[:3]:
        assert client.patch(f"/api/projects/{pid}/papers/{p['id']}", json={"status": "included"}).json()["status"] == "included"
    assert client.patch(f"/api/projects/{pid}/papers/{papers[3]['id']}", json={"status": "excluded"}).status_code == 400
    assert client.patch(f"/api/projects/{pid}/papers/99999", json={"status": "included"}).status_code == 404
    assert client.get(f"/api/projects/{pid}/prisma").json()["included"] == 3
    review = client.post(f"/api/projects/{pid}/review").json()
    assert len(review["references"]) == 3
    assert client.post(f"/api/projects/{pid}/search", json={"query": "x", "sources": []}).status_code == 422


def test_review_page_links_citations_to_references():
    store = Store(":memory:")
    pid = seed_demo_project(store)
    page = TestClient(create_app(Settings(_env_file=None), store)).get(f"/projects/{pid}/review").text
    assert '<a href="#ref1">1</a>' in page and 'id="ref1"' in page and "fictional" in page.lower()


@respx.mock
def test_llm_review_via_anthropic_api():
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "claude-test",
                "content": [
                    {"type": "text", "text": '{"themes": [], "sections": [{"heading": "Introduction", "paragraphs": ["Studied [1]."]}]}'}
                ],
            },
        )
    )
    store = Store(":memory:")
    pid = seed_demo_project(store, review=False)
    s = Settings(llm_provider="anthropic", llm_api_key="sk-test-key", _env_file=None)
    c = TestClient(create_app(s, store))
    r = c.post(f"/projects/{pid}/review", data={"method": "llm"})
    assert r.status_code == 200 and "LLM (claude-test)" in r.text
    req = route.calls[0].request
    assert req.headers["x-api-key"] == "sk-test-key" and req.headers["anthropic-version"] == "2023-06-01"


@respx.mock
def test_llm_errors_do_not_leak_the_api_key():
    respx.post("https://api.anthropic.com/v1/messages").mock(return_value=httpx.Response(401, text="invalid key sk-secret-123"))
    complete = make_completer(Settings(llm_provider="anthropic", llm_api_key="sk-secret-123", _env_file=None))
    try:
        complete("s", "p")
    except LLMError as e:
        assert "sk-secret-123" not in str(e) and "HTTP 401" in str(e)
    else:
        raise AssertionError("expected LLMError")


@respx.mock
def test_openai_compatible_endpoint():
    respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"model": "llama3", "choices": [{"message": {"content": "hello"}}]})
    )
    complete = make_completer(Settings(llm_provider="openai", llm_base_url="http://localhost:11434/v1", llm_model="llama3", _env_file=None))
    assert complete("s", "p") == ("hello", "llama3")
