"""FastAPI application: server-rendered UI plus a small JSON API."""

import csv
import hashlib
import hmac
import io
import re
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from urllib.parse import quote, urlparse

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from pydantic import BaseModel, Field

from . import __version__, citations, pipeline, synthesis
from .config import Settings, get_settings
from .db import Store
from .embeddings import make_embedder
from .llm import LLMError, make_completer
from .sources import PROVIDERS, run_search


class SearchIn(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    sources: list[str] = Field(default_factory=lambda: ["openalex"], min_length=1)
    limit: int = Field(25, ge=1, le=100)
    year_from: int | None = Field(None, ge=1900, le=2100)
    year_to: int | None = Field(None, ge=1900, le=2100)


class ScreenIn(BaseModel):
    status: str = Field(pattern="^(pending|included|excluded)$")
    reason: str = Field("", max_length=200)
    notes: str | None = Field(None, max_length=2000)


class ProjectIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=10, max_length=1000)
    inclusion: str = ""
    exclusion: str = ""


HERE = Path(__file__).parent
COOKIE = "litscout_session"
EXCLUSION_REASONS = [
    "Not relevant to the research question",
    "Wrong population or context",
    "Wrong study type",
    "Outside the date range",
    "Not in English",
    "Full text unavailable",
    "Duplicate (not detected automatically)",
]


def create_app(settings: Settings | None = None, store: Store | None = None, http_client=None, llm_client=None, embedder=None) -> FastAPI:
    settings = settings or get_settings()
    store = store or Store(settings.db_path)
    if embedder is not None or store.embedder is None:
        store.embedder = embedder or make_embedder(settings)
    secret = (settings.secret_key or secrets.token_hex(32)).encode()
    app = FastAPI(title="LitScout API", version=__version__, docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.globals.update(version=__version__, providers=PROVIDERS, settings=settings)
    templates.env.filters["cite_links"] = cite_links
    templates.env.filters["italics"] = italics

    def session_token() -> str:
        return hmac.new(secret, hashlib.sha256(settings.access_password.encode()).digest(), "sha256").hexdigest()

    # ------------------------------------------------------------ middleware
    @app.middleware("http")
    async def guard(request: Request, call_next):
        # Same-origin check for state-changing requests (CSRF defence that works with or without a password)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin") or request.headers.get("referer")
            if origin and urlparse(origin).netloc != request.url.netloc:
                return PlainTextResponse("Cross-site request blocked", status_code=403)
        open_path = request.url.path in ("/login", "/healthz") or request.url.path.startswith("/static/")
        if settings.access_password and not open_path:
            if not hmac.compare_digest(request.cookies.get(COOKIE, ""), session_token()):
                if request.url.path.startswith("/api/"):
                    return JSONResponse({"detail": "Not authenticated"}, status_code=401)
                return RedirectResponse("/login", status_code=303)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if not request.url.path.startswith("/api/docs"):
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; form-action 'self'; frame-ancestors 'none'",
            )
        return response

    def render(request: Request, name: str, code: int = 200, **ctx):
        msg = request.query_params.get("msg", "")[:200]
        return templates.TemplateResponse(request, name, {"msg": msg, "auth": bool(settings.access_password), **ctx}, status_code=code)

    def back(url: str, msg: str = "") -> RedirectResponse:
        return RedirectResponse(f"{url}{'&' if '?' in url else '?'}msg={quote(msg)}" if msg else url, status_code=303)

    def get_project(pid: int) -> dict:
        p = store.project(pid)
        if not p:
            raise HTTPException(404, "Project not found")
        return dict(p)

    Project = Annotated[dict, Depends(get_project)]

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        return render(request, "error.html", code=exc.status_code, title=str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": exc.errors()}, status_code=422)
        fields = ", ".join(sorted({str(e["loc"][-1]).replace("_", " ") for e in exc.errors()}))
        return render(request, "error.html", code=422, title=f"Please check: {fields}")

    # ------------------------------------------------------------ auth
    @app.get("/healthz", include_in_schema=False)
    def health():
        return {"ok": True}

    @app.get("/login", response_class=HTMLResponse, include_in_schema=False)
    def login_page(request: Request):
        return render(request, "login.html", error="")

    @app.post("/login", include_in_schema=False)
    def login(request: Request, password: Annotated[str, Form()] = ""):
        if not settings.access_password:
            return back("/")
        if not hmac.compare_digest(password.encode(), settings.access_password.encode()):
            return render(request, "login.html", code=401, error="Incorrect password")
        r = back("/")
        r.set_cookie(COOKIE, session_token(), httponly=True, samesite="strict", secure=request.url.scheme == "https", max_age=7 * 86400)
        return r

    @app.post("/logout", include_in_schema=False)
    def logout():
        r = back("/login")
        r.delete_cookie(COOKIE)
        return r

    # ------------------------------------------------------------ projects
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(request: Request):
        return render(request, "index.html", projects=store.projects())

    @app.post("/projects", include_in_schema=False)
    def create_project(
        title: Annotated[str, Form(min_length=1, max_length=200)],
        question: Annotated[str, Form(min_length=10, max_length=1000)],
        inclusion: Annotated[str, Form(max_length=2000)] = "",
        exclusion: Annotated[str, Form(max_length=2000)] = "",
    ):
        pid = store.create_project(title.strip(), question.strip(), inclusion.strip(), exclusion.strip())
        return back(f"/projects/{pid}", "Project created. Start by searching for papers.")

    @app.get("/projects/{pid}", response_class=HTMLResponse, include_in_schema=False)
    def project_page(request: Request, project: Project):
        return render(
            request,
            "project.html",
            project=project,
            prisma=store.prisma(project["id"]),
            searches=store.searches(project["id"]),
            tab="search",
        )

    @app.post("/projects/{pid}/edit", include_in_schema=False)
    def edit_project(
        project: Project,
        title: Annotated[str, Form(min_length=1, max_length=200)],
        question: Annotated[str, Form(min_length=10, max_length=1000)],
        inclusion: Annotated[str, Form(max_length=2000)] = "",
        exclusion: Annotated[str, Form(max_length=2000)] = "",
    ):
        store.update_project(project["id"], title.strip(), question.strip(), inclusion.strip(), exclusion.strip())
        return back(f"/projects/{project['id']}", "Project updated")

    @app.post("/projects/{pid}/delete", include_in_schema=False)
    def delete_project(project: Project):
        store.delete_project(project["id"])
        return back("/", f"Deleted “{project['title']}”")

    # ------------------------------------------------------------ search
    def do_search(project: dict, query: str, sources: list[str], limit: int, year_from: int | None, year_to: int | None) -> dict:
        if year_from and year_to and year_from > year_to:
            raise HTTPException(400, "'From' year must not be after 'to' year")
        results = run_search(settings, query, sources, limit, year_from, year_to, client=http_client)
        counts = store.add_search(
            project["id"],
            query,
            sources,
            year_from,
            year_to,
            {r.source: r.papers for r in results},
            {r.source: r.error for r in results if r.error},
        )
        return {
            **counts,
            "per_source": {r.source: len(r.papers) for r in results},
            "errors": {r.source: r.error for r in results if r.error},
        }

    @app.post("/projects/{pid}/search", include_in_schema=False)
    def search_form(
        project: Project,
        query: Annotated[str, Form(min_length=2, max_length=300)],
        sources: Annotated[list[str], Form()] = [],  # noqa: B006 - FastAPI form default
        limit: Annotated[int, Form(ge=5, le=100)] = 25,
        year_from: Annotated[str, Form()] = "",
        year_to: Annotated[str, Form()] = "",
    ):
        if not sources:
            return back(f"/projects/{project['id']}", "Select at least one source")
        yf, yt = _year(year_from), _year(year_to)
        r = do_search(project, query.strip(), sources, limit, yf, yt)
        msg = f"Found {r['identified']} records, {r['added']} new ({r['duplicates']} duplicates skipped)."
        if r["errors"]:
            msg += " " + " ".join(f"{PROVIDERS[s][0]}: {e}." for s, e in r["errors"].items() if s in PROVIDERS)
        return back(f"/projects/{project['id']}/screen?status=pending", msg)

    # ------------------------------------------------------------ screening
    @app.get("/projects/{pid}/screen", response_class=HTMLResponse, include_in_schema=False)
    def screen_page(request: Request, project: Project, status: str = "pending", q: str = "", sort: str = "relevance"):
        status = status if status in ("pending", "included", "excluded", "all") else "pending"
        papers = store.papers(project["id"], None if status == "all" else status, q[:100], sort)
        return render(
            request,
            "screen.html",
            project=project,
            papers=papers,
            status=status,
            q=q,
            sort=sort,
            prisma=store.prisma(project["id"]),
            reasons=EXCLUSION_REASONS,
            tab="screen",
        )

    @app.post("/projects/{pid}/papers/{paper_id}/screen", include_in_schema=False)
    def screen_paper(
        request: Request,
        project: Project,
        paper_id: int,
        status: Annotated[str, Form()],
        reason: Annotated[str, Form(max_length=200)] = "",
        notes: Annotated[str | None, Form(max_length=2000)] = None,
        next: Annotated[str, Form()] = "",
    ):  # noqa: A002
        try:
            ok = store.screen(project["id"], paper_id, status, reason, notes)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        if not ok:
            raise HTTPException(404, "Paper not found")
        target = next if next.startswith(f"/projects/{project['id']}/") else f"/projects/{project['id']}/screen"
        return RedirectResponse(target, status_code=303)

    # ------------------------------------------------------------ review
    @app.get("/projects/{pid}/review", response_class=HTMLResponse, include_in_schema=False)
    def review_page(request: Request, project: Project):
        return render(
            request,
            "review.html",
            project=project,
            review=store.latest_review(project["id"]),
            prisma=store.prisma(project["id"]),
            tab="review",
            llm=settings.llm_enabled,
        )

    @app.post("/projects/{pid}/review", include_in_schema=False)
    def generate_review(project: Project, method: Annotated[str, Form()] = "extractive"):
        r = make_review(project, method)
        return back(f"/projects/{project['id']}/review", f"Draft generated from {len(r['references'])} included papers.")

    def make_review(project: dict, method: str) -> dict:
        papers = store.papers(project["id"], "included")
        try:
            if method == "llm" and not settings.llm_enabled:
                raise HTTPException(400, "No LLM is configured. Set LITSCOUT_LLM_PROVIDER and LITSCOUT_LLM_API_KEY.")
            used = sorted({PROVIDERS[s][0] for x in store.searches(project["id"]) for s in x["sources"].split(",") if s in PROVIDERS})
            review = pipeline.run_review(
                project,
                papers,
                "llm" if method == "llm" else "extractive",
                complete=make_completer(settings, llm_client) if method == "llm" else None,
                retriever=lambda q, k: store.hybrid_search(project["id"], q, k, status="included"),
                prisma=store.prisma(project["id"]),
                sources_used=used,
                max_sources=settings.review_max_sources,
                max_attempts=settings.review_max_attempts,
            )
        except (ValueError, LLMError) as e:
            raise HTTPException(400, str(e)) from None
        store.save_review(project["id"], review["method"], review["model"], review)
        return review

    # ------------------------------------------------------------ exports
    def _slug(project: dict) -> str:
        return "".join(c if c.isalnum() else "-" for c in project["title"].lower()).strip("-")[:40] or "review"

    @app.get("/projects/{pid}/export/review.md", include_in_schema=False)
    def export_md(project: Project, style: str = "apa"):
        review = store.latest_review(project["id"])
        if not review:
            raise HTTPException(404, "Generate a review first")
        md = synthesis.to_markdown(project, review["content"], "ieee" if style == "ieee" else "apa")
        return _download(md, f"{_slug(project)}-review.md", "text/markdown")

    @app.get("/projects/{pid}/export/references.bib", include_in_schema=False)
    def export_bib(project: Project):
        papers = synthesis.numbered(store.papers(project["id"], "included"))
        return _download(citations.bibtex(papers), f"{_slug(project)}.bib", "application/x-bibtex")

    @app.get("/projects/{pid}/export/screening.csv", include_in_schema=False)
    def export_csv(project: Project):
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["status", "reason", "relevance", "year", "title", "authors", "venue", "doi", "url", "source", "notes"])
        for p in store.papers(project["id"], None, "", "relevance"):
            row = [
                p["status"],
                p["reason"],
                p["relevance"],
                p["year"],
                p["title"],
                "; ".join(p["authors"]),
                p["venue"],
                p["doi"],
                p["url"],
                p["source"],
                p["notes"],
            ]
            w.writerow([_csv_safe(v) for v in row])
        return _download(buf.getvalue(), f"{_slug(project)}-screening.csv", "text/csv")

    # ------------------------------------------------------------ JSON API
    @app.get("/api/projects", tags=["projects"])
    def api_projects():
        return [dict(r) for r in store.projects()]

    @app.post("/api/projects", tags=["projects"], status_code=201)
    def api_create(body: ProjectIn):
        return {"id": store.create_project(body.title, body.question, body.inclusion, body.exclusion)}

    @app.post("/api/projects/{pid}/search", tags=["search"])
    def api_search(project: Project, body: SearchIn):
        return do_search(project, body.query, body.sources, body.limit, body.year_from, body.year_to)

    @app.get("/api/projects/{pid}/papers", tags=["screening"])
    def api_papers(project: Project, status: str | None = None, q: str = "", sort: str = "relevance"):
        return store.papers(project["id"], status, q, sort)

    @app.get("/api/projects/{pid}/retrieve", tags=["search"])
    def api_retrieve(project: Project, q: str, k: int = 10, status: str | None = None):
        """Hybrid retrieval over the project's papers: BM25, plus vector similarity fused with RRF when embeddings are on."""
        if status not in (None, "pending", "included", "excluded"):
            raise HTTPException(400, "status must be pending, included or excluded")
        return {
            "mode": "hybrid" if store.embedder else "bm25",
            "results": store.hybrid_search(project["id"], q, max(1, min(k, 100)), status),
        }

    @app.patch("/api/projects/{pid}/papers/{paper_id}", tags=["screening"])
    def api_screen(project: Project, paper_id: int, body: ScreenIn):
        try:
            if not store.screen(project["id"], paper_id, body.status, body.reason, body.notes):
                raise HTTPException(404, "Paper not found")
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return store.paper(project["id"], paper_id)

    @app.get("/api/projects/{pid}/prisma", tags=["screening"])
    def api_prisma(project: Project):
        return store.prisma(project["id"])

    @app.post("/api/projects/{pid}/review", tags=["review"])
    def api_review(project: Project, method: str = "extractive"):
        return make_review(project, method)

    return app


def cite_links(text: str) -> Markup:
    """Escape text, then turn [1, 3] citations into links to the reference list."""
    safe = str(escape(text))
    return Markup(  # noqa: S704 - input escaped above; only fixed <a> tags with digits are added
        synthesis.CITE_RE.sub(
            lambda m: (
                "["
                + ", ".join(f'<a href="#ref{n.strip()}">{n.strip()}</a>' if n.strip().isdigit() else n for n in m.group(1).split(","))
                + "]"
            ),
            safe,
        )
    )


def italics(text: str) -> Markup:
    """Render *venue* emphasis from formatted references, safely."""
    return Markup(re.sub(r"\*([^*]+)\*", r"<i>\1</i>", str(escape(text))))  # noqa: S704 - escaped first


def _year(v: str) -> int | None:
    v = v.strip()
    if not v:
        return None
    if not v.isdigit() or not 1900 <= int(v) <= 2100:
        raise HTTPException(400, "Years must be between 1900 and 2100")
    return int(v)


def _csv_safe(v) -> str:
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s


def _download(text: str, filename: str, media: str) -> Response:
    return Response(
        text,
        media_type=f"{media}; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"', "Cache-Control": "no-store"},
    )


def stamp() -> str:
    return datetime.now(UTC).isoformat()
