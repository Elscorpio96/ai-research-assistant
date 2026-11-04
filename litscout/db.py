"""SQLite persistence. One file, standard library only."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .models import Paper
from .text import BM25

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS projects (
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  question TEXT NOT NULL,
  inclusion TEXT NOT NULL DEFAULT '',
  exclusion TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS searches (
  id INTEGER PRIMARY KEY,
  project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  query TEXT NOT NULL,
  sources TEXT NOT NULL,
  year_from INTEGER, year_to INTEGER,
  identified INTEGER NOT NULL,   -- records returned by all sources
  duplicates INTEGER NOT NULL,   -- already in project or repeated across sources
  added INTEGER NOT NULL,
  errors TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS papers (
  id INTEGER PRIMARY KEY,
  project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  dedupe_key TEXT NOT NULL,
  title TEXT NOT NULL,
  abstract TEXT NOT NULL DEFAULT '',
  authors TEXT NOT NULL DEFAULT '[]',
  year INTEGER, venue TEXT NOT NULL DEFAULT '', doi TEXT NOT NULL DEFAULT '', url TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL, source_id TEXT NOT NULL DEFAULT '', citations INTEGER,
  relevance REAL NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'included', 'excluded')),
  reason TEXT NOT NULL DEFAULT '',
  notes TEXT NOT NULL DEFAULT '',
  added_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE (project_id, dedupe_key)
);
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY,
  project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  method TEXT NOT NULL,
  model TEXT NOT NULL DEFAULT '',
  content TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_papers_project ON papers(project_id, status);
"""


class Store:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self.conn:
            yield self.conn

    # ---------------------------------------------------------------- projects
    def create_project(self, title: str, question: str, inclusion: str = "", exclusion: str = "") -> int:
        with self.tx() as c:
            return int(
                c.execute(
                    "INSERT INTO projects (title, question, inclusion, exclusion) VALUES (?, ?, ?, ?)",
                    (title, question, inclusion, exclusion),
                ).lastrowid
            )

    def update_project(self, pid: int, title: str, question: str, inclusion: str, exclusion: str) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE projects SET title = ?, question = ?, inclusion = ?, exclusion = ? WHERE id = ?",
                (title, question, inclusion, exclusion, pid),
            )
        self.rerank(pid)

    def delete_project(self, pid: int) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM projects WHERE id = ?", (pid,))

    def project(self, pid: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()

    def projects(self) -> list[sqlite3.Row]:
        return self.conn.execute("""
            SELECT p.*, COUNT(x.id) AS papers, SUM(x.status = 'included') AS included, SUM(x.status = 'pending') AS pending
            FROM projects p LEFT JOIN papers x ON x.project_id = p.id GROUP BY p.id ORDER BY p.id DESC""").fetchall()

    # ---------------------------------------------------------------- papers
    def add_search(
        self,
        pid: int,
        query: str,
        sources: list[str],
        year_from: int | None,
        year_to: int | None,
        results: dict[str, list[Paper]],
        errors: dict[str, str],
    ) -> dict[str, int]:
        """Store a search run and its new, de-duplicated records. Returns PRISMA-style counts."""
        identified = sum(len(v) for v in results.values())
        seen: set[str] = set()
        added = 0
        with self.tx() as c:
            existing = {r[0] for r in c.execute("SELECT dedupe_key FROM papers WHERE project_id = ?", (pid,))}
            for papers in results.values():
                for p in papers:
                    k = p.dedupe_key
                    if k in seen or k in existing:
                        continue
                    seen.add(k)
                    c.execute(
                        """INSERT INTO papers (project_id, dedupe_key, title, abstract, authors, year, venue, doi, url,
                                 source, source_id, citations) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            pid,
                            k,
                            p.title,
                            p.abstract,
                            json.dumps(p.authors),
                            p.year,
                            p.venue,
                            p.doi,
                            p.url,
                            p.source,
                            p.source_id,
                            p.citations,
                        ),
                    )
                    added += 1
            counts = {"identified": identified, "duplicates": identified - added, "added": added}
            c.execute(
                """INSERT INTO searches (project_id, query, sources, year_from, year_to, identified, duplicates, added, errors)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, query, ",".join(sources), year_from, year_to, identified, identified - added, added, json.dumps(errors)),
            )
        self.rerank(pid)
        return counts

    def rerank(self, pid: int) -> None:
        """Score every paper in the project against the research question and all search queries (BM25)."""
        proj = self.project(pid)
        if not proj:
            return
        queries = " ".join(r[0] for r in self.conn.execute("SELECT query FROM searches WHERE project_id = ?", (pid,)))
        rows = self.conn.execute("SELECT id, title, abstract FROM papers WHERE project_id = ?", (pid,)).fetchall()
        if not rows:
            return
        scores = BM25([f"{r['title']} {r['title']} {r['abstract']}" for r in rows]).scores(f"{proj['question']} {queries}")
        top = max(scores) or 1.0
        with self.tx() as c:
            c.executemany(
                "UPDATE papers SET relevance = ? WHERE id = ?", [(round(s / top, 4), r["id"]) for s, r in zip(scores, rows, strict=True)]
            )

    def papers(self, pid: int, status: str | None = None, q: str = "", sort: str = "relevance") -> list[dict]:
        order = {
            "relevance": "relevance DESC",
            "year": "year DESC, relevance DESC",
            "citations": "COALESCE(citations, -1) DESC",
            "title": "title COLLATE NOCASE",
        }.get(sort, "relevance DESC")
        sql = "SELECT * FROM papers WHERE project_id = ?"
        args: list = [pid]
        if status:
            sql += " AND status = ?"
            args.append(status)
        if q:
            sql += " AND (title LIKE ? OR abstract LIKE ? OR authors LIKE ?)"
            args += [f"%{q}%"] * 3
        return [_paper_row(r) for r in self.conn.execute(f"{sql} ORDER BY {order}", args)]

    def paper(self, pid: int, paper_id: int) -> dict | None:
        r = self.conn.execute("SELECT * FROM papers WHERE project_id = ? AND id = ?", (pid, paper_id)).fetchone()
        return _paper_row(r) if r else None

    def screen(self, pid: int, paper_id: int, status: str, reason: str = "", notes: str | None = None) -> bool:
        if status not in ("pending", "included", "excluded"):
            raise ValueError("invalid status")
        if status == "excluded" and not reason.strip():
            raise ValueError("An exclusion reason is required")
        with self.tx() as c:
            cur = c.execute(
                "UPDATE papers SET status = ?, reason = ?, notes = COALESCE(?, notes) WHERE project_id = ? AND id = ?",
                (status, reason.strip() if status == "excluded" else "", notes, pid, paper_id),
            )
            return cur.rowcount == 1

    def searches(self, pid: int) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM searches WHERE project_id = ? ORDER BY id DESC", (pid,)).fetchall()
        return [{**dict(r), "errors": json.loads(r["errors"])} for r in rows]

    def prisma(self, pid: int) -> dict:
        s = self.conn.execute(
            "SELECT COALESCE(SUM(identified), 0), COALESCE(SUM(duplicates), 0) FROM searches WHERE project_id = ?", (pid,)
        ).fetchone()
        st = dict(self.conn.execute("SELECT status, COUNT(*) FROM papers WHERE project_id = ? GROUP BY status", (pid,)).fetchall())
        reasons = self.conn.execute(
            """SELECT reason, COUNT(*) AS n FROM papers WHERE project_id = ? AND status = 'excluded'
                                       GROUP BY reason ORDER BY n DESC""",
            (pid,),
        ).fetchall()
        screened = sum(st.values())
        return {
            "identified": s[0],
            "duplicates": s[1],
            "screened": screened,
            "pending": st.get("pending", 0),
            "excluded": st.get("excluded", 0),
            "included": st.get("included", 0),
            "reasons": [dict(r) for r in reasons],
        }

    # ---------------------------------------------------------------- reviews
    def save_review(self, pid: int, method: str, model: str, content: dict) -> int:
        with self.tx() as c:
            return int(
                c.execute(
                    "INSERT INTO reviews (project_id, method, model, content) VALUES (?, ?, ?, ?)",
                    (pid, method, model, json.dumps(content)),
                ).lastrowid
            )

    def latest_review(self, pid: int) -> dict | None:
        r = self.conn.execute("SELECT * FROM reviews WHERE project_id = ? ORDER BY id DESC LIMIT 1", (pid,)).fetchone()
        return {**dict(r), "content": json.loads(r["content"])} if r else None


def _paper_row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["authors"] = json.loads(d["authors"])
    return d
