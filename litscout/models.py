from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class Paper:
    """A bibliographic record normalised from any source."""

    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    url: str = ""
    source: str = ""
    source_id: str = ""
    citations: int | None = None

    def __post_init__(self) -> None:
        self.title = clean_text(self.title)
        self.abstract = clean_text(self.abstract)
        self.doi = normalise_doi(self.doi)

    @property
    def dedupe_key(self) -> str:
        """DOI if present, otherwise a normalised title + year."""
        return f"doi:{self.doi}" if self.doi else f"title:{normalise_title(self.title)}:{self.year or ''}"


def clean_text(s: str | None) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)  # JATS/HTML tags in Crossref abstracts
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip()


def normalise_doi(doi: str | None) -> str:
    if not doi:
        return ""
    doi = doi.strip().lower()
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi)


def normalise_title(title: str) -> str:
    t = unicodedata.normalize("NFKD", title.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", t).strip()
