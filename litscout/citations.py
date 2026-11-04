"""Reference formatting: APA 7, IEEE and BibTeX."""

from __future__ import annotations

import re
import unicodedata


def _split(name: str) -> tuple[str, str]:
    """('Ada K. Lovelace') -> ('Lovelace', 'Ada K.'); handles 'Lovelace, Ada' too."""
    name = name.strip()
    if "," in name:
        last, first = (x.strip() for x in name.split(",", 1))
        return last, first
    parts = name.split()
    return (parts[-1], " ".join(parts[:-1])) if len(parts) > 1 else (name, "")


def _initials(first: str) -> str:
    return " ".join(f"{p[0]}." for p in re.split(r"[\s\-]+", first.replace(".", " ")) if p)


def apa(p: dict) -> str:
    names = [f"{last}, {_initials(first)}".rstrip(", ") for last, first in map(_split, p["authors"])]
    if not names:
        who = ""
    elif len(names) == 1:
        who = names[0]
    elif len(names) <= 20:
        who = ", ".join(names[:-1]) + ", & " + names[-1]
    else:
        who = ", ".join(names[:19]) + ", ... " + names[-1]
    year = p.get("year") or "n.d."
    title = p["title"].rstrip(".")
    parts = [f"{who} ({year}). {title}." if who else f"{title}. ({year})."]
    if p.get("venue"):
        parts.append(f"*{p['venue']}*.")
    if p.get("doi"):
        parts.append(f"https://doi.org/{p['doi']}")
    elif p.get("url"):
        parts.append(p["url"])
    return " ".join(parts)


def ieee(p: dict, n: int) -> str:
    names = [f"{_initials(first)} {last}".strip() for last, first in map(_split, p["authors"])]
    who = names[0] + " et al." if len(names) > 6 else (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else "".join(names))
    bits = [f"[{n}]", f"{who}," if who else "", f"“{p['title'].rstrip('.')},”"]
    if p.get("venue"):
        bits.append(f"*{p['venue']}*,")
    bits.append(f"{p.get('year') or 'n.d.'}.")
    if p.get("doi"):
        bits.append(f"doi: {p['doi']}.")
    return " ".join(b for b in bits if b)


def bibtex_key(p: dict, used: set[str]) -> str:
    last = _split(p["authors"][0])[0] if p["authors"] else "anon"
    last = unicodedata.normalize("NFKD", last).encode("ascii", "ignore").decode().lower()
    word = next((w for w in re.findall(r"[a-z]+", p["title"].lower()) if len(w) > 3), "paper")
    base = f"{re.sub(r'[^a-z]', '', last) or 'anon'}{p.get('year') or ''}{word}"
    key, i = base, 1
    while key in used:
        i += 1
        key = f"{base}{chr(96 + i)}"
    used.add(key)
    return key


def _bib_escape(s: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", s)


def bibtex(papers: list[dict]) -> str:
    used: set[str] = set()
    out = []
    for p in papers:
        kind = "misc" if "arxiv" in (p.get("venue") or "").lower() or p.get("source") == "arxiv" else "article"
        fields = {
            "title": "{" + _bib_escape(p["title"]) + "}",
            "author": " and ".join(_bib_escape(a) for a in p["authors"]),
            "year": str(p.get("year") or ""),
            "journal" if kind == "article" else "howpublished": _bib_escape(p.get("venue") or ""),
            "doi": p.get("doi") or "",
            "url": p.get("url") or "",
        }
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
        out.append(f"@{kind}{{{bibtex_key(p, used)},\n{body}\n}}")
    return "\n\n".join(out) + "\n"
