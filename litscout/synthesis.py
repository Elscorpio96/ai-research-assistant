"""Draft a literature review from the included papers.

Two drafters share one output format:
  * extractive (default, offline): themes by k-means over TF-IDF, sentences quoted from the papers' own abstracts.
  * LLM (optional): an abstractive draft whose citations are checked against the numbered source list;
    any citation to a source that does not exist is removed and reported.

Every claim in either draft is attached to a numbered source, so a human can verify it.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime

from . import citations
from .text import _TOKEN, STOPWORDS, cosine, kmeans, sentences, stem, tfidf_vectors, tokens

FINDING_CUES = re.compile(
    r"\b(\d+(\.\d+)?\s?%|accuracy|f1|kappa|outperform\w*|reduc\w+|improv\w+|found|show\w*|achiev\w+|predict\w*|"
    r"significant\w*|break\w* even|faster|lower|higher|missed|evade\w*|blocked)\b",
    re.I,
)
LIMIT_CUES = re.compile(
    r"\b(however|limit\w*|drop\w*|requires?|too high|few|fragile|insufficient|sensitive|need\w*|"
    r"lack|false positives?)\b",
    re.I,
)
DISCLAIMER = (
    "This is a machine-generated draft. Check every statement against the cited source before using it, and add your own critical analysis."
)


# ------------------------------------------------------------------ helpers
def numbered(papers: list[dict]) -> list[dict]:
    """Assign stable citation numbers: oldest first, then by first author."""
    ordered = sorted(papers, key=lambda p: (p.get("year") or 9999, (p["authors"] or [""])[0], p["title"]))
    return [{**p, "n": i + 1} for i, p in enumerate(ordered)]


def short_cite(p: dict) -> str:
    """Narrative citation: 'Okafor and Brandt (2021)' / 'Mensah et al. (2022)'."""
    lasts = [citations._split(a)[0] for a in p["authors"]]
    who = (
        lasts[0]
        if len(lasts) == 1
        else f"{lasts[0]} and {lasts[1]}"
        if len(lasts) == 2
        else f"{lasts[0]} et al."
        if lasts
        else "An untitled source"
    )
    return f"{who} ({p.get('year') or 'n.d.'})"


def cite_list(ns: list[int]) -> str:
    return "[" + ", ".join(str(n) for n in sorted(set(ns))) + "]"


def third_person(sentence: str) -> str:
    s = re.sub(r"^We\b", "The authors", sentence)
    s = re.sub(r"\bwe\b", "they", s)
    s = re.sub(r"\bour\b", "their", s)
    s = re.sub(r"\bOur\b", "Their", s)
    return s


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:] if len(s) > 1 and s[0].isupper() and s[1].islower() else s


def surface_forms(texts: list[str]) -> dict[str, str]:
    """Map each stem to its most common original word, so theme labels read naturally."""
    forms: dict[str, Counter[str]] = defaultdict(Counter)
    for t in texts:
        for w in _TOKEN.findall(t.lower()):
            if w not in STOPWORDS and len(w) > 2:
                forms[stem(w)][w] += 1
    return {k: v.most_common(1)[0][0] for k, v in forms.items()}


def join_words(words: list[str]) -> str:
    if not words:
        return ""
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _sentence_vec(s: str, idf: dict[str, float]) -> dict[str, float]:
    c = Counter(tokens(s))
    v = {t: (1 + math.log(tf)) * idf.get(t, 1.0) for t, tf in c.items()}
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {t: x / norm for t, x in v.items()}


ANAPHORA = re.compile(r"^(it|they|this|these|those|most|all|such|both|the latter|the former)\b", re.I)


def best_sentence(abstract: str, focus: dict[str, float], idf: dict[str, float], cue: re.Pattern, skip: set[str] = frozenset()) -> str:
    """Pick the sentence closest to the focus, preferring findings and avoiding sentences that depend on context."""
    cands = [s for s in sentences(abstract) if s not in skip] or sentences(abstract) or [abstract]
    return max(
        cands, key=lambda s: cosine(_sentence_vec(s, idf), focus) + (0.25 if cue.search(s) else 0) - (0.3 if ANAPHORA.match(s) else 0)
    )


def as_clause(sentence: str) -> str:
    """Make a quoted sentence fit after 'In X (2021) [n], ...' or 'X note that ...'."""
    s = re.sub(r"^(However|Moreover|Furthermore|Therefore|Thus|In addition|Overall),\s*", "", third_person(sentence))
    return _lower_first(s).rstrip(".")


# ------------------------------------------------------------------ extractive drafter
def extractive_review(project: dict, papers: list[dict], prisma: dict, sources_used: list[str]) -> dict:
    if len(papers) < 2:
        raise ValueError("Include at least two papers before drafting a review.")
    ps = numbered(papers)
    docs = [f"{p['title']}. {p['abstract']}" for p in ps]
    vecs, idf = tfidf_vectors([f"{p['title']}. {p['title']}. {p['abstract']}" for p in ps])  # titles count double
    k = 1 if len(ps) < 4 else min(5, max(2, round(math.sqrt(len(ps) / 2))))
    labels = kmeans(vecs, k)
    forms = surface_forms(docs)
    question_terms = set(tokens(project["question"]))

    themes = []
    for j in sorted(set(labels)):
        members = [p for p, lab in zip(ps, labels, strict=True) if lab == j]
        mvecs = [v for v, lab in zip(vecs, labels, strict=True) if lab == j]
        others = [v for v, lab in zip(vecs, labels, strict=True) if lab != j]
        # Label with terms that distinguish this theme from the rest and from the question everyone shares
        terms = _theme_terms(mvecs, others, question_terms) or _theme_terms(mvecs, others, set())
        words = [forms.get(t, t) for t in terms]
        themes.append(
            {
                "label": f"Theme {j + 1}: {', '.join(words[:3])}" if words else f"Theme {j + 1}",
                "terms": [forms.get(t, t) for t in terms],
                "papers": [p["n"] for p in members],
                "_vec": _centroid(mvecs),
            }
        )

    years = [p["year"] for p in ps if p.get("year")]
    span = f"between {min(years)} and {max(years)}" if years else "in the sources found"
    intro = f"This review synthesises {len(ps)} studies published {span} that address the question: “{project['question'].rstrip('?')}?” "
    if prisma.get("identified"):
        intro += (
            f"Searches of {join_words(sources_used) if sources_used else 'the selected sources'} identified "
            f"{prisma['identified']} records; after removing {prisma['duplicates']} duplicates and excluding "
            f"{prisma['excluded']} at title and abstract screening, {prisma['included']} were included. "
        )
    intro += (
        f"Clustering of titles and abstracts groups the studies into {len(themes)} theme{'s' if len(themes) != 1 else ''}, "
        "each labelled below by its most distinctive key terms; rename the themes once you have read the papers."
    )

    sections = [{"heading": "Introduction", "paragraphs": [intro]}]
    used_sents: set[str] = set()
    by_n = {p["n"]: p for p in ps}
    for t in themes:
        members = sorted((by_n[n] for n in t["papers"]), key=lambda p: (p.get("year") or 0, p["n"]))
        opening = f"This theme groups {len(members)} {'study' if len(members) == 1 else 'studies'} {cite_list(t['papers'])}" + (
            f" sharing the key terms {join_words(t['terms'][:4])}." if t["terms"] else "."
        )
        body = []
        for p in members:
            s = best_sentence(p["abstract"] or p["title"], t["_vec"], idf, FINDING_CUES, used_sents)
            used_sents.add(s)
            body.append(f"In {short_cite(p)} [{p['n']}], {as_clause(s)}.")
        sections.append({"heading": t["label"], "paragraphs": [opening + " " + " ".join(body)]})

    # Limitations and gaps quoted from the papers themselves
    limits = []
    for p in ps:
        for s in sentences(p["abstract"]):
            if LIMIT_CUES.search(s) and s not in used_sents and not ANAPHORA.match(s):
                limits.append((p, s))
                break
    gap_par = []
    if limits:
        gap_par.append(
            "The included studies report several limitations. "
            + " ".join(f"{short_cite(p)} note that {as_clause(s)} [{p['n']}]." for p, s in limits[:5])
        )
    meta = []
    if years and max(years) < datetime.now(UTC).year - 2:
        meta.append(f"the most recent included study is from {max(years)}, so newer work may be missing")
    preprints = [p["n"] for p in ps if "arxiv" in (p.get("venue") or "").lower()]
    if preprints:
        meta.append(
            f"{len(preprints)} source{'s are' if len(preprints) > 1 else ' is a'} preprint{'s' if len(preprints) > 1 else ''} "
            f"that may not be peer reviewed {cite_list(preprints)}"
        )
    missing = [p["n"] for p in ps if not p["abstract"]]
    if missing:
        meta.append(
            f"{len(missing)} source{'s have' if len(missing) > 1 else ' has'} no abstract and should be read in full {cite_list(missing)}"
        )
    if meta:
        gap_par.append("Regarding the evidence base itself: " + "; ".join(meta) + ".")
    if gap_par:
        sections.append({"heading": "Limitations and research gaps", "paragraphs": gap_par})

    sections.append(
        {
            "heading": "Conclusion",
            "paragraphs": [
                f"The {len(ps)} included studies fall into {len(themes)} thematic cluster{'s' if len(themes) != 1 else ''}. "
                "The evidence summarised above is drawn directly from the studies' abstracts; a full review should read each paper, "
                "appraise its methods and quality, and compare findings across themes."
            ],
        }
    )

    return _package(project, ps, sections, themes, "extractive", "", [DISCLAIMER], idf)


GENERIC = {
    stem(w)
    for w in """model models data dataset accuracy report reports reported performance evaluate evaluation evaluated
    system systems analysis tool tools tested test approach framework average mean high low small large learning""".split()
}


def _theme_terms(members: list[dict[str, float]], others: list[dict[str, float]], exclude: set[str]) -> list[str]:
    """Rank terms by lift: how much more weight they carry inside the theme than outside, times how many members use them."""
    inside: Counter[str] = Counter()
    present: Counter[str] = Counter()
    for v in members:
        inside.update(v)
        present.update(v.keys())
    outside: Counter[str] = Counter()
    for v in others:
        outside.update(v)
    n_in, n_out = len(members), max(1, len(others))
    score = {
        t: (w / n_in - outside.get(t, 0) / n_out) * (present[t] / n_in)
        for t, w in inside.items()
        if t not in exclude and t not in GENERIC and (present[t] >= 2 or n_in == 1)
    }
    return [t for t, sc in sorted(score.items(), key=lambda kv: -kv[1]) if sc > 0][:4]


def _centroid(vecs: list[dict[str, float]]) -> dict[str, float]:
    c: Counter[str] = Counter()
    for v in vecs:
        c.update(v)
    norm = math.sqrt(sum(x * x for x in c.values())) or 1.0
    return {t: x / norm for t, x in c.items()}


def _package(
    project: dict,
    ps: list[dict],
    sections: list[dict],
    themes: list[dict],
    method: str,
    model: str,
    warnings: list[str],
    idf: dict[str, float] | None = None,
) -> dict:
    if idf is None:
        _, idf = tfidf_vectors([f"{p['title']}. {p['abstract']}" for p in ps])
    table = []
    for p in ps:
        focus = _sentence_vec(p["title"], idf)
        finding = best_sentence(p["abstract"], focus, idf, FINDING_CUES) if p["abstract"] else "(no abstract available)"
        table.append({"n": p["n"], "cite": short_cite(p), "title": p["title"], "finding": third_person(finding)})
    return {
        "question": project["question"],
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "method": method,
        "model": model,
        "sections": sections,
        "themes": [{k: v for k, v in t.items() if not k.startswith("_")} for t in themes],
        "table": table,
        "references": [
            {
                "n": p["n"],
                "paper_id": p["id"],
                "apa": citations.apa(p),
                "ieee": citations.ieee(p, p["n"]),
                "doi": p.get("doi", ""),
                "url": p.get("url", ""),
                "demo": p.get("source") == "demo",
            }
            for p in ps
        ],
        "warnings": warnings
        + (
            ["Some sources come from the FICTIONAL demo corpus and must not be cited."]
            if any(p.get("source") == "demo" for p in ps)
            else []
        ),
    }


# ------------------------------------------------------------------ LLM drafter
CITE_RE = re.compile(r"\[(\d+(?:\s*[,–\-]\s*\d+)*)\]")

SYSTEM = (
    "You are an academic writing assistant drafting a literature review. Use ONLY the numbered sources provided. "
    "Every factual sentence must end with one or more citations in square brackets, e.g. [3] or [2, 5]. "
    "Never invent sources, numbers, authors or findings that are not in the provided abstracts. "
    "If the sources do not support a point, say the evidence is limited. Write in formal academic English."
)


def build_prompt(project: dict, ps: list[dict]) -> str:
    src = "\n\n".join(
        f"[{p['n']}] {short_cite(p)}. {p['title']}. {p.get('venue') or ''}\nAbstract: {p['abstract'] or '(no abstract available)'}"
        for p in ps
    )
    crit = ""
    if project.get("inclusion") or project.get("exclusion"):
        crit = f"\nInclusion criteria: {project.get('inclusion') or '-'}\nExclusion criteria: {project.get('exclusion') or '-'}\n"
    return (
        f"Research question: {project['question']}\n{crit}\nSources:\n\n{src}\n\n"
        "Write a thematic literature review. Respond with JSON only, in this exact shape:\n"
        '{"themes": [{"label": "short theme name", "papers": [1, 2]}], '
        '"sections": [{"heading": "Introduction", "paragraphs": ["..."]}, {"heading": "<theme>", "paragraphs": ["..."]}, '
        '{"heading": "Limitations and research gaps", "paragraphs": ["..."]}, {"heading": "Conclusion", "paragraphs": ["..."]}]}'
    )


def validate_citations(text: str, valid: set[int]) -> tuple[str, list[int]]:
    """Remove citation numbers that are not in `valid`. Returns cleaned text and the removed numbers."""
    removed: list[int] = []

    def fix(m: re.Match) -> str:
        nums: list[int] = []
        for part in re.split(r"\s*,\s*", m.group(1)):
            rng = re.split(r"\s*[–\-]\s*", part)
            if len(rng) == 2 and rng[0].isdigit() and rng[1].isdigit() and int(rng[1]) - int(rng[0]) < 50:
                nums += list(range(int(rng[0]), int(rng[1]) + 1))
            elif part.strip().isdigit():
                nums.append(int(part))
        good = [n for n in nums if n in valid]
        removed.extend(n for n in nums if n not in valid)
        return cite_list(good) if good else ""

    return re.sub(r"\s+([.,;])", r"\1", CITE_RE.sub(fix, text)), removed


def parse_llm_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("The model did not return JSON")
    data = json.loads(m.group(0))
    if not isinstance(data.get("sections"), list) or not data["sections"]:
        raise ValueError("The model's JSON has no sections")
    return data


def llm_review(project: dict, papers: list[dict], complete, feedback: str = "") -> dict:
    """`complete(system, prompt) -> (text, model_name)` is injected so it can be mocked in tests.

    `feedback` is appended to the prompt when the review pipeline asks the model to redraft after failed citation checks.
    """
    if len(papers) < 2:
        raise ValueError("Include at least two papers before drafting a review.")
    ps = numbered(papers)
    prompt = build_prompt(project, ps)
    if feedback:
        prompt += f"\n\nYour previous draft failed these checks; fix them in this draft:\n{feedback}"
    raw, model = complete(SYSTEM, prompt)
    data = parse_llm_json(raw)
    valid = {p["n"] for p in ps}
    removed: list[int] = []
    uncited = 0
    sections = []
    for s in data["sections"]:
        paras = []
        for para in s.get("paragraphs", []):
            clean, bad = validate_citations(str(para), valid)
            removed += bad
            if not CITE_RE.search(clean) and s.get("heading", "").lower() not in ("conclusion",):
                uncited += 1
            paras.append(clean)
        sections.append({"heading": str(s.get("heading", "")), "paragraphs": paras})
    themes = [
        {"label": str(t.get("label", "")), "papers": sorted({int(n) for n in t.get("papers", []) if int(n) in valid}), "terms": []}
        for t in data.get("themes", [])
        if isinstance(t, dict)
    ]
    warnings = [DISCLAIMER]
    if removed:
        warnings.append(f"The model cited sources that do not exist ({cite_list(sorted(set(removed)))}); these citations were removed.")
    if uncited:
        warnings.append(f"{uncited} paragraph(s) contain no citation. Treat them as unsupported until checked.")
    review = _package(project, ps, sections, themes, "llm", model, warnings)
    review["checks"] = {"invented_citations": sorted(set(removed)), "uncited_paragraphs": uncited}
    return review


# ------------------------------------------------------------------ export
def to_markdown(project: dict, review: dict, style: str = "apa") -> str:
    drafter = f"LLM: {review['model']}" if review["method"] == "llm" else "offline extractive drafter"
    out = [
        f"# {project['title']}",
        "",
        f"**Research question:** {review['question']}",
        "",
        f"*Drafted {review['generated_at']} by LitScout ({drafter}).*",
        "",
    ]
    out += [f"> **Note:** {w}" for w in review["warnings"]] + [""]
    for s in review["sections"]:
        out += [f"## {s['heading']}", ""] + [p + "\n" for p in s["paragraphs"]]
    out += ["## Summary of included studies", "", "| # | Study | Key finding (from abstract) |", "|---|---|---|"]
    out += [f"| {r['n']} | {r['cite']} | {r['finding'].replace('|', '/')} |" for r in review["table"]]
    out += ["", "## References", ""]
    out += [f"{r['n']}. {r['apa']}" if style == "apa" else r["ieee"] for r in review["references"]]
    return "\n".join(out) + "\n"
