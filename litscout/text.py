"""Dependency-free text utilities: tokenising, BM25 ranking, TF-IDF, k-means themes, extractive summaries."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from collections.abc import Sequence

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before being below between
    both but by can could did do does doing down during each few for from further had has have having he her here hers
    him his how i if in into is it its itself just me more most my no nor not now of off on once only or other our ours
    out over own same she should so some such than that the their theirs them then there these they this those through
    to too under until up very was we were what when where which while who whom why will with would you your yours
    using use used based paper study studies propose proposed approach method methods results result show shows shown
    present presents new novel however among within also via one two three may well various several et al thus therefore
    research article work findings finding provide provides including include includes without yet whether
    across per first second less many much given make makes made""".split()
)

_TOKEN = re.compile(r"[a-z][a-z0-9\-]+")


def tokens(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 2]


def stem(word: str) -> str:
    """A light suffix stripper; good enough to merge 'detection'/'detecting', 'attacks'/'attack'."""
    for suf in ("ations", "ation", "ions", "ion", "ments", "ment", "ings", "ing", "ies", "ers", "ed", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: -len(suf)] + ("y" if suf == "ies" else "")
    return word


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if len(p.split()) >= 5]


class BM25:
    """Okapi BM25 over a small in-memory corpus."""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokens(d)) for d in docs]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.docs else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokens(query)
        out = []
        for d, length in zip(self.docs, self.lens, strict=True):
            s = 0.0
            for t in q:
                tf = d.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / (self.avg or 1)))
            out.append(s)
        return out


def tfidf_vectors(docs: Sequence[str]) -> tuple[list[dict[str, float]], dict[str, float]]:
    counts = [Counter(tokens(d)) for d in docs]
    df: Counter[str] = Counter()
    for c in counts:
        df.update(c.keys())
    n = len(docs)
    idf = {t: math.log((1 + n) / (1 + f)) + 1 for t, f in df.items()}
    vecs = []
    for c in counts:
        v = {t: (1 + math.log(tf)) * idf[t] for t, tf in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({t: x / norm for t, x in v.items()})
    return vecs, idf


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())


def _centroid(vecs: list[dict[str, float]]) -> dict[str, float]:
    c: Counter[str] = Counter()
    for v in vecs:
        c.update(v)
    norm = math.sqrt(sum(x * x for x in c.values())) or 1.0
    return {t: x / norm for t, x in c.items()}


def kmeans(vecs: list[dict[str, float]], k: int, seed: int = 7, iters: int = 25) -> list[int]:
    """Spherical k-means with k-means++ seeding. Deterministic for a given seed."""
    n = len(vecs)
    if n == 0:
        return []
    k = max(1, min(k, n))
    rng = random.Random(seed)  # noqa: S311 - deterministic clustering, not security
    centres = [vecs[rng.randrange(n)]]
    while len(centres) < k:
        d = [1 - max(cosine(v, c) for c in centres) for v in vecs]
        total = sum(d)
        if total <= 0:
            break
        r, acc = rng.random() * total, 0.0
        for i, w in enumerate(d):
            acc += w
            if acc >= r:
                centres.append(vecs[i])
                break
    labels = [0] * n
    for _ in range(iters):
        new = [max(range(len(centres)), key=lambda j, v=v: cosine(v, centres[j])) for v in vecs]
        if new == labels and _ > 0:
            break
        labels = new
        centres = [_centroid([v for v, lab in zip(vecs, labels, strict=True) if lab == j]) or centres[j] for j in range(len(centres))]
    # Renumber clusters by size, largest first, for stable output
    order = [lab for lab, _ in Counter(labels).most_common()]
    remap = {old: new for new, old in enumerate(order)}
    return [remap[lab] for lab in labels]


def top_terms(vecs: list[dict[str, float]], n: int = 4, exclude: set[str] | None = None) -> list[str]:
    c = _centroid(vecs)
    exclude = exclude or set()
    return [t for t, _ in sorted(c.items(), key=lambda kv: -kv[1]) if t not in exclude][:n]


def key_sentence(text: str, focus: dict[str, float], idf: dict[str, float]) -> str:
    """The sentence of `text` most similar to a theme centroid; falls back to the first sentence."""
    sents = sentences(text)
    if not sents:
        return text.strip()
    best, best_s = sents[0], -1.0
    for i, s in enumerate(sents):
        c = Counter(tokens(s))
        v = {t: (1 + math.log(tf)) * idf.get(t, 1.0) for t, tf in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        score = cosine({t: x / norm for t, x in v.items()}, focus) + (0.02 if i == 0 else 0)
        if score > best_s:
            best, best_s = s, score
    return best
