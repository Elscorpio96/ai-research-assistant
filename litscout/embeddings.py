"""Dense embeddings and hybrid retrieval.

LitScout ranks papers with BM25 out of the box. When an embedding model is configured, it also embeds every paper,
stores the vectors in SQLite, and fuses the keyword ranking with the vector-similarity ranking using Reciprocal
Rank Fusion (RRF). This catches papers that answer the question in different words (e.g. "SQLi" vs "SQL injection")
while keeping BM25's precision on exact terms.

Any OpenAI-compatible `/embeddings` endpoint works: OpenAI, Ollama (`nomic-embed-text`), LM Studio, vLLM, etc.
"""

from __future__ import annotations

import logging
import math
from array import array
from collections.abc import Sequence
from typing import Protocol

import httpx

from .config import Settings

log = logging.getLogger(__name__)

RRF_K = 60  # standard constant from Cormack et al. (2009)


class EmbeddingError(Exception):
    pass


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbedder:
    """Calls POST {base_url}/embeddings in batches."""

    def __init__(self, model: str, base_url: str, api_key: str = "", client: httpx.Client | None = None, batch: int = 64) -> None:
        self.model, self.base_url, self.api_key, self.batch = model, base_url.rstrip("/"), api_key, batch
        self.http = client or httpx.Client(timeout=60)
        self.name = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        for i in range(0, len(texts), self.batch):
            chunk = [t[:8000] or " " for t in texts[i : i + self.batch]]
            try:
                r = self.http.post(f"{self.base_url}/embeddings", headers=headers, json={"model": self.model, "input": chunk})
                r.raise_for_status()
                data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
                vecs = [d["embedding"] for d in data]
            except httpx.HTTPStatusError as e:
                detail = e.response.text[:200].replace(self.api_key or "\0", "***")
                raise EmbeddingError(f"Embedding API returned HTTP {e.response.status_code}: {detail}") from None
            except httpx.HTTPError as e:
                raise EmbeddingError(f"Could not reach the embedding API ({type(e).__name__}).") from None
            except (KeyError, TypeError, ValueError) as e:
                raise EmbeddingError(f"Unexpected response from the embedding API: {e}") from None
            if len(vecs) != len(chunk):
                raise EmbeddingError(f"Embedding API returned {len(vecs)} vectors for {len(chunk)} inputs.")
            out += vecs
        return out


def make_embedder(settings: Settings, client: httpx.Client | None = None) -> Embedder | None:
    if settings.embed_provider == "none":
        return None
    return OpenAICompatibleEmbedder(
        model=settings.embed_model,
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key or (settings.llm_api_key if settings.llm_provider == "openai" else ""),
        client=client,
    )


# ------------------------------------------------------------------ vector maths
def normalise(v: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def pack(v: Sequence[float]) -> bytes:
    return array("f", v).tobytes()


def unpack(b: bytes) -> list[float]:
    a = array("f")
    a.frombytes(b)
    return a.tolist()


def ranks(scores: Sequence[float]) -> list[int]:
    """1-based rank of each item (highest score = rank 1)."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    out = [0] * len(scores)
    for r, i in enumerate(order, start=1):
        out[i] = r
    return out


def rrf(*score_lists: Sequence[float], k: int = RRF_K) -> list[float]:
    """Reciprocal Rank Fusion of several score lists over the same items.

    An item only earns credit from a list that actually retrieved it (score > 0), so a paper with no keyword match
    is not rewarded by an arbitrary BM25 tie-break.
    """
    n = len(score_lists[0])
    fused = [0.0] * n
    for scores in score_lists:
        for i, r in enumerate(ranks(scores)):
            if scores[i] > 0:
                fused[i] += 1.0 / (k + r)
    return fused


def paper_text(title: str, abstract: str) -> str:
    return f"{title}. {abstract}".strip()
