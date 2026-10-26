"""
evaluation/retrievers.py - Embedders, Retrievers & Chunking Baselines
======================================================================

Provides:
1. `Embedder` interface with `OllamaEmbedder` (using asymmetric `search_document:` / `search_query:`
   prefixes for `nomic-embed-text`) and `HashingEmbedder` (offline character n-gram smoke test).
2. Self-contained `BM25Index` (Okapi BM25) and `DenseIndex` (FAISS `IndexFlatIP` with NumPy fallback).
3. `RetrieverAdapter` plug-in interface and the three ablation modes (`BM25Retriever`,
   `FaissRetriever`, `HybridRRFRetriever`) plus `ProductionHybridRetrieverAdapter` wrapping
   `retriever.HybridRetriever`.
4. Table-chunking ablation baselines: `NaiveChunker` vs. `header_pinned_chunks` (`DualModeChunker`).
"""

from __future__ import annotations

import math
import re
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .metrics import reciprocal_rank_fusion
from .models import Chunk, RetrievalResult, RetrievedChunk

try:  # FAISS is used when installed; exact inner-product NumPy fallback otherwise.
    import faiss  # type: ignore

    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore
    _HAS_FAISS = False

_STOPWORDS = frozenset(
    "a an the and or of to in on for with at by from is are was were be been it its this that these those "
    "what which who whom how when where why do does did can could should would will may might must i we you "
    "our your their my me us they them as if then than so not no yes about into over any all each per out "
    "up down off use used using get gets got has have had having give gives please explain tell write "
    "describe fast long often much many".split()
)


def tokenize(text: str) -> List[str]:
    """Identifier-preserving tokenizer (mirrors retriever.HybridRetriever.tokenize_text)."""
    return re.findall(r"\b[A-Za-z0-9_.-]+\b", text.lower())


def content_tokens(text: str) -> List[str]:
    """Stopword-filtered content tokens used by BM25 and lexical-coverage guardrails."""
    return [t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1]


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        """Return float32 array (n, d); rows need not be normalised."""


class OllamaEmbedder(Embedder):
    """Local Ollama embedding client with connection retries and Nomic asymmetric task prefixes."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434",
                 timeout: int = 120, batch_size: int = 16) -> None:
        self.model, self.base_url, self.timeout, self.batch_size = model, base_url.rstrip("/"), timeout, batch_size
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1.5, status_forcelist=[500, 502, 503, 504],
                        allowed_methods=frozenset(["POST", "GET"]))
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _prefix(self, text: str, is_query: bool) -> str:
        if "nomic-embed" in self.model.lower():
            tag = "search_query: " if is_query else "search_document: "
            return text if text.startswith(tag) else f"{tag}{text}"
        return text

    def embed(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        prefixed = [self._prefix(t, is_query) for t in texts]
        out: List[List[float]] = []
        for i in range(0, len(prefixed), self.batch_size):
            batch = prefixed[i:i + self.batch_size]
            res = self.session.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": batch},
                                    timeout=self.timeout)
            res.raise_for_status()
            out.extend(res.json()["embeddings"])
        return np.asarray(out, dtype=np.float32)


class HashingEmbedder(Embedder):
    """Offline stand-in (character n-gram hashing). Smoke-testing only - no real semantics."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            t = f" {text.lower()} "
            for n in (3, 4):
                for j in range(len(t) - n + 1):
                    mat[row, hash(t[j:j + n]) % self.dim] += 1.0
        return mat


class RetrieverAdapter(ABC):
    """Plug-in point. Subclass and implement `retrieve` to benchmark your own RAG pipeline."""

    name: str = "unnamed"

    @abstractmethod
    def retrieve(self, query: str, k: int) -> RetrievalResult:
        ...


class BM25Index:
    """Okapi BM25 with Robertson-Sparck Jones IDF (Lucene-style +1 smoothing)."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks, self.k1, self.b = list(chunks), k1, b
        self.docs = [content_tokens(c.content) for c in self.chunks]
        self.doc_len = np.array([len(d) for d in self.docs], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean())
        df: Dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        n = len(self.docs)
        self.idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [{t: d.count(t) for t in set(d)} for d in self.docs]

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for t in set(content_tokens(query)):
            if t not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    out[i] += self.idf[t] * f * (self.k1 + 1) / denom
        return out


class DenseIndex:
    """Exact inner-product index over L2-normalised vectors (cosine). FAISS if installed, else NumPy."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        self.chunks, self.embedder = list(chunks), embedder
        vecs = embedder.embed([c.content for c in self.chunks]).astype(np.float32)
        self._normalize(vecs)
        self._vecs = vecs
        self._index = None
        if _HAS_FAISS:
            self._index = faiss.IndexFlatIP(vecs.shape[1])
            self._index.add(vecs)

    @staticmethod
    def _normalize(m: np.ndarray) -> None:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        m /= np.maximum(norms, 1e-12)

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        q = self.embedder.embed([query], is_query=True).astype(np.float32)
        self._normalize(q)
        k = min(k, len(self.chunks))
        if self._index is not None:
            sc, ix = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ix[0], sc[0]) if i != -1]
        sims = (self._vecs @ q[0])
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]


class BM25Retriever(RetrieverAdapter):
    name = "BM25"

    def __init__(self, bm25: BM25Index) -> None:
        self.bm25 = bm25

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        sc = self.bm25.scores(query)
        order = [i for i in np.argsort(-sc)[:k] if sc[i] > 0]
        chunks = [RetrievedChunk(self.bm25.chunks[i].chunk_id, self.bm25.chunks[i].content, float(sc[i]), r)
                  for r, i in enumerate(order, 1)]
        return RetrievalResult(chunks, {"max_dense_cosine": None})


class FaissRetriever(RetrieverAdapter):
    name = "FAISS"

    def __init__(self, dense: DenseIndex) -> None:
        self.dense = dense

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        hits = self.dense.search(query, k)
        chunks = [RetrievedChunk(self.dense.chunks[i].chunk_id, self.dense.chunks[i].content, s, r)
                  for r, (i, s) in enumerate(hits, 1)]
        return RetrievalResult(chunks, {"max_dense_cosine": hits[0][1] if hits else 0.0})


class HybridRRFRetriever(RetrieverAdapter):
    name = "Hybrid"

    def __init__(self, bm25: BM25Index, dense: DenseIndex, rrf_k: int = 60, candidate_pool: int = 20) -> None:
        self.bm25, self.dense, self.rrf_k, self.pool = bm25, dense, rrf_k, candidate_pool

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        n_chunks = len(self.bm25.chunks)
        eff_pool = min(self.pool, max(k, n_chunks // 2)) if n_chunks > k else n_chunks
        sc = self.bm25.scores(query)
        sparse_rank = [int(i) for i in np.argsort(-sc)[:eff_pool] if sc[i] > 0]
        dense_hits = self.dense.search(query, eff_pool)
        dense_rank = [i for i, _ in dense_hits]
        fused = reciprocal_rank_fusion([sparse_rank, dense_rank], k=self.rrf_k)[:k]
        chunks = [RetrievedChunk(self.bm25.chunks[i].chunk_id, self.bm25.chunks[i].content, s, r)
                  for r, (i, s) in enumerate(fused, 1)]
        return RetrievalResult(chunks, {"max_dense_cosine": dense_hits[0][1] if dense_hits else 0.0})


class ProductionHybridRetrieverAdapter(RetrieverAdapter):
    """
    Wraps the repository's `retriever.HybridRetriever` (already indexed on disk).
    `mode` in {"bm25", "faiss", "hybrid"}. Chunk IDs come from `chunk["chunk_id"]`.
    """

    def __init__(self, retriever: Any, mode: str) -> None:
        assert mode in {"bm25", "faiss", "hybrid"}
        self.r, self.mode = retriever, mode
        self.name = {"bm25": "BM25", "faiss": "FAISS", "hybrid": "Hybrid"}[mode]

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        chunks_meta = self.r.chunks
        if self.mode == "hybrid":
            res = self.r.hybrid_search(query, top_k=k)
            out = [RetrievedChunk(str(c["chunk_id"]), c["content"], c["rrf_score"], c["final_rank"]) for c in res]
            cos = [c["dense_stats"]["cosine_sim"] for c in res]
            return RetrievalResult(out, {"max_dense_cosine": max(cos) if cos else 0.0})
        hits = self.r.search_sparse(query, k) if self.mode == "bm25" else self.r.search_dense(query, k)
        out = [RetrievedChunk(str(chunks_meta[i]["chunk_id"]), chunks_meta[i]["content"], s, rk)
               for rk, (i, s) in enumerate(hits, 1)]
        return RetrievalResult(out, {"max_dense_cosine": (hits[0][1] if hits and self.mode == "faiss" else None)})


class NaiveChunker:
    """
    Baseline for `--suite chunking`: line-aware fixed-size splitter. Keeps table rows intact,
    but does NOT re-inject the Markdown table header or section heading into subsequent segments.
    """

    def __init__(self, max_chars: int = 600) -> None:
        self.max_chars = max_chars

    def chunk(self, markdown_text: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        buf: List[str] = []
        size = 0
        for line in markdown_text.splitlines():
            if buf and size + len(line) + 1 > self.max_chars:
                chunks.append(Chunk(f"naive_chunk_{len(chunks) + 1:05d}", "\n".join(buf).strip()))
                buf, size = [], 0
            buf.append(line)
            size += len(line) + 1
        if "".join(buf).strip():
            chunks.append(Chunk(f"naive_chunk_{len(chunks) + 1:05d}", "\n".join(buf).strip()))
        return [c for c in chunks if c.content]


def header_pinned_chunks(markdown_text: str, rows_per_chunk: int = 10) -> List[Chunk]:
    """Wraps `chunker.DualModeChunker` (re-injects column headers + parent heading into every table slice)."""
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from chunker import DualModeChunker  # type: ignore

    with tempfile.TemporaryDirectory() as tmp:
        ch = DualModeChunker(table_rows_per_chunk=rows_per_chunk, table_row_overlap=0, output_dir=tmp)
        raw = ch.chunk_markdown(markdown_text, source_filename="ledger")
    return [Chunk(c["chunk_id"], c["content"]) for c in raw]

