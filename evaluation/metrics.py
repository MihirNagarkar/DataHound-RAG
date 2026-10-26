"""
evaluation/metrics.py - Classical IR & Rank-Aware RAG Metrics
==============================================================

Pure, deterministic mathematical functions for:
1. Reciprocal Rank Fusion (RRF) — score-agnostic rank aggregation across sparse and dense streams.
2. Classical IR Metrics — Hit Rate@k, Mean Reciprocal Rank (MRR@k via per-query RR@k), and Recall@k.
3. Rank-Aware Context Precision@k (CP@k) — average precision over LLM-judged relevant context chunks.
4. Deterministic Table-Cell Answer Accuracy (`answer_contains_value`).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .models import EvalItem


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

    For a document d and a set R of rankers, each returning a 1-based rank r_i(d):

        RRF(d) = sum_{i in R, d in ranking_i}  1 / (k + r_i(d))

    Args:
        rankings: One best-first list of document indices per ranker.
        k: Smoothing constant (default 60 damps top-rank outliers; absent docs contribute 0).
    Returns:
        List of (doc_index, rrf_score) sorted by descending score (ties broken by lower doc_index).
    """
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def hit_at_k(retrieved: Sequence[str], golden: Sequence[str], k: int) -> float:
    """
    Hit Rate@k (per query):

        Hit@k(q) = 1.0  if |top_k(q) ∩ G_q| > 0  else 0.0
    """
    return float(bool(set(retrieved[:k]) & set(golden)))


def reciprocal_rank_at_k(retrieved: Sequence[str], golden: Sequence[str], k: int) -> float:
    """
    Reciprocal Rank @ k (per query; macro-average over queries Q equals MRR@k):

        RR@k(q) = 1 / min{ r : doc_at_rank(r) in G_q, r <= k }   (0.0 if none)
    """
    g = set(golden)
    for rank, cid in enumerate(retrieved[:k], 1):
        if cid in g:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: Sequence[str], golden: Sequence[str], k: int) -> float:
    """
    Recall@k (multi-hop coverage of golden evidence chunks G_q):

        Recall@k(q) = |top_k(q) ∩ G_q| / |G_q|
    """
    g = set(golden)
    return len(set(retrieved[:k]) & g) / len(g) if g else float("nan")


def context_precision_at_k(relevance: Sequence[int]) -> float:
    """
    RAGAS-style rank-aware Context Precision@k for binary relevance judgments rel_i in {0, 1}:

        CP@k = ( sum_{i=1..k} P@i * rel_i ) / ( sum_{i=1..k} rel_i ),
        where P@i = ( sum_{j=1..i} rel_j ) / i

    Rewards placing relevant chunks at Rank #1 rather than Rank #k. Returns 0.0 if nothing is relevant.
    """
    hits, acc = 0, 0.0
    for i, rel in enumerate(relevance, 1):
        if rel:
            hits += 1
            acc += hits / i
    return acc / hits if hits else 0.0


def answer_contains_value(item: EvalItem, answer: str) -> float:
    """Deterministic correctness for table lookups: 1.0 if the gold cell value appears verbatim in the answer."""
    return float(item.ground_truth_answer.lower() in answer.lower())

