"""Benchmark and chunking ablation orchestration runners."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .datasets import MOCK_CORPUS, build_chunking_items, build_ledger_markdown, validate_benchmark
from .guardrails import DeterministicGuardrail
from .judge import GEN_PROMPT, OllamaLLM, judge_triad, _is_abstention
from .metrics import answer_contains_value, hit_at_k, recall_at_k, reciprocal_rank_at_k
from .models import EvalItem, GuardrailDecision, REFUSAL_MESSAGE
from .retrievers import (
    BM25Index,
    BM25Retriever,
    DenseIndex,
    Embedder,
    FaissRetriever,
    HashingEmbedder,
    HybridRRFRetriever,
    NaiveChunker,
    OllamaEmbedder,
    ProductionHybridRetrieverAdapter,
    RetrieverAdapter,
    header_pinned_chunks,
)

logger = logging.getLogger("eval_rag")


def build_retrievers(args: argparse.Namespace) -> Tuple[List[RetrieverAdapter], List[str]]:
    """Returns (retrievers, known_chunk_ids)."""
    if args.use_production_index:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from retriever import HybridRetriever  # type: ignore

        prod = HybridRetriever(ollama_base_url=args.ollama_url, embedding_model=args.embed_model, rrf_k=args.rrf_k)
        prod.load_index(args.use_production_index)
        return ([ProductionHybridRetrieverAdapter(prod, m) for m in ("bm25", "faiss", "hybrid")],
                [str(c["chunk_id"]) for c in prod.chunks])
    embedder: Embedder = HashingEmbedder() if args.offline else OllamaEmbedder(args.embed_model, args.ollama_url)
    bm25, dense = BM25Index(MOCK_CORPUS), DenseIndex(MOCK_CORPUS, embedder)
    return ([BM25Retriever(bm25), FaissRetriever(dense), HybridRRFRetriever(bm25, dense, args.rrf_k)],
            [c.chunk_id for c in MOCK_CORPUS])


def run_benchmark(
    retrievers: Sequence[RetrieverAdapter],
    items: Sequence[EvalItem],
    guardrail: Optional[DeterministicGuardrail],
    llm: Optional[OllamaLLM],
    judge_llm: Optional[Any] = None,
    k: int = 5,
    context_k: int = 3,
    guardrail_reference: Optional[RetrieverAdapter] = None,
    answer_checker: Optional[Callable[[EvalItem, str], float]] = None,
    run_judges: bool = True,
) -> pd.DataFrame:
    """
    Run every retriever on every item; return one row per (query, mode).

    guardrail_reference: if given, the guardrail is evaluated ONCE per query on this retriever's
    evidence (outside all latency timers) and the decision is applied identically to every mode, so
    the ablation isolates retrieval quality rather than guardrail/retriever coupling. If None, the
    guardrail sees each mode's own context (per-mode scope). guardrail=None disables refusals.
    answer_checker: optional deterministic correctness check (item, answer) -> 0/1 -> `answer_correct`.
    run_judges: set False to skip the LLM-as-a-Judge calls (faster; used by the chunking suite).
    """
    rows: List[Dict[str, Any]] = []
    for item in items:
        shared: Optional[GuardrailDecision] = None
        if guardrail is None:
            shared = GuardrailDecision(False, "disabled")
        elif guardrail_reference is not None:
            shared = guardrail.evaluate(item.query, guardrail_reference.retrieve(item.query, k))
        for r in retrievers:
            t0 = time.perf_counter()
            result = r.retrieve(item.query, k)
            retrieval_ms = (time.perf_counter() - t0) * 1000
            ids = [c.chunk_id for c in result.chunks]
            decision = shared if shared is not None else guardrail.evaluate(item.query, result)  # type: ignore[union-attr]
            refused = decision.refuse
            ctx = result.chunks[:context_k]
            row: Dict[str, Any] = {
                "query_id": item.query_id, "category": item.category, "mode": r.name,
                "expected_refusal": item.expected_refusal, "refused": refused,
                "refusal_correct": refused == item.expected_refusal, "guardrail_reason": decision.reason,
                "retrieved_ids": ",".join(ids), "retrieval_ms": retrieval_ms, "generation_ms": float("nan"),
                "answer": None,
                "query_text": item.query,
                "ground_truth": item.ground_truth_answer,
            }
            if item.expected_refusal:  # IR metrics undefined; excluded from macro-averages
                row.update(hit=float("nan"), rr=float("nan"), recall=float("nan"))
            else:
                row.update(hit=hit_at_k(ids, item.golden_chunk_ids, k),
                           rr=reciprocal_rank_at_k(ids, item.golden_chunk_ids, k),
                           recall=recall_at_k(ids, item.golden_chunk_ids, k))
            row.update(faithfulness=float("nan"), answer_relevance=float("nan"), context_precision=float("nan"),
                       answer_correct=float("nan"))

            if llm is not None:
                if refused:
                    row["answer"] = REFUSAL_MESSAGE
                else:
                    context = "\n\n".join(f"[{c.chunk_id}] {c.content}" for c in ctx)
                    t1 = time.perf_counter()
                    try:
                        answer = llm.generate(GEN_PROMPT.format(context=context, query=item.query))
                    except Exception as exc:
                        logger.error("Generation failed (%s/%s): %s", item.query_id, r.name, exc)
                        answer = ""
                    row["generation_ms"] = (time.perf_counter() - t1) * 1000
                    row["answer"] = answer
                    if answer_checker is not None and not item.expected_refusal:
                        row["answer_correct"] = answer_checker(item, answer)
                    actual_judge = judge_llm if judge_llm is not None else llm
                    if run_judges and actual_judge is not None and answer and not item.expected_refusal:
                        row.update(judge_triad(actual_judge, item, ctx, answer))
            if item.expected_refusal:
                ans = str(row.get("answer") or "")
                row["end_to_end_abstention"] = True if (refused or _is_abstention(ans)) else False
            else:
                row["end_to_end_abstention"] = float("nan")
            rows.append(row)
            logger.info("%s | %-6s | hit=%s rr=%s refused=%s", item.query_id, r.name,
                        row["hit"], f"{row['rr']:.2f}" if row["rr"] == row["rr"] else "-", refused)
    return pd.DataFrame(rows)


def run_chunking_suite(args: argparse.Namespace, llm: Optional[OllamaLLM]) -> pd.DataFrame:
    md, rows = build_ledger_markdown()
    embedder: Embedder = HashingEmbedder() if args.offline else OllamaEmbedder(args.embed_model, args.ollama_url)
    strategies = {"Naive": NaiveChunker().chunk(md), "Header-Pinned": header_pinned_chunks(md)}
    ctx_k = args.context_k if getattr(args, "context_k", None) is not None else int(os.getenv("CHUNKING_CONTEXT_K", "1"))
    frames: List[pd.DataFrame] = []
    for name, chunks in strategies.items():
        logger.info("Chunking strategy %-13s -> %d chunks", name, len(chunks))
        items = build_chunking_items(rows, chunks)
        validate_benchmark(items, [c.chunk_id for c in chunks])
        bm25, dense = BM25Index(chunks), DenseIndex(chunks, embedder)
        retrievers = [BM25Retriever(bm25), FaissRetriever(dense), HybridRRFRetriever(bm25, dense, args.rrf_k)]
        df = run_benchmark(retrievers, items, None, llm, k=3, context_k=ctx_k,
                           answer_checker=answer_contains_value, run_judges=False)
        df.insert(0, "strategy", name)
        df["n_chunks"] = len(chunks)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

