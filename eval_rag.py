#!/usr/bin/env python3
"""
eval_rag.py - CLI Entrypoint for the Hybrid RAG Ablation Benchmark
==================================================================

This script is a thin CLI wrapper around the modular `evaluation/` package:
  - evaluation/models.py     : Dataclasses (Chunk, EvalItem, RetrievalResult, GuardrailDecision, etc.)
  - evaluation/datasets.py   : MOCK_CORPUS, BENCHMARK, CALIBRATION, 60-row ledger generator, validator
  - evaluation/metrics.py    : Pure math (RRF, Hit@k, MRR@k, Recall@k, Context Precision@k)
  - evaluation/retrievers.py : Embedders, BM25/Dense indices, RetrieverAdapter & ProductionAdapter, NaiveChunker
  - evaluation/guardrails.py : DeterministicGuardrail, extract_entities, calibrate_tau_dense
  - evaluation/judge.py      : OllamaLLM, RAG Triad judge prompts & parsers, preflight_ollama
  - evaluation/runner.py     : run_benchmark and run_chunking_suite orchestration
  - evaluation/reporting.py  : Pandas aggregation, ASCII terminal dashboard, and GFM Markdown exporter

Usage
-----
    python eval_rag.py                              # full run (needs Ollama)
    python eval_rag.py --skip-generation            # retrieval + guardrail only
    python eval_rag.py --offline                    # no Ollama: hashing embedder (smoke test)
    python eval_rag.py --suite chunking             # Naive vs Header-Pinned table chunking ablation
    python eval_rag.py --use-production-index NAME  # benchmark repo's HybridRetriever index
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

# Re-export everything from `evaluation` so existing scripts and unit tests (`import eval_rag as ev`) work seamlessly.
from evaluation import (  # noqa: F401
    BENCHMARK,
    BM25Index,
    BM25Retriever,
    CALIBRATION,
    CATEGORIES,
    CalibrationItem,
    Chunk,
    DenseIndex,
    DeterministicGuardrail,
    Embedder,
    EvalItem,
    FaissRetriever,
    GEN_PROMPT,
    GuardrailDecision,
    HashingEmbedder,
    HybridRRFRetriever,
    JUDGE_PRECISION_PROMPT,
    JUDGE_TRIAD_PROMPT,
    MOCK_CORPUS,
    MODE_ORDER,
    NaiveChunker,
    OllamaEmbedder,
    OllamaLLM,
    ProductionHybridRetrieverAdapter,
    REFUSAL_MESSAGE,
    RetrievalResult,
    RetrievedChunk,
    RetrieverAdapter,
    TABLE_COLUMNS,
    _escape_md_cell,
    _fmt,
    _is_abstention,
    _md_table,
    _norm_likert,
    _parse_json,
    answer_contains_value,
    build_chunking_items,
    build_ledger_markdown,
    build_retrievers,
    calibrate_tau_dense,
    calibration_note,
    category_matrix,
    chunking_table,
    content_tokens,
    context_precision_at_k,
    export_markdown,
    extract_entities,
    header_pinned_chunks,
    hit_at_k,
    judge_coverage_note,
    judge_triad,
    load_benchmark,
    overall_table,
    preflight_ollama,
    print_dashboard,
    recall_at_k,
    reciprocal_rank_at_k,
    reciprocal_rank_fusion,
    render_ascii_table,
    report_chunking,
    run_benchmark,
    run_chunking_suite,
    tokenize,
    validate_benchmark,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("eval_rag")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Hybrid RAG ablation benchmark")
    p.add_argument("--suite", choices=["retrieval", "chunking"], default="retrieval",
                   help="retrieval: BM25/FAISS/Hybrid ablation; chunking: Naive vs Header-Pinned tables")
    p.add_argument("--k", type=int, default=5, help="cutoff for Hit/MRR/Recall")
    p.add_argument("--context-k", type=int, default=None,
                   help="chunks passed to the generator (default: equals --k so IR@k and generation match)")
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--model", default=os.getenv("LLM_MODEL", "llama3.1:8b"),
                   help="generator/judge tag (default: $LLM_MODEL as in app.py, else llama3.1:8b)")
    p.add_argument("--embed-model", default="nomic-embed-text")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--tau-dense", type=float, default=None,
                   help="guardrail dense-cosine threshold (default: auto-calibrated on held-out CALIBRATION set)")
    p.add_argument("--tau-lex", type=float, default=0.34, help="guardrail lexical-coverage threshold")
    p.add_argument("--calibrate", action="store_true", help="print calibration statistics and exit")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--offline", action="store_true", help="hashing embedder, no Ollama (implies --skip-generation)")
    p.add_argument("--use-production-index", metavar="DATASET", default=None,
                   help="benchmark repo's HybridRetriever index (golden ids must match its chunk_ids)")
    p.add_argument("--benchmark-file", default=None,
                   help="JSON list of EvalItem objects (REQUIRED with --use-production-index)")
    p.add_argument("--calibration-file", default=None,
                   help="JSON list of {query, in_domain}; REQUIRED to auto-calibrate on a production index")
    p.add_argument("--guardrail-scope", choices=["shared", "per-mode"], default="shared",
                   help="shared: one mode-independent guardrail decision per query (fair ablation)")
    p.add_argument("--output", default=None, help="markdown report path (default depends on --suite)")
    p.add_argument("--csv", default=None, help="optional path for the raw per-query DataFrame")
    args = p.parse_args(argv)

    skip_gen = args.skip_generation or args.offline
    if args.use_production_index and not args.benchmark_file:
        raise SystemExit("[ERROR] --use-production-index requires --benchmark-file with golden ids "
                         "matching that index's chunk_ids (the built-in benchmark targets the mock corpus).")
    if not args.offline:
        args.model = preflight_ollama(args.ollama_url, None if skip_gen else args.model, args.embed_model)
    llm = None if skip_gen else OllamaLLM(args.model, args.ollama_url)

    if args.suite == "chunking":
        df = run_chunking_suite(args, llm)
        out = Path(args.output or "rag_chunking_report.md")
        report_chunking(df, out, None if skip_gen else args.model)
        if args.csv:
            df.to_csv(args.csv, index=False)
        print(f"\nMarkdown report written to: {out.resolve()}")
        return 0

    items = load_benchmark(Path(args.benchmark_file)) if args.benchmark_file else BENCHMARK
    retrievers, known_ids = build_retrievers(args)
    validate_benchmark(items, known_ids)

    notes: List[str] = []
    cal_items: Optional[List[CalibrationItem]] = None
    if args.calibration_file:
        cal_items = [CalibrationItem(**r) for r in json.loads(Path(args.calibration_file).read_text("utf-8"))]
    elif not args.use_production_index:
        cal_items = CALIBRATION
    tau_dense = args.tau_dense
    if cal_items is not None and (tau_dense is None or args.calibrate):
        probe = next(r for r in retrievers if r.name == "FAISS")
        tau_cal, stats = calibrate_tau_dense(probe, cal_items)
        note = calibration_note(stats, len(cal_items))
        print("CALIBRATION:", note)
        if args.calibrate:
            return 0
        if tau_dense is None:
            tau_dense = tau_cal
            notes.append(note)
    if tau_dense is None:
        tau_dense = 0.58
        logger.warning("No calibration set for the production index; using uncalibrated tau_dense=0.58 "
                       "(pass --calibration-file or --tau-dense).")
        notes.append("tau_dense=0.58 is an UNCALIBRATED default.")
    elif args.tau_dense is not None:
        notes.append(f"tau_dense={tau_dense:.3f} supplied manually.")

    reference = next(r for r in retrievers if r.name == "Hybrid") if args.guardrail_scope == "shared" else None
    ctx_k = args.k if args.context_k is None else args.context_k
    df = run_benchmark(retrievers, items, DeterministicGuardrail(args.tau_lex, tau_dense), llm,
                       k=args.k, context_k=ctx_k, guardrail_reference=reference)
    model_label = None if skip_gen else args.model
    print_dashboard(df, args.k, args.rrf_k, model_label, notes)
    out = Path(args.output or "rag_benchmark_report.md")
    export_markdown(df, out, args.k, args.rrf_k, model_label, notes, corpus_size=len(known_ids))
    if args.csv:
        df.to_csv(args.csv, index=False)
    print(f"\nMarkdown report written to: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
