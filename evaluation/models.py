"""
evaluation/models.py - Data Schemas & Value Objects
====================================================

Defines the immutable input records (`Chunk`, `EvalItem`, `CalibrationItem`) and runtime output
structures (`RetrievedChunk`, `RetrievalResult`, `GuardrailDecision`) shared across all modules in
the evaluation package. Keeping data schemas in a standalone dependency-free module prevents
circular imports between retrievers, guardrails, metrics, and runners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

REFUSAL_MESSAGE = "I cannot answer this: the knowledge base does not contain supporting evidence."


@dataclass(frozen=True)
class Chunk:
    """A single indexed text or table segment in the knowledge base."""

    chunk_id: str
    content: str


@dataclass(frozen=True)
class EvalItem:
    """A labelled benchmark question with ground-truth answer and golden chunk IDs."""

    query_id: str
    category: str
    query: str
    ground_truth_answer: str
    golden_chunk_ids: List[str]
    expected_refusal: bool
    lexical_overlap: Optional[float] = None


@dataclass
class RetrievedChunk:
    """A single ranked chunk returned by a `RetrieverAdapter`."""

    chunk_id: str
    content: str
    score: float
    rank: int


@dataclass
class RetrievalResult:
    """Ranked chunks plus optional retriever diagnostics (e.g. `max_dense_cosine`) consumed by guardrails."""

    chunks: List[RetrievedChunk]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailDecision:
    """Outcome of the pre-generation anti-hallucination guardrail check."""

    refuse: bool
    reason: str


@dataclass(frozen=True)
class CalibrationItem:
    """A held-out query (disjoint from the test benchmark) used strictly to tune `tau_dense`."""

    query: str
    in_domain: bool

