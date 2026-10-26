"""
evaluation/guardrails.py - Deterministic Anti-Hallucination Guardrail & Calibration
====================================================================================

Implements a fast, reproducible, LLM-free guardrail that gates queries before generation:
1. Prompt-injection regex detection (`ignore previous instructions`, `reveal system prompt`, etc.).
2. Empty-context rejection.
3. Opaque entity-code verification (`E-4017`, `TR6`, `2024-INV-001`): refuses if a queried code is
   absent from retrieved context while ignoring version/tech tokens (`v2`, `h200`, `sha256`).
4. Dual lexical + dense evidence gate (`coverage < tau_lex` AND `max_dense_cosine < tau_dense`),
   with `calibrate_tau_dense` tuning `tau_dense` on a held-out calibration set.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Sequence, Tuple

from .models import CalibrationItem, GuardrailDecision, RetrievalResult
from .retrievers import RetrieverAdapter, content_tokens, tokenize

logger = logging.getLogger("eval_rag")

_INJECTION_PATTERNS = [re.compile(p, re.I) for p in (
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"(reveal|print|show|repeat).{0,30}(system|hidden) prompt",
    r"you are now\b", r"\bjailbreak\b", r"disregard .{0,20}(rules|instructions)",
)]

# Identifier = alphanumeric token(s) joined by '-'/'_' containing >=1 letter AND >=1 digit (leading digits
# allowed, e.g. 2024-INV-001). Kept only if it looks like an opaque ID: a separator is present OR >=2 letters
# (TR6, OPR12) - this drops v2, s3, h200. Common technical terms are whitelisted so e.g. "sha256" cannot
# trigger a spurious entity-absent refusal. Slash-delimited IDs (A/12) are intentionally not matched.
_ENTITY_RE = re.compile(r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*\b")
_ENTITY_IGNORE = frozenset({"sha1", "sha256", "sha512", "md5", "utf8", "ipv4", "ipv6", "http2", "http3",
                            "oauth2", "base64", "tls12", "tls13", "x86", "arm64", "amd64", "ec2", "s3"})


def extract_entities(query: str) -> List[str]:
    """Opaque identifier codes in the query that the guardrail must verify in retrieved evidence."""
    out: List[str] = []
    for m in _ENTITY_RE.findall(query):
        letters = sum(ch.isalpha() for ch in m)
        if m.lower() in _ENTITY_IGNORE or not (("-" in m or "_" in m) or letters >= 2):
            continue
        out.append(m)
    return out


class DeterministicGuardrail:
    """
    Rule-based, LLM-free, post-retrieval guardrail. Refuses when:
      1. Prompt-injection pattern matches the query.
      2. Retrieval returned no context.
      3. A specific entity code (e.g. 'E-9999') in the query is absent from retrieved context.
      4. Evidence is weak: lexical coverage < tau_lex AND max dense cosine < tau_dense
         (dense signal ignored when the retriever has none, i.e. BM25-only).
    """

    def __init__(self, tau_lex: float = 0.34, tau_dense: float = 0.58) -> None:
        self.tau_lex, self.tau_dense = tau_lex, tau_dense

    def evaluate(self, query: str, result: RetrievalResult) -> GuardrailDecision:
        if any(p.search(query) for p in _INJECTION_PATTERNS):
            return GuardrailDecision(True, "prompt_injection")
        if not result.chunks:
            return GuardrailDecision(True, "empty_context")
        ctx = " ".join(c.content for c in result.chunks).lower()
        for ent in extract_entities(query):
            if ent.lower() not in ctx:
                return GuardrailDecision(True, f"entity_absent:{ent}")
        q_tokens = set(content_tokens(query))
        ctx_tokens = set(tokenize(ctx))
        coverage = len(q_tokens & ctx_tokens) / len(q_tokens) if q_tokens else 0.0
        dense = result.diagnostics.get("max_dense_cosine")
        if coverage < self.tau_lex and (dense is None or dense < self.tau_dense):
            return GuardrailDecision(True, f"weak_evidence(cov={coverage:.2f},cos={dense})")
        return GuardrailDecision(False, "ok")


def calibrate_tau_dense(probe: RetrieverAdapter, items: Sequence[CalibrationItem]) -> Tuple[float, Dict[str, Any]]:
    """
    Choose tau_dense as the midpoint between the weakest in-domain top-1 cosine and the strongest
    out-of-domain top-1 cosine measured on the held-out calibration set:

        tau = (min_{q in ID} cos_1(q) + max_{q in OOD} cos_1(q)) / 2
    """
    id_cos, ood_cos = [], []
    for it in items:
        cos = probe.retrieve(it.query, 1).diagnostics.get("max_dense_cosine")
        if cos is None:
            raise ValueError("Calibration probe must provide max_dense_cosine (use the FAISS retriever).")
        (id_cos if it.in_domain else ood_cos).append(float(cos))
    lo, hi = min(id_cos), max(ood_cos)
    stats = {"in_domain_cos": id_cos, "ood_cos": ood_cos, "min_in_domain": lo, "max_ood": hi,
             "separable": lo > hi, "tau_dense": (lo + hi) / 2.0}
    if not stats["separable"]:
        logger.warning("Calibration classes overlap (min ID cos %.3f <= max OOD cos %.3f); tau is unreliable.", lo, hi)
    return stats["tau_dense"], stats


def calibration_note(stats: Dict[str, Any], n: int) -> str:
    return (f"tau_dense={stats['tau_dense']:.3f} calibrated on {n} held-out calibration queries "
            f"(min in-domain top-1 cosine {stats['min_in_domain']:.3f}, max OOD {stats['max_ood']:.3f}, "
            f"separable={stats['separable']}); none of the benchmark queries were used for tuning.")

