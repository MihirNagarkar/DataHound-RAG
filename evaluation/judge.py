"""Ollama LLM client, generation prompt, and JSON-constrained RAG Triad LLM-as-a-Judge."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .metrics import context_precision_at_k
from .models import EvalItem, REFUSAL_MESSAGE, RetrievedChunk

logger = logging.getLogger("eval_rag")


class OllamaLLM:
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434",
                 timeout: int = 180) -> None:
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=2.0, status_forcelist=[500, 502, 503, 504],
                        allowed_methods=frozenset(["POST", "GET"]))
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        payload: Dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": False,
                                   "options": {"temperature": 0.0, "num_ctx": 4096, "seed": 7}}
        if json_mode:
            payload["format"] = "json"
        res = self.session.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        res.raise_for_status()
        return res.json().get("response", "").strip()


GEN_PROMPT = (
    "You are a strict enterprise documentation assistant. Answer ONLY from the CONTEXT. "
    "If the context lacks the answer, reply exactly: \"" + REFUSAL_MESSAGE + "\" Be concise.\n\n"
    "CONTEXT:\n{context}\n\nQUESTION: {query}\nANSWER:"
)

JUDGE_TRIAD_PROMPT = (
    "You are an impartial RAG evaluator. Score the ANSWER on a 1-5 integer scale.\n"
    "- faithfulness: 5 = every claim is supported by CONTEXT; 1 = mostly unsupported/contradicted.\n"
    "- answer_relevance: 5 = directly and completely addresses QUESTION; 1 = off-topic.\n"
    "Return ONLY JSON: {{\"faithfulness\": <int>, \"answer_relevance\": <int>, \"rationale\": \"<short>\"}}\n\n"
    "QUESTION: {query}\n\nCONTEXT:\n{context}\n\nANSWER: {answer}\n"
)

JUDGE_PRECISION_PROMPT = (
    "For each numbered CONTEXT passage, decide if it contains information useful for answering the QUESTION "
    "(reference answer given). Return ONLY JSON: {{\"relevance\": [<0 or 1 per passage, in order>]}}\n\n"
    "QUESTION: {query}\nREFERENCE ANSWER: {gt}\n\n{passages}\n"
)


def _parse_json(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}


def _norm_likert(v: Any) -> float:
    """Map a 1-5 Likert score to [0,1]; NaN if invalid."""
    try:
        return (min(5.0, max(1.0, float(v))) - 1.0) / 4.0
    except (TypeError, ValueError):
        return float("nan")


def _is_abstention(answer: str) -> bool:
    return REFUSAL_MESSAGE.lower() in answer.strip().lower()


def judge_triad(llm: OllamaLLM, item: EvalItem, chunks: Sequence[RetrievedChunk], answer: str) -> Dict[str, float]:
    nan = float("nan")
    context = "\n\n".join(f"[{c.chunk_id}] {c.content}" for c in chunks)
    scores = {"faithfulness": nan, "answer_relevance": nan, "context_precision": nan}
    if not item.expected_refusal and _is_abstention(answer):
        # Honest abstention when retrieved context lacked evidence: zero hallucination (faithfulness=1.0),
        # but failed to answer an answerable in-domain question (answer_relevance=0.0).
        scores["faithfulness"] = 1.0
        scores["answer_relevance"] = 0.0
    else:
        try:
            j = _parse_json(llm.generate(JUDGE_TRIAD_PROMPT.format(query=item.query, context=context, answer=answer),
                                         json_mode=True))
            scores["faithfulness"] = _norm_likert(j.get("faithfulness"))
            scores["answer_relevance"] = _norm_likert(j.get("answer_relevance"))
        except Exception as exc:  # judge failures must not kill the benchmark
            logger.warning("Triad judge failed for %s: %s", item.query_id, exc)
    try:
        passages = "\n".join(f"{i}. {c.content}" for i, c in enumerate(chunks, 1))
        j = _parse_json(llm.generate(JUDGE_PRECISION_PROMPT.format(query=item.query, gt=item.ground_truth_answer,
                                                                   passages=passages), json_mode=True))
        rel = [int(bool(int(x))) for x in j.get("relevance", [])][: len(chunks)]
        rel += [0] * (len(chunks) - len(rel))
        scores["context_precision"] = context_precision_at_k(rel)
    except Exception as exc:
        logger.warning("Precision judge failed for %s: %s", item.query_id, exc)
    return scores


def preflight_ollama(base_url: str, model: Optional[str], embed_model: Optional[str]) -> Optional[str]:
    """
    Verify Ollama is reachable and the requested models are pulled (avoids a 404 on every call).
    If the generator tag is missing but another tag of the same family is pulled
    (e.g. llama3.1:8b vs llama3.1:8b-instruct-q4_K_M), use it with a warning. Returns resolved model.
    """
    try:
        res = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
        res.raise_for_status()
    except Exception as exc:
        raise SystemExit(f"[ERROR] Cannot reach Ollama at {base_url}: {exc}")
    pulled = [m["name"] for m in res.json().get("models", [])]

    def has(tag: str) -> bool:
        return tag in pulled or (":" not in tag and f"{tag}:latest" in pulled)

    if embed_model and not has(embed_model):
        raise SystemExit(f"[ERROR] Embedding model '{embed_model}' not pulled. Available: {pulled}")
    if model and not has(model):
        family = [m for m in pulled if m.split(":")[0] == model.split(":")[0]]
        if not family:
            raise SystemExit(f"[ERROR] LLM '{model}' not pulled. Available: {pulled}. Use --model.")
        logger.warning("LLM tag '%s' not found; falling back to pulled tag '%s'", model, family[0])
        return family[0]
    return model

