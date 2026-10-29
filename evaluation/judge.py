"""Ollama LLM client, Gemini LLM judge client, generation prompt, and JSON-constrained RAG Triad LLM-as-a-Judge."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from pathlib import Path
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


class GeminiLLM:
    """Gemini-based LLM client for decoupled LLM-as-a-Judge evaluation."""

    def __init__(
        self,
        model: Optional[str] = None,
        rpm: Optional[int] = None,
        cache_dir: str = ".judge_cache",
    ) -> None:
        # Load .env via python-dotenv if available, else a minimal parser; never override existing env vars
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            env_file = Path(".env")
            if env_file.exists():
                for line in env_file.read_text("utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for GeminiLLM.")

        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError("google-genai package is required. Install via `pip install -r requirements-eval.txt`.") from e

        self.genai = genai
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.temperature = 0.0
        self.seed = 7

        # Model selection: env GEMINI_JUDGE_MODEL if set; else try "gemini-2.5-flash";
        # if it fails with not-found/deprecated, call models.list() and choose highest-version id matching ^gemini-[0-9.]+-flash$
        env_model = model or os.getenv("GEMINI_JUDGE_MODEL")
        if env_model:
            self.model = env_model
        else:
            chosen = None
            try:
                self.client.models.get(model="gemini-2.5-flash")
                chosen = "gemini-2.5-flash"
            except Exception as exc:
                candidates = []
                pattern = re.compile(r"^models/gemini-[0-9.]+-flash$")
                for m in self.client.models.list():
                    name = m.name or ""
                    if pattern.match(name) and "generateContent" in (m.supported_actions or []):
                        candidates.append(name.replace("models/", ""))
                if candidates:
                    chosen = sorted(candidates)[-1]
                else:
                    chosen = "gemini-2.5-flash"
            self.model = chosen

        logger.info("GeminiLLM initialized with model '%s' (temp=%.1f, seed=%d)", self.model, self.temperature, self.seed)

        # Rate limiting: env GEMINI_RPM default 4 with a min-interval throttle
        env_rpm = os.getenv("GEMINI_RPM")
        self.rpm = rpm or (int(env_rpm) if env_rpm and env_rpm.isdigit() else 4)
        self.min_interval = 60.0 / max(1, self.rpm)
        self.last_call_time = 0.0

        # Stats
        self.judge_calls = 0
        self.judge_failures = 0

        # Disk cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "cache.jsonl"
        self._cache: Dict[str, str] = {}
        if self.cache_file.exists():
            try:
                for line in self.cache_file.read_text("utf-8").splitlines():
                    if line.strip():
                        record = json.loads(line)
                        k = record.get("key")
                        resp = record.get("response")
                        if k and resp is not None:
                            self._cache[k] = resp
            except Exception as exc:
                pass

    def _cache_key(self, prompt: str) -> str:
        s = f"{self.model}|{prompt}|{self.temperature}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        key = self._cache_key(prompt)
        if key in self._cache:
            return self._cache[key]

        self.judge_calls += 1
        max_retries = 6
        base_delay = 2.0
        config = self.types.GenerateContentConfig(
            temperature=self.temperature,
            seed=self.seed,
            response_mime_type="application/json" if json_mode else None,
        )

        for attempt in range(max_retries + 1):
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            try:
                self.last_call_time = time.time()
                res = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                text = (res.text or "").strip()
                self._cache[key] = text
                try:
                    with open(self.cache_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"key": key, "model": self.model, "prompt": prompt, "temperature": self.temperature, "response": text}) + "\n")
                except Exception:
                    pass
                return text
            except Exception as exc:
                err_str = str(exc)
                is_retryable = any(code in err_str for code in ["429", "500", "502", "503", "504", "ResourceExhausted", "InternalServerError", "ServiceUnavailable"])
                if attempt < max_retries and is_retryable:
                    sleep_s = (base_delay * (2 ** attempt)) + random.uniform(0.1, 1.0)
                    m = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
                    if m:
                        sleep_s = max(sleep_s, float(m.group(1)) + 1.0)
                    elif any(c in err_str for c in ["429", "ResourceExhausted"]):
                        sleep_s = max(sleep_s, 15.0)
                    logger.warning("Gemini API error (%s); retrying in %.2fs (attempt %d/%d)", exc, sleep_s, attempt + 1, max_retries)
                    time.sleep(sleep_s)
                else:
                    logger.error("Gemini API call failed persistently after %d retries: %s", attempt, exc)
                    self.judge_failures += 1
                    if self.judge_calls >= 5 and (self.judge_failures / self.judge_calls) > 0.10:
                        raise RuntimeError(f"Judge failure rate exceeded 10% ({self.judge_failures}/{self.judge_calls}). Aborting run.") from exc
                    return ""


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
    if not raw:
        return {}
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {}


def _norm_likert(v: Any) -> float:
    """Map a 1-5 Likert score to [0,1]; NaN if invalid."""
    try:
        return (min(5.0, max(1.0, float(v))) - 1.0) / 4.0
    except (TypeError, ValueError):
        return float("nan")


def _is_abstention(answer: str) -> bool:
    return REFUSAL_MESSAGE.lower() in answer.strip().lower()


def judge_triad(llm: Any, item: EvalItem, chunks: Sequence[RetrievedChunk], answer: str) -> Dict[str, float]:
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
            raw = llm.generate(JUDGE_TRIAD_PROMPT.format(query=item.query, context=context, answer=answer),
                               json_mode=True)
            if raw:
                j = _parse_json(raw)
                scores["faithfulness"] = _norm_likert(j.get("faithfulness"))
                scores["answer_relevance"] = _norm_likert(j.get("answer_relevance"))
        except Exception as exc:  # judge failures must not kill the benchmark
            logger.warning("Triad judge failed for %s: %s", item.query_id, exc)
    try:
        passages = "\n".join(f"{i}. {c.content}" for i, c in enumerate(chunks, 1))
        raw = llm.generate(JUDGE_PRECISION_PROMPT.format(query=item.query, gt=item.ground_truth_answer,
                                                         passages=passages), json_mode=True)
        if raw:
            j = _parse_json(raw)
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
