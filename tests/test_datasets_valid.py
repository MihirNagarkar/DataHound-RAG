"""Unit tests verifying dataset integrity and frozen benchmark properties."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import pytest

from evaluation.datasets import build_ledger_markdown, build_chunking_items, TABLE_COLUMNS
from evaluation.retrievers import NaiveChunker, header_pinned_chunks

ENGLISH_STOPWORDS = {
    "about", "above", "after", "again", "against", "all", "also", "among", "and", "another",
    "any", "because", "been", "before", "being", "between", "both", "came", "come", "could",
    "each", "even", "from", "further", "give", "given", "have", "having", "here", "into",
    "just", "like", "made", "make", "many", "more", "most", "much", "must", "never", "only",
    "other", "over", "same", "should", "some", "such", "than", "that", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "under", "upon", "very",
    "well", "were", "what", "when", "where", "which", "while", "with", "would", "your",
}


def stem(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def get_content_tokens(text: str) -> set:
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    return {stem(w) for w in words if w not in ENGLISH_STOPWORDS}


def compute_lexical_overlap(query: str, gold_chunk_content: str) -> float:
    q_tokens = get_content_tokens(query)
    if not q_tokens:
        return 0.0
    c_tokens = get_content_tokens(gold_chunk_content)
    return len(q_tokens & c_tokens) / len(q_tokens)


def token_jaccard(q1: str, q2: str) -> float:
    t1 = set(re.findall(r"\w+", q1.lower()))
    t2 = set(re.findall(r"\w+", q2.lower()))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def test_ledger_dataset_valid():
    md, rows = build_ledger_markdown(60, seed=7)
    naive_chunks = NaiveChunker().chunk(md)
    hp_chunks = header_pinned_chunks(md)
    items = build_chunking_items(rows, hp_chunks)

    assert len(items) == 30, f"Expected 30 ledger queries, got {len(items)}"
    assert len(set(it.query_id for it in items)) == 30

    # Test presence of gold in both chunkings
    naive_items = build_chunking_items(rows, naive_chunks)
    for n_item, hp_item in zip(naive_items, items):
        assert len(n_item.golden_chunk_ids) == 1
        assert len(hp_item.golden_chunk_ids) == 1
        nc = next(c for c in naive_chunks if c.chunk_id == n_item.golden_chunk_ids[0])
        hpc = next(c for c in hp_chunks if c.chunk_id == hp_item.golden_chunk_ids[0])
        assert hp_item.ground_truth_answer in nc.content
        assert hp_item.ground_truth_answer in hpc.content


def test_nvidia_eval_dataset_valid():
    chunks_path = Path("chunks/NVIDIA_CFO_chunks.json")
    if not chunks_path.exists():
        pytest.skip("chunks/NVIDIA_CFO_chunks.json not present")
    chunks = {c["chunk_id"]: c["content"] for c in json.loads(chunks_path.read_text(encoding="utf-8"))}

    eval_path = Path("benchmarks/nvidia_cfo_eval.json")
    items = json.loads(eval_path.read_text(encoding="utf-8"))

    assert len(items) == 34, f"Expected 34 items, got {len(items)}"
    counts = Counter(it["category"] for it in items)
    assert counts["exact_keyword_match"] == 8
    assert counts["semantic_intent"] == 8
    assert counts["complex_multi_hop"] == 8
    assert counts["out_of_domain_attack"] == 10

    # Semantic overlap
    for it in items:
        if it["category"] == "semantic_intent":
            gold_chunk = chunks[it["golden_chunk_ids"][0]]
            ov = compute_lexical_overlap(it["query"], gold_chunk)
            assert ov <= 0.40, f"Query {it['query_id']} overlap {ov:.3f} > 0.40"

    # Multi-hop chunk count
    for it in items:
        if it["category"] == "complex_multi_hop":
            assert len(it["golden_chunk_ids"]) >= 2

    from entities import extract_entities
    absent_ids = [
        e for it in items
        if it["category"] == "out_of_domain_attack"
        for e in extract_entities(it["query"])
        ]
    assert len(absent_ids) >= 5
    for aid in absent_ids:
        for cid, content in chunks.items():
            assert aid.lower() not in content.lower(), f"Absent ID {aid} in {cid}"

    # Near-domain false premise terms
    false_terms = ["cancellation", "Blackwell Ultra", "Qualcomm"]
    for term in false_terms:
        for cid, content in chunks.items():
            assert term.lower() not in content.lower(), f"False term {term} in {cid}"

    # Special pattern check
    patterns = [r"\b4th quarter\b", r"\b10-K\b", r"\bQ4-FY25\b"]
    matched = sum(1 for it in items if not it["expected_refusal"] and any(re.search(p, it["query"], re.I) for p in patterns))
    assert matched >= 3


def test_calibration_dataset_valid():
    cal_path = Path("benchmarks/nvidia_cfo_calibration.json")
    cal_items = json.loads(cal_path.read_text(encoding="utf-8"))
    eval_path = Path("benchmarks/nvidia_cfo_eval.json")
    eval_items = json.loads(eval_path.read_text(encoding="utf-8"))

    assert len(cal_items) == 20
    assert sum(1 for c in cal_items if c["in_domain"]) == 10
    assert sum(1 for c in cal_items if not c["in_domain"]) == 10

    eval_queries = [it["query"] for it in eval_items]
    for c in cal_items:
        cq = c["query"]
        for eq in eval_queries:
            assert cq.strip().lower() != eq.strip().lower(), f"Identical query in cal and eval: {cq}"
            j = token_jaccard(cq, eq)
            assert j < 0.60, f"Token Jaccard {j:.3f} >= 0.60 between '{cq}' and '{eq}'"


def test_dataset_hashes_match():
    hash_file = Path("benchmarks/DATASET_HASHES.txt")
    assert hash_file.exists(), "DATASET_HASHES.txt does not exist"
    for line in hash_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected_hash, rel_path = line.split(None, 1)    
        actual_hash = hashlib.sha256(Path(rel_path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}: {actual_hash} != {expected_hash}"

