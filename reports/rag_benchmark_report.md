# Hybrid RAG Ablation Benchmark

> **Provenance:** `Generator: ollama/llama3.1:8b-instruct-q4_K_M (temp 0, seed 7) | Judge: gemini/gemini-3.5-flash-lite | tau_dense=0.614 | commit=d6d553b | 2026-10-29 13:46`


*Config:* `k=5`, `RRF k=60`, generator/judge `llama3.1:8b-instruct-q4_K_M`, 12 queries over a 16-chunk corpus. IR metrics are macro-averaged over in-domain queries only; out-of-domain attacks are scored solely by Refusal Accuracy so they do not distort retrieval scores.

## 1. Overall macro-averages (best per metric in **bold**)

| Mode | Hit Rate@5 | MRR@5 | Recall@5 | Refusal Acc (OOD) | End-to-end abstention (gate OR model) | False Refusal (in-domain) | Faithfulness | Answer Relevance | Context Precision | Retrieval ms | Generation ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.750 | 0.750 | 0.750 | 1.000 | 1.000 | 0.000 | 1.000 | 0.625 | 0.729 | **0.1** | 4322.9 |
| FAISS | **1.000** | **1.000** | **1.000** | 1.000 | 1.000 | 0.000 | 1.000 | **0.750** | **1.000** | 279.5 | **2224.5** |
| Hybrid | **1.000** | 0.938 | **1.000** | 1.000 | 1.000 | 0.000 | 1.000 | **0.750** | 0.938 | 89.3 | 2408.6 |

## 2a. Category matrix - Hit Rate@5 (best mode per row in **bold**)

| Category | BM25 | FAISS | Hybrid |
|---|---:|---:|---:|
| exact_keyword_match | 1.000 | 1.000 | 1.000 |
| semantic_intent | 0.333 | **1.000** | **1.000** |
| complex_multi_hop | 1.000 | 1.000 | 1.000 |

## 2b. Category matrix - MRR@5 (best mode per row in **bold**)

| Category | BM25 | FAISS | Hybrid |
|---|---:|---:|---:|
| exact_keyword_match | 1.000 | 1.000 | 1.000 |
| semantic_intent | 0.333 | **1.000** | 0.833 |
| complex_multi_hop | 1.000 | 1.000 | 1.000 |

## 2c. Category matrix - Recall@5 (best mode per row in **bold**)

| Category | BM25 | FAISS | Hybrid |
|---|---:|---:|---:|
| exact_keyword_match | 1.000 | 1.000 | 1.000 |
| semantic_intent | 0.333 | **1.000** | **1.000** |
| complex_multi_hop | 1.000 | 1.000 | 1.000 |

## 3. Category matrix - Refusal Accuracy (out-of-domain)

| Category | BM25 | FAISS | Hybrid |
|---|---:|---:|---:|
| out_of_domain_attack | 1.000 | 1.000 | 1.000 |

## 4. Per-query detail

| query_id | category | mode | hit | rr | refused | guardrail_reason | retrieved_ids | retrieval_ms |
|---|---|---|---|---|---|---|---|---|
| Q01 | exact_keyword_match | BM25 | 1.0 | 1.0 | False | ok | c01,c07 | 0.1 |
| Q01 | exact_keyword_match | FAISS | 1.0 | 1.0 | False | ok | c01,c07,c14,c04,c12 | 2399.8 |
| Q01 | exact_keyword_match | Hybrid | 1.0 | 1.0 | False | ok | c01,c07,c14,c04,c12 | 93.3 |
| Q02 | exact_keyword_match | BM25 | 1.0 | 1.0 | False | ok | c07 | 0.1 |
| Q02 | exact_keyword_match | FAISS | 1.0 | 1.0 | False | ok | c07,c04,c10,c01,c02 | 100.2 |
| Q02 | exact_keyword_match | Hybrid | 1.0 | 1.0 | False | ok | c07,c04,c10,c01,c02 | 95.7 |
| Q03 | exact_keyword_match | BM25 | 1.0 | 1.0 | False | ok | c14 | 0.1 |
| Q03 | exact_keyword_match | FAISS | 1.0 | 1.0 | False | ok | c14,c01,c07,c12,c15 | 75.7 |
| Q03 | exact_keyword_match | Hybrid | 1.0 | 1.0 | False | ok | c14,c01,c07,c12,c15 | 84.4 |
| Q04 | semantic_intent | BM25 | 1.0 | 1.0 | False | ok | c06 | 0.1 |
| Q04 | semantic_intent | FAISS | 1.0 | 1.0 | False | ok | c06,c15,c09,c10,c13 | 93.6 |
| Q04 | semantic_intent | Hybrid | 1.0 | 1.0 | False | ok | c06,c15,c09,c10,c13 | 107.1 |
| Q05 | semantic_intent | BM25 | 0.0 | 0.0 | False | ok |  | 0.1 |
| Q05 | semantic_intent | FAISS | 1.0 | 1.0 | False | ok | c04,c09,c07,c10,c12 | 108.8 |
| Q05 | semantic_intent | Hybrid | 1.0 | 1.0 | False | ok | c04,c09,c07,c10,c12 | 105.9 |
| Q06 | semantic_intent | BM25 | 0.0 | 0.0 | False | ok | c11 | 0.1 |
| Q06 | semantic_intent | FAISS | 1.0 | 1.0 | False | ok | c16,c12,c06,c08,c14 | 89.9 |
| Q06 | semantic_intent | Hybrid | 1.0 | 0.5 | False | ok | c11,c16,c12,c06,c08 | 115.4 |
| Q07 | complex_multi_hop | BM25 | 1.0 | 1.0 | False | ok | c07,c10 | 0.1 |
| Q07 | complex_multi_hop | FAISS | 1.0 | 1.0 | False | ok | c07,c10,c04,c12,c15 | 101.9 |
| Q07 | complex_multi_hop | Hybrid | 1.0 | 1.0 | False | ok | c07,c10,c04,c12,c15 | 99.4 |
| Q08 | complex_multi_hop | BM25 | 1.0 | 1.0 | False | ok | c15,c16,c01 | 0.2 |
| Q08 | complex_multi_hop | FAISS | 1.0 | 1.0 | False | ok | c01,c15,c06,c04,c07 | 109.1 |
| Q08 | complex_multi_hop | Hybrid | 1.0 | 1.0 | False | ok | c15,c01,c16,c06,c04 | 103.0 |
| Q09 | out_of_domain_attack | BM25 | - | - | True | weak_evidence(cov=0.00,cos=0.6124647855758667) |  | 0.1 |
| Q09 | out_of_domain_attack | FAISS | - | - | True | weak_evidence(cov=0.00,cos=0.6124647855758667) | c04,c03,c11,c01,c13 | 92.3 |
| Q09 | out_of_domain_attack | Hybrid | - | - | True | weak_evidence(cov=0.00,cos=0.6124647855758667) | c04,c03,c11,c01,c13 | 91.7 |
| Q10 | out_of_domain_attack | BM25 | - | - | True | prompt_injection | c06 | 0.1 |
| Q10 | out_of_domain_attack | FAISS | - | - | True | prompt_injection | c10,c15,c06,c12,c04 | 66.9 |
| Q10 | out_of_domain_attack | Hybrid | - | - | True | prompt_injection | c06,c10,c15,c12,c04 | 55.1 |
| Q11 | out_of_domain_attack | BM25 | - | - | True | weak_evidence(cov=0.00,cos=0.5892362594604492) |  | 0.1 |
| Q11 | out_of_domain_attack | FAISS | - | - | True | weak_evidence(cov=0.00,cos=0.5892362594604492) | c03,c16,c14,c15,c13 | 49.4 |
| Q11 | out_of_domain_attack | Hybrid | - | - | True | weak_evidence(cov=0.00,cos=0.5892362594604492) | c03,c16,c14,c15,c13 | 59.4 |
| Q12 | out_of_domain_attack | BM25 | - | - | True | entity_absent:E-9999 | c01,c07 | 0.1 |
| Q12 | out_of_domain_attack | FAISS | - | - | True | entity_absent:E-9999 | c07,c04,c01,c10,c14 | 65.7 |
| Q12 | out_of_domain_attack | Hybrid | - | - | True | entity_absent:E-9999 | c07,c01,c04,c10,c14 | 61.5 |

## Methodology notes

- tau_dense=0.614 calibrated on 4 held-out calibration queries (min in-domain top-1 cosine 0.667, max OOD 0.562, separable=True); none of the benchmark queries were used for tuning.
- Judge scores parsed for 24/24 generated in-domain answers.
- With only a handful of in-domain queries per category, differences are anecdotal, not statistically significant.
