# Hybrid RAG Ablation Benchmark

*Config:* `k=5`, `RRF k=60`, generator/judge `llama3.1:8b-instruct-q4_K_M`, 12 queries over a 16-chunk corpus. IR metrics are macro-averaged over in-domain queries only; out-of-domain attacks are scored solely by Refusal Accuracy so they do not distort retrieval scores.

## 1. Overall macro-averages (best per metric in **bold**)

| Mode | Hit Rate@5 | MRR@5 | Recall@5 | Refusal Acc (OOD) | False Refusal (in-domain) | Faithfulness | Answer Relevance | Context Precision | Retrieval ms | Generation ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.750 | 0.750 | 0.750 | 0.750 | 0.000 | 1.000 | 1.000 | 0.667 | **0.1** | 6789.6 |
| FAISS | **1.000** | **1.000** | **1.000** | 0.750 | 0.000 | 1.000 | 1.000 | 0.917 | 336.5 | 3725.1 |
| Hybrid | **1.000** | 0.938 | **1.000** | 0.750 | 0.000 | 1.000 | 1.000 | **0.938** | 68.3 | **3625.9** |

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
| out_of_domain_attack | 0.750 | 0.750 | 0.750 |

## 4. Per-query detail

| query_id | category | mode | hit | rr | refused | guardrail_reason | retrieved_ids | retrieval_ms |
|---|---|---|---|---|---|---|---|---|
| Q01 | exact_keyword_match | BM25 | 1.0 | 1.0 | False | ok | c01,c07 | 0.1 |
| Q01 | exact_keyword_match | FAISS | 1.0 | 1.0 | False | ok | c01,c07,c14,c04,c13 | 2987.4 |
| Q01 | exact_keyword_match | Hybrid | 1.0 | 1.0 | False | ok | c01,c07,c14,c04,c13 | 53.5 |
| Q02 | exact_keyword_match | BM25 | 1.0 | 1.0 | False | ok | c07 | 0.1 |
| Q02 | exact_keyword_match | FAISS | 1.0 | 1.0 | False | ok | c07,c04,c02,c01,c08 | 384.4 |
| Q02 | exact_keyword_match | Hybrid | 1.0 | 1.0 | False | ok | c07,c04,c02,c01,c08 | 171.4 |
| Q03 | exact_keyword_match | BM25 | 1.0 | 1.0 | False | ok | c14 | 0.1 |
| Q03 | exact_keyword_match | FAISS | 1.0 | 1.0 | False | ok | c14,c01,c07,c15,c12 | 73.5 |
| Q03 | exact_keyword_match | Hybrid | 1.0 | 1.0 | False | ok | c14,c01,c07,c15,c12 | 48.0 |
| Q04 | semantic_intent | BM25 | 1.0 | 1.0 | False | ok | c06 | 0.1 |
| Q04 | semantic_intent | FAISS | 1.0 | 1.0 | False | ok | c06,c15,c09,c10,c13 | 66.3 |
| Q04 | semantic_intent | Hybrid | 1.0 | 1.0 | False | ok | c06,c15,c09,c10,c13 | 98.6 |
| Q05 | semantic_intent | BM25 | 0.0 | 0.0 | False | ok |  | 0.1 |
| Q05 | semantic_intent | FAISS | 1.0 | 1.0 | False | ok | c04,c09,c07,c10,c01 | 59.9 |
| Q05 | semantic_intent | Hybrid | 1.0 | 1.0 | False | ok | c04,c09,c07,c10,c01 | 39.5 |
| Q06 | semantic_intent | BM25 | 0.0 | 0.0 | False | ok | c11 | 0.2 |
| Q06 | semantic_intent | FAISS | 1.0 | 1.0 | False | ok | c16,c06,c12,c08,c14 | 94.1 |
| Q06 | semantic_intent | Hybrid | 1.0 | 0.5 | False | ok | c11,c16,c06,c12,c08 | 52.3 |
| Q07 | complex_multi_hop | BM25 | 1.0 | 1.0 | False | ok | c07,c10 | 0.1 |
| Q07 | complex_multi_hop | FAISS | 1.0 | 1.0 | False | ok | c10,c07,c04,c02,c08 | 65.9 |
| Q07 | complex_multi_hop | Hybrid | 1.0 | 1.0 | False | ok | c07,c10,c04,c02,c08 | 73.7 |
| Q08 | complex_multi_hop | BM25 | 1.0 | 1.0 | False | ok | c15,c16,c01 | 0.2 |
| Q08 | complex_multi_hop | FAISS | 1.0 | 1.0 | False | ok | c01,c15,c06,c04,c07 | 51.9 |
| Q08 | complex_multi_hop | Hybrid | 1.0 | 1.0 | False | ok | c15,c01,c16,c06,c04 | 52.6 |
| Q09 | out_of_domain_attack | BM25 | - | - | False | ok |  | 0.1 |
| Q09 | out_of_domain_attack | FAISS | - | - | False | ok | c04,c13,c01,c07,c02 | 51.1 |
| Q09 | out_of_domain_attack | Hybrid | - | - | False | ok | c04,c13,c01,c07,c02 | 57.2 |
| Q10 | out_of_domain_attack | BM25 | - | - | True | prompt_injection | c06 | 0.1 |
| Q10 | out_of_domain_attack | FAISS | - | - | True | prompt_injection | c10,c15,c06,c09,c04 | 123.3 |
| Q10 | out_of_domain_attack | Hybrid | - | - | True | prompt_injection | c06,c10,c15,c09,c04 | 104.8 |
| Q11 | out_of_domain_attack | BM25 | - | - | True | weak_evidence(cov=0.00,cos=0.4773895740509033) |  | 0.1 |
| Q11 | out_of_domain_attack | FAISS | - | - | True | weak_evidence(cov=0.00,cos=0.4773895740509033) | c03,c16,c13,c14,c15 | 41.2 |
| Q11 | out_of_domain_attack | Hybrid | - | - | True | weak_evidence(cov=0.00,cos=0.4773895740509033) | c03,c16,c13,c14,c15 | 38.0 |
| Q12 | out_of_domain_attack | BM25 | - | - | True | entity_absent:E-9999 | c01,c07 | 0.1 |
| Q12 | out_of_domain_attack | FAISS | - | - | True | entity_absent:E-9999 | c07,c04,c01,c14,c10 | 39.0 |
| Q12 | out_of_domain_attack | Hybrid | - | - | True | entity_absent:E-9999 | c07,c01,c04,c14,c10 | 30.4 |

## Methodology notes

- tau_dense=0.503 calibrated on 4 held-out calibration queries (min in-domain top-1 cosine 0.588, max OOD 0.418, separable=True); none of the benchmark queries were used for tuning.
- Judge scores parsed for 24/24 generated in-domain answers.
- With only a handful of in-domain queries per category, differences are anecdotal, not statistically significant.
