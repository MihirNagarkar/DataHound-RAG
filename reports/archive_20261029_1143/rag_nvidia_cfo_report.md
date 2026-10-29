# Hybrid RAG Ablation Benchmark

*Config:* `k=5`, `RRF k=60`, generator/judge `llama3.1:8b-instruct-q4_K_M`, 12 queries over a 27-chunk corpus. IR metrics are macro-averaged over in-domain queries only; out-of-domain attacks are scored solely by Refusal Accuracy so they do not distort retrieval scores.

## 1. Overall macro-averages (best per metric in **bold**)

| Mode | Hit Rate@5 | MRR@5 | Recall@5 | Refusal Acc (OOD) | False Refusal (in-domain) | Faithfulness | Answer Relevance | Context Precision | Retrieval ms | Generation ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.625 | 0.562 | 0.625 | 0.750 | 0.000 | 1.000 | 0.625 | 0.683 | **0.5** | 6710.7 |
| FAISS | **0.875** | **0.812** | **0.875** | 0.750 | 0.000 | 1.000 | **0.750** | **0.793** | 327.4 | 3743.2 |
| Hybrid | **0.875** | 0.692 | **0.875** | 0.750 | 0.000 | 1.000 | **0.750** | 0.685 | 77.2 | **3535.4** |

## 2a. Category matrix - Hit Rate@5 (best mode per row in **bold**)

| Category | BM25 | FAISS | Hybrid |
|---|---:|---:|---:|
| exact_keyword_match | 1.000 | 1.000 | 1.000 |
| semantic_intent | 0.000 | **0.667** | **0.667** |
| complex_multi_hop | 1.000 | 1.000 | 1.000 |

## 2b. Category matrix - MRR@5 (best mode per row in **bold**)

| Category | BM25 | FAISS | Hybrid |
|---|---:|---:|---:|
| exact_keyword_match | **1.000** | 0.833 | **1.000** |
| semantic_intent | 0.000 | **0.667** | 0.178 |
| complex_multi_hop | 0.750 | **1.000** | **1.000** |

## 2c. Category matrix - Recall@5 (best mode per row in **bold**)

| Category | BM25 | FAISS | Hybrid |
|---|---:|---:|---:|
| exact_keyword_match | 1.000 | 1.000 | 1.000 |
| semantic_intent | 0.000 | **0.667** | **0.667** |
| complex_multi_hop | 1.000 | 1.000 | 1.000 |

## 3. Category matrix - Refusal Accuracy (out-of-domain)

| Category | BM25 | FAISS | Hybrid |
|---|---:|---:|---:|
| out_of_domain_attack | 0.750 | 0.750 | 0.750 |

## 4. Per-query detail

| query_id | category | mode | hit | rr | refused | guardrail_reason | retrieved_ids | retrieval_ms |
|---|---|---|---|---|---|---|---|---|
| P01 | exact_keyword_match | BM25 | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00009,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00011 | 0.4 |
| P01 | exact_keyword_match | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00003 | 3162.1 |
| P01 | exact_keyword_match | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00009 | 86.9 |
| P02 | exact_keyword_match | BM25 | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00007 | 0.5 |
| P02 | exact_keyword_match | FAISS | 1.0 | 0.5 | False | ok | NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00004 | 49.9 |
| P02 | exact_keyword_match | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00005 | 112.6 |
| P03 | exact_keyword_match | BM25 | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00007 | 0.6 |
| P03 | exact_keyword_match | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00027 | 61.9 |
| P03 | exact_keyword_match | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00023 | 61.5 |
| P04 | semantic_intent | BM25 | 0.0 | 0.0 | False | ok | NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00018,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00011 | 0.4 |
| P04 | semantic_intent | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00020,NVIDIA_CFO_chunk_00023 | 139.8 |
| P04 | semantic_intent | Hybrid | 1.0 | 0.3333333333333333 | False | ok | NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00010 | 124.4 |
| P05 | semantic_intent | BM25 | 0.0 | 0.0 | False | ok | NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00007 | 0.4 |
| P05 | semantic_intent | FAISS | 0.0 | 0.0 | False | ok | NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00007 | 73.4 |
| P05 | semantic_intent | Hybrid | 0.0 | 0.0 | False | ok | NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00007 | 86.4 |
| P06 | semantic_intent | BM25 | 0.0 | 0.0 | False | ok | NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00026,NVIDIA_CFO_chunk_00020 | 0.7 |
| P06 | semantic_intent | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00009,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00023 | 100.9 |
| P06 | semantic_intent | Hybrid | 1.0 | 0.2 | False | ok | NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00009 | 134.3 |
| P07 | complex_multi_hop | BM25 | 1.0 | 0.5 | False | ok | NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00007 | 0.5 |
| P07 | complex_multi_hop | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00005 | 73.5 |
| P07 | complex_multi_hop | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00007 | 56.8 |
| P08 | complex_multi_hop | BM25 | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00013,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00027,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00009 | 0.7 |
| P08 | complex_multi_hop | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00013,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00009 | 52.6 |
| P08 | complex_multi_hop | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00013,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00009,NVIDIA_CFO_chunk_00010 | 72.2 |
| P09 | out_of_domain_attack | BM25 | - | - | True | weak_evidence(cov=0.00,cos=0.5623) |  | 0.3 |
| P09 | out_of_domain_attack | FAISS | - | - | True | weak_evidence(cov=0.00,cos=0.5623) | NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00006,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00012 | 50.7 |
| P09 | out_of_domain_attack | Hybrid | - | - | True | weak_evidence(cov=0.00,cos=0.5623) | NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00006,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00012 | 36.5 |
| P10 | out_of_domain_attack | BM25 | - | - | True | prompt_injection | NVIDIA_CFO_chunk_00015 | 0.5 |
| P10 | out_of_domain_attack | FAISS | - | - | True | prompt_injection | NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00006,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00001 | 51.2 |
| P10 | out_of_domain_attack | Hybrid | - | - | True | prompt_injection | NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00006,NVIDIA_CFO_chunk_00001 | 49.7 |
| P11 | out_of_domain_attack | BM25 | - | - | True | empty_context | NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00026,NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00004 | 0.6 |
| P11 | out_of_domain_attack | FAISS | - | - | True | empty_context | NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00020 | 52.5 |
| P11 | out_of_domain_attack | Hybrid | - | - | True | empty_context |  | 56.0 |
| P12 | out_of_domain_attack | BM25 | - | - | False | ok | NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00016 | 0.3 |
| P12 | out_of_domain_attack | FAISS | - | - | False | ok | NVIDIA_CFO_chunk_00006,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00011 | 59.8 |
| P12 | out_of_domain_attack | Hybrid | - | - | False | ok | NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00006,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00004 | 48.7 |

## Methodology notes

- tau_dense=0.571 calibrated on 4 held-out calibration queries (min in-domain top-1 cosine 0.644, max OOD 0.499, separable=True); none of the benchmark queries were used for tuning.
- Judge scores parsed for 24/24 generated in-domain answers.
- With only a handful of in-domain queries per category, differences are anecdotal, not statistically significant.
