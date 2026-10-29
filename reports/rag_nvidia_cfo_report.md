# Hybrid RAG Ablation Benchmark

> **Provenance:** `Generator: ollama/llama3.1:8b-instruct-q4_K_M (temp 0, seed 7) | Judge: gemini/gemini-3.5-flash-lite | tau_dense=0.633 | commit=d6d553b | 2026-10-29 13:33`


*Config:* `k=5`, `RRF k=60`, generator/judge `llama3.1:8b-instruct-q4_K_M`, 34 queries over a 27-chunk corpus. IR metrics are macro-averaged over in-domain queries only; out-of-domain attacks are scored solely by Refusal Accuracy so they do not distort retrieval scores.

## 1. Overall macro-averages (best per metric in **bold**)

| Mode | Hit Rate@5 | MRR@5 | Recall@5 | Refusal Acc (OOD) | End-to-end abstention (gate OR model) | False Refusal (in-domain) | Faithfulness | Answer Relevance | Context Precision | Retrieval ms | Generation ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.792 | 0.585 | 0.750 | 0.700 | 1.000 | 0.042 | 0.957 | 0.696 | 0.636 | **0.7** | 3919.5 |
| FAISS | 0.833 | **0.764** | 0.812 | 0.700 | 1.000 | 0.042 | **1.000** | **0.783** | **0.766** | 194.9 | 3741.1 |
| Hybrid | **0.917** | 0.701 | **0.875** | 0.700 | 1.000 | 0.042 | **1.000** | **0.783** | 0.721 | 112.0 | **3108.7** |

## 2a. Category matrix - Hit Rate@5 (best mode per row in **bold**)

| Category | BM25 | FAISS | Hybrid |
|---|---:|---:|---:|
| exact_keyword_match | 1.000 | 1.000 | 1.000 |
| semantic_intent | 0.375 | 0.625 | **0.750** |
| complex_multi_hop | **1.000** | 0.875 | **1.000** |

## 2b. Category matrix - MRR@5 (best mode per row in **bold**)

| Category | BM25 | FAISS | Hybrid |
|---|---:|---:|---:|
| exact_keyword_match | 0.844 | **0.938** | 0.875 |
| semantic_intent | 0.198 | **0.562** | 0.321 |
| complex_multi_hop | 0.713 | 0.792 | **0.906** |

## 2c. Category matrix - Recall@5 (best mode per row in **bold**)

| Category | BM25 | FAISS | Hybrid |
|---|---:|---:|---:|
| exact_keyword_match | 1.000 | 1.000 | 1.000 |
| semantic_intent | 0.375 | 0.625 | **0.750** |
| complex_multi_hop | **0.875** | 0.812 | **0.875** |

## 3. Category matrix - Refusal Accuracy (out-of-domain)

| Category | BM25 | FAISS | Hybrid |
|---|---:|---:|---:|
| out_of_domain_attack | 0.700 | 0.700 | 0.700 |

## 4. Per-query detail

| query_id | category | mode | hit | rr | refused | guardrail_reason | retrieved_ids | retrieval_ms |
|---|---|---|---|---|---|---|---|---|
| P01 | exact_keyword_match | BM25 | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00009,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00011 | 0.4 |
| P01 | exact_keyword_match | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00003 | 2525.0 |
| P01 | exact_keyword_match | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00009 | 104.2 |
| P02 | exact_keyword_match | BM25 | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00007 | 0.4 |
| P02 | exact_keyword_match | FAISS | 1.0 | 0.5 | False | ok | NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00004 | 132.7 |
| P02 | exact_keyword_match | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00005 | 98.9 |
| P03 | exact_keyword_match | BM25 | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00007 | 0.5 |
| P03 | exact_keyword_match | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00027 | 126.9 |
| P03 | exact_keyword_match | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00023 | 117.5 |
| P04 | semantic_intent | BM25 | 0.0 | 0.0 | False | ok | NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00018,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00011 | 0.6 |
| P04 | semantic_intent | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00020,NVIDIA_CFO_chunk_00023 | 532.9 |
| P04 | semantic_intent | Hybrid | 1.0 | 0.3333333333333333 | False | ok | NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00010 | 104.2 |
| P05 | semantic_intent | BM25 | 0.0 | 0.0 | False | ok | NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00007 | 0.6 |
| P05 | semantic_intent | FAISS | 0.0 | 0.0 | False | ok | NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00007 | 109.4 |
| P05 | semantic_intent | Hybrid | 0.0 | 0.0 | False | ok | NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00007 | 110.7 |
| P06 | semantic_intent | BM25 | 0.0 | 0.0 | False | ok | NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00026,NVIDIA_CFO_chunk_00020 | 0.8 |
| P06 | semantic_intent | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00009,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00023 | 393.5 |
| P06 | semantic_intent | Hybrid | 1.0 | 0.2 | False | ok | NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00009 | 244.4 |
| P07 | complex_multi_hop | BM25 | 1.0 | 0.5 | False | ok | NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00007 | 0.9 |
| P07 | complex_multi_hop | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00005 | 134.9 |
| P07 | complex_multi_hop | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00007 | 124.3 |
| P08 | complex_multi_hop | BM25 | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00013,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00027,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00009 | 1.3 |
| P08 | complex_multi_hop | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00013,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00009 | 136.1 |
| P08 | complex_multi_hop | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00013,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00009,NVIDIA_CFO_chunk_00010 | 105.8 |
| P09 | out_of_domain_attack | BM25 | - | - | True | weak_evidence(cov=0.00,cos=0.5623) |  | 0.8 |
| P09 | out_of_domain_attack | FAISS | - | - | True | weak_evidence(cov=0.00,cos=0.5623) | NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00006,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00012 | 101.4 |
| P09 | out_of_domain_attack | Hybrid | - | - | True | weak_evidence(cov=0.00,cos=0.5623) | NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00006,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00012 | 98.8 |
| P10 | out_of_domain_attack | BM25 | - | - | True | prompt_injection | NVIDIA_CFO_chunk_00015 | 0.7 |
| P10 | out_of_domain_attack | FAISS | - | - | True | prompt_injection | NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00006,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00001 | 97.6 |
| P10 | out_of_domain_attack | Hybrid | - | - | True | prompt_injection | NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00006,NVIDIA_CFO_chunk_00001 | 95.2 |
| P11 | out_of_domain_attack | BM25 | - | - | True | empty_context | NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00026,NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00004 | 1.2 |
| P11 | out_of_domain_attack | FAISS | - | - | True | empty_context | NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00020 | 95.9 |
| P11 | out_of_domain_attack | Hybrid | - | - | True | empty_context |  | 93.1 |
| P12 | out_of_domain_attack | BM25 | - | - | False | ok | NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00016 | 0.3 |
| P12 | out_of_domain_attack | FAISS | - | - | False | ok | NVIDIA_CFO_chunk_00006,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00011 | 66.7 |
| P12 | out_of_domain_attack | Hybrid | - | - | False | ok | NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00006,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00004 | 67.6 |
| P13 | exact_keyword_match | BM25 | 1.0 | 0.25 | False | ok | NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00020,NVIDIA_CFO_chunk_00009 | 0.9 |
| P13 | exact_keyword_match | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00020,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00002 | 99.4 |
| P13 | exact_keyword_match | Hybrid | 1.0 | 0.5 | False | ok | NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00020,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00027 | 124.5 |
| P14 | exact_keyword_match | BM25 | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00006 | 0.6 |
| P14 | exact_keyword_match | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00009,NVIDIA_CFO_chunk_00011 | 97.1 |
| P14 | exact_keyword_match | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00009,NVIDIA_CFO_chunk_00010 | 100.1 |
| P15 | exact_keyword_match | BM25 | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00010 | 0.6 |
| P15 | exact_keyword_match | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00012 | 97.7 |
| P15 | exact_keyword_match | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00015 | 97.9 |
| P16 | exact_keyword_match | BM25 | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00027,NVIDIA_CFO_chunk_00013,NVIDIA_CFO_chunk_00009,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00004 | 0.4 |
| P16 | exact_keyword_match | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00027,NVIDIA_CFO_chunk_00009,NVIDIA_CFO_chunk_00013,NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00020 | 144.8 |
| P16 | exact_keyword_match | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00027,NVIDIA_CFO_chunk_00013,NVIDIA_CFO_chunk_00009,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00020 | 116.4 |
| P17 | exact_keyword_match | BM25 | 1.0 | 0.5 | False | ok | NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00007 | 0.5 |
| P17 | exact_keyword_match | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00003 | 97.6 |
| P17 | exact_keyword_match | Hybrid | 1.0 | 0.5 | False | ok | NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00002 | 107.9 |
| P18 | semantic_intent | BM25 | 0.0 | 0.0 | False | ok | NVIDIA_CFO_chunk_00018,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00016 | 0.4 |
| P18 | semantic_intent | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00009,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00007 | 91.6 |
| P18 | semantic_intent | Hybrid | 1.0 | 0.3333333333333333 | False | ok | NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00018,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00016 | 123.3 |
| P19 | semantic_intent | BM25 | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00022,NVIDIA_CFO_chunk_00026,NVIDIA_CFO_chunk_00021 | 0.4 |
| P19 | semantic_intent | FAISS | 1.0 | 0.5 | False | ok | NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00026,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00027 | 107.6 |
| P19 | semantic_intent | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00026,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00016 | 104.9 |
| P20 | semantic_intent | BM25 | 1.0 | 0.3333333333333333 | False | ok | NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00026,NVIDIA_CFO_chunk_00015 | 0.8 |
| P20 | semantic_intent | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00026,NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00017,NVIDIA_CFO_chunk_00011 | 133.8 |
| P20 | semantic_intent | Hybrid | 1.0 | 0.5 | False | ok | NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00026,NVIDIA_CFO_chunk_00017,NVIDIA_CFO_chunk_00011 | 145.2 |
| P21 | semantic_intent | BM25 | 1.0 | 0.25 | False | ok | NVIDIA_CFO_chunk_00027,NVIDIA_CFO_chunk_00024,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00026 | 0.5 |
| P21 | semantic_intent | FAISS | 0.0 | 0.0 | False | ok | NVIDIA_CFO_chunk_00024,NVIDIA_CFO_chunk_00027,NVIDIA_CFO_chunk_00026,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00020 | 104.4 |
| P21 | semantic_intent | Hybrid | 1.0 | 0.2 | False | ok | NVIDIA_CFO_chunk_00027,NVIDIA_CFO_chunk_00024,NVIDIA_CFO_chunk_00026,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00025 | 97.6 |
| P22 | semantic_intent | BM25 | 0.0 | 0.0 | True | weak_evidence(cov=0.11,cos=0.6283) | NVIDIA_CFO_chunk_00016 | 0.4 |
| P22 | semantic_intent | FAISS | 0.0 | 0.0 | True | weak_evidence(cov=0.11,cos=0.6283) | NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00006,NVIDIA_CFO_chunk_00009 | 104.6 |
| P22 | semantic_intent | Hybrid | 0.0 | 0.0 | True | weak_evidence(cov=0.11,cos=0.6283) | NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00006 | 101.1 |
| P23 | complex_multi_hop | BM25 | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00009,NVIDIA_CFO_chunk_00011 | 0.7 |
| P23 | complex_multi_hop | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00006,NVIDIA_CFO_chunk_00009 | 139.1 |
| P23 | complex_multi_hop | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00009,NVIDIA_CFO_chunk_00008 | 111.7 |
| P24 | complex_multi_hop | BM25 | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00007 | 0.9 |
| P24 | complex_multi_hop | FAISS | 1.0 | 0.3333333333333333 | False | ok | NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00003 | 122.8 |
| P24 | complex_multi_hop | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00002 | 425.7 |
| P25 | complex_multi_hop | BM25 | 1.0 | 0.5 | False | ok | NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00003 | 1.2 |
| P25 | complex_multi_hop | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00002,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00012 | 100.6 |
| P25 | complex_multi_hop | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00003,NVIDIA_CFO_chunk_00005,NVIDIA_CFO_chunk_00002 | 108.2 |
| P26 | complex_multi_hop | BM25 | 1.0 | 0.2 | False | ok | NVIDIA_CFO_chunk_00027,NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00020,NVIDIA_CFO_chunk_00009,NVIDIA_CFO_chunk_00023 | 0.8 |
| P26 | complex_multi_hop | FAISS | 0.0 | 0.0 | False | ok | NVIDIA_CFO_chunk_00024,NVIDIA_CFO_chunk_00027,NVIDIA_CFO_chunk_00020,NVIDIA_CFO_chunk_00026,NVIDIA_CFO_chunk_00022 | 101.9 |
| P26 | complex_multi_hop | Hybrid | 1.0 | 0.25 | False | ok | NVIDIA_CFO_chunk_00027,NVIDIA_CFO_chunk_00020,NVIDIA_CFO_chunk_00024,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00011 | 105.0 |
| P27 | complex_multi_hop | BM25 | 1.0 | 0.5 | False | ok | NVIDIA_CFO_chunk_00013,NVIDIA_CFO_chunk_00027,NVIDIA_CFO_chunk_00004,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00021 | 1.3 |
| P27 | complex_multi_hop | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00027,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00020,NVIDIA_CFO_chunk_00026,NVIDIA_CFO_chunk_00009 | 101.2 |
| P27 | complex_multi_hop | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00027,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00020,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00009 | 101.5 |
| P28 | complex_multi_hop | BM25 | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00020,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00027 | 0.8 |
| P28 | complex_multi_hop | FAISS | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00020,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00027 | 107.8 |
| P28 | complex_multi_hop | Hybrid | 1.0 | 1.0 | False | ok | NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00020,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00027 | 94.0 |
| P29 | out_of_domain_attack | BM25 | - | - | True | empty_context | NVIDIA_CFO_chunk_00001,NVIDIA_CFO_chunk_00011 | 1.3 |
| P29 | out_of_domain_attack | FAISS | - | - | True | empty_context | NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00016 | 135.4 |
| P29 | out_of_domain_attack | Hybrid | - | - | True | empty_context |  | 97.3 |
| P30 | out_of_domain_attack | BM25 | - | - | True | empty_context | NVIDIA_CFO_chunk_00022,NVIDIA_CFO_chunk_00016 | 0.5 |
| P30 | out_of_domain_attack | FAISS | - | - | True | empty_context | NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00027,NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00020 | 66.6 |
| P30 | out_of_domain_attack | Hybrid | - | - | True | empty_context |  | 58.1 |
| P31 | out_of_domain_attack | BM25 | - | - | True | empty_context | NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00026,NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00016 | 0.5 |
| P31 | out_of_domain_attack | FAISS | - | - | True | empty_context | NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00027,NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00023 | 58.8 |
| P31 | out_of_domain_attack | Hybrid | - | - | True | empty_context |  | 61.6 |
| P32 | out_of_domain_attack | BM25 | - | - | True | empty_context | NVIDIA_CFO_chunk_00024,NVIDIA_CFO_chunk_00019,NVIDIA_CFO_chunk_00022,NVIDIA_CFO_chunk_00026,NVIDIA_CFO_chunk_00027 | 0.7 |
| P32 | out_of_domain_attack | FAISS | - | - | True | empty_context | NVIDIA_CFO_chunk_00023,NVIDIA_CFO_chunk_00025,NVIDIA_CFO_chunk_00021,NVIDIA_CFO_chunk_00020,NVIDIA_CFO_chunk_00027 | 66.0 |
| P32 | out_of_domain_attack | Hybrid | - | - | True | empty_context |  | 74.7 |
| P33 | out_of_domain_attack | BM25 | - | - | False | ok | NVIDIA_CFO_chunk_00018,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00015 | 0.6 |
| P33 | out_of_domain_attack | FAISS | - | - | False | ok | NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00006,NVIDIA_CFO_chunk_00008 | 44.8 |
| P33 | out_of_domain_attack | Hybrid | - | - | False | ok | NVIDIA_CFO_chunk_00016,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00008 | 44.7 |
| P34 | out_of_domain_attack | BM25 | - | - | False | ok | NVIDIA_CFO_chunk_00012,NVIDIA_CFO_chunk_00011,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00010 | 0.5 |
| P34 | out_of_domain_attack | FAISS | - | - | False | ok | NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00016 | 49.9 |
| P34 | out_of_domain_attack | Hybrid | - | - | False | ok | NVIDIA_CFO_chunk_00007,NVIDIA_CFO_chunk_00008,NVIDIA_CFO_chunk_00010,NVIDIA_CFO_chunk_00015,NVIDIA_CFO_chunk_00016 | 43.4 |

## Methodology notes

- tau_dense=0.633 calibrated on 20 held-out calibration queries (min in-domain top-1 cosine 0.640, max OOD 0.625, separable=True); none of the benchmark queries were used for tuning.
- Judge scores parsed for 69/69 generated in-domain answers.
- With only a handful of in-domain queries per category, differences are anecdotal, not statistically significant.
