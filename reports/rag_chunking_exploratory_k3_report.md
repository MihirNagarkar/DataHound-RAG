# Chunking Ablation: Naive vs Header-Pinned Table Chunks

> **Provenance:** `Generator: ollama/llama3.1:8b-instruct-q4_K_M (temp 0, seed 7) | Judge: ollama/llama3.1:8b-instruct-q4_K_M | context_k=3 | commit=d6d553b | 2026-10-29 12:48`

*Setup:* 60-row transaction ledger, 30 single-row lookup questions, chunks per strategy: {'Header-Pinned': 7, 'Naive': 7}. Relevance = the chunk contains the full gold row. Generation uses only the top-1 chunk as context (`llama3.1:8b-instruct-q4_K_M`); correctness = gold cell value appears in the answer.

| Strategy / Mode | Hit@3 | MRR@3 | Answer Correct (top-1 ctx) | Retrieval ms |
|---|---:|---:|---:|---:|
| Naive / BM25 | **1.000** | 0.900 | 0.733 | 0.2 |
| Naive / FAISS | 0.467 | 0.311 | 0.400 | 58.4 |
| Naive / Hybrid | 0.967 | 0.689 | 0.700 | 57.1 |
| Header-Pinned / BM25 | **1.000** | **1.000** | 0.967 | **0.2** |
| Header-Pinned / FAISS | 0.567 | 0.372 | 0.400 | 52.6 |
| Header-Pinned / Hybrid | **1.000** | 0.917 | **1.000** | 49.3 |

- Fee and tax are both small decimals, so a chunk without column headers is ambiguous to the reader.
- 30 queries is a targeted sample.
