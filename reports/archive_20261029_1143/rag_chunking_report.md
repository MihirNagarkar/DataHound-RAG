# Chunking Ablation: Naive vs Header-Pinned Table Chunks

*Setup:* 60-row transaction ledger, 8 single-row lookup questions, chunks per strategy: {'Header-Pinned': 7, 'Naive': 7}. Relevance = the chunk contains the full gold row. Generation uses only the top-1 chunk as context (`llama3.1:8b-instruct-q4_K_M`); correctness = gold cell value appears in the answer.

| Strategy / Mode | Hit@3 | MRR@3 | Answer Correct (top-1 ctx) | Retrieval ms |
|---|---:|---:|---:|---:|
| Naive / BM25 | **1.000** | 0.938 | 0.500 | 0.4 |
| Naive / FAISS | 0.375 | 0.229 | 0.125 | 468.9 |
| Naive / Hybrid | **1.000** | 0.688 | 0.375 | 85.1 |
| Header-Pinned / BM25 | **1.000** | **1.000** | **1.000** | **0.2** |
| Header-Pinned / FAISS | 0.500 | 0.354 | 0.250 | 74.2 |
| Header-Pinned / Hybrid | **1.000** | 0.938 | 0.875 | 69.1 |

- Fee and tax are both small decimals, so a chunk without column headers is ambiguous to the reader.
- 8 queries is an anecdotal sample, not a statistically significant result.
