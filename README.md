# DataHound-RAG

A local-first **Hybrid Retrieval-Augmented Generation (RAG)** pipeline engineered for financial filings, audit ledgers, and technical documents containing dense tables, featuring deterministic guardrails to reduce hallucination risk.

Standard RAG pipelines break on tabular enterprise documents for four reasons:
1. **Layout destruction:** Naive PDF text extractors flatten multi-column tables into unreadable token streams.
2. **Schema amnesia:** Fixed-size text splitters sever rows 26–500 from their column headers on row 1, leaving downstream chunks with unlabelled numbers.
3. **Single-retriever blind spots:** Dense embeddings (`FAISS`) collapse on exact alphanumeric identifiers (`eg. Transaction IDs like TR109`) and adjacent decimal columns, while lexical search (`BM25`) fails when users paraphrase domain concepts with zero keyword overlap.
4. **Row-substitution hallucinations:** When a queried identifier is absent from retrieved chunks, LLMs silently substitute values from neighbouring rows.

This repository systematically mitigates each failure mode—running locally on **Ollama** (`nomic-embed-text` + `Llama 3.1 8B`)—and ships with a modular IR, chunking, and RAG Triad evaluation suite (`eval_rag.py`) to benchmark retrieval trade-offs and error modes empirically.

---

## System Architecture

```mermaid
flowchart LR
    subgraph Offline ["1. Offline Ingestion & Indexing"]
        direction TB
        Docs["Raw Filings (.pdf, .xlsx, .csv)"] --> Parser["Docling & MarkItDown Parsers"]
        Parser --> Chunker["DualModeChunker"]
        Chunker --> Prose["Prose Windows (Heading-Anchored)"]
        Chunker --> Tables["Table Slices (25-Row Header-Pinned)"]
        Prose --> Corpus["Chunk Corpus (chunks/*.json)"]
        Tables --> Corpus
        Corpus --> BM25["BM25Okapi Index (.pkl, 5x ID Boost)"]
        Corpus --> FAISS["FAISS IndexFlatIP (.index, Cosine)"]
    end

    subgraph Online ["2. Online Retrieval & Verification"]
        direction TB
        Query(["User Audit Query"]) --> Retriever["HybridRetriever"]
        Retriever --> Sparse["Sparse Lexical Search (BM25)"]
        Retriever --> Dense["Dense Semantic Search (nomic-embed)"]
        Sparse --> RRF["Reciprocal Rank Fusion (k = 60)"]
        Dense --> RRF
        RRF --> Guard["Deterministic Guardrail (Entity Gate + Dense Cosine Gate)"]
        Guard --> Reject(["Refusal: Absent ID / OOD (0 LLM Tokens)"])
        Guard --> LLM["Llama 3.1 8B (temp=0.0, Cited Answer)"]
    end

    Offline --> Online

    style Guard fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style Reject fill:#450a0a,stroke:#ef4444,stroke-width:2px,color:#fecaca
    style LLM fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#d1fae5
```

### Key Engineering Decisions

* **Header-Pinned Table Chunking (`DualModeChunker`):** Markdown tables are sliced into `25`-row windows (`2`-row overlap) with both the parent section heading and the Markdown header/separator rows re-injected at the top of *every* chunk so row #200 retains full column semantics.
* **Asymmetric Dense + Boosted Sparse Retrieval (`HybridRetriever`):**
  * **Sparse stream:** Identifier-preserving tokenizer (`TR109`, `05-01-2016`, `E-4017`) with stopword removal and a $5\times$ term-frequency boost on opaque record codes, excluding fiscal/calendar period shorthands (`FY25`, `Q4`) from destructive filtering.
  * **Dense stream:** L2-normalized `FAISS IndexFlatIP` (exact cosine similarity) using `nomic-embed-text` with asymmetric `search_document:` and `search_query:` prefixes and GPU batching.
  * **Reciprocal Rank Fusion (RRF):** Combines rankers without score-scale calibration via $\text{RRF}(d) = \sum_{i \in R} \frac{1}{k + r_i(d)}$ ($k = 60$), capping candidate pools on small corpora so bottom-half documents cannot accumulate phantom consensus credit.
* **Three-Layer Hallucination Mitigation Defense:**
  1. *Retrieval gate:* Returns empty context when a queried record ID (`TR9999`, `E-9999`) is absent across the entire index; ranks exact-ID chunks first and backfills remaining slots for multi-hop context when present.
  2. *Deterministic Python guardrail:* Rejects prompt injections, missing entity codes, and low-evidence out-of-domain queries (`lexical coverage < tau_lex` and `max_dense_cosine < tau_dense`, calibrated on held-out queries with zero test-set leakage) before calling the LLM.
  3. *Forensic system prompt:* Locks `temperature=0.0` (`num_ctx=4096`) and requires a verbatim `### Extracted Evidence` block prior to synthesis to constrain generation strictly to retrieved text.

---

## Empirical Benchmark Results

All generation was executed locally against **Ollama** using `nomic-embed-text` (embeddings) and `llama3.1:8b-instruct-q4_K_M` (generation locked at `temperature=0.0`, `seed=7`). RAG Triad faithfulness, relevance, and precision were independently evaluated using decoupled **Gemini 3.5 Flash Lite** (locked at `temperature=0.0`, `seed=7`, rate-limited to 4 RPM with local disk caching). Note that the Gemini judge sends the question, retrieved chunks, and model answer to Google's API for evaluation; for fully offline or air-gapped runs, `--judge-provider ollama` runs the judge locally with Llama 3.1. Pure benchmark questions and public financial statements are evaluated—no private data is transmitted. Out-of-domain attacks are evaluated strictly via **Refusal Accuracy** and **End-to-End Abstention** so they never distort in-domain IR macro-averages.

> [!NOTE]
> **Evaluation Scope & Provenance:** These benchmark metrics evaluate comparative ablation across retrieval and chunking strategies on expanded, frozen test sets (Ledger $N=30$, NVIDIA $N=34$, DevOps $N=12$). For full statistical methodology, Wilson 95% confidence intervals, and paired McNemar tests, see [`reports/VERIFICATION_FINAL.md`](reports/VERIFICATION_FINAL.md).

### 1. Table-Chunking Ablation: Naive vs. Header-Pinned (`--suite chunking`)

60-row financial settlement ledger (`10` rows/chunk with `0` overlap for ablation contrast vs `25` rows / `2` overlap production default), evaluated on 30 single-row lookups (T01–T30) across 4 columns (`fee_usd`, `tax_usd`, `channel`, `risk_label`), passing the top-1 chunk (`context_k=1`) to the generator. Full report: [`reports/rag_chunking_report.md`](reports/rag_chunking_report.md).

| Strategy / Mode | Hit@1 | Hit@3 | MRR@3 | Answer Correct (Top-1 Context) | Wilson 95% CI | Avg Latency |
|---|---:|---:|---:|---:|---|---:|
| Naive / BM25 | 0.800 | **1.000** | 0.900 | 16/30 (53.3%) | [36.1%, 69.8%] | **0.2 ms** |
| Naive / FAISS | 0.200 | 0.467 | 0.311 | 6/30 (20.0%) | [9.5%, 37.3%] | 135.4 ms |
| Naive / Hybrid | 0.433 | 0.967 | 0.689 | 11/30 (36.7%) | [21.9%, 54.5%] | 49.7 ms |
| **Header-Pinned / BM25** | **1.000** | **1.000** | **1.000** | **30/30 (100.0%)** | [88.6%, 100.0%] | **0.2 ms** |
| Header-Pinned / FAISS | 0.233 | 0.567 | 0.372 | 7/30 (23.3%) | [11.8%, 40.9%] | 51.3 ms |
| **Header-Pinned / Hybrid** | 0.833 | **1.000** | 0.917 | **25/30 (83.3%)** | [66.4%, 92.7%] | 46.7 ms |

* **Why Dense-only struggles on numeric tables:** `FAISS` drops to `Hit@3 = 0.467–0.567` (`Answer Correct = 0.200–0.233`) on exact tabular lookups, whereas `BM25` and `Hybrid` achieve `Hit@3 = 0.967–1.000`.
* **Lexical Superiority on Numeric Lookups:** Because the 30 ledger queries target rows by exact numeric amount (23 queries) or opaque transaction ID (7 queries), Header-Pinned BM25 (30/30, 100.0%) beat Header-Pinned Hybrid (25/30, 83.3%) under top-1 context (`context_k=1`).
* **Impact of Header-Pinning on Tabular Accuracy:** The drop in Hybrid answer accuracy from **83.3% (25/30)** to **36.7% (11/30)** reflects two compounding factors: lower rank-1 retrieval under naive chunking (Hit@1 drops from 0.833 to 0.433) and lost column identities in headerless chunks (McNemar paired test $p = 0.0005$, $b=15, c=1$). BM25 serves as the cleaner test of header-loss: while BM25 retains high retrieval under both chunkings (Hit@1 0.800 vs 1.000; Hit@3 1.000 in both), stripped headers account for up to 8 of the 14 misses (correctness drops from 100.0% to 53.3%, $p = 0.0001$, $b=14, c=0$) due to column ambiguity between `fee_usd` and `tax_usd`, with the remaining 6 being rank-1 retrieval misses.
* **Multi-Chunk Exploratory Suite (`context_k=3`):** Under top-1 context, Header-Pinned Hybrid had 5 misses where the relevant chunk was retrieved at rank 2 (`rr=0.50`, `Hit@3=1.000`). Expanding to `context_k=3` fully resolves all 5 misses, achieving **30/30 (100.0%)** correctness.

### 2. Dual-Stream Retrieval Ablation (`--suite retrieval`)

16-chunk technical corpus evaluated across 12 queries (`exact_keyword_match`, `semantic_intent`, `complex_multi_hop`, `out_of_domain_attack`), with `tau_dense = 0.614` calibrated on 4 disjoint held-out queries. Full report: [`reports/rag_benchmark_report.md`](reports/rag_benchmark_report.md).

| Mode | Hit Rate@5 | MRR@5 | Recall@5 | Refusal Acc (Gate) | False Refusal | Faithfulness | Answer Relevance | Context Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.750 | 0.750 | 0.750 | **1.000** | 0.000 | 1.000 | 0.625 | 0.729 |
| FAISS | **1.000** | **1.000** | **1.000** | **1.000** | 0.000 | 1.000 | **0.750** | **1.000** |
| **Hybrid (RRF)** | **1.000** | 0.938 | **1.000** | **1.000** | 0.000 | 1.000 | **0.750** | 0.938 |

* On `semantic_intent` queries (zero lexical overlap), `BM25` collapses to `Hit@5 = 0.333`, while `FAISS` and `Hybrid` achieve `1.000` (`Hybrid` achieves rank-aware `Context Precision` at `0.938`).

### 3. Real Enterprise PDF: NVIDIA Q4 FY25 CFO Commentary (`--use-production-index NVIDIA_CFO`)

27-chunk index parsed via IBM Docling and `DualModeChunker` from [`data/NVIDIA_CFO.pdf`](data/NVIDIA_CFO.pdf), evaluated on [`benchmarks/nvidia_cfo_eval.json`](benchmarks/nvidia_cfo_eval.json) ($N=34$; 8 exact match, 8 semantic intent, 8 multi-hop, 10 OOD attacks), with `tau_dense = 0.633` calibrated on 20 held-out queries ([`benchmarks/nvidia_cfo_calibration.json`](benchmarks/nvidia_cfo_calibration.json)). Full report: [`reports/rag_nvidia_cfo_report.md`](reports/rag_nvidia_cfo_report.md).

| Mode | Hit Rate@5 | MRR@5 | Recall@5 | Refusal Acc (Gate) | End-to-End Abstention | False Refusal | Faithfulness | Answer Relevance | Context Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.792 | 0.585 | 0.750 | 0.700 | **1.000** | 0.042 | 0.957 | 0.696 | 0.636 |
| FAISS | 0.833 | **0.764** | 0.812 | 0.700 | **1.000** | 0.042 | **1.000** | **0.783** | **0.766** |
| **Hybrid (RRF + Gate)** | **0.917** | 0.701 | **0.875** | 0.700 | **1.000** | 0.042 | **1.000** | **0.783** | 0.721 |

* **Zero-Token Guardrail Defenses:** The pre-generation guardrail (entity gate and dense cosine gate with `tau_dense = 0.633`) deterministically intercepted 70.0% (7/10) of out-of-domain queries without invoking the LLM: 5 absent-ID queries (`empty_context`), 1 prompt injection (`P10`), and 1 low-similarity query (`P09`). The generator model's strict evidence citation prompt prompted abstention on the remaining 3 near-domain false premises (`P12` Bitcoin, `P33` Blackwell cancellation, `P34` Qualcomm acquisition), achieving **100.0% end-to-end abstention**. The only in-domain false refusal is 0.042 (1/24, query `P22`), a known-defective query asking about "commercial billings" where the source text reports revenue.
* **Semantic & Multi-Hop Superiority:** On semantic intent queries, BM25 collapses to Hit@5 = 0.375, while Hybrid reaches **0.750**; on complex multi-hop queries, Hybrid achieves **1.000 Hit Rate@5** (MRR@5 = 0.906).

---

## Repository Structure

```text
DataHound-RAG/
├── ingestion.py                       # Stage 1: Layout-aware PDF (Docling) & Office/CSV (MarkItDown) parser
├── chunker.py                         # Stage 2: Header-pinned table & heading-anchored prose chunker
├── retriever.py                       # Stage 3: BM25Okapi + FAISS (Ollama embeddings) + RRF engine
├── app.py                             # Stage 4: Flask auditor API & UI with deterministic guardrails
├── eval_rag.py                        # Stage 5: CLI entrypoint for IR, Guardrail & RAG Triad ablation suite
├── evaluation/                        # Modular evaluation package
│   ├── models.py                      # Dataclasses (Chunk, EvalItem, RetrievalResult, GuardrailDecision)
│   ├── datasets.py                    # Built-in corpora, 60-row ledger generator & JSON validator
│   ├── metrics.py                     # Pure math: RRF, Hit@k, MRR@k, Recall@k, Context Precision@k
│   ├── retrievers.py                  # Embedders, BM25/Dense indices, RetrieverAdapter & NaiveChunker
│   ├── guardrails.py                  # DeterministicGuardrail, entity extractor & tau_dense calibration
│   ├── judge.py                       # OllamaLLM & JSON-constrained RAG Triad LLM-as-a-Judge
│   ├── runner.py                      # Benchmark and table-chunking orchestration runners
│   └── reporting.py                   # Pandas aggregation, ASCII dashboard & GFM Markdown exporter
├── templates/
│   └── index.html                     # Web UI frontend
├── data/
│   └── NVIDIA_CFO.pdf                 # Sample financial filing for reproducible benchmarking
├── benchmarks/
│   ├── nvidia_cfo_eval.json           # 34 labelled evaluation items for NVIDIA_CFO
│   ├── nvidia_cfo_calibration.json    # 20 disjoint held-out queries for tau_dense calibration
│   ├── DATASET_HASHES.txt             # SHA-256 hashes of frozen benchmark datasets
│   └── DATASET_REVIEW.md              # Detailed provenance, category & overlap review
├── tests/                             # 63 unit tests verifying metrics, guardrails, datasets & stats
├── reports/
│   ├── VERIFICATION_FINAL.md          # Comprehensive statistical verification report
│   ├── rag_benchmark_report.md        # Generated retrieval ablation report (DevOps corpus)
│   ├── rag_chunking_report.md         # Generated chunking ablation report (60-row ledger)
│   └── rag_nvidia_cfo_report.md       # Generated real-PDF benchmark report (NVIDIA Q4 FY25)
└── requirements.txt                   # Core Python dependencies (offline RAG runtime)
```

---

## Security, Limitations & Threat Model Considerations

- **Lexical vs. Semantic Injections:** The pre-retrieval regex gate deterministically intercepts explicit override commands (e.g., prompt extraction or instruction overrides) and non-existent entity IDs with zero LLM token overhead. However, heuristic regex alone cannot defend against indirect prompt injections embedded within untrusted third-party document text or PDF metadata.
- **Serving Path Integration:** While out-of-domain detection and refusal behaviors (`tau_dense`, cosine gating, injection patterns) are benchmarked in `evaluation/guardrails.py`, enterprise production hardening requires porting these checks into synchronous middleware within the Flask serving path (`app.py`): entity and regex checks prior to retrieval, and dense cosine gating post-retrieval prior to LLM generation.
- **Parametric Context Boundaries & Tabular Reasoning:** Operating at `temperature=0.0` with a mandatory `### Extracted Evidence` citation protocol grounds Llama 3.1 8B to the retrieved Markdown context. However, multi-page financial reconciliation remains bounded by localized context window limits (`num_ctx=4096`) and the arithmetic limitations of autoregressive LLMs on dense tables. Complex queries requiring cross-statement math or full-ledger aggregation are best addressed by pairing hybrid retrieval with deterministic code execution (e.g., Pandas).
- **False Refusal Sensitivity & Defective Query P22:** Calibrating `tau_dense = 0.633` on held-out queries creates a tight boundary (in-domain min 0.640, OOD max 0.625, margin +0.015). Query `P22` is a known-defective benchmark query asking about *"aggregate commercial billings"* while the source text reports revenue growth (billings and revenue are distinct accounting concepts); its semantic divergence dips slightly below the threshold (`cos = 0.6283 < 0.633`), resulting in a 4.2% (1/24) in-domain false refusal rate.
- **Benchmark Freeze Protocol:** Benchmark queries were frozen prior to running evaluations. Documented defects (e.g., `P23` referencing a 10-K filing rather than earnings commentary) are intentionally preserved and transparently audited in [`reports/VERIFICATION_FINAL.md`](reports/VERIFICATION_FINAL.md) rather than retroactively tuned.

---

## Quickstart & Setup

### 1. Prerequisites

* **Python 3.10+**
* **[Ollama](https://ollama.com/)** running locally (`http://localhost:11434`) with the embedding and generation models pulled:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b-instruct-q4_K_M
```

### 2. Installation

```powershell
# Clone repository and create virtual environment (Windows PowerShell)
git clone https://github.com/MihirNagarkar/DataHound-RAG.git
cd DataHound-RAG
py -3 -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## End-to-End Pipeline Usage

### Step 1: Ingest PDF or Spreadsheet to Layout-Preserved Markdown

Routes `.pdf` files through IBM Docling and `.xlsx` / `.csv` / `.docx` files through Microsoft MarkItDown (20 MB max guardrail):

```powershell
python ingestion.py data/NVIDIA_CFO.pdf
# Output -> processed_md/NVIDIA_CFO.md
```

### Step 2: Apply Dual-Mode Schema-Preserving Chunking

Slices narrative sections and re-injects parent headings + table headers across every table segment:

```powershell
python chunker.py processed_md/NVIDIA_CFO.md
# Output -> chunks/NVIDIA_CFO_chunks.json
```

### Step 3: Build & Query the Hybrid Index

Builds and serializes `BM25Okapi` (`.pkl`), normalized `FAISS` vectors (`.index`), and chunk metadata (`_chunks_meta.json`) into `indices/`:

```powershell
# Build index
python retriever.py build chunks/NVIDIA_CFO_chunks.json

# Test CLI hybrid search
python retriever.py search NVIDIA_CFO "Compute & Networking segment revenue Q4 FY25"
```

### Step 4: Launch the Web Auditor (`app.py`)

```powershell
# Optional overrides: $env:LLM_MODEL="llama3.1:8b-instruct-q4_K_M"; $env:PORT="8000"
python app.py
```

Open `http://localhost:8000` in your browser. The web interface provides real-time verification and rank telemetry:

| Query & Extracted Evidence | Grounded Audit Answer |
| :---: | :---: |
| ![Query & Extracted Evidence](assets/DataHound-RAG_1.png) | ![Grounded Synthesis](assets/DataHound-RAG_2.png) |

| Multi-Stream Rank Telemetry | Header-Pinned Table Slices |
| :---: | :---: |
| ![Rank Telemetry & Prose Chunks](assets/DataHound-RAG_3.png) | ![Header-Pinned Table Chunk](assets/DataHound-RAG_4.png) |

Or query the REST endpoint directly via CLI:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "NVIDIA_CFO", "query": "How much Blackwell revenue was delivered in Q4 FY25?", "top_k": 3}'
```

---

## Running the Evaluation & Ablation Suite (`eval_rag.py`)

### Run Deterministic Unit Tests

Verifies RRF math, Hit@$k$, MRR@$k$, Recall@$k$, rank-aware Context Precision@$k$, entity extraction, guardrail rules, and Markdown table rendering without requiring Ollama:

```powershell
pytest tests -q
```

### Run Benchmarks

```powershell
# 1. Offline pipeline smoke test (character n-gram hashing embedder, no Ollama required)
python eval_rag.py --offline

# 2. Inspect held-out guardrail threshold calibration only
python eval_rag.py --calibrate --skip-generation

# 3. Full Dual-Stream Retrieval + Guardrail + RAG Triad benchmark -> reports/rag_benchmark_report.md
# (Default uses Gemini judge; pass --judge-provider ollama for fully offline local evaluation)
python eval_rag.py --output reports/rag_benchmark_report.md --csv rag_benchmark_raw.csv

# 4. Naive vs. Header-Pinned Table-Chunking ablation -> reports/rag_chunking_report.md
python eval_rag.py --suite chunking --output reports/rag_chunking_report.md --csv rag_chunking_raw.csv

# 5. Benchmark a production HybridRetriever index -> reports/rag_nvidia_cfo_report.md
python eval_rag.py `
  --use-production-index NVIDIA_CFO `
  --benchmark-file benchmarks/nvidia_cfo_eval.json `
  --calibration-file benchmarks/nvidia_cfo_calibration.json `
  --output reports/rag_nvidia_cfo_report.md `
  --csv rag_nvidia_cfo_raw.csv
```

### Plugging in a Custom Retriever

Subclass `RetrieverAdapter` in [`evaluation/retrievers.py`](evaluation/retrievers.py) and implement `retrieve(query, k) -> RetrievalResult`:

```python
from evaluation.retrievers import RetrieverAdapter
from evaluation.models import RetrievalResult, RetrievedChunk

class MyCustomRetriever(RetrieverAdapter):
    name = "CustomHybrid"

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        hits = my_pipeline.search(query, top_k=k)
        chunks = [
            RetrievedChunk(chunk_id=h.id, content=h.text, score=h.score, rank=idx)
            for idx, h in enumerate(hits, start=1)
        ]
        return RetrievalResult(chunks=chunks, diagnostics={"max_dense_cosine": hits[0].cosine if hits else 0.0})
```

