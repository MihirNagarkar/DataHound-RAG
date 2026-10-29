import os
import sys
import re
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rank_bm25 import BM25Okapi
from entities import extract_entities, _ENTITY_RE, _NON_RECORD_ENTITY_RE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Production Dual-Stream Hybrid Retrieval Engine:
    - Sparse Stream: BM25Okapi with alpha-numeric preservation (IDs, tickers, codes).
    - Dense Stream: FAISS IndexFlatIP (Cosine Similarity) via Ollama nomic-embed-text.
    - Batch GPU Embedding: Uses Ollama /api/embed batch endpoint with pre-warm protection.
    - Rank Fusion: Reciprocal Rank Fusion (RRF, k=60).
    """

    def __init__(
        self,
        index_dir: str = "indices",
        ollama_base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        rrf_k: int = 60,
        request_timeout: int = 180
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.rrf_k = rrf_k
        self.request_timeout = request_timeout

        # Configure resilient HTTP session with connection pooling and retries
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=[502, 503, 504]
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

        # In-memory stores
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.embedding_dim: int = 768


    # TOKENIZATION & EMBEDDING CLIENT

    _BM25_STOPWORDS = frozenset(
        "a an the and or of to in on for with at by from is are was were be been it its this that these those "
        "what which who whom how when where why do does did can could should would will may might must i we you "
        "our your their my me us they them as if then than so not no yes about into over any all each per out "
        "up down off use used using get gets got has have had having give gives please explain tell write "
        "describe fast long often much many".split()
    )
    _ENTITY_RE = _ENTITY_RE
    _NON_RECORD_ENTITY_RE = _NON_RECORD_ENTITY_RE
    extract_entities = staticmethod(extract_entities)

    @classmethod
    def tokenize_text(cls, text: str) -> List[str]:
        """
        Tokenizer designed for enterprise documents. Preserves transaction codes
        (TR109), operator codes (OPR16), dates (05-01-2016), and monetary digits,
        while filtering high-frequency stopwords and single-letter noise.
        """
        tokens = re.findall(r"\b[A-Za-z0-9_.-]+\b", text.lower())
        return [t for t in tokens if t not in cls._BM25_STOPWORDS and (len(t) > 1 or t.isdigit())]

    def _format_doc_text(self, text: str) -> str:
        """Applies asymmetric document prefix required by nomic-embed models."""
        if "nomic-embed" in self.embedding_model.lower() and not text.startswith("search_document:"):
            return f"search_document: {text}"
        return text

    def _format_query_text(self, query: str) -> str:
        """Applies asymmetric query prefix required by nomic-embed models."""
        if "nomic-embed" in self.embedding_model.lower() and not query.startswith("search_query:"):
            return f"search_query: {query}"
        return query

    def _warmup_ollama(self) -> None:
        """
        Forces Ollama to load model weights into VRAM before sending heavy tabular chunks.
        """
        logger.info(f"Warming up Ollama model '{self.embedding_model}' in VRAM...")
        try:
            # Try newer /api/embed endpoint first
            url = f"{self.ollama_base_url}/api/embed"
            payload = {"model": self.embedding_model, "input": "warmup"}
            res = self.session.post(url, json=payload, timeout=self.request_timeout)
            
            if res.status_code == 404:
                # Fallback to legacy /api/embeddings endpoint
                url = f"{self.ollama_base_url}/api/embeddings"
                payload = {"model": self.embedding_model, "prompt": "warmup"}
                res = self.session.post(url, json=payload, timeout=self.request_timeout)
            
            res.raise_for_status()
            logger.info("Ollama embedding engine is warm and responsive.")
        except Exception as e:
            logger.error(f"Ollama warmup failed. Is Ollama running? Error: {e}")
            raise ConnectionError(
                f"Could not connect to Ollama at {self.ollama_base_url}. "
                "Verify Ollama is running in your background terminal."
            ) from e

    def _get_batch_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Batches texts and queries Ollama using GPU parallelism.
        Tries batch endpoint (/api/embed); falls back to sequential if running older Ollama.
        """
        self._warmup_ollama()

        texts = [self._format_doc_text(t) for t in texts]
        total = len(texts)
        all_embeddings: List[List[float]] = []
        embed_url = f"{self.ollama_base_url}/api/embed"
        legacy_url = f"{self.ollama_base_url}/api/embeddings"

        # Check if batch endpoint is supported
        test_res = self.session.post(
            embed_url,
            json={"model": self.embedding_model, "input": ["test"]},
            timeout=self.request_timeout
        )
        supports_batch = (test_res.status_code == 200)

        logger.info(
            f"Generating embeddings for {total} chunks using '{self.embedding_model}' "
            f"({'GPU Batch Mode' if supports_batch else 'Sequential Mode'})..."
        )

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]

            if supports_batch:
                try:
                    payload = {"model": self.embedding_model, "input": batch}
                    res = self.session.post(embed_url, json=payload, timeout=self.request_timeout)
                    res.raise_for_status()
                    batch_vecs = res.json().get("embeddings", [])
                    all_embeddings.extend(batch_vecs)
                except Exception as err:
                    logger.warning(f"Batch call failed at chunk {i}. Retrying sequentially: {err}")
                    for single_text in batch:
                        vec = self._get_single_embedding_fallback(single_text, legacy_url)
                        all_embeddings.append(vec)
            else:
                for single_text in batch:
                    vec = self._get_single_embedding_fallback(single_text, legacy_url)
                    all_embeddings.append(vec)

            processed = min(i + batch_size, total)
            if processed % 32 == 0 or processed == total:
                logger.info(f"Embedded [{processed}/{total}] chunks...")

        embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
        return embeddings_matrix

    def _get_single_embedding_fallback(self, text: str, url: str) -> List[float]:
        """Fallback for individual embeddings."""
        payload = {"model": self.embedding_model, "prompt": text}
        res = self.session.post(url, json=payload, timeout=self.request_timeout)
        res.raise_for_status()
        vec = res.json().get("embedding")
        if not vec:
            raise ValueError("Ollama returned an empty embedding vector.")
        return vec

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Embeds a single query string for runtime search."""
        query = self._format_query_text(query)
        embed_url = f"{self.ollama_base_url}/api/embed"
        legacy_url = f"{self.ollama_base_url}/api/embeddings"

        try:
            res = self.session.post(
                embed_url,
                json={"model": self.embedding_model, "input": query},
                timeout=self.request_timeout
            )
            if res.status_code == 200:
                data = res.json()
                vec = data.get("embeddings", [[]])[0]
                return np.array(vec, dtype=np.float32)
        except Exception:
            pass

        # Fallback to /api/embeddings
        res = self.session.post(
            legacy_url,
            json={"model": self.embedding_model, "prompt": query},
            timeout=self.request_timeout
        )
        res.raise_for_status()
        return np.array(res.json()["embedding"], dtype=np.float32)

    
    # INDEX BUILDING & PERSISTENCE:
    
    def build_index(self, chunks_json_path_str: str) -> str:
        """
        Builds and serializes both BM25 and FAISS indices from a chunks JSON file.
        """
        chunks_path = Path(chunks_json_path_str).resolve()
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        if not self.chunks:
            raise ValueError(f"No chunks found in {chunks_path.name}")

        dataset_name = self.chunks[0].get("source", chunks_path.stem.replace("_chunks", ""))
        logger.info(f"Building Hybrid Index for '{dataset_name}' ({len(self.chunks)} chunks)...")

        # 1. Build Sparse BM25 Index
        corpus = [chunk["content"] for chunk in self.chunks]
        tokenized_corpus = [self.tokenize_text(doc) for doc in corpus]
        logger.info("Building BM25Okapi index...")
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Build Dense FAISS Index via Batch Embedding
        embeddings = self._get_batch_embeddings(corpus, batch_size=16)
        self.embedding_dim = embeddings.shape[1]

        # L2-normalize vectors so Inner Product (IP) calculates Cosine Similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(embeddings)
        logger.info(f"FAISS Index created: {self.faiss_index.ntotal} vectors added (Dim: {self.embedding_dim}).")

        # 3. Serialize artifacts to disk
        bm25_file = self.index_dir / f"{dataset_name}_bm25.pkl"
        faiss_file = self.index_dir / f"{dataset_name}_faiss.index"
        chunks_file = self.index_dir / f"{dataset_name}_chunks_meta.json"

        with open(bm25_file, "wb") as f:
            pickle.dump(self.bm25, f)

        faiss.write_index(self.faiss_index, str(faiss_file))

        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

        logger.info(f"All index artifacts successfully saved to '{self.index_dir}/'")
        return dataset_name

    def load_index(self, dataset_name: str) -> None:
        """
        Loads pre-built BM25, FAISS, and metadata files into memory.
        """
        bm25_file = self.index_dir / f"{dataset_name}_bm25.pkl"
        faiss_file = self.index_dir / f"{dataset_name}_faiss.index"
        chunks_file = self.index_dir / f"{dataset_name}_chunks_meta.json"

        if not (bm25_file.exists() and faiss_file.exists() and chunks_file.exists()):
            raise FileNotFoundError(
                f"Missing index files for dataset '{dataset_name}' in '{self.index_dir}'. "
                "Ensure build_index() was executed successfully."
            )

        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        with open(bm25_file, "rb") as f:
            self.bm25 = pickle.load(f)

        self.faiss_index = faiss.read_index(str(faiss_file))
        self.embedding_dim = self.faiss_index.d

        logger.info(
            f"Successfully loaded index '{dataset_name}': {len(self.chunks)} chunks, "
            f"FAISS vectors: {self.faiss_index.ntotal}"
        )

    
    # RETRIEVAL & RECIPROCAL RANK FUSION (RRF)
    
    def search_sparse(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Executes BM25 search with Entity-Code Boosting.
        Pours extra weight into exact transaction codes, operator codes, and numbers.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not initialized. Call load_index() first.")

        tokens = self.tokenize_text(query)
        if not tokens:
            return []

        # ANTI-HALLUCINATION BOOST:
        # If the user mentioned a specific entity code (e.g., TR6, OPR12, E-4017, 2024-INV-001),
        # boost its weight by repeating it so generic column words don't dilute it.
        entity_tokens = {e.lower() for e in self.extract_entities(query)}
        boosted_tokens: List[str] = []
        for t in tokens:
            boosted_tokens.append(t)
            if t in entity_tokens:
                boosted_tokens.extend([t] * 5)  # 5x BM25 Weight Boost for exact IDs

        scores = self.bm25.get_scores(boosted_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0.0]

    def search_dense(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Executes Cosine Similarity search on FAISS and returns sorted (chunk_index, score)."""
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not initialized. Call load_index() first.")

        query_vec = self._get_query_embedding(query).reshape(1, -1)
        faiss.normalize_L2(query_vec)

        scores, indices = self.faiss_index.search(query_vec, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:
                results.append((int(idx), float(score)))
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Executes parallel Sparse and Dense retrieval, then merges via RRF.
        Includes a Hard Filter for exact Entity IDs to guarantee precision.
        """
        if not self.chunks:
            raise RuntimeError("Retriever contains no loaded chunks.")

        # 1. Extract opaque record IDs (e.g. ['TR6', 'E-4017', '2024-INV-001']), excluding period/tech tags like FY25 or H200
        required_entities = [ent.upper() for ent in self.extract_entities(query)]

        # Cap candidate pool on small corpora so bottom-ranked chunks do not receive RRF consensus credit
        n_chunks = len(self.chunks)
        effective_pool = min(candidate_pool, max(top_k, n_chunks // 2)) if n_chunks > top_k else n_chunks

        sparse_results = self.search_sparse(query, top_k=effective_pool)
        dense_results = self.search_dense(query, top_k=effective_pool)

        rrf_scores: Dict[int, float] = {}
        meta_ranks: Dict[int, Dict[str, Any]] = {}

        # Merge Sparse
        for rank, (chunk_idx, score) in enumerate(sparse_results, start=1):
            rrf_contribution = 1.0 / (self.rrf_k + rank)
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + rrf_contribution
            if chunk_idx not in meta_ranks:
                meta_ranks[chunk_idx] = {}
            meta_ranks[chunk_idx]["bm25_rank"] = rank
            meta_ranks[chunk_idx]["bm25_score"] = round(score, 4)

        # Merge Dense
        for rank, (chunk_idx, score) in enumerate(dense_results, start=1):
            rrf_contribution = 1.0 / (self.rrf_k + rank)
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + rrf_contribution
            if chunk_idx not in meta_ranks:
                meta_ranks[chunk_idx] = {}
            meta_ranks[chunk_idx]["dense_rank"] = rank
            meta_ranks[chunk_idx]["dense_cosine_sim"] = round(score, 4)

        # 2. Apply Entity-Aware Ordering & Anti-Hallucination Gate
        # If the user asked for a specific record ID (e.g. TR6 or E-9999):
        #   - If NO candidate chunk contains it, return [] so downstream guardrails refuse cleanly.
        #   - Otherwise rank exact-ID chunks first, then backfill up to top_k with top RRF chunks
        #     so multi-hop queries preserve secondary context chunks.
        if required_entities:
            matching_indices = []
            non_matching_indices = []
            for idx in rrf_scores.keys():
                chunk_content = self.chunks[idx]["content"]
                if all(
                    re.search(rf"\b{re.escape(entity)}\b", chunk_content, flags=re.IGNORECASE)
                    for entity in required_entities
                ):
                    matching_indices.append(idx)
                else:
                    non_matching_indices.append(idx)
            if not matching_indices:
                sorted_indices = []
            else:
                sorted_indices = (
                    sorted(matching_indices, key=lambda i: rrf_scores[i], reverse=True)
                    + sorted(non_matching_indices, key=lambda i: rrf_scores[i], reverse=True)
                )[:top_k]
        else:
            sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]

        merged_results: List[Dict[str, Any]] = []
        for final_rank, idx in enumerate(sorted_indices, start=1):
            chunk_data = self.chunks[idx].copy()
            chunk_data["final_rank"] = final_rank
            chunk_data["rrf_score"] = round(rrf_scores[idx], 6)
            chunk_data["sparse_stats"] = {
                "rank": meta_ranks[idx].get("bm25_rank", None),
                "score": meta_ranks[idx].get("bm25_score", 0.0)
            }
            chunk_data["dense_stats"] = {
                "rank": meta_ranks[idx].get("dense_rank", None),
                "cosine_sim": meta_ranks[idx].get("dense_cosine_sim", 0.0)
            }
            merged_results.append(chunk_data)

        return merged_results


# CLI TEST RUNNER

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUsage:")
        print("  1. Build Index : python retriever.py build <path_to_chunks.json>")
        print("  2. Test Search : python retriever.py search <dataset_name> \"<query>\"\n")
        print("Examples:")
        print("  python retriever.py build chunks/FraudDetection_CaseStudy_chunks.json")
        print("  python retriever.py search FraudDetection_CaseStudy \"TR6 Zariyah Richardson Fraud\"\n")
        sys.exit(1)

    action = sys.argv[1].lower()
    retriever = HybridRetriever()

    if action == "build":
        chunks_file_arg = sys.argv[2]
        try:
            name = retriever.build_index(chunks_file_arg)
            print(f"\n[SUCCESS] Index for '{name}' created and ready for querying.")
        except Exception as err:
            logger.error(f"Build failed: {err}")
            sys.exit(1)

    elif action == "search":
        if len(sys.argv) < 4:
            print("[ERROR] Please provide both dataset_name and query string.")
            sys.exit(1)

        dataset = sys.argv[2]
        query_text = sys.argv[3]

        try:
            retriever.load_index(dataset)
            results = retriever.hybrid_search(query_text, top_k=3)

            print("\n" + "=" * 70)
            print(f"HYBRID RETRIEVAL RESULTS FOR: \"{query_text}\"")
            print("=" * 70)

            for item in results:
                print(f"\n[RANK #{item['final_rank']}] | RRF Score: {item['rrf_score']}")
                print(f"BM25 Rank: {item['sparse_stats']['rank']} (Score: {item['sparse_stats']['score']}) | "
                      f"Dense Rank: {item['dense_stats']['rank']} (Cosine: {item['dense_stats']['cosine_sim']})")
                print(f"Row Range: {item.get('row_range')} | Chunk ID: {item['chunk_id']}")
                print("-" * 70)
                print(item["content"][:450] + "\n...")

        except Exception as err:
            logger.error(f"Search failed: {err}")
            sys.exit(1)