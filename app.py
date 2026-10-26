import os
import sys
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from flask import Flask, request, jsonify, render_template
import requests
from retriever import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")

retriever = HybridRetriever(
    index_dir="indices",
    ollama_base_url=OLLAMA_BASE_URL
)


SYSTEM_PROMPT = """You are a strict, forensic Enterprise Document Intelligence Auditor.
You must answer the user's question with 100% factual accuracy based ONLY on the provided Markdown context.

CRITICAL PROTOCOL TO ELIMINATE HALLUCINATIONS:
1. MANDATORY VERBATIM CITATION: 
   Before providing your answer, you MUST copy-paste the exact Markdown table row(s) or prose sentence(s) containing the relevant data inside an "### Extracted Evidence" block.
2. If the user asks for a specific identifier or data point and that EXACT data is NOT present in the retrieved context:
   - Output: "### VERIFICATION FAILED: The requested information is not present in the retrieved evidence."
   - DO NOT substitute another row.
   - DO NOT guess or infer what the data might be.
3. Keep your explanation concise, strictly reporting only what is confirmed in your cited evidence.
"""


# PYTHON FACT-CHECKING GUARDRAIL (DETERMINISTIC VERIFICATION)

def get_available_datasets() -> List[str]:
    indices_dir = Path("indices")
    if not indices_dir.exists():
        return []
    return [f.stem.replace("_faiss", "") for f in indices_dir.glob("*_faiss.index")]

def extract_query_entities(query: str) -> List[str]:
    """
    Finds exact alphanumeric codes in the query for the guardrail.
    Generic pattern: 2+ Uppercase letters followed by numbers (e.g., AB123, ID90, INV405).
    Developers can customize this regex for specific domains (CUSIP, SSN, etc.).
    """
    return re.findall(r"\b([A-Z]{2,}\d+)\b", query)

def verify_evidence_presence(target_entity: str, context_chunks: list) -> bool:
    """Scans the retrieved chunks to ensure the requested entity code actually exists."""
    pattern = rf"\b{re.escape(target_entity)}\b"
    for chunk in context_chunks:
        if re.search(pattern, chunk["content"], flags=re.IGNORECASE):
            return True
    return False

def call_local_llm(prompt: str, context_chunks: list) -> str:
    """Calls Ollama Llama 3.1 8B with locked 4096 context and 0.0 temperature."""
    formatted_context = "\n\n"
    for i, c in enumerate(context_chunks, start=1):
        formatted_context += (
            f"--- [RETRIEVED CONTEXT SEGMENT #{i} | Chunk: {c['chunk_id']} | "
            f"RRF Score: {c['rrf_score']:.5f}] ---\n"
            f"{c['content']}\n\n"
        )

    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"### RETRIEVED EVIDENCE:\n{formatted_context}\n"
        f"### USER QUERY:\n{prompt}\n\n"
        f"### AUDIT ANSWER:"
    )

    payload = {
        "model": LLM_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "num_ctx": 4096,
            "temperature": 0.0,
            "stop": ["### USER QUERY:", "### RETRIEVED EVIDENCE:"]
        }
    }

    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        raise RuntimeError(f"Ollama generation failed: {str(e)}") from e


# API ROUTES

@app.route("/", methods=["GET"])
def index():
    available = get_available_datasets()
    default_dataset = available[0] if available else ""
    return render_template("index.html", default_dataset=default_dataset)

@app.route("/api/query", methods=["POST"])
def query_endpoint():
    data = request.get_json(silent=True)
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' parameter."}), 400

    user_query = str(data["query"]).strip()
    dataset_name = data.get("dataset_name", "").strip()
    top_k = int(data.get("top_k", 3))

    if not dataset_name:
        return jsonify({"error": "Missing dataset name."}), 400

    try:
        # Lazy load the index if the user switches datasets in the UI
        if not retriever.chunks or (retriever.chunks[0].get("source") != dataset_name):
            retriever.load_index(dataset_name)

        # 1. Hybrid Retrieval
        retrieved_chunks = retriever.hybrid_search(user_query, top_k=top_k)

        # 2. Layer 3 Python Guardrail Check
        query_entities = extract_query_entities(user_query)
        if query_entities:
            target = query_entities[0].upper()
            found_in_context = verify_evidence_presence(target, retrieved_chunks)

            if not found_in_context:
                logger.warning(f"Guardrail Tripped: Entity '{target}' absent from Top-{top_k} chunks.")
                return jsonify({
                    "query": user_query,
                    "dataset": dataset_name,
                    "model": "Deterministic Guardrail (Bypassed LLM)",
                    "answer": (
                        f"⚠️ **AUDIT REFUSAL (Zero-Hallucination Guardrail)**\n\n"
                        f"The requested identifier **`{target}`** was not found in the Top-{top_k} retrieved chunks.\n\n"
                        f"To prevent hallucinating inaccurate data, the response was terminated before LLM synthesis."
                    ),
                    "retrieved_chunks": retrieved_chunks
                }), 200

        # 3. Call LLM with Verbatim Citation Protocol
        llm_answer = call_local_llm(user_query, retrieved_chunks)

        return jsonify({
            "query": user_query,
            "dataset": dataset_name,
            "model": LLM_MODEL,
            "answer": llm_answer,
            "retrieved_chunks": retrieved_chunks
        }), 200

    except Exception as err:
        logger.error(f"Error processing query: {err}")
        return jsonify({"error": str(err)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)