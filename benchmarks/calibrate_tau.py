"""Calibrate tau_dense threshold strictly on held-out calibration queries."""
import json
import logging
import sys
from pathlib import Path

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retriever import HybridRetriever
from evaluation.retrievers import ProductionHybridRetrieverAdapter
from evaluation.models import CalibrationItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("calibrate_tau")


def main() -> None:
    cal_file = Path("benchmarks/nvidia_cfo_calibration.json")
    if not cal_file.exists():
        raise FileNotFoundError(f"Missing calibration file: {cal_file}")

    raw_cal = json.loads(cal_file.read_text(encoding="utf-8"))
    items = [CalibrationItem(**r) for r in raw_cal]
    logger.info("Loaded %d calibration queries from %s", len(items), cal_file)

    # Initialize production retriever
    r = HybridRetriever(index_dir="indices", ollama_base_url="http://localhost:11434")
    r.load_index("NVIDIA_CFO")

    probe = ProductionHybridRetrieverAdapter(r, "faiss")

    print("\n" + "=" * 80)
    print(f"{'CALIBRATION PER-QUERY DENSE COSINE':^80}")
    print("=" * 80)
    print(f"{'Idx':<4} | {'Domain':<10} | {'Max Cosine':<10} | {'Query'}")
    print("-" * 80)

    id_cos = []
    ood_cos = []
    per_query_results = []

    for i, it in enumerate(items, 1):
        res = probe.retrieve(it.query, k=1)
        cos = res.diagnostics.get("max_dense_cosine")
        if cos is None:
            raise ValueError(f"Probe returned None max_dense_cosine for '{it.query}'")
        cos = float(cos)
        domain_label = "In-Domain" if it.in_domain else "Out-of-Domain"
        if it.in_domain:
            id_cos.append(cos)
        else:
            ood_cos.append(cos)
        print(f"{i:<4} | {domain_label:<10} | {cos:<10.4f} | {it.query[:50]}")
        per_query_results.append({"index": i, "in_domain": it.in_domain, "cosine": cos, "query": it.query})

    lo = min(id_cos)
    hi = max(ood_cos)
    margin = lo - hi
    separable = lo > hi

    print("-" * 80)
    print(f"In-domain minimum cosine : {lo:.4f}")
    print(f"OOD maximum cosine       : {hi:.4f}")
    print(f"Margin (min_ID - max_OOD): {margin:.4f}")
    print(f"Separable                : {separable}")

    if separable:
        new_tau = (lo + hi) / 2.0
        status_msg = f"SEPARABLE (midpoint tau = {new_tau:.4f})"
    else:
        # Maximize balanced accuracy: 0.5 * (TPR + TNR), ties -> higher tau
        all_cos = sorted(set(id_cos + ood_cos))
        best_bacc = -1.0
        best_tau = 0.58
        for candidate in all_cos:
            tpr = sum(1 for c in id_cos if c >= candidate) / len(id_cos)
            tnr = sum(1 for c in ood_cos if c < candidate) / len(ood_cos)
            bacc = 0.5 * (tpr + tnr)
            if bacc >= best_bacc:
                best_bacc = bacc
                best_tau = candidate
        new_tau = best_tau
        status_msg = f"NOT SEPARABLE (best balanced accuracy = {best_bacc:.3f} at tau = {new_tau:.4f})"

    print(f"Result                   : {status_msg}")
    print("=" * 80 + "\n")

    old_tau = 0.58
    out_record = {
        "old_tau": old_tau,
        "new_tau": round(new_tau, 3),
        "margin": round(margin, 3),
        "separable": separable,
        "n": len(items),
        "min_in_domain": round(lo, 3),
        "max_ood": round(hi, 3),
    }

    out_file = Path("benchmarks/tau_calibration_result.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_record, f, indent=2)
    print(f"Saved calibration result to {out_file}:\n{json.dumps(out_record, indent=2)}\n")


if __name__ == "__main__":
    main()
