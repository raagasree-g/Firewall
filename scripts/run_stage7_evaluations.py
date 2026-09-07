"""Run Stage 7's fixed, before-versus-after retrieval and retrieval-to-NLI comparisons."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.retrieval_evaluation import fixed_subset, retrieval_metrics, end_to_end_evaluation, save_json
from src.retrieval import TfidfEvidenceRetriever, RobustTfidfEvidenceRetriever
from src.verification.data_loader import load_fever_data, load_averitec_data
from src.verification.nli_verifier import NLIVerifier

METRICS = ROOT / "results" / "metrics"
SEED, SIZE, THRESHOLD = 42, 200, 0.50

def _delta(improved, original):
    return round(improved - original, 4)

def _retrieval_comparison(name, records):
    subset = fixed_subset(records, SIZE, SEED)
    output = {"dataset": name, "split": "validation" if name == "FEVER" else "dev", "seed": SEED,
              "sample_count": len(subset), "methodology": "Exact whitespace-normalized gold-evidence hit in a held-out split corpus."}
    for label, cls in (("original", TfidfEvidenceRetriever), ("improved", RobustTfidfEvidenceRetriever)):
        metrics, _, _ = retrieval_metrics(subset, records, retriever_class=cls)
        output[label] = metrics
    output["absolute_delta"] = {
        "recall_at_k": {key: _delta(output["improved"]["recall_at_k"][key], output["original"]["recall_at_k"][key]) for key in ("1", "3", "5", "10")},
        "evidence_hit_rate": _delta(output["improved"]["evidence_hit_rate"], output["original"]["evidence_hit_rate"]),
        "retrieval_failure_rate": _delta(output["improved"]["retrieval_failure_rate"], output["original"]["retrieval_failure_rate"]),
    }
    return output, subset

def _nli_comparison(name, records, subset, verifier):
    output = {"dataset": name, "split": "validation" if name == "FEVER" else "dev", "seed": SEED,
              "sample_count": len(subset), "methodology": "Top-1 retrieval followed by the Stage 3 NLI verifier; improved run applies the production 0.50 relevance gate."}
    original, _ = end_to_end_evaluation(subset, records, verifier, f"Stage7_{name}_Original_Retrieval_NLI", retriever_class=TfidfEvidenceRetriever)
    improved, _ = end_to_end_evaluation(subset, records, verifier, f"Stage7_{name}_Improved_Retrieval_NLI", retriever_class=RobustTfidfEvidenceRetriever, min_retrieval_score=THRESHOLD)
    output["original"], output["improved"] = original, improved
    output["absolute_delta"] = {key: _delta(improved[key], original[key]) for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")}
    return output

def run():
    _, fever = load_fever_data(seed=SEED)
    _, averitec = load_averitec_data()
    retrieval_results, subsets = {}, {}
    for name, records in (("FEVER", fever), ("AVeriTeC", averitec)):
        retrieval_results[name], subsets[name] = _retrieval_comparison(name, records)
    save_json(retrieval_results, METRICS / "stage7_retrieval_before_after.json")
    verifier = NLIVerifier()
    nli_results = {name: _nli_comparison(name, records, subsets[name], verifier) for name, records in (("FEVER", fever), ("AVeriTeC", averitec))}
    save_json(nli_results, METRICS / "stage7_retrieval_nli_before_after.json")
    print(json.dumps({"retrieval": retrieval_results, "retrieval_nli": nli_results}, indent=2))

if __name__ == "__main__":
    run()
