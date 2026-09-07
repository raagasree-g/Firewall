"""Run the remaining Stage 5 baselines and bounded evaluations."""
from pathlib import Path
import json

from src.evaluation.retrieval_evaluation import fixed_subset, retrieval_metrics, end_to_end_evaluation, save_json
from src.verification.data_loader import load_fever_data, load_averitec_data
from src.verification.halueval_detector import run_halueval_baseline
from src.verification.nli_verifier import NLIVerifier

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "datasets" / "processed"
METRICS = ROOT / "results" / "metrics"
PREDICTIONS = ROOT / "results" / "predictions"


def run():
    run_halueval_baseline(PROCESSED / "HaluEval" / "halueval_normalized.jsonl")
    _, fever_val = load_fever_data(seed=42)
    _, averitec_dev = load_averitec_data()
    for name, records, corpus in [
        ("FEVER", fever_val, fever_val),
        ("AVeriTeC", averitec_dev, averitec_dev),
    ]:
        subset = fixed_subset(records, size=200, seed=42)
        metrics, rows, _ = retrieval_metrics(subset, corpus)
        metrics.update({"dataset": name, "seed": 42, "subset_size": len(subset),
                        "protocol": "A fixed 200-record subset; corpus is the corresponding held-out FEVER validation or AVeriTeC dev split."})
        save_json(metrics, METRICS / f"{name}_bounded_retrieval_metrics.json")
        with (PREDICTIONS / f"{name}_bounded_retrieval_predictions.jsonl").open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row) + "\n")
    verifier = NLIVerifier()
    for name, records in [("FEVER", fever_val), ("AVeriTeC", averitec_dev)]:
        subset = fixed_subset(records, size=200, seed=42)
        metrics, _ = end_to_end_evaluation(subset, records, verifier, f"{name}_Bounded_Retrieval_NLI")
        metrics.update({"dataset": name, "seed": 42, "subset_size": len(subset)})
        save_json(metrics, METRICS / f"{name}_Bounded_Retrieval_NLI_metrics.json")


if __name__ == "__main__":
    run()
