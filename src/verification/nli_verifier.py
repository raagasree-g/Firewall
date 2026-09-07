"""Reusable NLI verifier for VeriLLM Stage 3.

Evidence is the NLI premise and the claim is the NLI hypothesis. Mapping:
ENTAILMENT -> SUPPORTED; CONTRADICTION -> CONTRADICTED; NEUTRAL -> UNSUPPORTED.
This operational mapping does not make source annotation schemes identical.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from src.evaluation.evaluate_classifier import evaluate_predictions, update_model_comparison_csv
from src.verification.data_loader import load_averitec_data, load_fever_data

with open(os.path.join(PROJECT_ROOT, "configs", "verification_config.yaml"), encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)

NLI_TO_VERDICT = {"ENTAILMENT": "SUPPORTED", "CONTRADICTION": "CONTRADICTED", "NEUTRAL": "UNSUPPORTED"}


def get_cross_encoder():
    """Import lazily so utility code and tests do not require model loading."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise RuntimeError("sentence-transformers is required for NLI verification") from error
    return CrossEncoder


def resolve_offline_model(model_name: str) -> str:
    """Use a complete locally cached Hugging Face snapshot when available.

    Stage evaluations must be reproducible and must not download models.  The
    returned path is still the same model revision named in configuration.
    """
    cache_root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    model_cache = cache_root / ("models--" + model_name.replace("/", "--"))
    revision_file = model_cache / "refs" / "main"
    if revision_file.exists():
        snapshot = model_cache / "snapshots" / revision_file.read_text(encoding="utf-8").strip()
        if (snapshot / "config.json").exists() and (snapshot / "model.safetensors").exists():
            return str(snapshot)
    return model_name


class NLIVerifier:
    """Classify arbitrary evidence/claim pairs with the mandated NLI model."""

    def __init__(self, model_name: str = CONFIG["nli_model"]["name"], max_length: int = CONFIG["nli_model"]["max_length"]):
        self.model_name = model_name
        model_source = resolve_offline_model(model_name)
        self.model = get_cross_encoder()(model_source, max_length=max_length, local_files_only=True)
        self.id2label = {int(index): str(label).upper() for index, label in self.model.model.config.id2label.items()}
        unknown = set(self.id2label.values()) - set(NLI_TO_VERDICT)
        if unknown:
            raise ValueError(f"Unexpected NLI labels from {model_name}: {sorted(unknown)}")

    @staticmethod
    def format_pair(claim: str, evidence: str) -> tuple[str, str]:
        """Format a pair as (premise=evidence, hypothesis=claim)."""
        return ((evidence or "").strip(), (claim or "").strip())

    def predict(self, claim: str, evidence: str) -> Dict[str, Any]:
        return self.predict_many([{"claim": claim, "evidence": evidence}])[0]

    def predict_many(self, records: Sequence[Dict[str, Any]], batch_size: int = CONFIG["nli_model"]["batch_size"]) -> List[Dict[str, Any]]:
        pairs = [self.format_pair(record.get("claim", ""), record.get("evidence", "")) for record in records]
        if not pairs:
            return []
        raw_scores = np.asarray(self.model.predict(pairs, batch_size=batch_size, show_progress_bar=True))
        # CrossEncoder returns raw logits for this model. Convert them to
        # probabilities before exposing the maximum as a confidence value.
        shifted = raw_scores - np.max(raw_scores, axis=1, keepdims=True)
        scores = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        indices, confidences = np.argmax(scores, axis=1), np.max(scores, axis=1)
        outputs = []
        for record, index, confidence in zip(records, indices, confidences):
            nli_label = self.id2label[int(index)]
            outputs.append({"claim": record.get("claim", ""), "evidence": record.get("evidence", ""), "nli_label": nli_label, "verdict": NLI_TO_VERDICT[nli_label], "confidence": float(confidence)})
        return outputs


def run_nli_evaluation(eval_records: List[Dict[str, Any]], verifier: NLIVerifier, experiment_name: str) -> Dict[str, Any]:
    """Evaluate a reusable verifier and persist measured prediction artifacts."""
    predictions = verifier.predict_many(eval_records)
    return evaluate_predictions(
        y_true=[record["label"] for record in eval_records],
        y_pred=[prediction["verdict"] for prediction in predictions],
        experiment_name=experiment_name,
        records=eval_records,
        prediction_details=predictions,
    )


def main() -> None:
    verifier = NLIVerifier()
    _, fever_validation = load_fever_data(seed=CONFIG["seed"])
    _, averitec_dev = load_averitec_data()
    run_nli_evaluation(fever_validation, verifier, "NLI_FEVER_InDomain")
    run_nli_evaluation(averitec_dev, verifier, "NLI_AVeriTeC_InDomain")
    metrics_dir = os.path.join(PROJECT_ROOT, "results", "metrics")
    all_metrics = []
    for filename in os.listdir(metrics_dir):
        if filename.endswith("_metrics.json"):
            with open(os.path.join(metrics_dir, filename), encoding="utf-8") as metrics_file:
                all_metrics.append(json.load(metrics_file))
    update_model_comparison_csv(all_metrics)


if __name__ == "__main__":
    main()
