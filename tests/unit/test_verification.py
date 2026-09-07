"""Deterministic unit tests for Stage 3 verification components."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_classifier
from src.evaluation.evaluate_classifier import evaluate_predictions, update_model_comparison_csv
from src.verification.data_loader import format_model_input, load_averitec_data, load_fever_data, validate_label
from src.verification.nli_verifier import NLIVerifier, resolve_offline_model


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_dataset_loading_and_reproducible_fever_split(tmp_path):
    labels = ["SUPPORTED", "CONTRADICTED", "UNSUPPORTED"] * 2
    records = [{"claim": f"claim {i}", "evidence": f"evidence {i}", "label": label} for i, label in enumerate(labels)]
    _write_jsonl(tmp_path / "FEVER" / "fever_normalized.jsonl", records)
    first_train, first_val = load_fever_data(str(tmp_path), val_size=0.5, seed=7)
    second_train, second_val = load_fever_data(str(tmp_path), val_size=0.5, seed=7)
    assert [record["claim"] for record in first_train] == [record["claim"] for record in second_train]
    assert [record["claim"] for record in first_val] == [record["claim"] for record in second_val]
    assert all("formatted_input" in record for record in first_train + first_val)


def test_averitec_loader_only_returns_train_and_dev(tmp_path):
    records = [
        {"claim": "a", "evidence": "b", "label": "SUPPORTED", "metadata": {"split": "train"}},
        {"claim": "c", "evidence": "d", "label": "CONTRADICTED", "metadata": {"split": "dev"}},
        {"claim": "e", "evidence": "f", "label": "UNSUPPORTED", "metadata": {"split": "test"}},
    ]
    _write_jsonl(tmp_path / "AVeriTeC" / "averitec_normalized.jsonl", records)
    train, dev = load_averitec_data(str(tmp_path))
    assert [record["claim"] for record in train] == ["a"]
    assert [record["claim"] for record in dev] == ["c"]


def test_label_validation_and_input_formatting():
    assert validate_label("SUPPORTED") == "SUPPORTED"
    assert validate_label("CONFLICTING_EVIDENCE") == "CONFLICTING_EVIDENCE"
    with pytest.raises(ValueError, match="Unsupported verification label"):
        validate_label("MAYBE")
    assert format_model_input("  claim ", " evidence  ") == "Claim: claim [SEP] Evidence: evidence"
    assert NLIVerifier.format_pair(" claim ", " evidence ") == ("evidence", "claim")


def test_offline_model_resolution_falls_back_when_cache_is_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert resolve_offline_model("org/model") == "org/model"


@patch("src.verification.nli_verifier.get_cross_encoder")
def test_nli_prediction_schema(mock_get_cross_encoder):
    model = MagicMock()
    model.model.config.id2label = {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}
    model.predict.return_value = [[0.1, 0.8, 0.1]]
    mock_get_cross_encoder.return_value.return_value = model
    result = NLIVerifier("test-model").predict("A claim", "An evidence")
    assert result["claim"] == "A claim"
    assert result["evidence"] == "An evidence"
    assert result["nli_label"] == "ENTAILMENT"
    assert result["verdict"] == "SUPPORTED"
    assert result["confidence"] == pytest.approx(0.5017, abs=0.0001)


def test_metric_calculation_is_deterministic():
    metrics = evaluate_predictions(["SUPPORTED", "CONTRADICTED"], ["SUPPORTED", "UNSUPPORTED"], "unit", save_artifacts=False)
    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"] == [[0, 0, 1], [0, 1, 0], [0, 0, 0]]


def test_model_comparison_preserves_existing_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_classifier, "METRICS_DIR", str(tmp_path))
    (tmp_path / "model_comparison.csv").write_text(
        "experiment_name,accuracy,macro_precision,macro_recall,macro_f1,weighted_f1\n"
        "Classical,0.8,0.8,0.8,0.8,0.8\n",
        encoding="utf-8",
    )
    update_model_comparison_csv([{
        "experiment_name": "NLI_FEVER_InDomain", "accuracy": 0.2,
        "macro_precision": 0.2, "macro_recall": 0.2, "macro_f1": 0.2,
        "weighted_f1": 0.2,
    }])
    rows = (tmp_path / "model_comparison.csv").read_text(encoding="utf-8")
    assert "Classical" in rows
    assert "NLI_FEVER_InDomain" in rows
