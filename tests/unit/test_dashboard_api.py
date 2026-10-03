from fastapi.testclient import TestClient
import pytest

from backend import app as api_module
from backend.dashboard_data import page_data


class FixedRetriever:
    def retrieve(self, claim, top_k=3):
        return [{"evidence_id": "test", "evidence": "Test evidence.", "score": 0.8}][:top_k]


class FixedVerifier:
    def predict_many(self, records):
        return [
            {
                "claim": record["claim"],
                "evidence": record["evidence"],
                "nli_label": "ENTAILMENT",
                "verdict": "SUPPORTED",
                "confidence": 0.9,
            }
            for record in records
        ]


def test_comparison_page_reads_compatible_persisted_artifact():
    data = page_data("comparison")

    assert data["comparison"]["comparable"] is True
    assert {entry["model_name"] for entry in data["comparison"]["entries"]} == {
        "TF-IDF Logistic Regression",
        "TF-IDF Linear SVM",
    }


def test_retrieval_page_uses_persisted_bounded_evaluation():
    data = page_data("retrieval")

    assert data["configuration"] == {
        "seed": 42,
        "sample_count_per_dataset": 200,
        "datasets": ["FEVER validation", "AVeriTeC dev"],
    }
    assert data["retrieval"]["FEVER"]["sample_count"] == 200
    assert data["retrieval_nli"]["AVeriTeC"]["improved"]["gated_to_unsupported"] == 164


def test_api_page_lists_registered_routes_and_request_schema():
    data = page_data("api")

    assert any(route["path"] == "/api/verify" and route["methods"] == ["POST"] for route in data["routes"])
    assert "min_retrieval_score" in data["verify_request_schema"]["properties"]


def test_verify_endpoint_reports_real_pipeline_fields(monkeypatch):
    monkeypatch.setattr(api_module, "_resolve_processed_corpus", lambda: [{"evidence_id": "test", "evidence": "Test evidence."}])
    monkeypatch.setattr(api_module, "_runtime_components", lambda: (FixedRetriever(), FixedVerifier()))
    client = TestClient(api_module.app)

    response = client.post("/api/verify", json={"response": "This is a factual claim."})

    assert response.status_code == 200
    claim = response.json()["claims"][0]
    assert claim["verdict"] == "SUPPORTED"
    assert claim["nli_evaluated"] is True
    assert claim["nli_confidence"] == 0.9
    assert claim["confidence"] == pytest.approx(0.72)
    assert response.json()["metadata"]["min_retrieval_score"] == 0.5


def test_verify_endpoint_rejects_malformed_evidence_records():
    client = TestClient(api_module.app)

    response = client.post(
        "/api/verify",
        json={"response": "This is a factual claim.", "evidence": [{"id": "test", "text": "Test evidence."}]},
    )

    assert response.status_code == 422
