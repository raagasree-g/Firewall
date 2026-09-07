import json
import pytest

from src.claim_extraction import SentenceClaimExtractor
from src.retrieval import TfidfEvidenceRetriever, load_processed_evidence
from src.utils.explanations import build_explanation
from src.verillm_pipeline import analyze_response
from src.evaluation.confidence_analysis import analyze_confidence
from src.evaluation.retrieval_evaluation import fixed_subset, retrieval_metrics
from src.scoring.hallucination_taxonomy import classify_hallucination
from src.scoring.severity import assess_severity
from src.scoring.review_flags import review_flags
from src.verification.ragtruth_analyzer import analyze_ragtruth


class StubVerifier:
    def predict_many(self, records):
        return [{"claim": r["claim"], "evidence": r["evidence"], "nli_label": "ENTAILMENT", "verdict": "SUPPORTED", "confidence": 0.9} for r in records]


def test_claim_extraction_handles_sentences_empty_and_duplicates():
    extractor = SentenceClaimExtractor()
    assert extractor.extract("") == []
    claims = extractor.extract("Earth is round. Mars is red. Earth is round.")
    assert [claim["claim"] for claim in claims] == ["Earth is round.", "Mars is red."]
    assert [claim["claim_id"] for claim in claims] == ["claim_001", "claim_002"]


def test_processed_corpus_loading_and_deduplication(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        {"sample_id": "one", "evidence": "  Earth is round. "},
        {"sample_id": "two", "evidence": "Earth is round."},
        {"sample_id": "three", "evidence": "Mars is red."},
    ]) + "\n", encoding="utf-8")
    assert load_processed_evidence([path]) == [
        {"evidence_id": "one", "evidence": "Earth is round."},
        {"evidence_id": "three", "evidence": "Mars is red."},
    ]


def test_retrieval_top_k_behavior():
    retriever = TfidfEvidenceRetriever([
        {"evidence_id": "earth", "evidence": "The Earth is round."},
        {"evidence_id": "mars", "evidence": "Mars is red."},
    ])
    results = retriever.retrieve("Earth is round", top_k=1)
    assert len(results) == 1
    assert results[0]["evidence_id"] == "earth"


def test_explanations_are_evidence_grounded():
    assert build_explanation("SUPPORTED", [{"evidence": "x"}]) == "The retrieved evidence supports this claim."
    assert "conflicting" in build_explanation("UNSUPPORTED", [{"evidence": "x"}], conflicting=True).lower()
    assert "No evidence" in build_explanation("UNSUPPORTED", [])


def test_end_to_end_pipeline_schema_with_injected_verifier():
    corpus = [{"evidence_id": "earth", "evidence": "The Earth is round."}]
    result = analyze_response("Earth is round.", corpus, top_k=1, verifier=StubVerifier())
    assert result["summary"] == {"total_claims": 1, "supported": 1, "contradicted": 0, "unsupported": 0, "high_severity_count": 0, "human_review_count": 0}
    claim = result["claims"][0]
    assert {"claim_id", "claim", "verdict", "confidence", "evidence", "retrieval_score", "explanation", "hallucination_type", "severity", "severity_reasons", "needs_human_review", "review_reasons"} <= set(claim)
    assert json.loads(json.dumps(result))["claims"][0]["verdict"] == "SUPPORTED"


def test_retrieval_metrics_are_deterministic_and_define_failures():
    records = [
        {"sample_id": "a", "claim": "Earth shape", "evidence": "Earth is round.", "label": "SUPPORTED"},
        {"sample_id": "b", "claim": "Mars colour", "evidence": "Mars is red.", "label": "SUPPORTED"},
    ]
    metrics, _, _ = retrieval_metrics(fixed_subset(records, 2), records)
    assert metrics["recall_at_k"]["1"] == 1.0
    assert metrics["retrieval_failure_rate"] == 0.0


def test_stage5_scoring_and_confidence_components(tmp_path):
    assert classify_hallucination("x", "UNSUPPORTED", "") != "HALLUCINATION"
    assert assess_severity("CONTRADICTED", .9, "FACTUAL_CONTRADICTION", .9)["severity"] in {"HIGH", "CRITICAL"}
    assert review_flags(.1, .1, "LOW")["needs_human_review"] is True
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(json.dumps({"true_label": "SUPPORTED", "predicted_label": "SUPPORTED", "confidence": .9}) + "\n", encoding="utf-8")
    assert analyze_confidence([pred])["brier_score"] == pytest.approx(0.01)


def test_ragtruth_analysis_preserves_unknown_type(tmp_path):
    path = tmp_path / "ragtruth.jsonl"
    path.write_text(json.dumps({"label": "HALLUCINATED", "metadata": {"hallucination_spans": [{"start": 1, "end": 4}]}}) + "\n", encoding="utf-8")
    result = analyze_ragtruth(path)
    assert result["hallucination_type_distribution"] == {"UNKNOWN": 1}
