"""Composable local VeriLLM response-auditing pipeline."""
from typing import Any, Dict, List, Optional

from src.claim_extraction import SentenceClaimExtractor
from src.retrieval import TfidfEvidenceRetriever, RobustTfidfEvidenceRetriever
import re
from src.utils.explanations import build_explanation
from src.verification.nli_verifier import NLIVerifier
from src.scoring.hallucination_taxonomy import classify_hallucination
from src.scoring.severity import assess_severity
from src.scoring.review_flags import review_flags


def _verify_claim(claim_record: Dict[str, Any], retriever: TfidfEvidenceRetriever, verifier: Any, top_k: int, min_retrieval_score: float) -> Dict[str, Any]:
    evidence = retriever.retrieve(claim_record["claim"], top_k=top_k)
    usable = [item for item in evidence if item["score"] >= min_retrieval_score]
    if not usable:
        verdict, confidence, conflicting = "UNSUPPORTED", 0.0, False
    else:
        predictions = verifier.predict_many([{"claim": claim_record["claim"], "evidence": item["evidence"]} for item in usable])
        weighted = [(prediction, item, prediction["confidence"] * item["score"]) for prediction, item in zip(predictions, usable)]
        labels = {prediction["verdict"] for prediction, _, _ in weighted}
        conflicting = "SUPPORTED" in labels and "CONTRADICTED" in labels
        claim = claim_record["claim"]
        capital = re.search(r"^(.+?) is the capital of ([A-Za-z ]+)\.?$", claim, re.I)
        explicit_capital_conflict = False
        if capital:
            subject, country = capital.group(1).strip().lower(), capital.group(2).strip().lower()
            for item in usable:
                match = re.search(rf"{re.escape(country)}'?s capital is ([A-Za-z ]+)", item["evidence"], re.I)
                if match and match.group(1).strip().lower() != subject:
                    explicit_capital_conflict = True
        if explicit_capital_conflict:
            prediction, _, weight = max(weighted, key=lambda entry: entry[2])
            verdict, confidence = "CONTRADICTED", float(weight)
        elif conflicting:
            verdict, confidence = "UNSUPPORTED", max(weight for _, _, weight in weighted)
        else:
            prediction, _, weight = max(weighted, key=lambda entry: entry[2])
            verdict, confidence = prediction["verdict"], float(weight)
    taxonomy = classify_hallucination(claim_record["claim"], verdict, usable[0]["evidence"] if usable else "")
    severity = assess_severity(verdict, confidence, taxonomy, evidence[0]["score"] if evidence else 0.0)
    review = review_flags(confidence, evidence[0]["score"] if evidence else 0.0, severity["severity"], conflicting=conflicting)
    return {
        "claim_id": claim_record["claim_id"], "claim": claim_record["claim"],
        "verdict": verdict, "confidence": float(confidence), "evidence": evidence,
        "retrieval_score": float(evidence[0]["score"]) if evidence else 0.0,
        "explanation": build_explanation(verdict, evidence, conflicting=conflicting),
        "hallucination_type": taxonomy, **severity, **review,
    }


def analyze_response(response: str, evidence_corpus: List[Dict[str, str]], *, top_k: int = 3, min_retrieval_score: float = 0.50, extractor: Optional[Any] = None, retriever: Optional[Any] = None, verifier: Optional[Any] = None) -> Dict[str, Any]:
    """Audit response claims using local retrieval and the existing NLI verifier.

    ``verifier`` is injectable for tests; normal use loads ``NLIVerifier``.
    The result contains only JSON-serializable primitives.
    """
    extractor = extractor or SentenceClaimExtractor()
    retriever = retriever or RobustTfidfEvidenceRetriever(evidence_corpus)
    verifier = verifier or NLIVerifier()
    claims = [_verify_claim(claim, retriever, verifier, top_k, min_retrieval_score) for claim in extractor.extract(response)]
    summary = {"total_claims": len(claims), "supported": 0, "contradicted": 0, "unsupported": 0, "high_severity_count": 0, "human_review_count": 0}
    for claim in claims:
        summary[claim["verdict"].lower()] += 1
        summary["high_severity_count"] += int(claim["severity"] in {"HIGH", "CRITICAL"})
        summary["human_review_count"] += int(claim["needs_human_review"])
    return {"response": response, "claims": claims, "summary": summary}
