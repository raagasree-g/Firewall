"""Transparent response and batch reliability framework metrics."""
from statistics import fmean

OUTCOMES = ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED")

def _rate(rows, predicate):
    return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else 0.0

def reliability_metrics(claims, ground_truth_hallucination_labels=None):
    """Return outcome, risk, and optional ground-truth hallucination metrics."""
    claims = list(claims or [])
    verified = [c for c in claims if c.get("verdict") in OUTCOMES]
    confidences = [float(c["confidence"]) for c in verified if c.get("confidence") is not None]
    result = {"claim_count": len(claims), "verified_claim_count": len(verified),
        "support_rate": _rate(verified, lambda c: c.get("verdict") == "SUPPORTED"),
        "contradiction_rate": _rate(verified, lambda c: c.get("verdict") == "CONTRADICTED"),
        "unsupported_rate": _rate(verified, lambda c: c.get("verdict") == "UNSUPPORTED"),
        "evidence_coverage": _rate(claims, lambda c: bool(c.get("evidence"))),
        "mean_verification_confidence": fmean(confidences) if confidences else None,
        "high_severity_error_rate": _rate(claims, lambda c: c.get("severity") == "HIGH"),
        "critical_severity_error_rate": _rate(claims, lambda c: c.get("severity") == "CRITICAL"),
        "human_review_rate": _rate(claims, lambda c: c.get("needs_human_review") is True)}
    result["ground_truth_hallucination_rate"] = (_rate(list(ground_truth_hallucination_labels), lambda x: x == "HALLUCINATED")
        if ground_truth_hallucination_labels is not None else None)
    result["metric_groups"] = {"verification_outcomes": {k: result[k] for k in ("support_rate", "contradiction_rate", "unsupported_rate", "evidence_coverage", "mean_verification_confidence", "claim_count", "verified_claim_count")},
        "rule_based_risk": {k: result[k] for k in ("high_severity_error_rate", "critical_severity_error_rate", "human_review_rate")},
        "ground_truth_hallucination": {"ground_truth_hallucination_rate": result["ground_truth_hallucination_rate"]}}
    return result

def composite_reliability_score(metrics, weights):
    """Experimental bounded score, not a ground-truth or validated measure."""
    required = {"support_rate", "evidence_coverage", "mean_verification_confidence", "contradiction_penalty", "unsupported_penalty", "severe_error_penalty"}
    if set(weights) != required or any(float(v) < 0 for v in weights.values()) or abs(sum(weights.values()) - 1) > 1e-9:
        raise ValueError("Composite weights must be non-negative, sum to 1, and contain each required component.")
    if any(metrics.get(key) is None for key in ("mean_verification_confidence", "high_severity_error_rate", "critical_severity_error_rate")): return None
    value = (weights["support_rate"] * metrics["support_rate"] + weights["evidence_coverage"] * metrics["evidence_coverage"] + weights["mean_verification_confidence"] * metrics["mean_verification_confidence"] - weights["contradiction_penalty"] * metrics["contradiction_rate"] - weights["unsupported_penalty"] * metrics["unsupported_rate"] - weights["severe_error_penalty"] * (metrics["high_severity_error_rate"] + metrics["critical_severity_error_rate"]))
    return max(0.0, min(100.0, 100.0 * value))
