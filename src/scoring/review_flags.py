"""Configurable review routing based on interpretable risk conditions."""
DEFAULT_REVIEW_CONFIG={"low_confidence":.5,"weak_retrieval":.1}
def review_flags(confidence, retrieval_score, severity, conflicting=False, ambiguous=False, config=None, confidence_available=True):
    cfg={**DEFAULT_REVIEW_CONFIG,**(config or {})}; reasons=[]
    if confidence_available and confidence < cfg["low_confidence"]: reasons.append("Low NLI model confidence.")
    if not confidence_available: reasons.append("NLI verification was not run because no evidence met the retrieval threshold.")
    if retrieval_score < cfg["weak_retrieval"]: reasons.append("Weak retrieval relevance.")
    if conflicting: reasons.append("Retrieved evidence has conflicting verification signals.")
    if ambiguous: reasons.append("Evidence is ambiguous.")
    if severity in {"HIGH","CRITICAL"}: reasons.append(f"{severity.title()} severity requires review.")
    return {"needs_human_review":bool(reasons),"review_reasons":reasons}
