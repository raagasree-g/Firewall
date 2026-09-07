"""Transparent, non-ground-truth severity rules."""
def assess_severity(verdict, confidence, hallucination_type="UNKNOWN", evidence_strength=0.0, claim_importance="normal"):
    reasons=[]
    if verdict == "CONTRADICTED": reasons.append("Retrieved evidence contradicts the claim.")
    if hallucination_type in {"NUMERICAL_ERROR", "TEMPORAL_ERROR", "ENTITY_CONFUSION"}: reasons.append(f"{hallucination_type} can materially change interpretation.")
    if confidence >= .8: reasons.append("The verifier assigned high model confidence.")
    if evidence_strength >= .5: reasons.append("Retrieval relevance is strong.")
    if verdict == "CONTRADICTED" and claim_importance == "high": severity="CRITICAL"
    elif verdict == "CONTRADICTED" and (confidence >= .8 or evidence_strength >= .5): severity="HIGH"
    elif verdict == "CONTRADICTED" or verdict == "UNSUPPORTED": severity="MEDIUM"
    else: severity="LOW"
    return {"severity": severity, "severity_reasons": reasons or ["No elevated evidence-based risk factor was detected."]}
