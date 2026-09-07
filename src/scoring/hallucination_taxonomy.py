"""Conservative evidence-based error taxonomy."""
import re

TAXONOMY = {"FABRICATION", "CONTRADICTION", "NUMERICAL_ERROR", "TEMPORAL_ERROR", "ENTITY_CONFUSION", "SOURCE_MISMATCH", "CONTEXT_MISINTERPRETATION", "OTHER", "UNKNOWN"}

def classify_hallucination(claim: str, verdict: str, evidence: str = "") -> str:
    """Return UNKNOWN whenever evidence does not establish a specific error."""
    if verdict != "CONTRADICTED":
        return "UNKNOWN"
    text = f"{claim} {evidence}"
    if re.search(r"\b\d+(?:\.\d+)?\b|\b(percent|million|billion)\b", text, re.I): return "NUMERICAL_ERROR"
    if re.search(r"\b(19|20)\d{2}\b|\b(before|after|today|yesterday)\b", text, re.I): return "TEMPORAL_ERROR"
    if re.search(r"\b(not|no|false|incorrect|refuted|contradict)", evidence, re.I): return "CONTRADICTION"
    return "OTHER"
