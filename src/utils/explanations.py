"""Deterministic evidence-grounded user-facing explanations."""
from typing import Iterable


def build_explanation(verdict: str, evidence: Iterable[dict], conflicting: bool = False) -> str:
    """Explain a verdict without producing hidden reasoning or new facts."""
    if conflicting:
        return "Retrieved evidence gives conflicting verification signals; the claim remains unsupported."
    if verdict == "SUPPORTED":
        return "The retrieved evidence supports this claim."
    if verdict == "CONTRADICTED":
        return "The retrieved evidence contradicts this claim."
    if not list(evidence):
        return "No evidence was retrieved to verify this claim."
    return "No sufficiently relevant evidence was retrieved to verify this claim."
