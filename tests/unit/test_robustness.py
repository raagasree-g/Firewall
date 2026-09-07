from scripts.run_adversarial_evaluation import CASES
from src.retrieval import RobustTfidfEvidenceRetriever
from src.verillm_pipeline import analyze_response


class ControlledVerifier:
    """Deterministic verifier double used to test pipeline control flow, not model quality."""
    def predict_many(self, records):
        output = []
        for record in records:
            evidence = record["evidence"].lower()
            if "opened in 2018" in evidence and "did not" not in evidence and " is false" not in evidence:
                verdict = "SUPPORTED"
            elif any(token in evidence for token in ("did not", " is false", "3 errors", "2018.", "william shakespeare", "60 of 100")):
                verdict = "CONTRADICTED"
            elif "freezes at 0" in evidence or "opened in 2010" in evidence or "12 errors" in evidence:
                verdict = "SUPPORTED"
            else:
                verdict = "UNSUPPORTED"
            output.append({"claim": record["claim"], "evidence": record["evidence"], "nli_label": {"SUPPORTED": "ENTAILMENT", "CONTRADICTED": "CONTRADICTION", "UNSUPPORTED": "NEUTRAL"}[verdict], "verdict": verdict, "confidence": 0.9})
        return output


def _run_case(case):
    corpus = [{"evidence_id": f"{case['case_id']}_{i}", "evidence": evidence} for i, evidence in enumerate(case["evidence"], 1)]
    if not corpus:
        corpus = [{"evidence_id": "empty_fallback", "evidence": "Unrelated background information."}]
    return analyze_response(case["claim"], corpus, top_k=len(corpus), verifier=ControlledVerifier())["claims"][0]
def test_robust_retrieval_is_deterministic():
    r=RobustTfidfEvidenceRetriever([{"evidence_id":"a","evidence":"Earth is round."},{"evidence_id":"b","evidence":"Mars is red."}])
    assert r.retrieve("Earth is round",1)==r.retrieve("Earth is round",1)


def test_controlled_suite_has_exactly_the_required_twelve_categories():
    assert [case["category"] for case in CASES] == [
        "direct support", "direct contradiction", "unsupported claim", "weakly related evidence",
        "irrelevant evidence", "conflicting evidence", "numerical mismatch", "temporal mismatch",
        "entity confusion", "exact duplicate evidence", "multiple evidence passages", "empty/no evidence",
    ]
    assert all({"case_id", "claim", "evidence", "expected"} <= set(case) for case in CASES)


def test_controlled_cases_exercise_deterministic_verdicts():
    actual = {case["case_id"]: _run_case(case)["verdict"] for case in CASES}
    expected = {case["case_id"]: case["expected"] for case in CASES}
    assert actual == expected


def test_relevance_gate_preserves_unsupported_evidence_handling():
    result = analyze_response("Europa has liquid chocolate oceans.", [{"evidence_id": "europa", "evidence": "Europa is a moon of Jupiter."}], top_k=1, verifier=ControlledVerifier())
    assert result["claims"][0]["retrieval_score"] < 0.50
    assert result["claims"][0]["verdict"] == "UNSUPPORTED"


def test_explicit_contradiction_and_conflict_handling():
    direct = _run_case(CASES[1])
    conflict = _run_case(CASES[5])
    assert direct["verdict"] == "CONTRADICTED"
    assert conflict["verdict"] == "UNSUPPORTED"
    assert conflict["needs_human_review"] is True


def test_duplicate_evidence_does_not_change_deterministic_verdict():
    duplicate = _run_case(CASES[9])
    assert duplicate["verdict"] == "SUPPORTED"
    assert len(duplicate["evidence"]) == 2
