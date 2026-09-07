import pytest
from src.scoring.reliability import reliability_metrics, composite_reliability_score
W={"support_rate":.35,"evidence_coverage":.2,"mean_verification_confidence":.15,"contradiction_penalty":.15,"unsupported_penalty":.1,"severe_error_penalty":.05}
def test_empty_and_mixed_metrics():
    assert reliability_metrics([])["claim_count"] == 0
    m=reliability_metrics([{"verdict":"SUPPORTED","evidence":[1],"confidence":.8},{"verdict":"CONTRADICTED","evidence":[],"severity":"HIGH","needs_human_review":True}])
    assert m["support_rate"]==.5 and m["evidence_coverage"]==.5 and m["high_severity_error_rate"]==.5
def test_composite_valid_and_invalid():
    m=reliability_metrics([{"verdict":"SUPPORTED","evidence":[1],"confidence":.8}])
    assert 0 <= composite_reliability_score(m,W) <= 100
    with pytest.raises(ValueError): composite_reliability_score(m,{"support_rate":1})
