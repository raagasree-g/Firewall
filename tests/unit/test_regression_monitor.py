from src.evaluation.regression_monitor import compare
def test_threshold_boundaries_and_missing():
    r=compare({"accuracy":.8,"macro_f1":.8},{"accuracy":.78,"macro_f1":None},{"accuracy":.02,"macro_f1":.02})
    assert r["comparisons"][0]["status"]=="FAIL"
    assert r["comparisons"][1]["status"]=="WARN"
