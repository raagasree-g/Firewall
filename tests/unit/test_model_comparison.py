from src.evaluation.model_comparison import build_comparison
def test_comparability_detects_split_difference():
    a={"model_name":"a","metadata":{"dataset":"FEVER","task":"x","split":"v","seed":42,"retrieval_configuration":None,"evidence_configuration":"gold"}}
    b={"model_name":"b","metadata":{**a["metadata"],"split":"dev"}}
    assert build_comparison([a,b])["comparable"] is False
