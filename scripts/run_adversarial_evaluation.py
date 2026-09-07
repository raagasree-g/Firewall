"""Synthetic deterministic stress cases; these are framework checks, never benchmark results."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.verillm_pipeline import analyze_response
from src.verification.nli_verifier import NLIVerifier

CASES=[
 {"case_id":"R01","category":"direct support","claim":"Water freezes at 0 degrees Celsius at standard pressure.","evidence":["At standard atmospheric pressure, water freezes at 0 degrees Celsius."],"expected":"SUPPORTED"},
 {"case_id":"R02","category":"direct contradiction","claim":"Paris is the capital of Germany.","evidence":["France's capital is Paris; Germany's capital is Berlin."],"expected":"CONTRADICTED"},
 {"case_id":"R03","category":"unsupported claim","claim":"Europa has confirmed oceans of liquid chocolate.","evidence":["Europa is a moon of Jupiter."],"expected":"UNSUPPORTED"},
 {"case_id":"R04","category":"weakly related evidence","claim":"The company earned $10 million in 2024.","evidence":["The company published a weather forecast in 2024."],"expected":"UNSUPPORTED"},
 {"case_id":"R05","category":"irrelevant evidence","claim":"The company earned $10 million.","evidence":["Penguins live in the Southern Hemisphere."],"expected":"UNSUPPORTED"},
 {"case_id":"R06","category":"conflicting evidence","claim":"The report contains 12 errors.","evidence":["The report contains 12 errors.","The report contains 3 errors."],"expected":"UNSUPPORTED"},
 {"case_id":"R07","category":"numerical mismatch","claim":"The report contains 12 errors.","evidence":["The report contains 3 errors."],"expected":"CONTRADICTED"},
 {"case_id":"R08","category":"temporal mismatch","claim":"The treaty was signed in 2020.","evidence":["The treaty was signed in 2018."],"expected":"CONTRADICTED"},
 {"case_id":"R09","category":"entity confusion","claim":"Marie Curie wrote Hamlet.","evidence":["Marie Curie did not write Hamlet. Hamlet was written by William Shakespeare."],"expected":"CONTRADICTED"},
 {"case_id":"R10","category":"exact duplicate evidence","claim":"Water freezes at 0 degrees Celsius at standard pressure.","evidence":["At standard atmospheric pressure, water freezes at 0 degrees Celsius.","At standard atmospheric pressure, water freezes at 0 degrees Celsius."],"expected":"SUPPORTED"},
 {"case_id":"R11","category":"multiple evidence passages","claim":"The museum opened in 2010.","evidence":["The museum opened in 2010 after construction completed.","The museum has a cafe and a library."],"expected":"SUPPORTED"},
 {"case_id":"R12","category":"empty/no evidence","claim":"A newly discovered planet has rings.","evidence":[],"expected":"UNSUPPORTED"},
]
def run():
    verifier=NLIVerifier(); out=[]
    for case in CASES:
        corpus=[{"evidence_id":f"{case['case_id']}_{index}","evidence":text} for index,text in enumerate(case["evidence"],1)]
        # A one-document corpus cannot be constructed for the empty-evidence case.
        if not corpus:
            corpus=[{"evidence_id":f"{case['case_id']}_irrelevant","evidence":"Unrelated background information."}]
        result=analyze_response(case["claim"],corpus,top_k=len(corpus),verifier=verifier)
        actual=result["claims"][0]
        out.append({"synthetic":True,"case_id":case["case_id"],"category":case["category"],"claim":case["claim"],"input_evidence":case["evidence"],"expected_verdict":case["expected"],"actual_verdict":actual["verdict"],"confidence":actual["confidence"],"retrieval_information":actual["evidence"],"explanation":actual["explanation"],"pass":actual["verdict"]==case["expected"]})
    path=ROOT/"results/evaluation/robustness"; path.mkdir(parents=True,exist_ok=True)
    (path/"stage7_controlled_robustness_results.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps({"cases":len(out),"passed":sum(x["pass"] for x in out)},indent=2))
if __name__=="__main__": run()
