"""Generate Stage 6 reports, comparison, regression monitor, and plots from existing artifacts."""
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.reliability_evaluation import report_from_predictions, save_report
from src.evaluation.model_comparison import build_comparison, save_comparison
from src.evaluation.regression_monitor import compare

METRICS=ROOT/"results"/"metrics"
def metric(name): return json.loads((METRICS/name).read_text(encoding="utf-8"))
def run():
    fever=report_from_predictions(ROOT/"results/predictions/FEVER_Bounded_Retrieval_NLI_predictions.jsonl","FEVER","CLAIM_VERIFICATION",{"split":"validation","seed":42,"retrieval_configuration":"TF-IDF top-1"})
    halu=report_from_predictions(ROOT/"results/predictions/HaluEval_TFIDF_LogReg_predictions.jsonl","HaluEval","RESPONSE_HALLUCINATION_DETECTION",{"split":"stratified holdout","seed":42})
    save_report(fever,"stage6_fever_reliability_report.json"); save_report(halu,"stage6_halueval_reliability_report.json")
    entries=[]
    for filename, model in [("FEVER_LogReg_InDomain_metrics.json","TF-IDF Logistic Regression"),("FEVER_LinearSVM_InDomain_metrics.json","TF-IDF Linear SVM")]:
        entries.append({"model_name":model,"model_version":"Stage 3","metrics":metric(filename),"metadata":{"dataset":"FEVER","task":"CLAIM_VERIFICATION","split":"validation","seed":42,"retrieval_configuration":None,"evidence_configuration":"gold normalized evidence"}})
    comp=build_comparison(entries); save_comparison(comp,ROOT/"results/comparisons","fever_in_domain_models")
    thresholds=yaml.safe_load((ROOT/"configs/regression_config.yaml").read_text())["thresholds"]
    b=metric("FEVER_LogReg_InDomain_metrics.json"); c=metric("FEVER_LinearSVM_InDomain_metrics.json")
    report=compare(b,c,thresholds); (ROOT/"results/metrics/stage6_fever_logreg_to_svm_regression.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    plots=ROOT/"results/plots"; plots.mkdir(parents=True,exist_ok=True)
    plt.bar([e["model_name"] for e in entries],[e["metrics"]["macro_f1"] for e in entries]); plt.ylabel("Macro F1"); plt.title("FEVER in-domain model comparison"); plt.tight_layout(); plt.savefig(plots/"fever_model_macro_f1.png",dpi=150); plt.close()
    r=fever["reliability"]; plt.bar(["Supported","Contradicted","Unsupported"],[r["support_rate"],r["contradiction_rate"],r["unsupported_rate"]]); plt.ylim(0,1); plt.ylabel("Rate"); plt.title("FEVER retrieval-to-NLI outcomes"); plt.tight_layout(); plt.savefig(plots/"fever_reliability_outcomes.png",dpi=150); plt.close()
    keys=["accuracy","macro_f1","weighted_f1"]; plt.bar(keys,[c[k]-b[k] for k in keys]); plt.axhline(0,color="black",linewidth=.8); plt.ylabel("SVM minus LogReg"); plt.title("FEVER model metric deltas"); plt.tight_layout(); plt.savefig(plots/"fever_regression_deltas.png",dpi=150); plt.close()
if __name__=="__main__": run()
