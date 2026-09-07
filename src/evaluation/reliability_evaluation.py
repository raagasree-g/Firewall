"""Build non-destructive Stage 6 reliability reports from prediction artifacts."""
import json
from pathlib import Path
import yaml
from src.scoring.reliability import reliability_metrics, composite_reliability_score
ROOT = Path(__file__).resolve().parents[2]

def load_jsonl(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]

def report_from_predictions(path, dataset, task, metadata=None):
    rows = load_jsonl(path)
    claims = [{**row, "verdict": row.get("predicted_label")} for row in rows]
    labels = [row.get("true_label") for row in rows] if task == "RESPONSE_HALLUCINATION_DETECTION" else None
    metrics = reliability_metrics(claims, labels)
    if rows and not any("severity" in row for row in rows):
        metrics["high_severity_error_rate"] = None
        metrics["critical_severity_error_rate"] = None
        metrics["human_review_rate"] = None
        metrics["metric_groups"]["rule_based_risk"] = {key: metrics[key] for key in ("high_severity_error_rate", "critical_severity_error_rate", "human_review_rate")}
    config = yaml.safe_load((ROOT / "configs" / "reliability_config.yaml").read_text(encoding="utf-8"))
    metrics["composite_reliability_score"] = composite_reliability_score(metrics, config["composite"]["weights"]) if config["composite"]["enabled"] else None
    return {"report_version": config["version"], "dataset": dataset, "task": task, "sample_count": len(rows), "metadata": metadata or {}, "reliability": metrics,
            "availability_notes": {"severity": "Unavailable in historical prediction artifacts unless stored by pipeline.", "ground_truth_hallucination": "Only populated for response-level hallucination benchmark labels."}}

def save_report(report, filename):
    path = ROOT / "results" / "metrics" / filename
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
