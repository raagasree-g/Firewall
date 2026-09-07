"""Rule-based regression monitor; it does not infer statistical significance."""
import math

LOWER_IS_WORSE = {"accuracy", "macro_f1", "weighted_f1", "support_rate", "evidence_coverage", "mean_confidence"}
HIGHER_IS_WORSE = {"contradiction_rate", "unsupported_rate", "high_severity_rate", "critical_severity_rate", "human_review_rate"}

def compare(baseline, candidate, thresholds):
    rows=[]
    for metric, threshold in thresholds.items():
        base, cand = baseline.get(metric), candidate.get(metric)
        if base is None or cand is None:
            rows.append({"metric": metric, "baseline_value": base, "candidate_value": cand, "status": "WARN", "reason": "Metric unavailable"}); continue
        delta = cand-base
        worsening = -delta if metric in LOWER_IS_WORSE else delta if metric in HIGHER_IS_WORSE else 0
        status = "FAIL" if worsening > threshold else "PASS"
        rows.append({"metric": metric, "baseline_value": base, "candidate_value": cand, "absolute_delta": delta,
                     "relative_delta": (delta / abs(base)) if base else None, "threshold": threshold, "status": status})
    overall = "FAIL" if any(r["status"] == "FAIL" for r in rows) else "WARN" if any(r["status"] == "WARN" for r in rows) else "PASS"
    return {"overall_regression_status": overall, "comparisons": rows, "methodology": "Configured absolute thresholds only; no significance test is claimed."}
