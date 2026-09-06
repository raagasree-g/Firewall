"""
VeriLLM Verification Classifier Evaluator
Location: src/evaluation/evaluate_classifier.py

Calculates comprehensive classification metrics (Accuracy, Precision, Recall, Macro F1, Weighted F1,
Per-class F1, Confusion Matrix) and saves structured metrics, predictions, and plots.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


def evaluate_predictions(
    y_true: List[str],
    y_pred: List[str],
    experiment_name: str,
    records: List[Dict[str, Any]] = None,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """Compute metrics, export prediction outputs, and render confusion matrix."""
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    per_class_metrics = {}
    for idx, lbl in enumerate(labels):
        per_class_metrics[lbl] = {
            "precision": round(float(per_class_p[idx]), 4),
            "recall": round(float(per_class_r[idx]), 4),
            "f1": round(float(per_class_f1[idx]), 4),
            "support": int(per_class_supp[idx])
        }
        
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    
    metrics = {
        "experiment_name": experiment_name,
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm,
        "labels": labels
    }
    
    if save_artifacts:
        # Save JSON metrics
        metrics_file = os.path.join(METRICS_DIR, f"{experiment_name}_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            
        # Save confusion matrix plot
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"Confusion Matrix: {experiment_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{experiment_name}_confusion_matrix.png"), dpi=300)
        plt.close()
        
        # Save predictions if records provided
        if records:
            pred_file = os.path.join(PREDICTIONS_DIR, f"{experiment_name}_predictions.jsonl")
            with open(pred_file, "w", encoding="utf-8") as f:
                for idx, r in enumerate(records):
                    out = {
                        "sample_id": r.get("sample_id"),
                        "claim": r.get("claim"),
                        "evidence": r.get("evidence"),
                        "true_label": y_true[idx],
                        "predicted_label": y_pred[idx],
                        "source_dataset": r.get("source_dataset")
                    }
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    
    return metrics


def update_model_comparison_csv(all_metrics: List[Dict[str, Any]]):
    """Update results/metrics/model_comparison.csv table."""
    csv_file = os.path.join(METRICS_DIR, "model_comparison.csv")
    rows = []
    for m in all_metrics:
        rows.append({
            "experiment_name": m["experiment_name"],
            "accuracy": m["accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"]
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    print(f"[OK] Updated model comparison table: {csv_file}")
