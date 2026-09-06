"""
VeriLLM Classical ML Verification Baseline & Cross-Dataset Experiments
Location: src/verification/tfidf_baseline.py

Trains TF-IDF + Logistic Regression and TF-IDF + Linear SVM models on FEVER and AVeriTeC,
evaluates in-domain performance, and executes cross-dataset generalization experiments.
"""

import sys
import os
import json
import yaml
import numpy as np
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.verification.data_loader import load_fever_data, load_averitec_data
from src.evaluation.evaluate_classifier import evaluate_predictions, update_model_comparison_csv

CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "verification_config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


def run_experiment(
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
    model_type: str,
    exp_name: str
) -> Dict[str, Any]:
    print(f"\n" + "=" * 70)
    print(f"  RUNNING EXPERIMENT: {exp_name} ({model_type.upper()})")
    print(f"  Train Count: {len(train_records):,} | Eval Count: {len(eval_records):,}")
    print("=" * 70)

    X_train = [r["formatted_input"] for r in train_records]
    y_train = [r["label"] for r in train_records]
    
    X_eval = [r["formatted_input"] for r in eval_records]
    y_eval = [r["label"] for r in eval_records]

    # Fit TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(config["tfidf"]["ngram_range"]),
        max_features=config["tfidf"]["max_features"],
        sublinear_tf=config["tfidf"]["sublinear_tf"],
        min_df=config["tfidf"]["min_df"]
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_eval_vec = vectorizer.transform(X_eval)

    # Initialize Classifier
    if model_type == "logistic_regression":
        model = LogisticRegression(
            C=config["logistic_regression"]["C"],
            max_iter=config["logistic_regression"]["max_iter"],
            class_weight=config["logistic_regression"]["class_weight"],
            random_state=config["seed"]
        )
    elif model_type == "linear_svm":
        model = LinearSVC(
            C=config["linear_svm"]["C"],
            max_iter=config["linear_svm"]["max_iter"],
            class_weight=config["linear_svm"]["class_weight"],
            random_state=config["seed"]
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_eval_vec)

    metrics = evaluate_predictions(
        y_true=y_eval,
        y_pred=list(y_pred),
        experiment_name=exp_name,
        records=eval_records
    )
    
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro F1 : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    return metrics


def main():
    print("=" * 70)
    print("  VERILLM CLASSICAL ML BASELINE & CROSS-DATASET EVALUATION")
    print("=" * 70)

    fever_train, fever_val = load_fever_data(seed=config["seed"])
    averitec_train, averitec_dev = load_averitec_data()

    all_metrics = []

    # 1. FEVER In-Domain
    m1 = run_experiment(fever_train, fever_val, "logistic_regression", "FEVER_LogReg_InDomain")
    all_metrics.append(m1)
    
    m2 = run_experiment(fever_train, fever_val, "linear_svm", "FEVER_LinearSVM_InDomain")
    all_metrics.append(m2)

    # 2. AVeriTeC In-Domain
    m3 = run_experiment(averitec_train, averitec_dev, "logistic_regression", "AVeriTeC_LogReg_InDomain")
    all_metrics.append(m3)
    
    m4 = run_experiment(averitec_train, averitec_dev, "linear_svm", "AVeriTeC_LinearSVM_InDomain")
    all_metrics.append(m4)

    # 3. Cross-Dataset: Train FEVER -> Eval AVeriTeC Dev
    m5 = run_experiment(fever_train, averitec_dev, "logistic_regression", "CrossDS_FEVER_to_AVeriTeC_LogReg")
    all_metrics.append(m5)

    # 4. Cross-Dataset: Train AVeriTeC -> Eval FEVER Val
    m6 = run_experiment(averitec_train, fever_val, "logistic_regression", "CrossDS_AVeriTeC_to_FEVER_LogReg")
    all_metrics.append(m6)

    update_model_comparison_csv(all_metrics)
    print("\n[OK] Classical ML baselines and cross-dataset evaluations complete!")


if __name__ == "__main__":
    main()
