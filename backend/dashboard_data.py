"""Read-only API data assembled from VeriLLM source artifacts."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
PREDICTIONS = RESULTS / "predictions"
PROCESSED = ROOT / "datasets" / "processed"
EVALUATION = RESULTS / "evaluation"

DATASET_FILES = {
    "FEVER": "FEVER/fever_normalized.jsonl",
    "AVeriTeC": "AVeriTeC/averitec_normalized.jsonl",
    "HaluEval": "HaluEval/halueval_normalized.jsonl",
    "RAGTruth": "RAGTruth/ragtruth_normalized.jsonl",
}

DOC_FILES = {
    "Core pipeline": "architecture/core_pipeline.md",
    "Dataset strategy": "architecture/dataset_strategy.md",
    "Verification strategy": "architecture/verification_strategy.md",
    "Hallucination analysis": "architecture/hallucination_analysis.md",
    "Reliability scoring": "architecture/reliability_scoring.md",
    "Model comparison": "architecture/model_comparison.md",
    "Regression monitoring": "architecture/regression_monitoring.md",
    "Retrieval and NLI robustness": "architecture/retrieval_nli_robustness.md",
}

PIPELINE_NODES = [
    {
        "id": "response",
        "label": "LLM RESPONSE",
        "purpose": "Accept a generated response for reliability analysis.",
        "implementation": "FastAPI request validation and the response passed to analyze_response.",
        "source": "backend/app.py",
        "inputs": "Response text and optional evaluation settings.",
        "outputs": "Response-level analysis containing extracted claims.",
    },
    {
        "id": "claims",
        "label": "CLAIM EXTRACTION",
        "purpose": "Extract ordered, de-duplicated sentence claims.",
        "implementation": "SentenceClaimExtractor; this is sentence segmentation, not guaranteed atomic decomposition.",
        "source": "src/claim_extraction/sentence_extractor.py",
        "inputs": "Response text.",
        "outputs": "Claim IDs, sentence text, and source positions.",
    },
    {
        "id": "retrieval",
        "label": "RETRIEVAL",
        "purpose": "Rank processed evidence by lexical TF-IDF cosine similarity.",
        "implementation": "RobustTfidfEvidenceRetriever using unigrams/bigrams, stop-word filtering, and accent normalization.",
        "source": "src/retrieval/tfidf_retriever.py",
        "inputs": "Claim text and the loaded processed evidence corpus.",
        "outputs": "Ranked evidence records and retrieval similarity scores.",
    },
    {
        "id": "evidence",
        "label": "EVIDENCE",
        "purpose": "Retain retrieved passages with their corpus IDs and similarity scores.",
        "implementation": "Retrieved evidence records returned by the TF-IDF retriever.",
        "source": "src/retrieval/tfidf_retriever.py",
        "inputs": "Claim and corpus.",
        "outputs": "Ranked passages; retrieval does not establish truth.",
    },
    {
        "id": "nli",
        "label": "NLI VERIFICATION",
        "purpose": "Classify evidence-premise and claim-hypothesis pairs.",
        "implementation": "cross-encoder/nli-distilroberta-base; only passages at or above the relevance gate are evaluated.",
        "source": "src/verification/nli_verifier.py",
        "inputs": "Claim/evidence pairs meeting the configured relevance threshold.",
        "outputs": "ENTAILMENT, CONTRADICTION, or NEUTRAL with model confidence.",
    },
    {
        "id": "hallucination",
        "label": "HALLUCINATION ANALYSIS",
        "purpose": "Apply the project's conservative claim taxonomy without equating unsupported with hallucinated.",
        "implementation": "classify_hallucination.",
        "source": "src/scoring/hallucination_taxonomy.py",
        "inputs": "Claim, verdict, and selected evidence.",
        "outputs": "Hallucination type or UNKNOWN.",
    },
    {
        "id": "severity",
        "label": "SEVERITY",
        "purpose": "Apply transparent evidence-based severity rules.",
        "implementation": "assess_severity and review_flags.",
        "source": "src/scoring/severity.py",
        "inputs": "Verdict, confidence, taxonomy, and retrieval score.",
        "outputs": "Severity reasons and human-review flags.",
    },
    {
        "id": "explanation",
        "label": "EXPLANATION",
        "purpose": "Build a deterministic explanation from the verification outcome.",
        "implementation": "build_explanation; it does not expose hidden chain-of-thought.",
        "source": "src/utils/explanations.py",
        "inputs": "Verdict, evidence, and conflict flag.",
        "outputs": "Evidence-grounded explanatory text.",
    },
    {
        "id": "reliability",
        "label": "RELIABILITY",
        "purpose": "Summarize measured outcomes and separately label experimental scoring.",
        "implementation": "reliability_metrics and composite_reliability_score.",
        "source": "src/scoring/reliability.py",
        "inputs": "Pipeline or persisted prediction records.",
        "outputs": "Outcome, coverage, confidence, and risk metrics.",
    },
    {
        "id": "monitoring",
        "label": "MONITORING",
        "purpose": "Compare persisted baseline and candidate metrics using configured thresholds.",
        "implementation": "Regression monitor with configured absolute thresholds.",
        "source": "src/evaluation/regression_monitor.py",
        "inputs": "Baseline, candidate, and regression configuration.",
        "outputs": "PASS, WARN, or FAIL status.",
    },
]


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_yaml(path: Path) -> Any:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@lru_cache(maxsize=4)
def _dataset_stats(dataset: str) -> dict[str, Any]:
    relative = DATASET_FILES[dataset]
    path = PROCESSED / relative
    if not path.is_file():
        return {
            "dataset": dataset,
            "availability": "DATA NOT AVAILABLE",
            "missing": f"Processed dataset file missing: datasets/processed/{relative}",
            "records": None,
            "labels": {},
            "splits": {},
            "task_types": {},
        }

    count = 0
    labels: dict[str, int] = {}
    splits: dict[str, int] = {}
    task_types: dict[str, int] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid processed JSONL at {path}:{line_number}") from error
            count += 1
            label = record.get("label")
            if label:
                labels[label] = labels.get(label, 0) + 1
            metadata = record.get("metadata") or {}
            split = metadata.get("split")
            if split:
                splits[split] = splits.get(split, 0) + 1
            task = record.get("task_type")
            if task:
                task_types[task] = task_types.get(task, 0) + 1

    return {
        "dataset": dataset,
        "availability": "AVAILABLE",
        "file": f"datasets/processed/{relative}",
        "records": count,
        "labels": labels,
        "splits": splits,
        "task_types": task_types,
    }


def _metric_artifacts() -> list[dict[str, Any]]:
    items = []
    if not METRICS.is_dir():
        return items
    for path in sorted(METRICS.glob("*.json")):
        data = _read_json(path)
        if isinstance(data, dict) and data.get("experiment_name") and "accuracy" in data:
            stat = path.stat()
            items.append(
                {
                    "name": data["experiment_name"],
                    "file": path.name,
                    "dataset": data.get("dataset"),
                    "sample_count": data.get("sample_count", data.get("subset_size")),
                    "metrics": data,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                }
            )
    return sorted(items, key=lambda item: item["modified_at"], reverse=True)


def _prediction_artifacts() -> list[str]:
    if not PREDICTIONS.is_dir():
        return []
    return sorted(path.name for path in PREDICTIONS.glob("*_predictions.jsonl"))


def _prediction_errors(filename: str, limit: int = 200) -> dict[str, Any]:
    path = PREDICTIONS / filename
    if not path.is_file():
        return {"rows": [], "total_errors": 0}
    rows: list[dict[str, Any]] = []
    total_errors = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid prediction JSONL at {path}:{line_number}") from error
            if record.get("true_label") != record.get("predicted_label"):
                total_errors += 1
                if len(rows) < limit:
                    rows.append(record)
    metrics_path = METRICS / filename.replace("_predictions.jsonl", "_metrics.json")
    metrics = _read_json(metrics_path)
    return {"rows": rows, "total_errors": total_errors, "metrics": metrics}


def _adversarial_file(path: Path, label: str) -> dict[str, Any]:
    data = _read_json(path)
    if data is None:
        return {"name": label, "availability": "DATA NOT AVAILABLE", "missing": str(path.relative_to(ROOT)), "cases": []}
    if isinstance(data, dict):
        records = data.get("cases", data.get("results", []))
        metadata = {key: value for key, value in data.items() if key not in {"cases", "results"}}
    elif isinstance(data, list):
        records, metadata = data, {}
    else:
        records, metadata = [], {}
    cases = []
    for index, record in enumerate(records):
        output = record.get("verification_output") or {}
        actual = record.get("actual_verdict") or record.get("model_prediction") or output.get("verdict")
        cases.append(
            {
                "case": record.get("case_id", record.get("expected_category", f"case_{index + 1}")),
                "category": record.get("category", record.get("expected_category")),
                "input": record.get("claim") or record.get("input"),
                "evidence": record.get("evidence")
                or "\n".join(record.get("input_evidence", []))
                or (output.get("evidence") or [{}])[0].get("evidence"),
                "expected": record.get("expected_verdict"),
                "actual": actual,
                "pass": record.get("pass", record.get("passed")),
                "verification_output": output,
            }
        )
    total = len(cases)
    passed = sum(case["pass"] is True for case in cases)
    failed = sum(case["pass"] is False for case in cases)
    return {
        "name": label,
        "availability": "AVAILABLE",
        "metadata": metadata,
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total else None,
        "cases": cases,
    }


def page_data(page: str, *, dataset: str | None = None, model: str | None = None, limit: int = 200) -> dict[str, Any]:
    """Return source-labeled data for an individual dashboard page."""
    metrics = _metric_artifacts()

    if page in {"benchmarks", "datasets"}:
        benchmarks = []
        definitions = {
            "FEVER": ("Claim verification", "CLAIM-LEVEL VERIFICATION", "Normalized train records; validation created as a deterministic holdout.", "Stage 3 classifiers and bounded retrieval/NLI evaluation."),
            "AVeriTeC": ("Real-world claim verification", "CLAIM-LEVEL VERIFICATION", "Normalized train and dev splits only; test split excluded.", "Stage 5/7 evaluation uses dev split only."),
            "HaluEval": ("Response hallucination detection", "RESPONSE-LEVEL HALLUCINATION DETECTION", "Task-aware normalized records; holdout metrics are from the persisted classifier artifact.", "Separate response-level TF-IDF + Logistic Regression baseline."),
            "RAGTruth": ("RAG response/span hallucination analysis", "RESPONSE-LEVEL HALLUCINATION DETECTION", "Normalized response records with preserved span metadata.", "Read-only response and span analytics."),
        }
        for name, (task, task_type, split, role) in definitions.items():
            stats = _dataset_stats(name)
            benchmarks.append(
                {
                    **stats,
                    "task": task,
                    "task_type": task_type,
                    "split_role": split,
                    "project_usage": role,
                    "metrics_available": [entry["name"] for entry in metrics if name.lower() in entry["name"].lower()],
                    "raw_dataset": {
                        "availability": "PRESENT" if (ROOT / "datasets" / "raw" / name).exists() else "DATA NOT AVAILABLE",
                        "editable": False,
                        "note": "Read-only benchmark source; this dashboard does not modify raw files.",
                    },
                    "processed_dataset": {
                        "availability": stats["availability"],
                        "path": stats.get("file"),
                        "records": stats.get("records"),
                    },
                }
            )
        return {
            "page": page,
            "benchmark_roles": {
                "claim_level": ["FEVER", "AVeriTeC"],
                "response_level": ["HaluEval", "RAGTruth"],
            },
            "datasets": benchmarks,
        }

    if page == "hallucination":
        return {
            "page": page,
            "halueval": {
                "source": "results/metrics/HaluEval_TFIDF_LogReg_metrics.json",
                "metrics": _read_json(METRICS / "HaluEval_TFIDF_LogReg_metrics.json"),
                "reliability_report": _read_json(METRICS / "stage6_halueval_reliability_report.json"),
            },
            "ragtruth": {
                "source": "results/metrics/ragtruth_summary.json",
                "summary": _read_json(METRICS / "ragtruth_summary.json"),
            },
            "pipeline_taxonomy": "Only available for live pipeline claims. Unsupported claims are not automatically labeled hallucinations.",
        }

    if page == "reliability":
        config = _read_yaml(ROOT / "configs" / "reliability_config.yaml")
        return {
            "page": page,
            "reports": [
                _read_json(METRICS / "stage6_fever_reliability_report.json"),
                _read_json(METRICS / "stage6_halueval_reliability_report.json"),
            ],
            "composite_config": config,
            "source": "Persisted Stage 6 prediction-artifact reliability reports.",
            "metric_caveat": "Experimental reliability score is a framework metric, not validated ground truth.",
        }

    if page == "retrieval":
        return {
            "page": page,
            "retrieval": _read_json(METRICS / "stage7_retrieval_before_after.json"),
            "retrieval_nli": _read_json(METRICS / "stage7_retrieval_nli_before_after.json"),
            "configuration": {"seed": 42, "sample_count_per_dataset": 200, "datasets": ["FEVER validation", "AVeriTeC dev"]},
        }

    if page == "nli":
        config = _read_yaml(ROOT / "configs" / "verification_config.yaml")
        return {
            "page": page,
            "model": config.get("nli_model", {}).get("name") if config else None,
            "mapping": {"ENTAILMENT": "SUPPORTED", "CONTRADICTION": "CONTRADICTED", "NEUTRAL": "UNSUPPORTED"},
            "metrics": {
                "FEVER": _read_json(METRICS / "NLI_FEVER_InDomain_metrics.json"),
                "AVeriTeC": _read_json(METRICS / "NLI_AVeriTeC_InDomain_metrics.json"),
            },
            "confidence_distribution": "DATA NOT AVAILABLE: persisted NLI prediction artifacts omit confidence for these NLI runs.",
        }

    if page == "comparison":
        selected = _read_json(RESULTS / "comparisons" / "fever_in_domain_models.json")
        return {
            "page": page,
            "comparison": selected,
            "available_comparison_files": sorted(path.name for path in (RESULTS / "comparisons").glob("*.json")),
        }

    if page == "regression":
        return {
            "page": page,
            "regression": _read_json(METRICS / "stage6_fever_logreg_to_svm_regression.json"),
            "configuration": _read_yaml(ROOT / "configs" / "regression_config.yaml"),
        }

    if page == "errors":
        files = _prediction_artifacts()
        selected = model if model in files else next((name for name in files if dataset and name.lower().startswith(dataset.lower())), files[0] if files else None)
        error_data = _prediction_errors(selected, max(1, min(limit, 500))) if selected else {"rows": [], "total_errors": 0, "metrics": None}
        return {
            "page": page,
            "files": files,
            "selected_file": selected,
            "errors": error_data["rows"],
            "total_errors": error_data["total_errors"],
            "metrics": error_data.get("metrics"),
            "availability": "AVAILABLE" if selected else "DATA NOT AVAILABLE",
        }

    if page == "adversarial":
        return {
            "page": page,
            "scope_note": "Synthetic framework robustness evaluation — not benchmark performance.",
            "evaluations": [
                _adversarial_file(EVALUATION / "adversarial" / "stage6_synthetic_adversarial_results.json", "Stage 6 synthetic adversarial"),
                _adversarial_file(EVALUATION / "adversarial" / "stage7_improved_adversarial_results.json", "Stage 7 improved adversarial"),
                _adversarial_file(EVALUATION / "robustness" / "stage7_controlled_robustness_results.json", "Stage 7 controlled robustness"),
            ],
        }

    if page == "experiments":
        return {
            "page": page,
            "experiments": metrics,
            "missing_metadata": ["configuration", "seed", "date"] ,
            "note": "Only values present in persisted metric artifacts are populated. Artifact modification time is shown as file metadata, not run time.",
        }

    if page == "pipeline":
        return {"page": page, "nodes": PIPELINE_NODES, "version": _read_json(METRICS / "pipeline_v0.2_baseline.json")}

    if page == "api":
        from backend.app import app
        from backend.app import VerifyRequest

        routes = []
        for route in app.routes:
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", None)
            if path and methods and (path.startswith("/api/") or path == "/health"):
                routes.append(
                    {
                        "methods": sorted(methods - {"HEAD", "OPTIONS"}),
                        "path": path,
                        "description": getattr(route, "summary", None) or getattr(route, "name", "API endpoint"),
                    }
                )
        return {
            "page": page,
            "base_url": os.environ.get("VERILLM_PUBLIC_URL", "http://127.0.0.1:8000"),
            "routes": routes,
            "verify_request_schema": VerifyRequest.model_json_schema(),
            "verify_response_fields": [
                "summary",
                "claims",
                "response",
                "mode",
                "metadata",
            ],
            "response_schema_note": "The FastAPI handler returns a JSON object; the live request panel below displays the actual response payload.",
        }

    if page == "docs":
        documents = []
        for title, relative in DOC_FILES.items():
            path = ROOT / "docs" / relative
            documents.append(
                {
                    "title": title,
                    "path": f"docs/{relative}",
                    "content": path.read_text(encoding="utf-8") if path.is_file() else None,
                }
            )
        return {"page": page, "documents": documents}

    if page == "overview":
        return {
            "page": page,
            "persisted_evaluations": len(metrics),
            "recent_evaluations": metrics[-5:],
            "datasets": [_dataset_stats(name) for name in DATASET_FILES],
            "available_prediction_artifacts": len(_prediction_artifacts()),
            "live_pipeline": "Loaded separately from the real /api/overview pipeline execution.",
        }

    return {"page": page}
