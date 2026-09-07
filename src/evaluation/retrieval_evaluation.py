"""Bounded, reproducible retrieval and retrieval-to-NLI evaluation utilities."""
import hashlib
import json
from pathlib import Path

from src.evaluation.evaluate_classifier import evaluate_predictions
from src.retrieval.tfidf_retriever import TfidfEvidenceRetriever, normalize_text


def fixed_subset(records, size=200, seed=42):
    """Choose a deterministic subset by a stable hash, independent of file order."""
    ranked = sorted(records, key=lambda item: hashlib.sha256(
        f"{seed}:{item.get('sample_id', '')}".encode("utf-8")
    ).hexdigest())
    return ranked[:min(size, len(ranked))]


def build_evidence_corpus(records):
    """Create a de-duplicated corpus while retaining every gold evidence text."""
    corpus, seen = [], set()
    for record in records:
        evidence = normalize_text(record.get("evidence", ""))
        if evidence and evidence not in seen:
            seen.add(evidence)
            corpus.append({"evidence_id": str(record.get("sample_id")), "evidence": evidence})
    return corpus


def retrieval_metrics(records, corpus_records, ks=(1, 3, 5, 10), retriever_class=TfidfEvidenceRetriever):
    """Measure exact normalized-gold-evidence recall against a fixed corpus."""
    corpus = build_evidence_corpus(corpus_records)
    retriever = retriever_class(corpus)
    ks = tuple(sorted(set(ks)))
    hits = {k: 0 for k in ks}
    rows = []
    for record in records:
        gold = normalize_text(record.get("evidence", ""))
        retrieved = retriever.retrieve(record.get("claim") or record.get("text", ""), top_k=max(ks))
        ids = [item["evidence_id"] for item in retrieved]
        row_hits = {k: gold in [item["evidence"] for item in retrieved[:k]] for k in ks}
        for k, hit in row_hits.items():
            hits[k] += int(hit)
        rows.append({"sample_id": record.get("sample_id"), "gold_evidence": gold,
                     "retrieved_evidence_ids": ids, "hits": row_hits})
    count = len(records)
    return ({"evaluation_count": count, "corpus_size": len(corpus),
             "evidence_hit_definition": "A hit occurs when an exact whitespace-normalized gold evidence string is returned in the top K results.",
             "recall_at_k": {str(k): round(hits[k] / count, 4) if count else 0.0 for k in ks},
             "evidence_hit_rate": round(hits[max(ks)] / count, 4) if count else 0.0,
             "retrieval_failure_rate": round(1 - hits[max(ks)] / count, 4) if count else 0.0}, rows, retriever)


def end_to_end_evaluation(records, corpus_records, verifier, experiment_name, top_k=1,
                          retriever_class=TfidfEvidenceRetriever, min_retrieval_score=None):
    """Evaluate top-1 retrieved evidence followed by NLI, optionally applying a relevance gate."""
    _, retrieval_rows, retriever = retrieval_metrics(
        records, corpus_records, ks=(top_k,), retriever_class=retriever_class
    )
    nli_inputs, kept_records = [], []
    for record in records:
        evidence = retriever.retrieve(record.get("claim") or record.get("text", ""), top_k=1)
        if evidence and (min_retrieval_score is None or evidence[0]["score"] >= min_retrieval_score):
            nli_inputs.append({"claim": record.get("claim") or record.get("text", ""), "evidence": evidence[0]["evidence"]})
            kept_records.append(record)
    predictions = verifier.predict_many(nli_inputs)
    prediction_by_id = {str(record.get("sample_id")): prediction for record, prediction in zip(kept_records, predictions)}
    full_predictions = []
    for record in records:
        prediction = prediction_by_id.get(str(record.get("sample_id")))
        full_predictions.append(prediction or {
            "claim": record.get("claim") or record.get("text", ""), "evidence": "",
            "nli_label": "NEUTRAL", "verdict": "UNSUPPORTED", "confidence": 0.0,
        })
    metrics = evaluate_predictions(
        [r["label"] for r in records], [p["verdict"] for p in full_predictions], experiment_name,
        records, prediction_details=full_predictions,
    )
    metrics["retrieval_protocol"] = "Top-1 retrieved evidence is passed to the existing NLI verifier."
    metrics["relevance_threshold"] = min_retrieval_score
    metrics["gated_to_unsupported"] = len(records) - len(kept_records)
    return metrics, retrieval_rows


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
