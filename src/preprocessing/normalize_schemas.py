"""
VeriLLM Schema Normalization Module (Revised Stage 2)
Location: src/preprocessing/normalize_schemas.py

Normalizes raw datasets into task-aware VeriLLM canonical JSONL schemas under datasets/processed/:
- FEVER      -> task_type: CLAIM_VERIFICATION (SUPPORTED, CONTRADICTED, UNSUPPORTED)
- AVeriTeC   -> task_type: REAL_WORLD_CLAIM_VERIFICATION (SUPPORTED, CONTRADICTED, UNSUPPORTED, CONFLICTING_EVIDENCE)
- HaluEval   -> task_type: RESPONSE_HALLUCINATION_DETECTION (CORRECT, HALLUCINATED)
- RAGTruth   -> task_type: RAG_HALLUCINATION_DETECTION (CLEAN, HALLUCINATED)
"""

import sys
import os
import json
from collections import Counter
from typing import Dict, Any, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "datasets", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_jsonl(records: List[Dict[str, Any]], filepath: str):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_stats(name: str, total_raw: int, total_processed: int, dropped: int, drop_reasons: Dict[str, int], label_dist: Counter):
    print(f"\n" + "=" * 70)
    print(f"  TASK-AWARE NORMALIZATION SUMMARY: {name}")
    print("=" * 70)
    print(f"  Total Raw Records       : {total_raw}")
    print(f"  Total Processed Records : {total_processed}")
    print(f"  Dropped Records         : {dropped}")
    if drop_reasons:
        print(f"  Drop Reasons            : {dict(drop_reasons)}")
    print(f"  Canonical Label Dist    : {dict(label_dist)}")
    print("=" * 70)


def normalize_fever() -> Tuple[int, int, List[Dict[str, Any]]]:
    raw_path = os.path.join(RAW_DIR, "FEVER", "train.jsonl")
    out_path = os.path.join(PROCESSED_DIR, "FEVER", "fever_normalized.jsonl")
    
    label_map = {
        "SUPPORTS": "SUPPORTED",
        "REFUTES": "CONTRADICTED",
        "NOT ENOUGH INFO": "UNSUPPORTED"
    }
    
    total_raw = 0
    records = []
    drop_reasons = Counter()
    label_dist = Counter()
    
    with open(raw_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total_raw += 1
            try:
                item = json.loads(line)
                claim = item.get("claim", "").strip()
                orig_label = item.get("label", "").strip()
                
                if not claim:
                    drop_reasons["missing_claim"] += 1
                    continue
                if orig_label not in label_map:
                    drop_reasons["invalid_label"] += 1
                    continue
                
                norm_label = label_map[orig_label]
                label_dist[norm_label] += 1
                
                rec = {
                    "sample_id": f"FEVER_TRAIN_{idx+1:06d}",
                    "text": claim,
                    "claim": claim,
                    "evidence": str(item.get("evidence", "")),
                    "task_type": "CLAIM_VERIFICATION",
                    "label": norm_label,
                    "original_label": orig_label,
                    "source_dataset": "FEVER",
                    "metadata": {
                        "fever_id": item.get("id"),
                        "verifiable": item.get("verifiable")
                    }
                }
                records.append(rec)
            except Exception as e:
                drop_reasons["malformed_json"] += 1

    write_jsonl(records, out_path)
    total_dropped = sum(drop_reasons.values())
    print_stats("FEVER", total_raw, len(records), total_dropped, drop_reasons, label_dist)
    return total_raw, len(records), records


def normalize_averitec() -> Tuple[int, int, List[Dict[str, Any]]]:
    averitec_dir = os.path.join(RAW_DIR, "AVeriTeC")
    out_path = os.path.join(PROCESSED_DIR, "AVeriTeC", "averitec_normalized.jsonl")
    
    label_map = {
        "Supported": "SUPPORTED",
        "Refuted": "CONTRADICTED",
        "Not Enough Evidence": "UNSUPPORTED",
        "Conflicting Evidence/Cherrypicking": "CONFLICTING_EVIDENCE"
    }
    
    total_raw = 0
    records = []
    drop_reasons = Counter()
    label_dist = Counter()
    
    splits = [("train.json", "train"), ("dev.json", "dev")]
    
    for fname, split_name in splits:
        fpath = os.path.join(averitec_dir, fname)
        if not os.path.exists(fpath):
            continue
            
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            total_raw += len(data)
            
            for idx, item in enumerate(data):
                claim = item.get("claim", "").strip()
                orig_label = item.get("label", "").strip()
                
                if not claim:
                    drop_reasons["missing_claim"] += 1
                    continue
                if orig_label not in label_map:
                    drop_reasons[f"unexpected_label_{orig_label}"] += 1
                    continue
                    
                norm_label = label_map[orig_label]
                label_dist[norm_label] += 1
                
                justification = item.get("justification", "")
                questions = item.get("questions", [])
                qa_summary = []
                if isinstance(questions, list):
                    for q in questions:
                        if isinstance(q, dict):
                            q_str = q.get("question", "")
                            answers = q.get("answers", [])
                            a_strs = [a.get("answer", "") for a in answers if isinstance(a, dict)]
                            if q_str and a_strs:
                                qa_summary.append(f"Q: {q_str} A: {' '.join(a_strs)}")
                
                full_evidence = f"Justification: {justification}\n" + "\n".join(qa_summary)
                
                rec = {
                    "sample_id": f"AVERITEC_{split_name.upper()}_{idx+1:05d}",
                    "text": claim,
                    "claim": claim,
                    "evidence": full_evidence.strip(),
                    "task_type": "REAL_WORLD_CLAIM_VERIFICATION",
                    "label": norm_label,
                    "original_label": orig_label,
                    "source_dataset": "AVeriTeC",
                    "metadata": {
                        "split": split_name,
                        "speaker": item.get("speaker"),
                        "claim_date": item.get("claim_date"),
                        "claim_types": item.get("claim_types"),
                        "fact_checking_article": item.get("fact_checking_article"),
                        "reporting_source": item.get("reporting_source"),
                        "questions": questions
                    }
                }
                records.append(rec)
                
    write_jsonl(records, out_path)
    total_dropped = sum(drop_reasons.values())
    print_stats("AVeriTeC (Train + Dev)", total_raw, len(records), total_dropped, drop_reasons, label_dist)
    return total_raw, len(records), records


def normalize_halueval() -> Tuple[int, int, List[Dict[str, Any]]]:
    halueval_dir = os.path.join(RAW_DIR, "HaluEval", "HaluEval-main", "data")
    out_path = os.path.join(PROCESSED_DIR, "HaluEval", "halueval_normalized.jsonl")
    
    total_raw = 0
    records = []
    drop_reasons = Counter()
    label_dist = Counter()
    
    paired_tasks = [
        ("qa_data.json", "qa", "knowledge", "question", "right_answer", "hallucinated_answer"),
        ("summarization_data.json", "summarization", "document", None, "right_summary", "hallucinated_summary"),
        ("dialogue_data.json", "dialogue", "knowledge", "dialogue_history", "right_response", "hallucinated_response")
    ]
    
    for fname, task_name, ev_field, ctx_field, pos_field, neg_field in paired_tasks:
        fpath = os.path.join(halueval_dir, fname)
        if not os.path.exists(fpath):
            continue
            
        with open(fpath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                total_raw += 1
                try:
                    item = json.loads(line)
                    ev_text = item.get(ev_field, "")
                    if ctx_field and item.get(ctx_field):
                        ev_text = f"{ev_text}\nContext: {item.get(ctx_field)}"
                    
                    pos_resp = item.get(pos_field, "").strip()
                    neg_resp = item.get(neg_field, "").strip()
                    
                    if pos_resp:
                        rec_pos = {
                            "sample_id": f"HALUEVAL_{task_name.upper()}_POS_{idx+1:05d}",
                            "text": pos_resp,
                            "claim": None,
                            "evidence": ev_text.strip(),
                            "task_type": "RESPONSE_HALLUCINATION_DETECTION",
                            "label": "CORRECT",
                            "original_label": "right",
                            "source_dataset": "HaluEval",
                            "metadata": {"task": task_name, "pair_type": "right"}
                        }
                        records.append(rec_pos)
                        label_dist["CORRECT"] += 1
                        
                    if neg_resp:
                        rec_neg = {
                            "sample_id": f"HALUEVAL_{task_name.upper()}_NEG_{idx+1:05d}",
                            "text": neg_resp,
                            "claim": None,
                            "evidence": ev_text.strip(),
                            "task_type": "RESPONSE_HALLUCINATION_DETECTION",
                            "label": "HALLUCINATED",
                            "original_label": "hallucinated",
                            "source_dataset": "HaluEval",
                            "metadata": {"task": task_name, "pair_type": "hallucinated"}
                        }
                        records.append(rec_neg)
                        label_dist["HALLUCINATED"] += 1
                except Exception as e:
                    drop_reasons["malformed_json"] += 1
                    
    gen_path = os.path.join(halueval_dir, "general_data.json")
    if os.path.exists(gen_path):
        with open(gen_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                total_raw += 1
                try:
                    item = json.loads(line)
                    resp = item.get("chatgpt_response", "").strip()
                    is_hallu = item.get("hallucination", "").lower() == "yes"
                    
                    if not resp:
                        drop_reasons["missing_claim"] += 1
                        continue
                        
                    norm_label = "HALLUCINATED" if is_hallu else "CORRECT"
                    label_dist[norm_label] += 1
                    
                    rec = {
                        "sample_id": f"HALUEVAL_GENERAL_{idx+1:05d}",
                        "text": resp,
                        "claim": None,
                        "evidence": f"User Query: {item.get('user_query', '')}",
                        "task_type": "RESPONSE_HALLUCINATION_DETECTION",
                        "label": norm_label,
                        "original_label": item.get("hallucination"),
                        "source_dataset": "HaluEval",
                        "metadata": {
                            "query_id": item.get("ID"),
                            "hallucination_spans": item.get("hallucination_spans")
                        }
                    }
                    records.append(rec)
                except Exception as e:
                    drop_reasons["malformed_json"] += 1

    write_jsonl(records, out_path)
    total_dropped = sum(drop_reasons.values())
    print_stats("HaluEval", total_raw, len(records), total_dropped, drop_reasons, label_dist)
    return total_raw, len(records), records


def normalize_ragtruth() -> Tuple[int, int, List[Dict[str, Any]]]:
    ragtruth_dir = os.path.join(RAW_DIR, "RAGTruth", "RAGTruth-main", "dataset")
    resp_path = os.path.join(ragtruth_dir, "response.jsonl")
    source_path = os.path.join(ragtruth_dir, "source_info.jsonl")
    out_path = os.path.join(PROCESSED_DIR, "RAGTruth", "ragtruth_normalized.jsonl")
    
    source_map = {}
    if os.path.exists(source_path):
        with open(source_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    source_map[item.get("source_id")] = item

    total_raw = 0
    records = []
    drop_reasons = Counter()
    label_dist = Counter()

    with open(resp_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total_raw += 1
            try:
                item = json.loads(line)
                resp_text = item.get("response", "").strip()
                labels_array = item.get("labels", [])
                source_id = item.get("source_id")
                
                if not resp_text:
                    drop_reasons["missing_response"] += 1
                    continue
                    
                is_hallucinated = len(labels_array) > 0
                norm_label = "HALLUCINATED" if is_hallucinated else "CLEAN"
                label_dist[norm_label] += 1
                
                source_info = source_map.get(source_id, {})
                ev_parts = []
                if source_info.get("source"):
                    ev_parts.append(f"Source: {source_info.get('source')}")
                if source_info.get("prompt"):
                    ev_parts.append(f"Prompt: {source_info.get('prompt')}")
                    
                rec = {
                    "sample_id": f"RAGTRUTH_{idx+1:06d}",
                    "text": resp_text,
                    "claim": None,
                    "evidence": "\n".join(ev_parts).strip(),
                    "task_type": "RAG_HALLUCINATION_DETECTION",
                    "label": norm_label,
                    "original_label": "hallucinated" if is_hallucinated else "clean",
                    "source_dataset": "RAGTruth",
                    "metadata": {
                        "ragtruth_id": item.get("id"),
                        "source_id": source_id,
                        "model": item.get("model"),
                        "temperature": item.get("temperature"),
                        "split": item.get("split"),
                        "quality": item.get("quality"),
                        "task_type": source_info.get("task_type"),
                        "hallucination_spans": labels_array
                    }
                }
                records.append(rec)
            except Exception as e:
                drop_reasons["malformed_json"] += 1

    write_jsonl(records, out_path)
    total_dropped = sum(drop_reasons.values())
    print_stats("RAGTruth", total_raw, len(records), total_dropped, drop_reasons, label_dist)
    return total_raw, len(records), records


def run_quality_checks():
    print("\n" + "=" * 70)
    print("  RUNNING TASK-AWARE QUALITY CONTROL AUDIT")
    print("=" * 70)
    
    datasets = ["FEVER", "AVeriTeC", "HaluEval", "RAGTruth"]
    all_sample_ids = set()
    total_processed_records = 0
    
    claim_level_count = 0
    response_level_count = 0
    evidence_count = 0
    span_annotation_count = 0
    
    task_label_dists = {}

    for ds in datasets:
        fname = f"{ds.lower()}_normalized.jsonl"
        fpath = os.path.join(PROCESSED_DIR, ds, fname)
        
        if not os.path.exists(fpath):
            print(f"[FAIL] Missing processed file: {fpath}")
            continue
            
        ds_records = 0
        missing_text = 0
        missing_labels = 0
        duplicate_ids = 0
        
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                ds_records += 1
                sid = rec.get("sample_id")
                ttype = rec.get("task_type")
                lbl = rec.get("label")
                
                if ttype not in task_label_dists:
                    task_label_dists[ttype] = Counter()
                task_label_dists[ttype][lbl] += 1
                
                if not rec.get("text"): missing_text += 1
                if not lbl: missing_labels += 1
                if sid in all_sample_ids: duplicate_ids += 1
                else: all_sample_ids.add(sid)
                
                if rec.get("claim"): claim_level_count += 1
                else: response_level_count += 1
                
                if rec.get("evidence"): evidence_count += 1
                
                spans = rec.get("metadata", {}).get("hallucination_spans")
                if spans and len(spans) > 0:
                    span_annotation_count += 1
                
        total_processed_records += ds_records
        print(f"[OK] {ds:<10}: {ds_records:>6} records | Missing Text: {missing_text} | Missing Labels: {missing_labels} | Duplicate IDs: {duplicate_ids}")

    print("\n" + "-" * 70)
    print("  TASK-LEVEL RECORD & METRIC BREAKDOWN:")
    print("-" * 70)
    print(f"  1. Total Processed Records Across All Tasks : {total_processed_records:,}")
    print(f"  2. Claim-Level Labeled Records             : {claim_level_count:,} (FEVER + AVeriTeC)")
    print(f"  3. Response-Level Labeled Records          : {response_level_count:,} (HaluEval + RAGTruth)")
    print(f"  4. Records Containing Non-Empty Evidence   : {evidence_count:,}")
    print(f"  5. Records Containing Hallucination Spans  : {span_annotation_count:,} (HaluEval + RAGTruth)")
    
    print("\n" + "-" * 70)
    print("  PER-TASK CANONICAL LABEL DISTRIBUTIONS:")
    print("-" * 70)
    for ttype, dist in task_label_dists.items():
        print(f"  * {ttype:<32}: {dict(dist)}")
    print("=" * 70 + "\n")


def main():
    print("=" * 70)
    print("  VERILLM REVISED TASK-AWARE SCHEMA NORMALIZATION")
    print("=" * 70)
    
    normalize_fever()
    normalize_averitec()
    normalize_halueval()
    normalize_ragtruth()
    
    run_quality_checks()


if __name__ == "__main__":
    main()
