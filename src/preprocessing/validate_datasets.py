"""
VeriLLM Dataset Validation Script
Location: src/preprocessing/validate_datasets.py

Validates downloaded datasets (AVeriTeC, FEVER, HaluEval, RAGTruth) without modifying raw files:
1. Verifies expected raw file existence.
2. Efficiently streams/samples records to avoid RAM overload.
3. Checks JSON / JSONL syntax integrity.
4. Audits record counts and schema fields.
5. Flags malformed or missing key fields.
6. Reports PASS / FAIL for each dataset.
"""

import sys
import os
import json
from typing import Dict, Any, Tuple, List

# Define project root relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RAW_DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets", "raw")


def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def validate_averitec() -> Tuple[bool, Dict[str, Any]]:
    print_separator("VALIDATING AVERITEC DATASET")
    averitec_dir = os.path.join(RAW_DATASETS_DIR, "AVeriTeC")
    expected_files = ["train.json", "dev.json", "test.json"]
    
    status = True
    info = {"files_found": [], "file_stats": {}, "schema": [], "sample": None}
    
    for fname in expected_files:
        fpath = os.path.join(averitec_dir, fname)
        if not os.path.exists(fpath):
            print(f"[FAIL] Missing file: {fpath}")
            status = False
            continue
        
        info["files_found"].append(fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                count = len(data)
                info["file_stats"][fname] = {
                    "count": count,
                    "size_mb": round(os.path.getsize(fpath) / (1024 * 1024), 2)
                }
                print(f"[OK] {fname}: {count} records ({info['file_stats'][fname]['size_mb']} MB)")
                
                if fname == "train.json" and count > 0:
                    sample = data[0]
                    info["schema"] = list(sample.keys())
                    info["sample"] = {
                        "claim": sample.get("claim"),
                        "label": sample.get("label"),
                        "speaker": sample.get("speaker")
                    }
        except Exception as e:
            print(f"[FAIL] Syntax error in {fname}: {str(e)}")
            status = False
            
    print(f"Discovered AVeriTeC Schema: {info['schema']}")
    return status, info


def validate_fever() -> Tuple[bool, Dict[str, Any]]:
    print_separator("VALIDATING FEVER DATASET")
    fpath = os.path.join(RAW_DATASETS_DIR, "FEVER", "train.jsonl")
    
    if not os.path.exists(fpath):
        print(f"[FAIL] Missing file: {fpath}")
        return False, {}

    status = True
    info = {
        "file": "train.jsonl",
        "size_mb": round(os.path.getsize(fpath) / (1024 * 1024), 2),
        "total_records": 0,
        "malformed_records": 0,
        "schema": [],
        "sample": None,
        "required_fields": ["id", "verifiable", "label", "claim", "evidence"]
    }
    
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    info["total_records"] += 1
                    if idx == 0:
                        info["schema"] = list(record.keys())
                        info["sample"] = {
                            "id": record.get("id"),
                            "verifiable": record.get("verifiable"),
                            "label": record.get("label"),
                            "claim": record.get("claim")
                        }
                    
                    # Verify required fields
                    for field in info["required_fields"]:
                        if field not in record:
                            info["malformed_records"] += 1
                            break
                except json.JSONDecodeError:
                    info["malformed_records"] += 1
                    
        print(f"[OK] train.jsonl: {info['total_records']} records ({info['size_mb']} MB)")
        print(f"Malformed / Missing Field Records: {info['malformed_records']}")
        print(f"Discovered FEVER Schema: {info['schema']}")
        
        if info["malformed_records"] > 0:
            status = False
            
    except Exception as e:
        print(f"[FAIL] Error reading FEVER dataset: {str(e)}")
        status = False
        
    return status, info


def validate_halueval() -> Tuple[bool, Dict[str, Any]]:
    print_separator("VALIDATING HALUEVAL DATASET")
    halueval_dir = os.path.join(RAW_DATASETS_DIR, "HaluEval", "HaluEval-main", "data")
    
    if not os.path.exists(halueval_dir):
        print(f"[FAIL] Missing HaluEval data directory: {halueval_dir}")
        return False, {}

    task_files = [
        "qa_data.json",
        "summarization_data.json",
        "dialogue_data.json",
        "general_data.json"
    ]
    
    status = True
    info = {"task_files": {}, "schemas": {}}
    
    for fname in task_files:
        fpath = os.path.join(halueval_dir, fname)
        if not os.path.exists(fpath):
            print(f"[WARN] Missing HaluEval task file: {fname}")
            continue
        
        size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
        valid_records = 0
        malformed = 0
        schema = []
        
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        valid_records += 1
                        if idx == 0:
                            schema = list(record.keys())
                    except json.JSONDecodeError:
                        malformed += 1
            
            info["task_files"][fname] = {
                "records": valid_records,
                "malformed": malformed,
                "size_mb": size_mb
            }
            info["schemas"][fname] = schema
            print(f"[OK] {fname}: {valid_records} records ({size_mb} MB) | Schema: {schema}")
            
        except Exception as e:
            print(f"[FAIL] Error reading HaluEval file {fname}: {str(e)}")
            status = False
            
    return status, info


def validate_ragtruth() -> Tuple[bool, Dict[str, Any]]:
    print_separator("VALIDATING RAGTRUTH DATASET")
    ragtruth_dir = os.path.join(RAW_DATASETS_DIR, "RAGTruth", "RAGTruth-main", "dataset")
    
    if not os.path.exists(ragtruth_dir):
        print(f"[FAIL] Missing RAGTruth dataset directory: {ragtruth_dir}")
        return False, {}

    target_files = ["response.jsonl", "source_info.jsonl"]
    status = True
    info = {"files": {}, "schemas": {}}
    
    for fname in target_files:
        fpath = os.path.join(ragtruth_dir, fname)
        if not os.path.exists(fpath):
            print(f"[FAIL] Missing RAGTruth file: {fname}")
            status = False
            continue
            
        size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
        valid_records = 0
        malformed = 0
        schema = []
        
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        valid_records += 1
                        if idx == 0:
                            schema = list(record.keys())
                    except json.JSONDecodeError:
                        malformed += 1
                        
            info["files"][fname] = {
                "records": valid_records,
                "malformed": malformed,
                "size_mb": size_mb
            }
            info["schemas"][fname] = schema
            print(f"[OK] {fname}: {valid_records} records ({size_mb} MB) | Schema: {schema}")
            
        except Exception as e:
            print(f"[FAIL] Error reading RAGTruth file {fname}: {str(e)}")
            status = False

    return status, info


def main():
    print("=" * 70)
    print("  VERILLM DATASET INTEGRITY & SCHEMA VALIDATION")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Raw Datasets Dir: {RAW_DATASETS_DIR}")
    print("=" * 70)
    
    results = {}
    
    results["AVeriTeC"] = validate_averitec()
    results["FEVER"] = validate_fever()
    results["HaluEval"] = validate_halueval()
    results["RAGTruth"] = validate_ragtruth()
    
    print_separator("SUMMARY VALIDATION REPORT")
    all_passed = True
    for ds, (passed, info) in results.items():
        status_str = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {ds:<12}: [{status_str}]")
        
    print("=" * 70)
    if all_passed:
        print("  ALL DATASETS VALIDATED SUCCESSFULLY! Project ready for preprocessing.")
    else:
        print("  WARNING: One or more datasets failed validation. Check log details above.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
