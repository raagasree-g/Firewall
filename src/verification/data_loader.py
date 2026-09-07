"""
VeriLLM Verification Data Loader
Location: src/verification/data_loader.py

Provides clean, reproducible data loading and splitting for Task 1 (FEVER) and Task 2 (AVeriTeC).
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")
# AVeriTeC also preserves its source-specific CONFLICTING_EVIDENCE label.
# NLI still emits only the three operational VeriLLM verdicts.
VALID_LABELS = {"SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "CONFLICTING_EVIDENCE"}


def format_model_input(claim: str, evidence: str) -> str:
    """Format claim and evidence into a structured input string."""
    claim_str = (claim or "").strip()
    ev_str = (evidence or "").strip()
    return f"Claim: {claim_str} [SEP] Evidence: {ev_str}"


def validate_label(label: str) -> str:
    """Validate the shared VeriLLM label vocabulary before model use."""
    if label not in VALID_LABELS:
        raise ValueError(f"Unsupported verification label: {label!r}")
    return label


def load_fever_data(processed_dir: str = PROCESSED_DIR, val_size: float = 0.2, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load FEVER normalized data and return stratified train/val splits."""
    fpath = os.path.join(processed_dir, "FEVER", "fever_normalized.jsonl")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"FEVER dataset not found at {fpath}")

    records = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                validate_label(item.get("label"))
                item["formatted_input"] = format_model_input(item.get("claim"), item.get("evidence"))
                records.append(item)

    labels = [r["label"] for r in records]
    train_records, val_records = train_test_split(
        records, test_size=val_size, random_state=seed, stratify=labels
    )
    return train_records, val_records


def load_averitec_data(processed_dir: str = PROCESSED_DIR) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load AVeriTeC dataset split strictly into train (train.json) and dev (dev.json)."""
    fpath = os.path.join(processed_dir, "AVeriTeC", "averitec_normalized.jsonl")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"AVeriTeC dataset not found at {fpath}")

    train_records = []
    dev_records = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                validate_label(item.get("label"))
                item["formatted_input"] = format_model_input(item.get("claim"), item.get("evidence"))
                split = item.get("metadata", {}).get("split")
                if split == "train":
                    train_records.append(item)
                elif split == "dev":
                    dev_records.append(item)

    return train_records, dev_records


if __name__ == "__main__":
    fever_train, fever_val = load_fever_data()
    averitec_train, averitec_dev = load_averitec_data()
    
    print(f"FEVER Train: {len(fever_train):,} | FEVER Val: {len(fever_val):,}")
    print(f"AVeriTeC Train: {len(averitec_train):,} | AVeriTeC Dev: {len(averitec_dev):,}")
