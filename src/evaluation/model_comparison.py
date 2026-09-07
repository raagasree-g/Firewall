"""Compatibility-aware model comparison; incompatible runs are never ranked together."""
import json
from pathlib import Path

CORE = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")

def compatible(a, b):
    keys = ("dataset", "task", "split", "seed", "retrieval_configuration", "evidence_configuration")
    return all(a.get("metadata", {}).get(k) == b.get("metadata", {}).get(k) for k in keys)

def build_comparison(entries):
    if not entries: return {"entries": [], "comparable": False, "reason": "No entries."}
    same = all(compatible(entries[0], item) for item in entries[1:])
    return {"entries": entries, "comparable": same, "reason": None if same else "Runs have incompatible evaluation metadata; no ranking is asserted."}

def save_comparison(comparison, root, name):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    lines = ["# Model comparison", "", f"Comparable: **{comparison['comparable']}**", ""]
    if comparison["reason"]: lines += [comparison["reason"], ""]
    lines += ["| Model | Accuracy | Macro F1 | Weighted F1 |", "|---|---:|---:|---:|"]
    for e in comparison["entries"]:
        m=e.get("metrics", {}); lines.append(f"| {e['model_name']} | {m.get('accuracy', '—')} | {m.get('macro_f1', '—')} | {m.get('weighted_f1', '—')} |")
    (root / f"{name}.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
