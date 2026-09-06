"""
VeriLLM EDA Visualizations & Artifact Generator (Task-Aware Version)
Location: scripts/generate_eda_artifacts.py

Generates exploratory data analysis plots reflecting 4 distinct task types:
- CLAIM_VERIFICATION (FEVER)
- REAL_WORLD_CLAIM_VERIFICATION (AVeriTeC)
- RESPONSE_HALLUCINATION_DETECTION (HaluEval)
- RAG_HALLUCINATION_DETECTION (RAGTruth)
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

datasets = ["FEVER", "AVeriTeC", "HaluEval", "RAGTruth"]
data_by_ds = {}

for ds in datasets:
    fpath = os.path.join(PROCESSED_DIR, ds, f"{ds.lower()}_normalized.jsonl")
    records = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    data_by_ds[ds] = records

# 1. Dataset & Task Size Comparison Plot
fig, ax = plt.subplots(figsize=(9, 5))
ds_names = ["FEVER\n(Claim Verif.)", "AVeriTeC\n(Real-World Verif.)", "HaluEval\n(Resp. Halluc.)", "RAGTruth\n(RAG Spans)"]
ds_sizes = [len(data_by_ds[ds]) for ds in datasets]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

bars = ax.bar(ds_names, ds_sizes, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
ax.set_ylabel('Number of Normalized Records (Log Scale)', fontsize=11, fontweight='bold')
ax.set_title('VeriLLM Task-Aware Dataset Size Comparison', fontsize=13, fontweight='bold', pad=15)
ax.set_yscale('log')
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:,}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "dataset_size_comparison.png"), dpi=300)
plt.close()
print("[OK] Saved results/dataset_size_comparison.png")

# 2. Task-Aware Label Distributions Plot
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

for idx, ds in enumerate(datasets):
    records = data_by_ds[ds]
    task_type = records[0]["task_type"]
    labels = [r['label'] for r in records]
    counts = Counter(labels)
    categories = list(counts.keys())
    values = [counts[c] for c in categories]
    
    ax = axes[idx]
    bars = ax.bar(categories, values, color=colors[idx], edgecolor='black', linewidth=1)
    ax.set_title(f'{ds} ({task_type})\nTotal: {len(labels):,} Records', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10)
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(labels)) * 100
        ax.annotate(f'{height:,}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "label_distributions.png"), dpi=300)
plt.close()
print("[OK] Saved results/label_distributions.png")

# 3. Text Length Distributions Plot
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

for idx, ds in enumerate(datasets):
    text_lens = [len(r['text'].split()) for r in data_by_ds[ds]]
    ax = axes[idx]
    sns.histplot(text_lens, ax=ax, bins=30, kde=True, color=colors[idx])
    ax.set_title(f'{ds} Evaluated Text Length (Words)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Word Count', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    
    median_val = np.median(text_lens)
    mean_val = np.mean(text_lens)
    ax.axvline(median_val, color='red', linestyle='--', linewidth=1.5, label=f'Median: {median_val:.0f}')
    ax.axvline(mean_val, color='black', linestyle=':', linewidth=1.5, label=f'Mean: {mean_val:.1f}')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "text_length_distributions.png"), dpi=300)
plt.close()
print("[OK] Saved results/text_length_distributions.png")
