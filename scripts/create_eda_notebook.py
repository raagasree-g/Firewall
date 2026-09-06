"""
Jupyter Notebook Generator for VeriLLM Stage 2 Revised EDA
Location: scripts/create_eda_notebook.py
"""

import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
NOTEBOOK_PATH = os.path.join(PROJECT_ROOT, "notebooks", "01_dataset_exploration", "01_eda_and_normalization.ipynb")

os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# VeriLLM Stage 2: Task-Aware Exploratory Data Analysis & Schema Normalization\n",
                "\n",
                "**Project**: VeriLLM: An Explainable RAG + ML Framework for LLM Reliability and Hallucination Detection  \n",
                "**Notebook**: `notebooks/01_dataset_exploration/01_eda_and_normalization.ipynb`  \n",
                "\n",
                "---  \n",
                "\n",
                "## 📌 Methodological Foundation & Task Taxonomy\n",
                "\n",
                "VeriLLM explicitly distinguishes between **Evidence-Based Factual Claim Verification** and **Response-Level Hallucination Detection**. Collapsing all datasets into a single 3-class (`SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`) classifier would corrupt task semantics, because an ungrounded hallucination in RAG is not necessarily an explicit evidence contradiction.\n",
                "\n",
                "### 🎯 The Four VeriLLM Task Semantics\n",
                "1. **`CLAIM_VERIFICATION` (FEVER)**: Atomic claim verification against Wikipedia evidence (`SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`).\n",
                "2. **`REAL_WORLD_CLAIM_VERIFICATION` (AVeriTeC)**: Complex real-world claim verification with multi-question decomposition (`SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`, `CONFLICTING_EVIDENCE`).\n",
                "3. **`RESPONSE_HALLUCINATION_DETECTION` (HaluEval)**: Paired or general LLM response fidelity (`CORRECT`, `HALLUCINATED`).\n",
                "4. **`RAG_HALLUCINATION_DETECTION` (RAGTruth)**: Full RAG output response and span-level grounding (`CLEAN`, `HALLUCINATED`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import json\n",
                "import pandas as pd\n",
                "from IPython.display import display, Image\n",
                "\n",
                "PROCESSED_DIR = '../../datasets/processed'\n",
                "RESULTS_DIR = '../../results'\n",
                "\n",
                "print('VeriLLM Task-Aware EDA Environment Active!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📊 1. Task-Level Dataset Overview & Summary Statistics"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "datasets = ['FEVER', 'AVeriTeC', 'HaluEval', 'RAGTruth']\n",
                "summary_data = []\n",
                "\n",
                "for ds in datasets:\n",
                "    fpath = os.path.join(PROCESSED_DIR, ds, f'{ds.lower()}_normalized.jsonl')\n",
                "    with open(fpath, 'r', encoding='utf-8') as f:\n",
                "        records = [json.loads(line) for line in f if line.strip()]\n",
                "        \n",
                "    df = pd.DataFrame(records)\n",
                "    df['text_words'] = df['text'].apply(lambda x: len(x.split()))\n",
                "    df['evidence_words'] = df['evidence'].apply(lambda x: len(x.split()) if x else 0)\n",
                "    \n",
                "    summary_data.append({\n",
                "        'Dataset': ds,\n",
                "        'Task Type': df['task_type'].iloc[0],\n",
                "        'Total Records': len(df),\n",
                "        'Canonical Label Distribution': dict(df['label'].value_counts()),\n",
                "        'Median Text Words': df['text_words'].median(),\n",
                "        'Mean Text Words': round(df['text_words'].mean(), 1),\n",
                "        'Median Evidence Words': df['evidence_words'].median()\n",
                "    })\n",
                "\n",
                "summary_df = pd.DataFrame(summary_data)\n",
                "display(summary_df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🖼️ 2. Task-Aware Visualizations\n",
                "\n",
                "### A. Dataset Size Comparison per Task"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "Image(filename=os.path.join(RESULTS_DIR, 'dataset_size_comparison.png'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### B. Canonical Label Distributions per Task"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "Image(filename=os.path.join(RESULTS_DIR, 'label_distributions.png'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### C. Evaluated Text Length Distributions"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "Image(filename=os.path.join(RESULTS_DIR, 'text_length_distributions.png'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🧬 3. VeriLLM Revised Canonical JSONL Schema\n",
                "\n",
                "```json\n",
                "{\n",
                "  \"sample_id\": \"AVERITEC_TRAIN_00001\",\n",
                "  \"text\": \"Hunter Biden had no experience in Ukraine or in the energy sector when he joined the board of Burisma.\",\n",
                "  \"claim\": \"Hunter Biden had no experience in Ukraine or in the energy sector when he joined the board of Burisma.\",\n",
                "  \"evidence\": \"Justification: Hunter Biden was appointed to the board of Burisma in 2014...\",\n",
                "  \"task_type\": \"REAL_WORLD_CLAIM_VERIFICATION\",\n",
                "  \"label\": \"SUPPORTED\",\n",
                "  \"original_label\": \"Supported\",\n",
                "  \"source_dataset\": \"AVeriTeC\",\n",
                "  \"metadata\": {\n",
                "    \"speaker\": \"Donald Trump\",\n",
                "    \"claim_types\": [\"Numerical claim\"],\n",
                "    \"questions\": [...]\n",
                "  }\n",
                "}\n",
                "```\n",
                "\n",
                "---  \n",
                "\n",
                "## 🧠 4. Why Datasets Must Not Be Concatenated Blindly\n",
                "1. **Semantic Divergence**: FEVER evaluates explicit factual truth vs. false context. HaluEval evaluates model output adherence. RAGTruth evaluates span grounding.\n",
                "2. **Granularity Variance**: Claims (9-15 words) vs LLM Responses (30-74 words).\n",
                "3. **Downstream ML Design**: VeriLLM will utilize specialized task heads and multi-task learning rather than a single naive 3-class model."
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"[OK] Revised notebook created: {NOTEBOOK_PATH}")
