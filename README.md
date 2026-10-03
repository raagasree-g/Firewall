# VeriLLM: An Explainable RAG + ML Framework for LLM Reliability and Hallucination Detection

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

**VeriLLM** is an open, modular, empirical framework designed to evaluate, verify, and quantify Large Language Model (LLM) reliability and hallucination severity. Built for both **Smart India Hackathon (SIH)** and **DNG Academic Research**, VeriLLM moves beyond naive API wrapping by centering core Natural Language Inference (NLI) and machine learning models to detect factual hallucinations with granular explainability.

---

## 🎯 Key Objectives

1. **Deconstruct LLM Outputs**: Extract atomic factual claims from generated responses.
2. **RAG Evidence Retrieval**: Fetch contextually relevant source evidence.
3. **ML/NLI Verification**: Classify each claim into distinct categories:
   - `SUPPORTED`
   - `CONTRADICTED`
   - `UNSUPPORTED / UNVERIFIABLE`
4. **Hallucination & Severity Scoring**: Distinguish between unverified context versus explicit refutation with mathematical formulation.
5. **Granular Explainability**: Map each verdict back to concrete evidence sentences and confidence metrics.
6. **Model Benchmarking & Regression Tracking**: Compare multiple LLM architectures and track reliability drift across prompt and RAG revisions.

---

## 🏗️ Pipeline Architecture

```
                       ┌─────────────────────────┐
                       │     LLM APPLICATION     │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │   GENERATED RESPONSE    │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │    CLAIM EXTRACTION     │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │    INDIVIDUAL CLAIMS    │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │  RAG EVIDENCE RETRIEVAL │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │        EVIDENCE         │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │   ML / NLI VERIFICATION │
                       └────────────┬────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
   ┌────────▼────────┐     ┌────────▼────────┐    ┌─────────▼────────┐
   │    SUPPORTED    │     │  CONTRADICTED   │    │   UNSUPPORTED    │
   └────────┬────────┘     └────────┬────────┘    └─────────┬────────┘
            └───────────────────────┼───────────────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │  HALLUCINATION ANALYSIS │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │  SEVERITY + ERROR TYPE  │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │     EXPLAINABILITY      │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │    RELIABILITY ENGINE   │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │    MODEL COMPARISON     │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │   REGRESSION TESTING    │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │   DEVELOPER DASHBOARD   │
                       └─────────────────────────┘
```

---

## 📊 Datasets

VeriLLM utilizes four premier fact verification and hallucination detection benchmarks:

| Dataset | Split / Files | Description & Schema Focus |
| :--- | :--- | :--- |
| **AVeriTeC** | `train.json`, `dev.json`, `test.json` | Real-world complex claims requiring multi-step question decomposition and web evidence. Stage 7 bounded evaluation uses dev only; test data is untouched. |
| **FEVER** | `train.jsonl` | Large-scale claim verification against Wikipedia ground truth. |
| **HaluEval** | `qa_data.json`, `summarization_data.json`, `dialogue_data.json`, `general_data.json` | Comprehensive LLM hallucination benchmark spanning QA, Dialogue, Summarization, and General query responses. |
| **RAGTruth** | `response.jsonl`, `source_info.jsonl` | Fine-grained RAG hallucination dataset with word/span-level annotations across multiple models and temperatures. |

---

## 🚀 Environment Setup & Installation

### Prerequisites
- Python 3.11.0

### 1. Virtual Environment Activation
Create and activate the dedicated virtual environment `.venv`:

**Windows (PowerShell / Command Prompt):**
```powershell
# Create environment (if not already present)
py -3.11 -m venv .venv

# Activate environment
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Dependency Installation
```bash
pip install -r requirements.txt
```

---

## 🔍 Dataset Validation

To verify raw dataset file integrity, schema compliance, and line syntax without altering raw files:

```bash
python src/preprocessing/validate_datasets.py
```

---

## 📁 Repository Structure

```
VeriLLM/
├── datasets/
│   ├── raw/                  # Preserved raw dataset downloads
│   └── processed/            # Preprocessed datasets (generated during pipeline)
├── src/                      # Core modular Python package
│   ├── preprocessing/        # Cleaners, tokenizers, schema normalizers
│   ├── claim_extraction/     # Claim extraction & decomposition
│   ├── retrieval/            # Offline TF-IDF evidence retrieval
│   ├── verification/         # NLI & ML classifiers
│   ├── scoring/              # Reliability & severity formulas
│   ├── evaluation/           # Metrics calculation
│   └── utils/                # Helper utilities
├── models/                   # Local model weights & checkpoints
├── notebooks/                # Exploratory & analysis Jupyter notebooks
├── backend/                  # API server implementation
├── frontend/                 # Developer dashboard UI
├── tests/                    # Unit, integration, and evaluation tests
├── docs/                     # SIH & DNG specific documentation
│   ├── SIH/                  # Hackathon pitch & technical docs
│   └── DNG/                  # Academic paper & research docs
├── results/                  # Evaluation run logs & prediction outputs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📅 Current Development Stage

- [x] Environment & directory setup
- [x] Raw dataset organization & extraction
- [x] Schema & integrity validation script
- [x] Exploratory data analysis and schema normalization
- [x] Baseline ML / NLI classifier development
- [x] Offline retrieval, hallucination taxonomy, severity and reliability scoring
- [x] Model comparison and regression monitoring
- [x] Stage 7 retrieval/NLI robustness evaluation

## Running locally

Activate the virtual environment, then run the test suite:

```powershell
python -m pytest -v
```

Run the Stage 7 reproducibility evaluations (uses the locally cached
`cross-encoder/nli-distilroberta-base` model and does not use AVeriTeC test
data):

```powershell
python scripts/run_adversarial_evaluation.py
python scripts/run_stage7_evaluations.py
```

The default pipeline in `src/verillm_pipeline.py` extracts sentence claims,
retrieves evidence using robust TF-IDF, gates evidence below relevance 0.50,
verifies usable evidence with NLI, and returns explanations, taxonomy labels,
severity, review flags, and reliability-oriented outputs.

## Running the VeriLLM product UI

The project includes a lightweight research dashboard that connects to the
existing VeriLLM analysis pipeline without duplicating the ML logic in the
frontend.

Start the API and UI locally:

```powershell
# from the project root
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Then open:

```text
http://127.0.0.1:8000/
```

The frontend posts to `/api/verify`, which runs the live VeriLLM pipeline and
returns claim-level evidence, NLI verdicts, severity, and reliability summary
cards. If the full offline NLI model cache is unavailable, the backend silently
falls back to a lightweight local verification mode so the dashboard remains
functional.

## Current limitations

The retriever is lexical and the NLI model is frozen. The 0.50 relevance gate
prevents weak evidence from deciding a verdict but is conservative on the
broad-corpus bounded evaluation. Stage 7's controlled robustness suite is a
synthetic framework check, not an external benchmark. See
`docs/architecture/retrieval_nli_robustness.md` for measured results and
per-class limitations.
