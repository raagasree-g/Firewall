# VeriLLM Dataset Strategy & Task-Aware Normalization Specification (Revised Stage 2)

## 1. Executive Summary & Core Mandate

**VeriLLM** ("An Explainable RAG + ML Framework for LLM Reliability and Hallucination Detection") relies on a task-aware data foundation. Rather than naively collapsing all datasets into a single 3-class (`SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`) classifier, VeriLLM explicitly distinguishes between **Evidence-Based Factual Claim Verification** and **Response-Level Hallucination Detection**.

An ungrounded hallucination in a RAG system is **not** semantically equivalent to an explicit evidence contradiction. Therefore, VeriLLM structures raw datasets into 4 distinct task types under a unified JSONL schema.

---

## 2. Four-Task Taxonomy & Conceptual Roles

| Dataset | Split / Files | Size (Raw) | Processed Records | Task Type (`task_type`) | Downstream VeriLLM Module |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FEVER** | `train.jsonl` | 145,449 | 145,449 | `CLAIM_VERIFICATION` | Pre-training NLI Baseline Classifiers |
| **AVeriTeC** | `train.json`, `dev.json` | 3,568 | 3,567 | `REAL_WORLD_CLAIM_VERIFICATION` | Multi-step QA Decomposition & Real-world Cross-Encoder |
| **HaluEval** | `qa`, `summarization`, `dialogue`, `general` | 34,507 | 64,507 | `RESPONSE_HALLUCINATION_DETECTION` | Task-Aware Response Hallucination Classifiers |
| **RAGTruth** | `response.jsonl`, `source_info.jsonl` | 17,790 | 17,790 | `RAG_HALLUCINATION_DETECTION` | Fine-grained RAG Span Detectors & Severity Analysis |

> [!IMPORTANT]
> **Test Set Isolation**: `AVeriTeC/test.json` (2,215 claims) remains strictly excluded from normalization and training. It is preserved untouched under `datasets/raw/AVeriTeC/test.json` for final out-of-domain evaluation.

---

## 3. Label Mapping & Semantic Task Matrix

### Task 1: `CLAIM_VERIFICATION` (FEVER)
- `SUPPORTS` ──► **`SUPPORTED`**
- `REFUTES` ──► **`CONTRADICTED`**
- `NOT ENOUGH INFO` ──► **`UNSUPPORTED`**

### Task 2: `REAL_WORLD_CLAIM_VERIFICATION` (AVeriTeC)
- `Supported` ──► **`SUPPORTED`**
- `Refuted` ──► **`CONTRADICTED`**
- `Not Enough Evidence` ──► **`UNSUPPORTED`**
- `Conflicting Evidence/Cherrypicking` ──► **`CONFLICTING_EVIDENCE`** *(Preserves multi-source evidence ambiguity)*

### Task 3: `RESPONSE_HALLUCINATION_DETECTION` (HaluEval)
- `right` / non-hallucinated / `no` ──► **`CORRECT`**
- `hallucinated` / `yes` ──► **`HALLUCINATED`**

### Task 4: `RAG_HALLUCINATION_DETECTION` (RAGTruth)
- `labels == []` (Clean output) ──► **`CLEAN`**
- `labels != []` (Span hallucinations present) ──► **`HALLUCINATED`** *(All span objects preserved in metadata for span-level evaluation)*

---

## 4. Revised VeriLLM Canonical Schema

Every record in `datasets/processed/` follows this canonical schema:

```json
{
  "sample_id": "STRING (Unique sample identifier e.g. FEVER_TRAIN_000001, AVERITEC_TRAIN_00001)",
  "text": "STRING (The actual evaluated text or response)",
  "claim": "STRING or NULL (Atomic claim text when available, else null)",
  "evidence": "STRING (Retrieved evidence, justification, or source document text)",
  "task_type": "STRING (CLAIM_VERIFICATION | REAL_WORLD_CLAIM_VERIFICATION | RESPONSE_HALLUCINATION_DETECTION | RAG_HALLUCINATION_DETECTION)",
  "label": "STRING (Task-specific canonical label)",
  "original_label": "STRING (Original label from raw dataset)",
  "source_dataset": "STRING (FEVER | AVeriTeC | HaluEval | RAGTruth)",
  "metadata": "OBJECT (Preserved task metadata e.g. questions, hallucination_spans, model, temperature)"
}
```

---

## 5. Downstream Multi-Model & Task-Head Architecture

VeriLLM will **not** train a single naive 3-class classifier blindly on all 231,313 samples. Instead:

```
                          ┌───────────────────────────┐
                          │   INPUT GENERATED TEXT    │
                          └─────────────┬─────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           │                            │                            │
 ┌─────────▼──────────┐       ┌─────────▼──────────┐       ┌─────────▼──────────┐
 │    CLAIM EXTRACTION│       │    TASK RESPONSE   │       │    RAG GROUNDING   │
 │   & NLI VERIFICATION│       │    HALLUCINATION   │       │    SPAN DETECTOR   │
 │ (FEVER + AVeriTeC) │       │     (HaluEval)     │       │     (RAGTruth)     │
 └─────────┬──────────┘       └─────────┬──────────┘       └─────────┬──────────┘
           │                            │                            │
           └────────────────────────────┼────────────────────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │ VERILLM RELIABILITY ENGINE│
                          │   & SEVERITY CALCULATOR   │
                          └───────────────────────────┘
```

1. **Claim Extraction & NLI Verification Module**: Trained on FEVER & AVeriTeC for fine-grained factual claim status (`SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`, `CONFLICTING_EVIDENCE`).
2. **Response-Level Hallucination Module**: Fine-tuned on HaluEval for overall response fidelity (`CORRECT`, `HALLUCINATED`).
3. **RAG Grounding & Span Severity Module**: Fine-tuned on RAGTruth for token/span-level hallucination extraction (`Evident Conflict`, `Evident Baseless Info`, `Subtle Baseless Info`).
