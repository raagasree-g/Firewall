# Stage 5: Hallucination Analysis and Evaluation

VeriLLM audits an existing LLM response. It extracts claims, retrieves evidence, and uses NLI to classify a claim as `SUPPORTED`, `CONTRADICTED`, or `UNSUPPORTED`. This is claim verification, not a RAG answer-generation system. In particular, `UNSUPPORTED` means the available evidence did not establish the claim; it is **not automatically HALLUCINATED**.

## Dataset boundaries

- **FEVER** supports three-way claim verification and its held-out validation split is used for bounded retrieval and retrieval-to-NLI checks.
- **AVeriTeC** supports real-world claim verification. Its train and dev records are used; raw `test.json` is never used for development or evaluation.
- **HaluEval** remains a separate binary response-level task (`CORRECT` / `HALLUCINATED`). The Stage 5 baseline is TF-IDF (word 1–2 grams) plus balanced logistic regression using a fixed seed and an 80/20 stratified split.
- **RAGTruth** remains a separate response-level hallucination dataset (`CLEAN` / `HALLUCINATED`). Its span metadata is retained. When source type metadata is missing, the analysis reports `UNKNOWN`; no type is fabricated and a hallucinated label is never remapped to `CONTRADICTED`.

## Retrieval and end-to-end protocol

For FEVER validation and AVeriTeC dev, the evaluator deterministically ranks records by a SHA-256 seed key and selects the first 200. The candidate corpus is the corresponding held-out split. An evidence hit means the exact whitespace-normalized gold evidence string occurs in the top K TF-IDF results. It reports Recall@1, @3, @5, @10, evidence hit rate (Recall@10), and retrieval failure rate (one minus Recall@10). The fixed subset size and seed are stored with each metric artifact.

The end-to-end experiment sends the top-1 retrieved evidence to the existing `cross-encoder/nli-distilroberta-base` verifier. It persists accuracy, macro precision, macro recall, macro F1, weighted F1, per-class metrics, predictions, and a confusion matrix. These results measure the retrieval-to-verdict chain rather than gold-evidence NLI alone.

## Explainability and review

The response pipeline records retrieved evidence and score, NLI verdict/confidence, a grounded explanation, taxonomy output, severity and reasons, and review flags. Severity and review flagging identify uncertain, low-retrieval, conflicting, and serious contradicted claims for human review. They are decision aids, not a composite reliability score.

## Confidence limitation

Existing Stage 3 classical prediction artifacts have no probabilities, so Brier score and expected calibration error are not reported for them. New NLI artifacts can contain model confidence; calibration diagnostics only operate when those measured values are present. No confidence values are invented.

## Limitations

TF-IDF is lexical and the bounded experiment does not represent an open-web retrieval benchmark. FEVER's normalized evidence can be structured source text rather than prose, which limits lexical retrieval. NLI labels are operational mappings and do not merge the source tasks or replace expert review.
