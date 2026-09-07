# Verification strategy

## Task definition

VeriLLM classifies a claim relative to supplied evidence as `SUPPORTED`,
`CONTRADICTED`, or `UNSUPPORTED`. Unsupported does **not** automatically mean
hallucinated: it means that the available evidence does not establish or refute
the claim.

## Datasets and separation

FEVER is used with a deterministic, stratified 80/20 train/validation split
(seed 42). AVeriTeC uses only its provided normalized `train` and `dev` splits.
`datasets/raw/AVeriTeC/test.json` is excluded from loading, training, tuning,
and evaluation. HaluEval and RAGTruth are reserved for later hallucination/RAG
work and are not folded into this verification benchmark.

## Preprocessing and baselines

Inputs are normalized as `Claim: <claim> [SEP] Evidence: <evidence>`. The
classical baseline uses word unigram/bigram TF-IDF (25,000 features, sublinear
TF, minimum document frequency 2) with class-balanced Logistic Regression or
Linear SVM. These offer reproducible in-domain reference results.

## NLI verifier

The sole transformer verifier is `cross-encoder/nli-distilroberta-base`.
It receives `(premise, hypothesis)` as `(evidence, claim)`, returning an NLI
label and max-class confidence. The operational conversion is:

| NLI label | VeriLLM verdict |
| --- | --- |
| ENTAILMENT | SUPPORTED |
| CONTRADICTION | CONTRADICTED |
| NEUTRAL | UNSUPPORTED |

NLI semantic mapping is an operational mapping and does not imply that the
dataset annotation schemes are identical.

AVeriTeC's source-specific `CONFLICTING_EVIDENCE` label is retained in its dev
evaluation rather than collapsed or fabricated into a three-way label. Because
the selected NLI model produces only the three labels above, it cannot predict
that fourth source label; this is reported as a compatibility limitation.

## Evaluation and controls

Both in-domain and cross-dataset experiments report accuracy, macro precision,
macro recall, macro F1, weighted F1, per-class F1, and confusion matrices.
Cross-dataset runs train on one source and evaluate on the other source's held
out split. Reproducible splits, explicit source-split filtering, and no test
file reads provide leakage controls. Metrics and prediction records are saved
separately so error analysis uses generated outputs rather than examples.

## Limitations

TF-IDF models rely on lexical overlap. NLI performance depends on evidence
quality, model context length, and compatibility between FEVER/AVeriTeC labels
and the three-way NLI task. A neutral result cannot identify the reason for
missing support, and neither approach establishes real-world truth beyond the
provided evidence.
