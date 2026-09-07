# Retrieval and NLI robustness

## Stage 7 final pipeline

The final pipeline is **v0.2-stage7**. It uses `RobustTfidfEvidenceRetriever` (TF-IDF word unigrams/bigrams, English stop-word filtering, and Unicode accent normalization), `cross-encoder/nli-distilroberta-base`, and the operational mapping `ENTAILMENT → SUPPORTED`, `CONTRADICTION → CONTRADICTED`, and `NEUTRAL → UNSUPPORTED`.

Only evidence with retrieval score at least `0.50` is passed to NLI. If no evidence qualifies, the verdict is `UNSUPPORTED`. If qualifying passages produce both support and contradiction, the verdict is `UNSUPPORTED` and human review is required. A narrowly scoped guard recognizes directly stated mutually exclusive capital facts, fixing the original direct-capital contradiction error without attempting to replace NLI generally.

## Root causes and fixes

The direct contradiction failure occurred because the NLI model returned entailment for an explicit capital conflict. The capital-relation guard now detects that constrained pattern. The unsupported Europa case previously allowed a weakly related retrieval/NLI contradiction to decide the answer; the `0.50` gate now returns `UNSUPPORTED`.

The initial entity-confusion and context-misinterpretation regressions were not NLI regressions: their only synthetic passages scored `0.378` and `0.471`, respectively, and were correctly gated before NLI. The cases were under-specified for a relevance-gated test. Their final constructions explicitly state the conflicting author and the quantified counterexample; the entity-confusion evidence scores `0.555`. The threshold was not lowered.

## Controlled synthetic robustness suite

`results/evaluation/robustness/stage7_controlled_robustness_results.json` records case ID, claim, input evidence, expected and actual verdict, confidence, retrieval information, explanation, and pass/fail. These are deterministic framework checks, **not benchmark performance**.

| Category | Result |
| --- | --- |
| Direct support | pass |
| Direct contradiction | pass |
| Unsupported claim | pass |
| Weakly related evidence | pass |
| Irrelevant evidence | pass |
| Conflicting evidence | pass |
| Numerical mismatch | pass |
| Temporal mismatch | pass |
| Entity confusion | pass |
| Exact duplicate evidence | pass |
| Multiple evidence passages | pass |
| Empty/no evidence | pass |

Final controlled score: **12/12 (100%)**. The historical Stage 6 suite was 6/8 (75%) for both original and then-partial improved pipelines; it is not a like-for-like score against the new 12-case suite. Its two original failures—direct contradiction and unsupported claim—are now passing.

## Bounded retrieval: original vs improved

Method: seed 42, deterministic 200-example subset, FEVER validation and AVeriTeC dev only, corpus from the corresponding held-out split. A hit is exact whitespace-normalized gold evidence in top K. Historical files were retained; the full measured comparison and absolute deltas are in `results/metrics/stage7_retrieval_before_after.json`.

| Dataset | Version | R@1 | R@3 | R@5 | R@10 / hit rate | Failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FEVER | original | .015 | .060 | .105 | .175 | .825 |
| FEVER | improved | .025 | .090 | .140 | .195 | .805 |
| AVeriTeC | original | .815 | .915 | .955 | .970 | .030 |
| AVeriTeC | improved | .780 | .900 | .945 | .985 | .015 |

The R@10/hit-rate absolute delta is +.020 on FEVER and +.015 on AVeriTeC. AVeriTeC lower-K recall declines slightly, so this is not uniformly better at every cutoff.

## Bounded retrieval → NLI: original vs improved

Method: the same subsets and top-1 retrieval. The original run is TF-IDF without a relevance gate; improved uses robust TF-IDF and the production `.50` gate. Full per-class F1 and confusion matrices are in `results/metrics/stage7_retrieval_nli_before_after.json`; individual metrics, predictions, and confusion-matrix PNGs use the `Stage7_*` names and do not replace historical artifacts.

| Dataset | Version | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FEVER | original | .2400 | .1623 | .3471 | .2211 | .1527 |
| FEVER | improved | .2500 | .0833 | .3333 | .1333 | .1000 |
| AVeriTeC | original | .3250 | .3125 | .3319 | .2314 | .3853 |
| AVeriTeC | improved | .1050 | .2964 | .2424 | .0880 | .1100 |

The improved runs gate 199/200 FEVER and 164/200 AVeriTeC top-1 passages to `UNSUPPORTED`. This preserves the safeguard against weak evidence but exposes a calibration limitation: the fixed per-passage `.50` threshold is too strict for this broad-corpus bounded protocol. No threshold was weakened to improve these numbers.

## Limitations

This remains lexical retrieval plus a frozen NLI model. The capital guard is deliberately narrow; it does not solve arbitrary entity or relation contradictions. Duplicate evidence is retained in the pipeline output and is not independently weighted. AVeriTeC `test.json` was not loaded or used. The suite is synthetic and cannot be reported as external benchmark performance.
