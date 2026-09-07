# Reliability scoring

Verification outcomes are reported separately from risk rules and benchmark ground truth. Support, contradiction and unsupported rates divide the relevant verified claims; evidence coverage divides claims with retrieved evidence by all claims; mean confidence averages only stored probability confidences. Empty collections return zero rates/counts and unavailable means.

The optional experimental score is: 100 times clamp(0,1, .35S + .20E + .15C - .15K - .10U - .05(H+R)), where S=support rate, E=evidence coverage, C=mean verification confidence, K=contradiction rate, U=unsupported rate, H=high-severity rate, and R=critical-severity rate. Weights live in reliability_config.yaml, sum to one, and are intentionally exposed for sensitivity analysis. This is a framework metric, not ground truth or a validated scientific reliability measure.

Ground-truth hallucination rate is populated only from response-level human/benchmark labels (such as HaluEval); prediction-derived contradiction or unsupported rates are never called ground-truth hallucination.
