# VeriLLM core pipeline

VeriLLM audits an LLM response using retrieved evidence. It is not simply RAG
used to generate a better response.

1. A deterministic sentence extractor yields structured, ordered claims. It is
   deliberately not presented as perfect atomic-claim decomposition.
2. The offline corpus loader reads existing processed FEVER or AVeriTeC JSONL
   data only, normalizes whitespace, and de-duplicates evidence.
3. A TF-IDF retriever ranks evidence by cosine similarity and exposes `top_k`.
4. The existing Stage 3 NLI verifier evaluates each usable claim/evidence pair.
   Evidence is the premise and the claim is the hypothesis; entailment maps to
   supported, contradiction to contradicted, and neutral to unsupported.
5. The pipeline selects the highest retrieval-weighted NLI confidence. Mixed
   supported/contradicted signals are retained as a conflict and return
   `UNSUPPORTED` rather than an invented resolution.
6. Deterministic explanations state only the retrieved-evidence outcome.

The response summary counts supported, contradicted, and unsupported claims;
it deliberately has no reliability score. Unsupported means insufficient
verification from retrieved evidence, not necessarily hallucination.

Limitations: sentence claims can combine propositions; lexical TF-IDF may miss
semantic matches; evidence corpus coverage constrains results; and NLI labels
are an operational mapping rather than proof that source schemes are identical.
